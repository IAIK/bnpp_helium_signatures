#include "signature.h"

#include "field.h"
#include "rain.h"
#include "tape.h"
#include "tree.h"
#include <algorithm>
#include <cassert>
#include <cstring>

extern "C" {
#include "kdf_shake.h"
#include "randomness.h"
}

namespace {
template <typename GF>
inline void hash_update_GF2E(hash_context *ctx,
                             const signature_instance_t &instance,
                             const GF &element) {
  std::array<uint8_t, GF::BYTE_SIZE> buffer;
  element.to_bytes(buffer.data());
  hash_update(ctx, buffer.data(), GF::BYTE_SIZE);
}

std::pair<salt_t, std::vector<std::vector<uint8_t>>>
generate_salt_and_seeds(const signature_instance_t &instance,
                        const keypair_t &keypair, const uint8_t *message,
                        size_t message_len) {
  // salt, seed_1, ..., seed_r = H(instance||sk||pk||m)
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update_uint16_le(&ctx, (uint16_t)instance.params);
  hash_update(&ctx, keypair.first.data(), keypair.first.size());
  hash_update(&ctx, keypair.second.data(), keypair.second.size());
  hash_update(&ctx, message, message_len);
  hash_final(&ctx);

  salt_t salt;
  hash_squeeze(&ctx, salt.data(), salt.size());
  std::vector<std::vector<uint8_t>> seeds;
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    std::vector<uint8_t> s(instance.seed_size);
    hash_squeeze(&ctx, s.data(), s.size());
    seeds.push_back(s);
  }
  return std::make_pair(salt, seeds);
}

void commit_to_party_seed(const signature_instance_t &instance,
                          const gsl::span<uint8_t> &seed, const salt_t &salt,
                          size_t rep_idx, size_t party_idx,
                          gsl::span<uint8_t> commitment) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update_uint16_le(&ctx, (uint16_t)rep_idx);
  hash_update_uint16_le(&ctx, (uint16_t)party_idx);
  hash_update(&ctx, seed.data(), seed.size());
  hash_final(&ctx);

  hash_squeeze(&ctx, commitment.data(), commitment.size());
}

void commit_to_4_party_seeds(
    const signature_instance_t &instance, const gsl::span<uint8_t> &seed0,
    const gsl::span<uint8_t> &seed1, const gsl::span<uint8_t> &seed2,
    const gsl::span<uint8_t> &seed3, const salt_t &salt, size_t rep_idx,
    size_t party_idx, gsl::span<uint8_t> com0, gsl::span<uint8_t> com1,
    gsl::span<uint8_t> com2, gsl::span<uint8_t> com3) {
  hash_context_x4 ctx;
  hash_init_x4(&ctx, instance.digest_size);
  hash_update_x4_1(&ctx, salt.data(), salt.size());
  hash_update_x4_uint16_le(&ctx, (uint16_t)rep_idx);
  const uint16_t party_idxs[4] = {
      (uint16_t)party_idx, (uint16_t)(party_idx + 1), (uint16_t)(party_idx + 2),
      (uint16_t)(party_idx + 3)};
  hash_update_x4_uint16s_le(&ctx, party_idxs);
  hash_update_x4_4(&ctx, seed0.data(), seed1.data(), seed2.data(), seed3.data(),
                   instance.seed_size);
  hash_final_x4(&ctx);

  hash_squeeze_x4_4(&ctx, com0.data(), com1.data(), com2.data(), com3.data(),
                    instance.digest_size);
}

template <typename GF>
std::vector<uint8_t>
phase_1_commitment(const signature_instance_t &instance, const salt_t &salt,
                   const std::vector<uint8_t> &pk, const uint8_t *message,
                   size_t message_len, const RepByteContainer &commitments,
                   const std::vector<std::vector<uint8_t>> &key_deltas,
                   const std::vector<std::vector<GF>> &t_deltas,
                   const std::vector<GF> &c_deltas) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_1);
  hash_update(&ctx, message, message_len);
  hash_update(&ctx, pk.data(), pk.size());
  hash_update(&ctx, salt.data(), salt.size());

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto commitment = commitments.get(repetition, party);
      hash_update(&ctx, commitment.data(), commitment.size());
    }
    hash_update(&ctx, key_deltas[repetition].data(),
                key_deltas[repetition].size());
    for (size_t ell = 0; ell < instance.block_cipher_params.num_sboxes - 1;
         ell++) {
      hash_update_GF2E(&ctx, instance, t_deltas[repetition][ell]);
    }
    hash_update_GF2E(&ctx, instance, c_deltas[repetition]);
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

// returns vecs of \epsilon values
template <typename GF>
std::vector<std::vector<GF>>
phase_1_expand(const signature_instance_t &instance,
               const std::vector<uint8_t> &h_1) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_1.data(), h_1.size());
  hash_final(&ctx);

  std::array<uint8_t, GF::BYTE_SIZE> buffer;
  std::vector<std::vector<GF>> epsilons(instance.num_repetitions);
  for (size_t e = 0; e < instance.num_repetitions; e++) {
    epsilons[e].resize(instance.block_cipher_params.num_sboxes);
    for (size_t ell = 0; ell < instance.block_cipher_params.num_sboxes; ell++) {
      hash_squeeze(&ctx, buffer.data(), buffer.size());
      epsilons[e][ell].from_bytes(buffer.data());
    }
  }
  return epsilons;
}

template <typename GF>
std::vector<uint8_t>
phase_2_commitment(const signature_instance_t &instance, const salt_t &salt,
                   const std::vector<uint8_t> &h_1,
                   const RepContainer<GF> &alpha_shares,
                   const std::vector<std::vector<GF>> &v_shares) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_2);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update(&ctx, h_1.data(), h_1.size());

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      for (size_t ell = 0; ell < instance.block_cipher_params.num_sboxes;
           ell++) {
        hash_update_GF2E(&ctx, instance,
                         alpha_shares.get(repetition, party)[ell]);
      }
      hash_update_GF2E(&ctx, instance, v_shares[repetition][party]);
    }
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

std::vector<uint16_t> phase_2_expand(const signature_instance_t &instance,
                                     const std::vector<uint8_t> &h_2) {
  assert(instance.num_MPC_parties < (1ULL << 16));
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_2.data(), h_2.size());
  hash_final(&ctx);
  size_t num_squeeze_bytes = instance.num_MPC_parties > 256 ? 2 : 1;

  std::vector<uint16_t> opened_parties;
  uint16_t mask = (1ULL << ceil_log2(instance.num_MPC_parties)) - 1;
  for (size_t e = 0; e < instance.num_repetitions; e++) {
    uint16_t party;
    do {
      hash_squeeze(&ctx, (uint8_t *)&party, num_squeeze_bytes);
      party = le16toh(party);
      party = party & mask;
    } while (party >= instance.num_MPC_parties);
    opened_parties.push_back(party);
  }
  return opened_parties;
}
} // namespace

keypair_t rainier_keygen(const signature_instance_t &instance) {
  std::vector<uint8_t> key(instance.block_cipher_params.key_size),
      pt(instance.block_cipher_params.block_size),
      ct(instance.block_cipher_params.block_size);

  while (true) {
    rand_bytes(key.data(), key.size());
    rand_bytes(pt.data(), pt.size());
    // 3 round variants
    if (instance.block_cipher_params.key_size == 16 &&
        instance.block_cipher_params.num_sboxes == 3) {
      if (RAIN_128_3::rain(key, pt, ct)) {
        break;
      }
    } else if (instance.block_cipher_params.key_size == 24 &&
               instance.block_cipher_params.num_sboxes == 3) {
      if (RAIN_192_3::rain(key, pt, ct)) {
        break;
      }
    } else if (instance.block_cipher_params.key_size == 32 &&
               instance.block_cipher_params.num_sboxes == 3) {
      if (RAIN_256_3::rain(key, pt, ct)) {
        break;
      }
    }
    // 4 round variants
    else if (instance.block_cipher_params.key_size == 16 &&
             instance.block_cipher_params.num_sboxes == 4) {
      if (RAIN_128_4::rain(key, pt, ct)) {
        break;
      }
    } else if (instance.block_cipher_params.key_size == 24 &&
               instance.block_cipher_params.num_sboxes == 4) {
      if (RAIN_192_4::rain(key, pt, ct)) {
        break;
      }
    } else if (instance.block_cipher_params.key_size == 32 &&
               instance.block_cipher_params.num_sboxes == 4) {
      if (RAIN_256_4::rain(key, pt, ct)) {
        break;
      }
    } else
      throw std::runtime_error("invalid parameters");
  }
  keypair_t keypair;
  keypair.first = key;
  keypair.second = pt;
  keypair.second.insert(keypair.second.end(), ct.begin(), ct.end());
  return keypair;
}
template <typename GF>
signature_t<GF> rainier_sign_template(const signature_instance_t &instance,
                                      const keypair_t &keypair,
                                      const uint8_t *message,
                                      size_t message_len) {

  // so we dont have to type the long variant everytime
  const size_t L = instance.block_cipher_params.num_sboxes;

  // grab aes key, pt and ct
  std::vector<uint8_t> key = keypair.first;
  GF key_gf;
  key_gf.from_bytes(key.data());
  std::vector<uint8_t> pt_ct = keypair.second;
  const size_t total_pt_ct_size = instance.block_cipher_params.block_size;
  std::vector<uint8_t> pt(total_pt_ct_size), ct(total_pt_ct_size),
      ct2(total_pt_ct_size);
  memcpy(pt.data(), keypair.second.data(), pt.size());
  memcpy(ct.data(), keypair.second.data() + pt.size(), ct.size());

  // get sbox inputs and outputs for aes evaluation
  std::pair<std::vector<GF>, std::vector<GF>> sbox_pairs;

  // below implementation only makes sense for those atm
  static_assert(std::is_same<GF, field::GF2_128>::value ||
                std::is_same<GF, field::GF2_192>::value ||
                std::is_same<GF, field::GF2_256>::value);

  if constexpr (std::is_same<GF, field::GF2_128>::value) {
    if (instance.block_cipher_params.key_size == 16 &&
        instance.block_cipher_params.num_sboxes == 3)
      sbox_pairs = RAIN_128_3::rain_with_sbox_output(key, pt, ct2);
    else if (instance.block_cipher_params.key_size == 16 &&
             instance.block_cipher_params.num_sboxes == 4)
      sbox_pairs = RAIN_128_4::rain_with_sbox_output(key, pt, ct2);
    else
      throw std::runtime_error("invalid parameters");
  } else if constexpr (std::is_same<GF, field::GF2_192>::value) {
    if (instance.block_cipher_params.key_size == 24 &&
        instance.block_cipher_params.num_sboxes == 3)
      sbox_pairs = RAIN_192_3::rain_with_sbox_output(key, pt, ct2);
    else if (instance.block_cipher_params.key_size == 24 &&
             instance.block_cipher_params.num_sboxes == 4)
      sbox_pairs = RAIN_192_4::rain_with_sbox_output(key, pt, ct2);
    else
      throw std::runtime_error("invalid parameters");
  } else if constexpr (std::is_same<GF, field::GF2_256>::value) {
    if (instance.block_cipher_params.key_size == 32 &&
        instance.block_cipher_params.num_sboxes == 3)
      sbox_pairs = RAIN_256_3::rain_with_sbox_output(key, pt, ct2);
    else if (instance.block_cipher_params.key_size == 32 &&
             instance.block_cipher_params.num_sboxes == 4)
      sbox_pairs = RAIN_256_4::rain_with_sbox_output(key, pt, ct2);
    else
      throw std::runtime_error("invalid parameters");
  } else {
    throw std::runtime_error("no implementation for type");
  }

  // sanity check, incoming keypair is valid
  assert(ct == ct2);

#ifndef NDEBUG
  for (size_t ell = 0; ell < L; ell++) {
    assert(sbox_pairs.first[ell] * sbox_pairs.second[ell] == GF(1));
  }
#endif

  // generate salt and master seeds for each repetition
  auto [salt, master_seeds] =
      generate_salt_and_seeds(instance, keypair, message, message_len);

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  seed_trees.reserve(instance.num_repetitions);
  // key share + (L-1)*T_share + (L-1) * a_share + z_share
  const size_t random_tape_size = instance.block_cipher_params.key_size +
                                  (L - 1) * GF::BYTE_SIZE +
                                  (L - 1) * GF::BYTE_SIZE + GF::BYTE_SIZE;

  RandomTapes random_tapes(instance.num_repetitions, instance.num_MPC_parties,
                           random_tape_size);

  RepByteContainer party_seed_commitments(
      instance.num_repetitions, instance.num_MPC_parties, instance.digest_size);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    // generate seed tree for the N parties
    seed_trees.emplace_back(master_seeds[repetition], instance.num_MPC_parties,
                            salt, repetition);

    // commit to each party's seed;
    {
      size_t party = 0;
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {

        commit_to_4_party_seeds(
            instance, seed_trees[repetition].get_leaf(party).value(),
            seed_trees[repetition].get_leaf(party + 1).value(),
            seed_trees[repetition].get_leaf(party + 2).value(),
            seed_trees[repetition].get_leaf(party + 3).value(), salt,
            repetition, party, party_seed_commitments.get(repetition, party),
            party_seed_commitments.get(repetition, party + 1),
            party_seed_commitments.get(repetition, party + 2),
            party_seed_commitments.get(repetition, party + 3));
      }
      for (; party < instance.num_MPC_parties; party++) {
        commit_to_party_seed(
            instance, seed_trees[repetition].get_leaf(party).value(), salt,
            repetition, party, party_seed_commitments.get(repetition, party));
      }
    }

    // create random tape for each party
    {
      size_t party = 0;
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
        random_tapes.generate_4_tapes(
            repetition, party, salt,
            seed_trees[repetition].get_leaf(party).value(),
            seed_trees[repetition].get_leaf(party + 1).value(),
            seed_trees[repetition].get_leaf(party + 2).value(),
            seed_trees[repetition].get_leaf(party + 3).value());
      }
      for (; party < instance.num_MPC_parties; party++) {
        random_tapes.generate_tape(
            repetition, party, salt,
            seed_trees[repetition].get_leaf(party).value());
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  // phase 1: commit to MPC executions and polynomials
  /////////////////////////////////////////////////////////////////////////////
  RepByteContainer rep_shared_keys(instance.num_repetitions,
                                   instance.num_MPC_parties,
                                   instance.block_cipher_params.key_size);
  RepContainer<GF> rep_shared_s(instance.num_repetitions,
                                instance.num_MPC_parties, L);
  RepContainer<GF> rep_shared_t(instance.num_repetitions,
                                instance.num_MPC_parties, L);
  std::vector<std::vector<uint8_t>> rep_key_deltas;
  rep_key_deltas.reserve(instance.num_repetitions);
  std::vector<std::vector<GF>> rep_t_deltas;
  rep_t_deltas.reserve(instance.num_repetitions);

  RepContainer<GF> rep_shared_dot_a(instance.num_repetitions,
                                    instance.num_MPC_parties, L - 1);
  RepContainer<GF> rep_shared_dot_c(instance.num_repetitions,
                                    instance.num_MPC_parties, 1);
  std::vector<GF> rep_c_deltas;
  rep_c_deltas.reserve(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {

    // generate sharing of secret key
    std::vector<uint8_t> key_delta = key;
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_key = rep_shared_keys.get(repetition, party);
      auto random_share =
          random_tapes.get_bytes(repetition, party, 0, shared_key.size_bytes());
      std::copy(std::begin(random_share), std::end(random_share),
                std::begin(shared_key));
      std::transform(std::begin(shared_key), std::end(shared_key),
                     std::begin(key_delta), std::begin(key_delta),
                     std::bit_xor<uint8_t>());
    }

    // fix first share
    auto first_share_key = rep_shared_keys.get(repetition, 0);
    std::transform(std::begin(key_delta), std::end(key_delta),
                   std::begin(first_share_key), std::begin(first_share_key),
                   std::bit_xor<uint8_t>());

    rep_key_deltas.push_back(key_delta);
    // generate sharing of t values
    std::vector<GF> t_deltas = sbox_pairs.second;
    // we no longer need the last t_delta
    t_deltas.pop_back();

    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_t = rep_shared_t.get(repetition, party);
      auto random_t_shares = random_tapes.get_bytes(
          repetition, party, instance.block_cipher_params.key_size,
          (L - 1) * instance.block_cipher_params.block_size);
      for (size_t ell = 0; ell < L - 1; ell++) {
        shared_t[ell].from_bytes(random_t_shares.data() +
                                 ell * instance.block_cipher_params.block_size);
      }
      for (size_t ell = 0; ell < L - 1; ell++) {
        t_deltas[ell] += shared_t[ell];
      }
    }
    // fix first share
    auto first_shared_t = rep_shared_t.get(repetition, 0);
    for (size_t ell = 0; ell < L - 1; ell++) {
      first_shared_t[ell] -= t_deltas[ell];
    }

    // get shares of sbox inputs by simulating MPC execution
    auto shared_s = rep_shared_s.get_repetition(repetition);
    auto shared_t = rep_shared_t.get_repetition(repetition);

    if constexpr (std::is_same<GF, field::GF2_128>::value) {
      if (instance.block_cipher_params.key_size == 16 &&
          instance.block_cipher_params.num_sboxes == 3)
        RAIN_128_3::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else if (instance.block_cipher_params.key_size == 16 &&
               instance.block_cipher_params.num_sboxes == 4)
        RAIN_128_4::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else
        throw std::runtime_error("invalid parameters");
    } else if constexpr (std::is_same<GF, field::GF2_192>::value) {
      if (instance.block_cipher_params.key_size == 24 &&
          instance.block_cipher_params.num_sboxes == 3)
        RAIN_192_3::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else if (instance.block_cipher_params.key_size == 24 &&
               instance.block_cipher_params.num_sboxes == 4)
        RAIN_192_4::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else
        throw std::runtime_error("invalid parameters");
    } else if constexpr (std::is_same<GF, field::GF2_256>::value) {
      if (instance.block_cipher_params.key_size == 32 &&
          instance.block_cipher_params.num_sboxes == 3)
        RAIN_256_3::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else if (instance.block_cipher_params.key_size == 32 &&
               instance.block_cipher_params.num_sboxes == 4)
        RAIN_256_4::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else
        throw std::runtime_error("invalid parameters");
    } else {
      throw std::runtime_error("no implementation for type");
    }

#ifndef NDEBUG
    // sanity check, all middle s and t values multiply to 1, and other ones are
    // according to opt c.5
    for (size_t ell = 0; ell < L; ell++) {
      GF test_S, test_T;
      for (size_t party = 0; party < instance.num_MPC_parties; party++) {
        test_S += rep_shared_s.get_repetition(repetition)[party][ell];
        test_T += rep_shared_t.get_repetition(repetition)[party][ell];
      }
      if (ell == 0 || ell == L - 1)
        assert(test_S * key_gf == test_T);
      else
        assert(test_S * test_T == GF(1));
    }
#endif
    rep_t_deltas.push_back(t_deltas);
  }

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {

    // also generate valid dot triple a,b,c and save c_delta
    // note that due to opt c5, the first b is k, the remaining ones are t
    std::vector<GF> a(L - 1);
    GF c(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto random_triple_bytes = random_tapes.get_bytes(
          repetition, party,
          instance.block_cipher_params.key_size + (L - 1) * GF::BYTE_SIZE,
          L * GF::BYTE_SIZE);
      auto a_share = rep_shared_dot_a.get(repetition, party);
      for (size_t ell = 0; ell < L - 1; ell++) {
        a_share[ell].from_bytes(random_triple_bytes.data() +
                                GF::BYTE_SIZE * ell);
        a[ell] += a_share[ell];
      }
      auto c_share = rep_shared_dot_c.get(repetition, party);
      c_share[0].from_bytes(random_triple_bytes.data() +
                            GF::BYTE_SIZE * (L - 1));
      c -= c_share[0];
    }

    // calculate c_delta = -c + a*b, c is negated above already
    // note that due to opt c5, the first b is k, the remaining ones are t
    c += a[0] * key_gf;
    for (size_t ell = 1; ell < L - 1; ell++) {
      c += a[ell] * sbox_pairs.second[ell];
    }
    // calculate c_delta that fixes the dot triple
    rep_c_deltas[repetition] = c;
    // fix party 0's share
    rep_shared_dot_c.get(repetition, 0)[0] += rep_c_deltas[repetition];
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 2: challenge the checking polynomials
  /////////////////////////////////////////////////////////////////////////////

  // commit to salt, (all commitments of parties seeds,
  // output_broadcasts, key_delta, t_delta, P_delta) for all
  // repetitions
  std::vector<uint8_t> h_1 = phase_1_commitment(
      instance, salt, keypair.second, message, message_len,
      party_seed_commitments, rep_key_deltas, rep_t_deltas, rep_c_deltas);

  // expand challenge hash to epsilon values
  std::vector<std::vector<GF>> epsilons = phase_1_expand<GF>(instance, h_1);

  /////////////////////////////////////////////////////////////////////////////
  // phase 3: commit to the views of the checking protocol
  /////////////////////////////////////////////////////////////////////////////

  RepContainer<GF> rep_alpha_shares(instance.num_repetitions,
                                    instance.num_MPC_parties, L);
  std::vector<std::vector<GF>> v_shares(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    v_shares[repetition].resize(instance.num_MPC_parties);

    auto s_shares = rep_shared_s.get_repetition(repetition);
    auto t_shares = rep_shared_t.get_repetition(repetition);
    auto dot_a_shares = rep_shared_dot_a.get_repetition(repetition);
    auto dot_c_share = rep_shared_dot_c.get_repetition(repetition);
    auto alpha_shares = rep_alpha_shares.get_repetition(repetition);

#ifndef NDEBUG
    // sanity check: a*t=c
    GF acc(0), c(0);
    std::vector<GF> a(L);
    for (size_t ell = 0; ell < L - 1; ell++) {
      for (size_t party = 0; party < instance.num_MPC_parties; party++) {
        a[ell] += dot_a_shares[party][ell];
      }
      if (ell == 0)
        acc += a[ell] * key_gf;
      else
        acc += a[ell] * sbox_pairs.second[ell];
    }
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      c += dot_c_share[party][0];
    }
    assert(acc == c);
#endif

    // execute sacrificing check protocol
    // alpha_j^i = eps_j * s_j^i + a_j^i
    std::vector<GF> alphas(L - 1);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      // first one is different due to opt c5
      {
        alpha_shares[party][0] =
            s_shares[party][0] * epsilons[repetition][0] +
            s_shares[party][L - 1] * epsilons[repetition][L - 1] +
            dot_a_shares[party][0];
        alphas[0] += alpha_shares[party][0];
      }
      for (size_t ell = 1; ell < L - 1; ell++) {
        alpha_shares[party][ell] =
            s_shares[party][ell] * epsilons[repetition][ell] +
            dot_a_shares[party][ell];
        alphas[ell] += alpha_shares[party][ell];
      }
    }
    // v^i = dot(eps, 1) - c^i + dot(alpha, t^i)
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      v_shares[repetition][party] -= dot_c_share[party][0];
      for (size_t ell = 1; ell < L - 1; ell++) {
        v_shares[repetition][party] -= alphas[ell] * t_shares[party][ell];
      }
      GF key_share;
      key_share.from_bytes(rep_shared_keys.get(repetition, party).data());
      v_shares[repetition][party] -= alphas[0] * key_share;
      // add repeated multiplier shares here
      v_shares[repetition][party] +=
          epsilons[repetition][0] * t_shares[party][0] +
          epsilons[repetition][L - 1] * t_shares[party][L - 1];
    }
    // remaining ones where z = 1 are only for party 1
    for (size_t ell = 1; ell < L - 1; ell++) {
      v_shares[repetition][0] += epsilons[repetition][ell];
    }

#ifndef NDEBUG
    for (size_t ell = 1; ell < L - 1; ell++) {
      assert(alphas[ell] ==
             epsilons[repetition][ell] * sbox_pairs.first[ell] - a[ell]);
    }
    // sanity check: vs are zero
    GF v(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      v += v_shares[repetition][party];
    }
    assert(v == GF(0));
#endif
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 4: challenge the views of the checking protocol
  /////////////////////////////////////////////////////////////////////////////

  std::vector<uint8_t> h_2 =
      phase_2_commitment(instance, salt, h_1, rep_alpha_shares, v_shares);

  std::vector<uint16_t> missing_parties = phase_2_expand(instance, h_2);

  /////////////////////////////////////////////////////////////////////////////
  // phase 5: Open the views of the checking protocol
  /////////////////////////////////////////////////////////////////////////////
  std::vector<reveal_list_t> seeds;
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    seeds.push_back(
        seed_trees[repetition].reveal_all_but(missing_parties[repetition]));
  }
  // build signature
  std::vector<repetition_proof_t<GF>> proofs;
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    size_t missing_party = missing_parties[repetition];
    std::vector<uint8_t> commitment(instance.digest_size);
    auto missing_commitment =
        party_seed_commitments.get(repetition, missing_party);
    std::copy(std::begin(missing_commitment), std::end(missing_commitment),
              std::begin(commitment));
    auto alpha_values = rep_alpha_shares.get(repetition, missing_party);
    std::vector<GF> missing_alphas(alpha_values.begin(), alpha_values.end());
    repetition_proof_t<GF> proof{
        seeds[repetition],        commitment,     rep_key_deltas[repetition],
        rep_t_deltas[repetition], missing_alphas, rep_c_deltas[repetition],
    };
    proofs.push_back(proof);
  }

  signature_t<GF> signature{salt, h_1, h_2, proofs};

  return signature;
}

template <typename GF>
bool rainier_verify_template(const signature_instance_t &instance,
                             const std::vector<uint8_t> &pk,
                             const signature_t<GF> &signature,
                             const uint8_t *message, size_t message_len) {

  // so we dont have to type the long variant everytime
  const size_t L = instance.block_cipher_params.num_sboxes;

  std::vector<uint8_t> pt(instance.block_cipher_params.block_size),
      ct(instance.block_cipher_params.block_size);
  memcpy(pt.data(), pk.data(), pt.size());
  memcpy(ct.data(), pk.data() + pt.size(), ct.size());

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  // key share + (L-1)*T_share + (L-1) * a_share + c_share
  const size_t random_tape_size = instance.block_cipher_params.key_size +
                                  (L - 1) * GF::BYTE_SIZE +
                                  (L - 1) * GF::BYTE_SIZE + GF::BYTE_SIZE;

  RandomTapes random_tapes(instance.num_repetitions, instance.num_MPC_parties,
                           random_tape_size);
  RepByteContainer party_seed_commitments(
      instance.num_repetitions, instance.num_MPC_parties, instance.digest_size);

  // h1 expansion
  std::vector<std::vector<GF>> epsilons =
      phase_1_expand<GF>(instance, signature.h_1);
  // h2 expansion already happened in deserialize to get missing parties
  std::vector<uint16_t> missing_parties =
      phase_2_expand(instance, signature.h_2);

  // rebuild SeedTrees
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF> &proof = signature.proofs[repetition];
    // regenerate generate seed tree for the N parties (except the missing
    // one)
    if (missing_parties[repetition] != proof.reveallist.second)
      throw std::runtime_error(
          "modified signature between deserialization and verify");
    seed_trees.push_back(SeedTree(proof.reveallist, instance.num_MPC_parties,
                                  signature.salt, repetition));
    // commit to each party's seed, fill up missing one with data from proof
    {
      std::vector<uint8_t> dummy(instance.seed_size);
      size_t party = 0;
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
        auto seed0 = seed_trees[repetition].get_leaf(party).value_or(dummy);
        auto seed1 = seed_trees[repetition].get_leaf(party + 1).value_or(dummy);
        auto seed2 = seed_trees[repetition].get_leaf(party + 2).value_or(dummy);
        auto seed3 = seed_trees[repetition].get_leaf(party + 3).value_or(dummy);
        commit_to_4_party_seeds(
            instance, seed0, seed1, seed2, seed3, signature.salt, repetition,
            party, party_seed_commitments.get(repetition, party),
            party_seed_commitments.get(repetition, party + 1),
            party_seed_commitments.get(repetition, party + 2),
            party_seed_commitments.get(repetition, party + 3));
      }
      for (; party < instance.num_MPC_parties; party++) {
        if (party != missing_parties[repetition]) {
          commit_to_party_seed(instance,
                               seed_trees[repetition].get_leaf(party).value(),
                               signature.salt, repetition, party,
                               party_seed_commitments.get(repetition, party));
        }
      }
    }
    auto com =
        party_seed_commitments.get(repetition, missing_parties[repetition]);
    std::copy(std::begin(proof.Com_e), std::end(proof.Com_e), std::begin(com));

    // create random tape for each party, dummy one for missing party
    {
      size_t party = 0;
      std::vector<uint8_t> dummy(instance.seed_size);
      for (; party < (instance.num_MPC_parties / 4) * 4; party += 4) {
        random_tapes.generate_4_tapes(
            repetition, party, signature.salt,
            seed_trees[repetition].get_leaf(party).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 1).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 2).value_or(dummy),
            seed_trees[repetition].get_leaf(party + 3).value_or(dummy));
      }
      for (; party < instance.num_MPC_parties; party++) {
        random_tapes.generate_tape(
            repetition, party, signature.salt,
            seed_trees[repetition].get_leaf(party).value_or(dummy));
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  // recompute commitments to executions of block cipher
  /////////////////////////////////////////////////////////////////////////////
  RepByteContainer rep_shared_keys(instance.num_repetitions,
                                   instance.num_MPC_parties,
                                   instance.block_cipher_params.key_size);
  RepContainer<GF> rep_shared_s(instance.num_repetitions,
                                instance.num_MPC_parties,
                                instance.block_cipher_params.num_sboxes);
  RepContainer<GF> rep_shared_t(instance.num_repetitions,
                                instance.num_MPC_parties,
                                instance.block_cipher_params.num_sboxes);

  RepContainer<GF> rep_shared_dot_a(instance.num_repetitions,
                                    instance.num_MPC_parties, L - 1);
  RepContainer<GF> rep_shared_dot_c(instance.num_repetitions,
                                    instance.num_MPC_parties, 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF> &proof = signature.proofs[repetition];

    // generate sharing of secret key
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_key = rep_shared_keys.get(repetition, party);
      auto random_share =
          random_tapes.get_bytes(repetition, party, 0, shared_key.size_bytes());
      std::copy(std::begin(random_share), std::end(random_share),
                std::begin(shared_key));
    }

    // fix first share
    auto first_key_share = rep_shared_keys.get(repetition, 0);
    std::transform(std::begin(proof.sk_delta), std::end(proof.sk_delta),
                   std::begin(first_key_share), std::begin(first_key_share),
                   std::bit_xor<uint8_t>());

    // generate sharing of t values
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_t = rep_shared_t.get(repetition, party);
      auto random_t_shares = random_tapes.get_bytes(
          repetition, party, instance.block_cipher_params.key_size,
          (L - 1) * instance.block_cipher_params.block_size);
      for (size_t ell = 0; ell < L - 1; ell++) {
        shared_t[ell].from_bytes(random_t_shares.data() +
                                 ell * instance.block_cipher_params.block_size);
      }
    }
    // fix first share
    auto first_shared_t = rep_shared_t.get(repetition, 0);
    for (size_t ell = 0; ell < L - 1; ell++) {
      first_shared_t[ell] -= proof.t_delta[ell];
    }

    // get shares of sbox inputs by executing MPC RAIN
    auto shared_s = rep_shared_s.get_repetition(repetition);
    auto shared_t = rep_shared_t.get_repetition(repetition);

    if constexpr (std::is_same<GF, field::GF2_128>::value) {
      if (instance.block_cipher_params.key_size == 16 &&
          instance.block_cipher_params.num_sboxes == 3)
        RAIN_128_3::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else if (instance.block_cipher_params.key_size == 16 &&
               instance.block_cipher_params.num_sboxes == 4)
        RAIN_128_4::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else
        throw std::runtime_error("invalid parameters");
    } else if constexpr (std::is_same<GF, field::GF2_192>::value) {
      if (instance.block_cipher_params.key_size == 24 &&
          instance.block_cipher_params.num_sboxes == 3)
        RAIN_192_3::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else if (instance.block_cipher_params.key_size == 24 &&
               instance.block_cipher_params.num_sboxes == 4)
        RAIN_192_4::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else
        throw std::runtime_error("invalid parameters");
    } else if constexpr (std::is_same<GF, field::GF2_256>::value) {
      if (instance.block_cipher_params.key_size == 32 &&
          instance.block_cipher_params.num_sboxes == 3)
        RAIN_256_3::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else if (instance.block_cipher_params.key_size == 32 &&
               instance.block_cipher_params.num_sboxes == 4)
        RAIN_256_4::rain_mpc(rep_shared_keys.get_repetition(repetition),
                             shared_t, pt, ct, shared_s);
      else
        throw std::runtime_error("invalid parameters");
    } else {
      throw std::runtime_error("no implementation for type");
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute shares of dot product
  /////////////////////////////////////////////////////////////////////////////

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF> &proof = signature.proofs[repetition];

    // also generate valid dot triple a,t,c and save c_delta
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_parties[repetition]) {
        auto random_triple_bytes = random_tapes.get_bytes(
            repetition, party,
            instance.block_cipher_params.key_size + (L - 1) * GF::BYTE_SIZE,
            L * GF::BYTE_SIZE);
        auto a_share = rep_shared_dot_a.get(repetition, party);
        for (size_t ell = 0; ell < L - 1; ell++) {
          a_share[ell].from_bytes(random_triple_bytes.data() +
                                  GF::BYTE_SIZE * ell);
        }
        auto c_share = rep_shared_dot_c.get(repetition, party);
        c_share[0].from_bytes(random_triple_bytes.data() +
                              GF::BYTE_SIZE * (L - 1));
      }
    }

    // fix party 0's share
    if (0 != missing_parties[repetition]) {
      // adjust first share with delta from signature
      rep_shared_dot_c.get(repetition, 0)[0] += proof.c_delta;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute views of sacrificing checks
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF> rep_alpha_shares(instance.num_repetitions,
                                    instance.num_MPC_parties, L);
  std::vector<std::vector<GF>> v_shares(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF> &proof = signature.proofs[repetition];
    size_t missing_party = missing_parties[repetition];

    v_shares[repetition].resize(instance.num_MPC_parties);
    std::vector<GF> alphas(L - 1);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alpha_shares = rep_alpha_shares.get(repetition, party);
      auto a_shares = rep_shared_dot_a.get(repetition, party);
      auto s_shares = rep_shared_s.get(repetition, party);
      if (party != missing_party) {
        // index 0 is opt c.5
        alpha_shares[0] = s_shares[0] * epsilons[repetition][0] +
                          s_shares[L - 1] * epsilons[repetition][L - 1] +
                          a_shares[0];
        alphas[0] += alpha_shares[0];
        for (size_t ell = 1; ell < L - 1; ell++) {
          alpha_shares[ell] =
              s_shares[ell] * epsilons[repetition][ell] - a_shares[ell];
          alphas[ell] += alpha_shares[ell];
        }
      }
    }
    // fill missing shares
    auto missing_alphas = rep_alpha_shares.get(repetition, missing_party);
    for (size_t ell = 0; ell < L - 1; ell++) {
      missing_alphas[ell] = proof.missing_alpha_shares[ell];
      alphas[ell] += proof.missing_alpha_shares[ell];
    }
    // v^i = eps * c^i + z^i - alpha*y^i - beta*x^i - alpha*beta
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_party) {
        auto t_shares = rep_shared_t.get(repetition, party);
        auto c_share = rep_shared_dot_c.get(repetition, party);
        v_shares[repetition][party] -= c_share[0];
        for (size_t ell = 1; ell < L - 1; ell++) {
          v_shares[repetition][party] -= alphas[ell] * t_shares[ell];
        }
        GF key_share;
        key_share.from_bytes(rep_shared_keys.get(repetition, party).data());
        v_shares[repetition][party] -= alphas[0] * key_share;
        // add repeated multiplier shares here
        v_shares[repetition][party] +=
            epsilons[repetition][0] * t_shares[0] +
            epsilons[repetition][L - 1] * t_shares[L - 1];
      }
    }
    if (missing_party != 0) {
      for (size_t ell = 1; ell < L - 1; ell++) {
        v_shares[repetition][0] += epsilons[repetition][ell];
      }
    }
    // calculate missing shares as 0 - sum_{i!=missing} v^i
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_party) {
        v_shares[repetition][missing_party] -= v_shares[repetition][party];
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  // recompute h_1 and h_2
  /////////////////////////////////////////////////////////////////////////////
  std::vector<std::vector<uint8_t>> sk_deltas;
  std::vector<std::vector<GF>> t_deltas;
  std::vector<GF> c_deltas;
  sk_deltas.reserve(instance.num_repetitions);
  t_deltas.reserve(instance.num_repetitions);
  c_deltas.reserve(instance.num_repetitions);
  for (const repetition_proof_t<GF> &proof : signature.proofs) {
    sk_deltas.push_back(proof.sk_delta);
    t_deltas.push_back(proof.t_delta);
    c_deltas.push_back(proof.c_delta);
  }
  std::vector<uint8_t> h_1 =
      phase_1_commitment(instance, signature.salt, pk, message, message_len,
                         party_seed_commitments, sk_deltas, t_deltas, c_deltas);

  std::vector<uint8_t> h_2 = phase_2_commitment(instance, signature.salt, h_1,
                                                rep_alpha_shares, v_shares);
  // do checks
  if (memcmp(h_1.data(), signature.h_1.data(), h_1.size()) != 0) {
    return false;
  }
  if (memcmp(h_2.data(), signature.h_2.data(), h_2.size()) != 0) {
    return false;
  }

  return true;
}

template <typename GF>
std::vector<uint8_t>
rainier_serialize_signature(const signature_instance_t &instance,
                            const signature_t<GF> &signature) {
  std::vector<uint8_t> serialized;
  // prealloc signature size
  const size_t signature_size =
      signature.salt.size() +    // salt
      instance.digest_size * 2 + // h_1, h_2
      instance.num_repetitions * // tau x
          (ceil_log2(instance.num_MPC_parties) *
               instance.seed_size +                // merkle tree path
           instance.digest_size +                  // Com_e
           instance.block_cipher_params.key_size + // delta sk
           (instance.block_cipher_params.num_sboxes - 1) *
               GF::BYTE_SIZE + // delta t
           (instance.block_cipher_params.num_sboxes - 1) *
               GF::BYTE_SIZE + // alpha
           GF::BYTE_SIZE);     // delta_c
  serialized.reserve(signature_size);

  serialized.insert(serialized.end(), signature.salt.begin(),
                    signature.salt.end());
  serialized.insert(serialized.end(), signature.h_1.begin(),
                    signature.h_1.end());
  serialized.insert(serialized.end(), signature.h_2.begin(),
                    signature.h_2.end());

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF> &proof = signature.proofs[repetition];
    for (const std::vector<uint8_t> &seed : proof.reveallist.first) {
      serialized.insert(serialized.end(), seed.begin(), seed.end());
    }
    serialized.insert(serialized.end(), proof.Com_e.begin(), proof.Com_e.end());
    serialized.insert(serialized.end(), proof.sk_delta.begin(),
                      proof.sk_delta.end());
    size_t current_size = serialized.size();
    serialized.resize(current_size +
                      (instance.block_cipher_params.num_sboxes - 1) *
                          GF::BYTE_SIZE);
    for (size_t ell = 0; ell < instance.block_cipher_params.num_sboxes - 1;
         ell++) {
      proof.t_delta[ell].to_bytes(serialized.data() + current_size +
                                  ell * GF::BYTE_SIZE);
    }
    current_size = serialized.size();
    serialized.resize(current_size +
                      (instance.block_cipher_params.num_sboxes - 1) *
                          GF::BYTE_SIZE);
    for (size_t k = 0; k < instance.block_cipher_params.num_sboxes - 1; k++) {
      proof.missing_alpha_shares[k].to_bytes(serialized.data() + current_size +
                                             k * GF::BYTE_SIZE);
    }
    current_size = serialized.size();

    serialized.resize(current_size + GF::BYTE_SIZE);
    proof.c_delta.to_bytes(serialized.data() + current_size);
    current_size = serialized.size();
  }
  // calculation as expected
  assert(signature_size == serialized.size());
  return serialized;
}

template <typename GF>
signature_t<GF>
rainier_deserialize_signature(const signature_instance_t &instance,
                              const std::vector<uint8_t> &serialized) {

  size_t current_offset = 0;
  salt_t salt{};
  memcpy(salt.data(), serialized.data() + current_offset, salt.size());
  current_offset += salt.size();
  std::vector<uint8_t> h_1(instance.digest_size), h_2(instance.digest_size);
  memcpy(h_1.data(), serialized.data() + current_offset, h_1.size());
  current_offset += h_1.size();
  memcpy(h_2.data(), serialized.data() + current_offset, h_2.size());
  current_offset += h_2.size();
  std::vector<repetition_proof_t<GF>> proofs;
  proofs.reserve(instance.num_repetitions);

  std::vector<uint16_t> missing_parties = phase_2_expand(instance, h_2);
  size_t reveallist_size = ceil_log2(instance.num_MPC_parties);
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    reveal_list_t reveallist;
    reveallist.first.reserve(reveallist_size);
    reveallist.second = missing_parties[repetition];
    for (size_t i = 0; i < reveallist_size; i++) {
      std::vector<uint8_t> seed(instance.seed_size);
      memcpy(seed.data(), serialized.data() + current_offset, seed.size());
      current_offset += seed.size();
      reveallist.first.push_back(seed);
    }
    std::vector<uint8_t> Com_e(instance.digest_size);
    memcpy(Com_e.data(), serialized.data() + current_offset, Com_e.size());
    current_offset += Com_e.size();

    std::vector<uint8_t> sk_delta(instance.block_cipher_params.key_size);
    memcpy(sk_delta.data(), serialized.data() + current_offset,
           sk_delta.size());
    current_offset += sk_delta.size();

    GF tmp;
    std::vector<GF> t_delta;
    t_delta.reserve(instance.block_cipher_params.num_sboxes - 1);
    for (size_t ell = 0; ell < instance.block_cipher_params.num_sboxes - 1;
         ell++) {
      tmp.from_bytes(serialized.data() + current_offset);
      current_offset += GF::BYTE_SIZE;
      t_delta.push_back(tmp);
    }

    std::vector<GF> missing_alpha_shares;
    missing_alpha_shares.reserve(instance.block_cipher_params.num_sboxes - 1);
    for (size_t k = 0; k < instance.block_cipher_params.num_sboxes - 1; k++) {
      tmp.from_bytes(serialized.data() + current_offset);
      current_offset += GF::BYTE_SIZE;
      missing_alpha_shares.push_back(tmp);
    }
    GF c_delta;
    c_delta.from_bytes(serialized.data() + current_offset);
    current_offset += GF::BYTE_SIZE;

    proofs.emplace_back(repetition_proof_t<GF>{
        reveallist,
        Com_e,
        sk_delta,
        t_delta,
        missing_alpha_shares,
        c_delta,
    });
  }
  assert(current_offset == serialized.size());
  signature_t<GF> signature{salt, h_1, h_2, proofs};
  return signature;
}

std::vector<uint8_t> rainier_sign(const signature_instance_t &instance,
                                  const keypair_t &keypair,
                                  const uint8_t *message, size_t message_len) {
  if (instance.block_cipher_params.block_size == 16) {
    auto sig = rainier_sign_template<field::GF2_128>(instance, keypair, message,
                                                     message_len);
    return rainier_serialize_signature(instance, sig);
  } else if (instance.block_cipher_params.block_size == 24) {
    auto sig = rainier_sign_template<field::GF2_192>(instance, keypair, message,
                                                     message_len);
    return rainier_serialize_signature(instance, sig);
  } else if (instance.block_cipher_params.block_size == 32) {
    auto sig = rainier_sign_template<field::GF2_256>(instance, keypair, message,
                                                     message_len);
    return rainier_serialize_signature(instance, sig);
  }
  throw std::runtime_error("parameter set not implemented");
}

bool rainier_verify(const signature_instance_t &instance,
                    const std::vector<uint8_t> &pk,
                    const std::vector<uint8_t> &signature,
                    const uint8_t *message, size_t message_len) {
  if (instance.block_cipher_params.block_size == 16) {
    auto sig =
        rainier_deserialize_signature<field::GF2_128>(instance, signature);
    return rainier_verify_template<field::GF2_128>(instance, pk, sig, message,
                                                   message_len);
  } else if (instance.block_cipher_params.block_size == 24) {
    auto sig =
        rainier_deserialize_signature<field::GF2_192>(instance, signature);
    return rainier_verify_template<field::GF2_192>(instance, pk, sig, message,
                                                   message_len);
  } else if (instance.block_cipher_params.block_size == 32) {
    auto sig =
        rainier_deserialize_signature<field::GF2_256>(instance, signature);
    return rainier_verify_template<field::GF2_256>(instance, pk, sig, message,
                                                   message_len);
  }
  throw std::runtime_error("parameter set not implemented");
}