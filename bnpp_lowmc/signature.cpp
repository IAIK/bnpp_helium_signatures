#include "signature.h"

#include "field.h"
#include "lowmc.h"
#include "tape.h"
#include "tree.h"
#include <algorithm>
#include <cassert>
#include <cstring>

extern "C" {
#include "kdf.h"
#include "randomness.h"
}

namespace {
template <typename GF>
inline void hash_update_GF2E(hash_context *ctx, const GF &element) {
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
                   const RepByteContainer &output_broadcasts,
                   const std::vector<std::vector<uint8_t>> &key_deltas,
                   const std::vector<std::vector<GF>> &z_deltas,
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
      auto output_broadcast = output_broadcasts.get(repetition, party);
      hash_update(&ctx, output_broadcast.data(), output_broadcast.size());
    }
    hash_update(&ctx, key_deltas[repetition].data(),
                key_deltas[repetition].size());
    for (size_t ell = 0; ell < instance.num_lifted_multiplications; ell++) {
      hash_update_GF2E(&ctx, z_deltas[repetition][ell]);
    }
    hash_update_GF2E(&ctx, c_deltas[repetition]);
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

// returns lists of epsilon values
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
    epsilons[e].reserve(instance.num_lifted_multiplications);
    for (size_t eps_idx = 0; eps_idx < instance.num_lifted_multiplications;
         eps_idx++) {
      hash_squeeze(&ctx, buffer.data(), buffer.size());
      GF eps;
      // eps does not have restrictions
      eps.from_bytes(buffer.data());
      epsilons[e].push_back(eps);
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
      auto alphas = alpha_shares.get(repetition, party);
      for (size_t ell = 0; ell < instance.num_lifted_multiplications; ell++) {
        hash_update_GF2E(&ctx, alphas[ell]);
      }
      hash_update_GF2E(&ctx, v_shares[repetition][party]);
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

keypair_t bnpp_keygen(const signature_instance_t &instance) {
  std::vector<uint8_t> key(instance.block_cipher_params.key_size),
      pt(instance.block_cipher_params.block_size),
      ct(instance.block_cipher_params.block_size);

  rand_bytes(key.data(), key.size());
  rand_bytes(pt.data(), pt.size());
  if (instance.block_cipher_params.key_size == 17 &&
      instance.block_cipher_params.num_sboxes == 172) {
    LOWMC_129_4::lowmc_with_sbox_output<field::GF2_51>(key, pt, ct);
  } else if (instance.block_cipher_params.key_size == 16 &&
             instance.block_cipher_params.num_sboxes == 200) {
    LOWMC_128_20::lowmc_with_sbox_output<field::GF2_51>(key, pt, ct);
  } else
    throw std::runtime_error("invalid parameters");
  keypair_t keypair;
  keypair.first = key;
  keypair.second = pt;
  keypair.second.insert(keypair.second.end(), ct.begin(), ct.end());
  return keypair;
}

template <typename GF>
signature_t<GF> bnpp_sign_template(const signature_instance_t &instance,
                                     const keypair_t &keypair,
                                     const uint8_t *message,
                                     size_t message_len) {

  // so we dont have to type the long variant everytime
  const size_t L = instance.num_lifted_multiplications;

  // grab lowmc key, pt and ct
  std::vector<uint8_t> key = keypair.first;
  std::vector<uint8_t> pt_ct = keypair.second;
  const size_t total_pt_ct_size = instance.block_cipher_params.block_size;
  std::vector<uint8_t> pt(total_pt_ct_size), ct(total_pt_ct_size),
      ct2(total_pt_ct_size);
  memcpy(pt.data(), keypair.second.data(), pt.size());
  memcpy(ct.data(), keypair.second.data() + pt.size(), ct.size());

  // get multiplication inputs and outputs for lowmc evaluation
  std::vector<std::array<GF, 3>> mult_triples;

  if (instance.block_cipher_params.key_size == 17 &&
      instance.block_cipher_params.num_sboxes == 172)
    mult_triples = LOWMC_129_4::lowmc_with_sbox_output<GF>(key, pt, ct2);
  else if (instance.block_cipher_params.key_size == 16 &&
           instance.block_cipher_params.num_sboxes == 200)
    mult_triples = LOWMC_128_20::lowmc_with_sbox_output<GF>(key, pt, ct2);
  else
    throw std::runtime_error("invalid parameters");

  // sanity check, incoming keypair is valid
  assert(ct == ct2);

#ifndef NDEBUG
  for (size_t ell = 0; ell < L; ell++) {
    assert(mult_triples[ell][0] * mult_triples[ell][1] == mult_triples[ell][2]);
  }
#endif

  // generate salt and master seeds for each repetition
  auto [salt, master_seeds] =
      generate_salt_and_seeds(instance, keypair, message, message_len);

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  seed_trees.reserve(instance.num_repetitions);
  // key share + L*z_share + L*a_share + c_share
  const size_t random_tape_size =
      instance.block_cipher_params.key_size +
      instance.num_lifted_multiplications * GF::BYTE_SIZE +
      instance.num_lifted_multiplications * GF::BYTE_SIZE + GF::BYTE_SIZE;

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
  RepByteContainer rep_output_broadcasts(
      instance.num_repetitions, instance.num_MPC_parties,
      instance.block_cipher_params.block_size);
  RepContainer<GF> rep_shared_x(instance.num_repetitions,
                                instance.num_MPC_parties, L);
  RepContainer<GF> rep_shared_y(instance.num_repetitions,
                                instance.num_MPC_parties, L);
  RepContainer<GF> rep_shared_z(instance.num_repetitions,
                                instance.num_MPC_parties, L);
  RepContainer<GF> rep_shared_dot_a(instance.num_repetitions,
                                    instance.num_MPC_parties, L);
  RepContainer<GF> rep_shared_dot_c(instance.num_repetitions,
                                    instance.num_MPC_parties, 1);
  std::vector<std::vector<uint8_t>> rep_key_deltas;
  rep_key_deltas.reserve(instance.num_repetitions);
  std::vector<std::vector<GF>> rep_z_deltas;
  rep_z_deltas.reserve(instance.num_repetitions);
  std::vector<GF> rep_c_deltas(instance.num_repetitions);

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
    // generate sharing of z values
    std::vector<GF> z_deltas(L);
    for (size_t ell = 0; ell < L; ell++) {
      z_deltas[ell] = mult_triples[ell][2];
    }
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_z = rep_shared_z.get(repetition, party);
      auto random_z_shares = random_tapes.get_bytes(
          repetition, party, instance.block_cipher_params.key_size,
          L * GF::BYTE_SIZE);
      for (size_t ell = 0; ell < L; ell++) {
        shared_z[ell].from_bytes(random_z_shares.data() + ell * GF::BYTE_SIZE);
      }
      std::transform(std::begin(shared_z), std::end(shared_z),
                     std::begin(z_deltas), std::begin(z_deltas),
                     std::plus<GF>());
    }
    // fix first share
    auto first_share_z = rep_shared_z.get(repetition, 0);
    std::transform(std::begin(z_deltas), std::end(z_deltas),
                   std::begin(first_share_z), std::begin(first_share_z),
                   std::minus<GF>());
    rep_z_deltas.push_back(z_deltas);

    // get shares of sbox inputs by simulating MPC execution
    auto ct_shares = rep_output_broadcasts.get_repetition(repetition);
    auto shared_x = rep_shared_x.get_repetition(repetition);
    auto shared_y = rep_shared_y.get_repetition(repetition);

    if (instance.block_cipher_params.key_size == 17 &&
        instance.block_cipher_params.num_sboxes == 172)
      LOWMC_129_4::lowmc_mpc(rep_shared_keys.get_repetition(repetition),
                             rep_shared_z.get_repetition(repetition), pt,
                             ct_shares, shared_x, shared_y);
    else if (instance.block_cipher_params.key_size == 16 &&
             instance.block_cipher_params.num_sboxes == 200)
      LOWMC_128_20::lowmc_mpc(rep_shared_keys.get_repetition(repetition),
                              rep_shared_z.get_repetition(repetition), pt,
                              ct_shares, shared_x, shared_y);
    else
      throw std::runtime_error("invalid parameters");

#ifndef NDEBUG
    // sanity check, mpc execution = plain one
    std::vector<uint8_t> ct_check(instance.block_cipher_params.block_size);
    memset(ct_check.data(), 0, ct_check.size());
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      std::transform(std::begin(ct_shares[party]), std::end(ct_shares[party]),
                     std::begin(ct_check), std::begin(ct_check),
                     std::bit_xor<uint8_t>());
    }

    assert(ct == ct_check);
    // sanity check, all x and y values multiply to z
    for (size_t ell = 0; ell < L; ell++) {
      GF test_X, test_Y, test_Z;
      for (size_t party = 0; party < instance.num_MPC_parties; party++) {
        test_X += rep_shared_x.get_repetition(repetition)[party][ell];
        test_Y += rep_shared_y.get_repetition(repetition)[party][ell];
        test_Z += rep_shared_z.get_repetition(repetition)[party][ell];
      }
      assert(test_X * test_Y == test_Z);
    }
#endif
  }

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {

    // also generate valid dot triple a,y,c and save c_delta
    std::vector<GF> a(instance.num_lifted_multiplications);
    GF c(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto random_dot_bytes = random_tapes.get_bytes(
          repetition, party,
          instance.block_cipher_params.key_size + L * GF::BYTE_SIZE,
          (L + 1) * GF::BYTE_SIZE);
      auto a_share = rep_shared_dot_a.get(repetition, party);
      auto c_share = rep_shared_dot_c.get(repetition, party);
      for (size_t ell = 0; ell < L; ell++) {
        a_share[ell].from_bytes(random_dot_bytes.data() + ell * GF::BYTE_SIZE);
        a[ell] += a_share[ell];
      }
      c_share[0].from_bytes(random_dot_bytes.data() + L * GF::BYTE_SIZE);
      c += c_share[0];
    }
    // calculate c_delta that fixes the dot triple
    for (size_t ell = 0; ell < L; ell++) {
      rep_c_deltas[repetition] += a[ell] * mult_triples[ell][1];
    }
    rep_c_deltas[repetition] -= c;
    // fix party 0's share
    rep_shared_dot_c.get(repetition, 0)[0] += rep_c_deltas[repetition];
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 2: challenge for the sacrificing check
  /////////////////////////////////////////////////////////////////////////////

  // commit to salt, (all commitments of parties seeds,
  // output_broadcasts, key_delta, t_delta, P_delta) for all
  // repetitions
  std::vector<uint8_t> h_1 =
      phase_1_commitment(instance, salt, keypair.second, message, message_len,
                         party_seed_commitments, rep_output_broadcasts,
                         rep_key_deltas, rep_z_deltas, rep_c_deltas);

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

#ifndef NDEBUG
    // sanity check: x_i*y_i=z_i
    for (size_t ell = 0; ell < L; ell++) {
      GF x(0), y(0), z(0);
      for (size_t party = 0; party < instance.num_MPC_parties; party++) {
        x += rep_shared_x.get(repetition, party)[ell];
        y += rep_shared_y.get(repetition, party)[ell];
        z += rep_shared_z.get(repetition, party)[ell];
      }
      assert(x * y == z);
    }
    // sanity check: dot(a,y)=c
    GF acc(0);
    for (size_t ell = 0; ell < L; ell++) {
      GF ai(0), yi(0);
      for (size_t party = 0; party < instance.num_MPC_parties; party++) {
        ai += rep_shared_dot_a.get(repetition, party)[ell];
        yi += rep_shared_y.get(repetition, party)[ell];
      }
      acc += ai * yi;
    }
    GF c(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      c += rep_shared_dot_c.get(repetition, party)[0];
    }
    assert(acc == c);
#endif

    std::vector<GF> alphas(L);
    // execute sacrificing check protocol
    // alpha^i = eps * x^i + a^i
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alpha_shares = rep_alpha_shares.get(repetition, party);
      auto x_shares = rep_shared_x.get(repetition, party);
      auto a_shares = rep_shared_dot_a.get(repetition, party);
      for (size_t ell = 0; ell < L; ell++) {
        alpha_shares[ell] =
            x_shares[ell] * epsilons[repetition][ell] + a_shares[ell];
        alphas[ell] += alpha_shares[ell];
      }
    }
    // v^i = dot(eps, z^i) - c^i - dot(alpha, y^i)
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      v_shares[repetition][party] -= rep_shared_dot_c.get(repetition, party)[0];
      auto y_shares = rep_shared_y.get(repetition, party);
      auto z_shares = rep_shared_z.get(repetition, party);
      for (size_t ell = 0; ell < L; ell++) {
        v_shares[repetition][party] +=
            epsilons[repetition][ell] * z_shares[ell] -
            alphas[ell] * y_shares[ell];
      }
    }
#ifndef NDEBUG
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
    auto missing_party_alpha = rep_alpha_shares.get(repetition, missing_party);
    std::vector<GF> missing_alphas(missing_party_alpha.begin(),
                                   missing_party_alpha.end());
    repetition_proof_t<GF> proof{
        seeds[repetition],        commitment,     rep_key_deltas[repetition],
        rep_z_deltas[repetition], missing_alphas, rep_c_deltas[repetition],
    };
    proofs.push_back(proof);
  }

  signature_t<GF> signature{salt, h_1, h_2, proofs};

  return signature;
}

template <typename GF>
bool bnpp_verify_template(const signature_instance_t &instance,
                            const std::vector<uint8_t> &pk,
                            const signature_t<GF> &signature,
                            const uint8_t *message, size_t message_len) {

  // so we dont have to type the long variant everytime
  const size_t L = instance.num_lifted_multiplications;

  std::vector<uint8_t> pt(instance.block_cipher_params.block_size),
      ct(instance.block_cipher_params.block_size);
  memcpy(pt.data(), pk.data(), pt.size());
  memcpy(ct.data(), pk.data() + pt.size(), ct.size());

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  // key share + L*z_share + L*a_share + c_share
  const size_t random_tape_size =
      instance.block_cipher_params.key_size +
      instance.num_lifted_multiplications * GF::BYTE_SIZE +
      instance.num_lifted_multiplications * GF::BYTE_SIZE + GF::BYTE_SIZE;

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
  RepContainer<GF> rep_shared_x(instance.num_repetitions,
                                instance.num_MPC_parties, L);
  RepContainer<GF> rep_shared_y(instance.num_repetitions,
                                instance.num_MPC_parties, L);
  RepContainer<GF> rep_shared_z(instance.num_repetitions,
                                instance.num_MPC_parties, L);
  RepByteContainer rep_output_broadcasts(
      instance.num_repetitions, instance.num_MPC_parties,
      instance.block_cipher_params.block_size);

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

    // generate sharing of z values
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_z = rep_shared_z.get(repetition, party);
      auto random_z_shares = random_tapes.get_bytes(
          repetition, party, instance.block_cipher_params.key_size,
          L * GF::BYTE_SIZE);
      for (size_t ell = 0; ell < L; ell++) {
        shared_z[ell].from_bytes(random_z_shares.data() + ell * GF::BYTE_SIZE);
      }
    }
    // fix first share
    auto first_shared_z = rep_shared_z.get(repetition, 0);
    std::transform(std::begin(proof.z_delta), std::end(proof.z_delta),
                   std::begin(first_shared_z), std::begin(first_shared_z),
                   std::minus<GF>());

    // get shares of sbox inputs by executing MPC AES
    auto ct_shares = rep_output_broadcasts.get_repetition(repetition);
    auto shared_x = rep_shared_x.get_repetition(repetition);
    auto shared_y = rep_shared_y.get_repetition(repetition);

    if (instance.block_cipher_params.key_size == 17 &&
        instance.block_cipher_params.num_sboxes == 172)
      LOWMC_129_4::lowmc_mpc(rep_shared_keys.get_repetition(repetition),
                             rep_shared_z.get_repetition(repetition), pt,
                             ct_shares, shared_x, shared_y);
    else if (instance.block_cipher_params.key_size == 16 &&
             instance.block_cipher_params.num_sboxes == 200)
      LOWMC_128_20::lowmc_mpc(rep_shared_keys.get_repetition(repetition),
                              rep_shared_z.get_repetition(repetition), pt,
                              ct_shares, shared_x, shared_y);
    else
      throw std::runtime_error("invalid parameters");

    // calculate missing output broadcast
    std::copy(ct.begin(), ct.end(),
              ct_shares[missing_parties[repetition]].begin());
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_parties[repetition])
        std::transform(std::begin(ct_shares[party]), std::end(ct_shares[party]),
                       std::begin(ct_shares[missing_parties[repetition]]),
                       std::begin(ct_shares[missing_parties[repetition]]),
                       std::bit_xor<uint8_t>());
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute dot-product triple, except missing shares
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF> rep_shared_dot_a(instance.num_repetitions,
                                    instance.num_MPC_parties, L);
  RepContainer<GF> rep_shared_dot_c(instance.num_repetitions,
                                    instance.num_MPC_parties, 1);
  // also generate valid dot triple a,y,c and save c_delta
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_parties[repetition]) {
        auto random_dot_bytes = random_tapes.get_bytes(
            repetition, party,
            instance.block_cipher_params.key_size + L * GF::BYTE_SIZE,
            (L + 1) * GF::BYTE_SIZE);
        auto a_share = rep_shared_dot_a.get(repetition, party);
        auto c_share = rep_shared_dot_c.get(repetition, party);
        for (size_t ell = 0; ell < L; ell++) {
          a_share[ell].from_bytes(random_dot_bytes.data() +
                                  ell * GF::BYTE_SIZE);
        }
        c_share[0].from_bytes(random_dot_bytes.data() + L * GF::BYTE_SIZE);
      }
    }
    // fix party 0's share
    if (0 != missing_parties[repetition]) {
      const repetition_proof_t<GF> &proof = signature.proofs[repetition];
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

    // execute sacrificing check protocol
    // alpha^i = eps * x^i + a^i
    std::vector<GF> alphas(L);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alpha_shares = rep_alpha_shares.get(repetition, party);
      if (party != missing_party) {
        auto x_shares = rep_shared_x.get(repetition, party);
        auto a_shares = rep_shared_dot_a.get(repetition, party);
        for (size_t ell = 0; ell < L; ell++) {
          alpha_shares[ell] =
              x_shares[ell] * epsilons[repetition][ell] + a_shares[ell];
          alphas[ell] += alpha_shares[ell];
        }
      } else {
        // fill missing shares
        for (size_t ell = 0; ell < L; ell++) {
          alpha_shares[ell] = proof.missing_alphas[ell];
          alphas[ell] += alpha_shares[ell];
        }
      }
    }
    v_shares[repetition].resize(instance.num_MPC_parties);
    // v^i = dot(eps, z^i) - c^i - dot(alpha, y^i)
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_party) {
        v_shares[repetition][party] -=
            rep_shared_dot_c.get(repetition, party)[0];
        auto y_shares = rep_shared_y.get(repetition, party);
        auto z_shares = rep_shared_z.get(repetition, party);
        for (size_t ell = 0; ell < L; ell++) {
          v_shares[repetition][party] +=
              epsilons[repetition][ell] * z_shares[ell] -
              alphas[ell] * y_shares[ell];
        }
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
  std::vector<std::vector<GF>> z_deltas;
  std::vector<GF> c_deltas;
  sk_deltas.reserve(instance.num_repetitions);
  z_deltas.reserve(instance.num_repetitions);
  c_deltas.reserve(instance.num_repetitions);
  for (const repetition_proof_t<GF> &proof : signature.proofs) {
    sk_deltas.push_back(proof.sk_delta);
    z_deltas.push_back(proof.z_delta);
    c_deltas.push_back(proof.c_delta);
  }
  std::vector<uint8_t> h_1 =
      phase_1_commitment(instance, signature.salt, pk, message, message_len,
                         party_seed_commitments, rep_output_broadcasts,
                         sk_deltas, z_deltas, c_deltas);

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
bnpp_serialize_signature(const signature_instance_t &instance,
                           const signature_t<GF> &signature) {
  std::vector<uint8_t> serialized;

  // prealloc signature size
  const size_t packed_GF_elements_size =
      (instance.num_repetitions *                                // tau times
           (instance.num_lifted_multiplications * GF::BIT_SIZE + // (delta zs +
            instance.num_lifted_multiplications * GF::BIT_SIZE + // alphas +
            GF::BIT_SIZE) +                                      // delta c)
       7) // round up to next byte
      / 8;
  const size_t signature_size =
      signature.salt.size() +    // salt
      instance.digest_size * 2 + // h_1, h_2
      instance.num_repetitions * // tau x
          (ceil_log2(instance.num_MPC_parties) *
               instance.seed_size +               // merkle tree path
           instance.digest_size +                 // Com_e
           instance.block_cipher_params.key_size) // delta sk
      + packed_GF_elements_size;
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
  }
  // pack GF elements tightly into the provided byte array

  // small buffer here so we can always write a full uint64_t in the buffer,
  // even at the end, removed later
  std::vector<uint8_t> packed_GF_elements(packed_GF_elements_size + 7);
  // current implementation only works for this constraint
  static_assert(GF::BIT_SIZE <= 64);
  size_t total_bits_written = 0;
  size_t current_shift = 0;
  const size_t uneven_bits = (GF::BIT_SIZE % 8);
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF> &proof = signature.proofs[repetition];
    // z_deltas
    for (const GF &ele : proof.z_delta) {
      uint64_t tmp = ele.data;
      tmp = tmp << current_shift;
      current_shift = (current_shift + uneven_bits) % 8;
      size_t current_byte_index = (total_bits_written) / 8;
      *((uint64_t *)(packed_GF_elements.data() + current_byte_index)) ^=
          htole64(tmp);
      total_bits_written += GF::BIT_SIZE;
    }
    // alphas
    for (const GF &ele : proof.missing_alphas) {
      uint64_t tmp = ele.data;
      tmp = tmp << current_shift;
      current_shift = (current_shift + uneven_bits) % 8;
      size_t current_byte_index = (total_bits_written) / 8;
      *((uint64_t *)(packed_GF_elements.data() + current_byte_index)) ^=
          htole64(tmp);
      total_bits_written += GF::BIT_SIZE;
    }
    // c_delta
    {
      uint64_t tmp = proof.c_delta.data;
      tmp = tmp << current_shift;
      current_shift = (current_shift + uneven_bits) % 8;
      size_t current_byte_index = (total_bits_written) / 8;
      *((uint64_t *)(packed_GF_elements.data() + current_byte_index)) ^=
          htole64(tmp);
      total_bits_written += GF::BIT_SIZE;
    }
  }

  serialized.insert(serialized.end(), packed_GF_elements.begin(),
                    packed_GF_elements.end() - 7);
  // calculation as expected
  assert(signature_size == serialized.size());
  return serialized;
}

template <typename GF>
signature_t<GF>
bnpp_deserialize_signature(const signature_instance_t &instance,
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

    std::vector<GF> z_delta(instance.num_lifted_multiplications);
    std::vector<GF> missing_alpha_shares(instance.num_lifted_multiplications);
    GF c_delta;

    proofs.emplace_back(repetition_proof_t<GF>{
        reveallist, Com_e, sk_delta, z_delta, missing_alpha_shares, c_delta});
  }
  // unpack tightly packed GF elements
  std::vector<uint8_t> packed_GF_elements(serialized.begin() + current_offset,
                                          serialized.end());
  // buffer so we can always read uint64_t values without invalid memory access
  packed_GF_elements.resize(packed_GF_elements.size() + 7);
  // current implementation only works for this constraint
  static_assert(GF::BIT_SIZE <= 64);
  size_t total_bits_read = 0;
  size_t current_shift = 0;
  const size_t uneven_bits = (GF::BIT_SIZE % 8);
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    repetition_proof_t<GF> &proof = proofs[repetition];
    // z_deltas
    for (GF &ele : proof.z_delta) {
      size_t current_byte_index = (total_bits_read) / 8;
      uint64_t tmp =
          *((uint64_t *)(packed_GF_elements.data() + current_byte_index));
      tmp = le64toh(tmp) >> current_shift;
      current_shift = (current_shift + uneven_bits) % 8;
      ele.data = tmp & GF::ELEMENT_MASK;
      total_bits_read += GF::BIT_SIZE;
    }
    // alphas
    for (GF &ele : proof.missing_alphas) {
      size_t current_byte_index = (total_bits_read) / 8;
      uint64_t tmp =
          *((uint64_t *)(packed_GF_elements.data() + current_byte_index));
      tmp = le64toh(tmp) >> current_shift;
      current_shift = (current_shift + uneven_bits) % 8;
      ele.data = tmp & GF::ELEMENT_MASK;
      total_bits_read += GF::BIT_SIZE;
    }
    // c_delta
    {
      size_t current_byte_index = (total_bits_read) / 8;
      uint64_t tmp =
          *((uint64_t *)(packed_GF_elements.data() + current_byte_index));
      tmp = le64toh(tmp) >> current_shift;
      current_shift = (current_shift + uneven_bits) % 8;
      proof.c_delta.data = tmp & GF::ELEMENT_MASK;
      total_bits_read += GF::BIT_SIZE;
    }
  }
  current_offset += (total_bits_read + 7) / 8;
  // the last byte might only have a few bits that are part of a FE
  // check that leftover bits are indeed 0
  size_t leftover_bits = (8 - (total_bits_read % 8)) % 8;
  if (leftover_bits != 0) {
    uint8_t last_byte = serialized.back();
    last_byte = last_byte >> (8 - leftover_bits);
    if (last_byte != 0)
      throw std::runtime_error("serialized signature is malformed");
  }

  assert(current_offset == serialized.size());
  signature_t<GF> signature{salt, h_1, h_2, proofs};
  return signature;
}

std::vector<uint8_t> bnpp_sign(const signature_instance_t &instance,
                                 const keypair_t &keypair,
                                 const uint8_t *message, size_t message_len) {
  if (instance.block_cipher_params.block_size == 16 ||
      instance.block_cipher_params.block_size == 17) {
    auto sig = bnpp_sign_template<field::GF2_51>(instance, keypair, message,
                                                   message_len);
    return bnpp_serialize_signature(instance, sig);
  }
  throw std::runtime_error("parameter set not implemented");
}

bool bnpp_verify(const signature_instance_t &instance,
                   const std::vector<uint8_t> &pk,
                   const std::vector<uint8_t> &signature,
                   const uint8_t *message, size_t message_len) {
  if (instance.block_cipher_params.block_size == 16 ||
      instance.block_cipher_params.block_size == 17) {
    auto sig = bnpp_deserialize_signature<field::GF2_51>(instance, signature);
    return bnpp_verify_template<field::GF2_51>(instance, pk, sig, message,
                                                 message_len);
  }
  throw std::runtime_error("parameter set not implemented");
}