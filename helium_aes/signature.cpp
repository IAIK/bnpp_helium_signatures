#include "signature.h"

#include "aes.h"
#include "field.h"
#include "precomputation.h"
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

std::vector<uint8_t>
phase_1_commitment(const signature_instance_t &instance, const salt_t &salt,
                   const std::vector<uint8_t> &pk, const uint8_t *message,
                   size_t message_len, const RepByteContainer &commitments,
                   const std::vector<std::vector<uint8_t>> &key_deltas,
                   const std::vector<std::vector<uint8_t>> &t_deltas,
                   const std::vector<std::vector<uint8_t>> &p_delta1,
                   const std::vector<std::vector<uint8_t>> &p_delta2) {

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
    hash_update(&ctx, t_deltas[repetition].data(), t_deltas[repetition].size());
    hash_update(&ctx, p_delta1[repetition].data(), p_delta1[repetition].size());
    hash_update(&ctx, p_delta2[repetition].data(), p_delta2[repetition].size());
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

// returns list of R values
template <typename GF>
std::vector<GF> phase_1_expand(const signature_instance_t &instance,
                               const std::vector<uint8_t> &h_1) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_1.data(), h_1.size());
  hash_final(&ctx);

  std::array<uint8_t, GF::BYTE_SIZE> buffer;
  std::vector<GF> r_values;
  r_values.reserve(instance.num_repetitions);
  for (size_t e = 0; e < instance.num_repetitions; e++) {
    hash_squeeze(&ctx, buffer.data(), buffer.size());
    GF r;
    // r does not have restrictions
    r.from_bytes(buffer.data());
    r_values.push_back(r);
  }
  return r_values;
}

template <typename GF>
std::vector<uint8_t> phase_2_commitment(const signature_instance_t &instance,
                                        const salt_t &salt,
                                        const std::vector<uint8_t> &h_1,
                                        const std::vector<GF> &c_deltas) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_2);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update(&ctx, h_1.data(), h_1.size());

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    hash_update_GF2E(&ctx, c_deltas[repetition]);
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

// returns list of eps values
template <typename GF>
std::vector<std::array<GF, 2>>
phase_2_expand(const signature_instance_t &instance,
               const std::vector<uint8_t> &h_2) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_2.data(), h_2.size());
  hash_final(&ctx);

  std::array<uint8_t, GF::BYTE_SIZE> buffer;
  std::vector<std::array<GF, 2>> eps_values;
  eps_values.reserve(instance.num_repetitions);
  for (size_t e = 0; e < instance.num_repetitions; e++) {
    hash_squeeze(&ctx, buffer.data(), buffer.size());
    std::array<GF, 2> eps;
    // eps does not have restrictions
    eps[0].from_bytes(buffer.data());
    hash_squeeze(&ctx, buffer.data(), buffer.size());
    // eps does not have restrictions
    eps[1].from_bytes(buffer.data());
    eps_values.push_back(eps);
  }
  return eps_values;
}

template <typename GF>
std::vector<uint8_t> phase_3_commitment(const signature_instance_t &instance,
                                        const salt_t &salt,
                                        const std::vector<uint8_t> &h_2,
                                        const RepContainer<GF> &alpha_shares,
                                        const RepContainer<GF> &v_shares) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_3);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update(&ctx, h_2.data(), h_2.size());

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alphas = alpha_shares.get(repetition, party);
      hash_update_GF2E(&ctx, alphas[0]);
      hash_update_GF2E(&ctx, alphas[1]);
      auto vs = v_shares.get(repetition, party);
      hash_update_GF2E(&ctx, vs[0]);
    }
  }
  hash_final(&ctx);

  std::vector<uint8_t> commitment(instance.digest_size);
  hash_squeeze(&ctx, commitment.data(), commitment.size());
  return commitment;
}

std::vector<uint16_t> phase_3_expand(const signature_instance_t &instance,
                                     const std::vector<uint8_t> &h_3) {
  assert(instance.num_MPC_parties <= (1ULL << 16));
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_3.data(), h_3.size());
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

keypair_t helium_keygen(const signature_instance_t &instance) {
  std::vector<uint8_t> key(instance.block_cipher_params.key_size),
      pt(instance.block_cipher_params.block_size),
      ct(instance.block_cipher_params.block_size);

  while (true) {
    rand_bytes(key.data(), key.size());
    rand_bytes(pt.data(), pt.size());
    if (instance.block_cipher_params.key_size == 16) {
      if (AES128::aes_128(key, pt, ct)) {
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

template <typename GF_CHECK>
signature_t<GF_CHECK> helium_sign_template(const signature_instance_t &instance,
                                          const keypair_t &keypair,
                                          const uint8_t *message,
                                          size_t message_len) {

  // so we dont have to type the long variant everytime
  const size_t L = instance.num_multiplications;
  if (L != 200)
    throw std::runtime_error("not implemented");

  // grab lowmc key, pt and ct
  std::vector<uint8_t> key = keypair.first;
  std::vector<uint8_t> pt_ct = keypair.second;
  const size_t total_pt_ct_size = instance.block_cipher_params.block_size;
  std::vector<uint8_t> pt(total_pt_ct_size), ct(total_pt_ct_size),
      ct2(total_pt_ct_size);
  memcpy(pt.data(), keypair.second.data(), pt.size());
  memcpy(ct.data(), keypair.second.data() + pt.size(), ct.size());

  // get multiplication inputs and outputs for lowmc evaluation
  std::pair<std::vector<uint8_t>, std::vector<uint8_t>> mult_triples;

  if (instance.block_cipher_params.key_size == 16)
    mult_triples = AES128::aes_128_with_sbox_output(key, pt, ct2);
  else
    throw std::runtime_error("invalid parameters");

  // sanity check, incoming keypair is valid
  assert(ct == ct2);

#ifndef NDEBUG
  for (size_t ell = 0; ell < L; ell++) {
    assert(field::GF2_8(mult_triples.first[ell]) *
               field::GF2_8(mult_triples.second[ell]) ==
           field::GF2_8(1));
  }
#endif

  // generate salt and master seeds for each repetition
  auto [salt, master_seeds] =
      generate_salt_and_seeds(instance, keypair, message, message_len);

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  seed_trees.reserve(instance.num_repetitions);
  // key share + L*z_share + (L-1)*p_share + 2*a_share + c_share
  const size_t random_tape_size =
      instance.block_cipher_params.key_size + instance.num_multiplications +
      (instance.num_multiplications - 2) + 3 * GF_CHECK::BYTE_SIZE;

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
  RepContainer<uint8_t> rep_shared_s(instance.num_repetitions,
                                     instance.num_MPC_parties, L);
  RepContainer<uint8_t> rep_shared_t(instance.num_repetitions,
                                     instance.num_MPC_parties, L);
  RepContainer<field::GF2_8> rep_shared_p_points1(
      instance.num_repetitions, instance.num_MPC_parties, L - 1);
  RepContainer<field::GF2_8> rep_shared_p_points2(
      instance.num_repetitions, instance.num_MPC_parties, L - 1);

  std::vector<std::vector<uint8_t>> rep_key_deltas;
  rep_key_deltas.reserve(instance.num_repetitions);
  std::vector<std::vector<uint8_t>> rep_t_deltas;
  rep_t_deltas.reserve(instance.num_repetitions);
  std::vector<std::vector<uint8_t>> rep_p_deltas1;
  std::vector<std::vector<uint8_t>> rep_p_deltas2;
  rep_p_deltas1.reserve(instance.num_repetitions);
  rep_p_deltas2.reserve(instance.num_repetitions);

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
    std::vector<uint8_t> t_deltas(L - 16);
    for (size_t ell = 0; ell < L - 16; ell++) {
      t_deltas[ell] = mult_triples.second[ell];
    }
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_t = rep_shared_t.get(repetition, party);
      auto random_t_shares = random_tapes.get_bytes(
          repetition, party, instance.block_cipher_params.key_size, L);
      for (size_t ell = 0; ell < L - 16; ell++) {
        shared_t[ell] = random_t_shares[ell];
      }
      std::transform(std::begin(shared_t), std::end(shared_t) - 16,
                     std::begin(t_deltas), std::begin(t_deltas),
                     std::bit_xor<uint8_t>());
    }
    // fix first share
    auto first_share_t = rep_shared_t.get(repetition, 0);
    std::transform(std::begin(t_deltas), std::end(t_deltas),
                   std::begin(first_share_t), std::begin(first_share_t),
                   std::bit_xor<uint8_t>());
    rep_t_deltas.push_back(t_deltas);

    // get shares of sbox inputs by simulating MPC execution
    auto shared_s = rep_shared_s.get_repetition(repetition);
    auto shared_t = rep_shared_t.get_repetition(repetition);

    if (instance.block_cipher_params.key_size == 16)
      AES128::aes_128_s_shares(rep_shared_keys.get_repetition(repetition),
                               shared_t, pt, ct, shared_s);
    else
      throw std::runtime_error("invalid parameters");

#ifndef NDEBUG
    // sanity check, all shared s and t values multiply to 1
    for (size_t ell = 0; ell < L; ell++) {
      field::GF2_8 test_S, test_T;
      for (size_t party = 0; party < instance.num_MPC_parties; party++) {
        test_S +=
            field::GF2_8(rep_shared_s.get_repetition(repetition)[party][ell]);
        test_T +=
            field::GF2_8(rep_shared_t.get_repetition(repetition)[party][ell]);
      }
      assert(test_S * test_T == field::GF2_8(1));
    }
#endif
  }

  std::vector<field::GF2_8> first_2L_min_1_field_elements =
      field::get_first_n_field_elements<field::GF2_8>(L - 1);

  // interpolate S and T
  std::vector<field::GF2_8> y_values_S1(L / 2);
  std::vector<field::GF2_8> y_values_T1(L / 2);
  std::vector<field::GF2_8> y_values_S2(L / 2);
  std::vector<field::GF2_8> y_values_T2(L / 2);
  for (size_t i = 0; i < L / 2; i++) {
    y_values_S1[i] = mult_triples.first[i];
    y_values_T1[i] = mult_triples.second[i];
  }
  for (size_t i = L / 2; i < L; i++) {
    y_values_S2[i - L / 2] = mult_triples.first[i];
    y_values_T2[i - L / 2] = mult_triples.second[i];
  }
  std::vector<field::GF2_8> S_poly1 = field::interpolate_with_precomputation(
      precomputation::precomputed_lagrange_polys_100, y_values_S1);
  std::vector<field::GF2_8> T_poly1 = field::interpolate_with_precomputation(
      precomputation::precomputed_lagrange_polys_100, y_values_T1);
  std::vector<field::GF2_8> S_poly2 = field::interpolate_with_precomputation(
      precomputation::precomputed_lagrange_polys_100, y_values_S2);
  std::vector<field::GF2_8> T_poly2 = field::interpolate_with_precomputation(
      precomputation::precomputed_lagrange_polys_100, y_values_T2);

  std::vector<field::GF2_8> P_poly1 = S_poly1 * T_poly1;
  std::vector<field::GF2_8> P_poly2 = S_poly2 * T_poly2;
  std::vector<field::GF2_8> P_at_k_1(L / 2 - 1);
  std::vector<field::GF2_8> P_at_k_2(L / 2 - 1);
  for (size_t ell = L / 2; ell < L - 1; ell++) {
    P_at_k_1[ell - L / 2] =
        field::eval(P_poly1, first_2L_min_1_field_elements[ell]);
    P_at_k_2[ell - L / 2] =
        field::eval(P_poly2, first_2L_min_1_field_elements[ell]);
  }

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    std::vector<field::GF2_8> p_delta1 = P_at_k_1;
    std::vector<field::GF2_8> p_delta2 = P_at_k_2;
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_p1 = rep_shared_p_points1.get(repetition, party);
      auto shared_p2 = rep_shared_p_points2.get(repetition, party);

      // first L/2 points are set to 1
      for (size_t ell = 0; ell < L / 2; ell++) {
        if (party == 0) {
          shared_p1[ell] = field::GF2_8(1);
          shared_p2[ell] = field::GF2_8(1);
        } else {
          shared_p1[ell] = field::GF2_8(0);
          shared_p2[ell] = field::GF2_8(0);
        }
      }
      // next L/2-1 points are read from the random tape
      auto random_P_shares = random_tapes.get_bytes(
          repetition, party, instance.block_cipher_params.key_size + L, L - 2);
      for (size_t ell = L / 2; ell < L - 1; ell++) {
        shared_p1[ell] = random_P_shares[(ell - L / 2) * 2];
        p_delta1[ell - L / 2] -= random_P_shares[(ell - L / 2) * 2];
        shared_p2[ell] = random_P_shares[(ell - L / 2) * 2 + 1];
        p_delta2[ell - L / 2] -= random_P_shares[(ell - L / 2) * 2 + 1];
      }
    }

    auto first_shared_p1 = rep_shared_p_points1.get(repetition, 0);
    auto first_shared_p2 = rep_shared_p_points2.get(repetition, 0);
    for (size_t ell = L / 2; ell < L - 1; ell++) {
      first_shared_p1[ell] += p_delta1[ell - L / 2];
      first_shared_p2[ell] += p_delta2[ell - L / 2];
    }
    std::vector<uint8_t> tmp1(p_delta1.size());
    std::vector<uint8_t> tmp2(p_delta2.size());
    for (size_t i = 0; i < p_delta1.size(); i++) {
      tmp1[i] = p_delta1[i].data;
    }
    rep_p_deltas1.push_back(tmp1);
    for (size_t i = 0; i < p_delta2.size(); i++) {
      tmp2[i] = p_delta2[i].data;
    }
    rep_p_deltas2.push_back(tmp2);
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 2: challenge for the polynomial evaluation
  /////////////////////////////////////////////////////////////////////////////

  // commit to salt, (all commitments of parties seeds,
  // output_broadcasts, key_delta, t_delta, P_delta) for all
  // repetitions
  std::vector<uint8_t> h_1 =
      phase_1_commitment(instance, salt, keypair.second, message, message_len,
                         party_seed_commitments, rep_key_deltas, rep_t_deltas,
                         rep_p_deltas1, rep_p_deltas2);

  // expand challenge hash to R values
  std::vector<GF_CHECK> r_values = phase_1_expand<GF_CHECK>(instance, h_1);

  /////////////////////////////////////////////////////////////////////////////
  // phase 3: evaluate the polynomial and calculate/commit to a checking
  // triple
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF_CHECK> powers_of_R(1, instance.num_repetitions, L - 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    auto powers = powers_of_R.get(0, repetition);

    powers[0] = GF_CHECK(1);
    for (size_t i = 1; i < L - 1; i++) {
      powers[i] = powers[i - 1] * r_values[repetition];
    }
  }

  std::vector<GF_CHECK> S1_at_R, S2_at_R;
  S1_at_R.reserve(instance.num_repetitions);
  S2_at_R.reserve(instance.num_repetitions);
  std::vector<GF_CHECK> T1_at_R, T2_at_R;
  T1_at_R.reserve(instance.num_repetitions);
  T2_at_R.reserve(instance.num_repetitions);
  std::vector<GF_CHECK> P1_at_R, P2_at_R;
  P1_at_R.reserve(instance.num_repetitions);
  P2_at_R.reserve(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    auto r_powers = powers_of_R.get(0, repetition);
    auto r_powers_L = r_powers.subspan(0, L / 2);
    S1_at_R.push_back(lifted_dot_product(r_powers_L, S_poly1));
    T1_at_R.push_back(lifted_dot_product(r_powers_L, T_poly1));
    P1_at_R.push_back(lifted_dot_product(r_powers, P_poly1));
    S2_at_R.push_back(lifted_dot_product(r_powers_L, S_poly2));
    T2_at_R.push_back(lifted_dot_product(r_powers_L, T_poly2));
    P2_at_R.push_back(lifted_dot_product(r_powers, P_poly2));
#ifndef NDEBUG
    assert(S1_at_R[repetition] * T1_at_R[repetition] == P1_at_R[repetition]);
    assert(S2_at_R[repetition] * T2_at_R[repetition] == P2_at_R[repetition]);
#endif
  }

  // build two triple a*b = c, with b = T(R)

  RepContainer<GF_CHECK> rep_shared_a(instance.num_repetitions,
                                      instance.num_MPC_parties, 2);
  RepContainer<GF_CHECK> rep_shared_c(instance.num_repetitions,
                                      instance.num_MPC_parties, 1);

  std::vector<GF_CHECK> rep_c_deltas;
  rep_c_deltas.reserve(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    GF_CHECK a1(0), a2(0), c(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {

      auto random_a_c_bytes = random_tapes.get_bytes(
          repetition, party,
          instance.block_cipher_params.key_size + L + (L - 2),
          3 * GF_CHECK::BYTE_SIZE);

      auto shared_a = rep_shared_a.get(repetition, party);
      auto shared_c = rep_shared_c.get(repetition, party);
      shared_a[0].from_bytes(random_a_c_bytes.data());
      shared_a[1].from_bytes(random_a_c_bytes.data() + GF_CHECK::BYTE_SIZE);
      shared_c[0].from_bytes(random_a_c_bytes.data() + 2 * GF_CHECK::BYTE_SIZE);
      a1 += shared_a[0];
      a2 += shared_a[1];
      c += shared_c[0];
    }
    // calc c_delta and fix first parties share
    GF_CHECK c_delta = a1 * T1_at_R[repetition] + a2 * T2_at_R[repetition] - c;
    rep_shared_c.get(repetition, 0)[0] += c_delta;
    rep_c_deltas.push_back(c_delta);
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 4: challenge for the multiplication check
  /////////////////////////////////////////////////////////////////////////////

  std::vector<uint8_t> h_2 =
      phase_2_commitment(instance, salt, h_1, rep_c_deltas);

  // expand challenge hash to epsilon values
  std::vector<std::array<GF_CHECK, 2>> epsilons =
      phase_2_expand<GF_CHECK>(instance, h_2);

  /////////////////////////////////////////////////////////////////////////////
  // phase 5: commit to the views of the checking protocol
  /////////////////////////////////////////////////////////////////////////////

  RepContainer<GF_CHECK> rep_alpha_shares(instance.num_repetitions,
                                          instance.num_MPC_parties, 2);
  RepContainer<GF_CHECK> rep_v_shares(instance.num_repetitions,
                                      instance.num_MPC_parties, 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {

#ifndef NDEBUG
    // sanity check: a * T(R) =c
    GF_CHECK acc(0);
    GF_CHECK a1(0), a2(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      a1 += rep_shared_a.get(repetition, party)[0];
      a2 += rep_shared_a.get(repetition, party)[1];
    }
    acc += a1 * T1_at_R[repetition];
    acc += a2 * T2_at_R[repetition];
    GF_CHECK c(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      c += rep_shared_c.get(repetition, party)[0];
    }
    assert(acc == c);
#endif

    auto r_powers = powers_of_R.get(0, repetition);
    auto r_powers_L = r_powers.subspan(0, L / 2);
    // since we did not yet calculate the individual shares of S,T,P for the
    // parties, we do so now
    std::vector<GF_CHECK> lag_L_eval_at_R;
    lag_L_eval_at_R.reserve(L / 2);
    for (const auto &lag_poly :
         precomputation::precomputed_lagrange_polys_100) {
      lag_L_eval_at_R.push_back(lifted_dot_product(r_powers_L, lag_poly));
    }
    std::vector<GF_CHECK> lag_2L_min_1_eval_at_R;
    lag_2L_min_1_eval_at_R.reserve(L - 1);
    for (const auto &lag_poly :
         precomputation::precomputed_lagrange_polys_199) {
      lag_2L_min_1_eval_at_R.push_back(lifted_dot_product(r_powers, lag_poly));
    }

    GF_CHECK alpha1, alpha2;
#ifndef NDEBUG
    GF_CHECK S1_check, T1_check, P1_check;
    GF_CHECK S2_check, T2_check, P2_check;
#endif
    // execute sacrificing check protocol
    // alpha^i = eps * x^i + a^i
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alpha_shares = rep_alpha_shares.get(repetition, party);
      auto s_shares = rep_shared_s.get(repetition, party);
      auto a_shares = rep_shared_a.get(repetition, party);

      // calculate S(R) via dot product of lag polys evaluated at R and lifted
      // S
      GF_CHECK S1_at_R_share =
          lifted_dot_product_uint8(lag_L_eval_at_R, s_shares.subspan(0, L / 2));
      GF_CHECK S2_at_R_share = lifted_dot_product_uint8(
          lag_L_eval_at_R, s_shares.subspan(L / 2, L / 2));
#ifndef NDEBUG
      S1_check += S1_at_R_share;
      S2_check += S2_at_R_share;
#endif

      alpha_shares[0] = S1_at_R_share * epsilons[repetition][0] + a_shares[0];
      alpha_shares[1] = S2_at_R_share * epsilons[repetition][1] + a_shares[1];
      alpha1 += alpha_shares[0];
      alpha2 += alpha_shares[1];
    }
    // v^i = dot(eps, z^i) - c^i - dot(alpha, y^i)
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto t_shares = rep_shared_t.get(repetition, party);
      auto p_points1 = rep_shared_p_points1.get(repetition, party);
      auto p_points2 = rep_shared_p_points2.get(repetition, party);

      // calculate T(R) via dot product of lag polys evaluated at R and lifted
      // T
      GF_CHECK T1_at_R_share =
          lifted_dot_product_uint8(lag_L_eval_at_R, t_shares.subspan(0, L / 2));
      GF_CHECK T2_at_R_share = lifted_dot_product_uint8(
          lag_L_eval_at_R, t_shares.subspan(L / 2, L / 2));

      GF_CHECK P1_at_R_share =
          lifted_dot_product(lag_2L_min_1_eval_at_R, p_points1);
      GF_CHECK P2_at_R_share =
          lifted_dot_product(lag_2L_min_1_eval_at_R, p_points2);
#ifndef NDEBUG
      T1_check += T1_at_R_share;
      P1_check += P1_at_R_share;
      T2_check += T2_at_R_share;
      P2_check += P2_at_R_share;
#endif
      auto v_shares = rep_v_shares.get(repetition, party);
      v_shares[0] -= rep_shared_c.get(repetition, party)[0];
      v_shares[0] +=
          epsilons[repetition][0] * P1_at_R_share - alpha1 * T1_at_R_share;
      v_shares[0] +=
          epsilons[repetition][1] * P2_at_R_share - alpha2 * T2_at_R_share;
    }
#ifndef NDEBUG
    // sanity check, shares of S(R),T(R),P(R) are correct
    assert(S1_check == S1_at_R[repetition]);
    assert(T1_check == T1_at_R[repetition]);
    assert(P1_check == P1_at_R[repetition]);
    // sanity check, shares of S(R),T(R),P(R) are correct
    assert(S2_check == S2_at_R[repetition]);
    assert(T2_check == T2_at_R[repetition]);
    assert(P2_check == P2_at_R[repetition]);
    // sanity check: vs are zero
    GF_CHECK v(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto v_shares = rep_v_shares.get(repetition, party);
      v += v_shares[0];
    }
    assert(v == GF_CHECK(0));
#endif
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 4: challenge the views of the checking protocol
  /////////////////////////////////////////////////////////////////////////////

  std::vector<uint8_t> h_3 =
      phase_3_commitment(instance, salt, h_2, rep_alpha_shares, rep_v_shares);

  std::vector<uint16_t> missing_parties = phase_3_expand(instance, h_3);

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
  std::vector<repetition_proof_t<GF_CHECK>> proofs;
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    size_t missing_party = missing_parties[repetition];
    std::vector<uint8_t> commitment(instance.digest_size);
    auto missing_commitment =
        party_seed_commitments.get(repetition, missing_party);
    std::copy(std::begin(missing_commitment), std::end(missing_commitment),
              std::begin(commitment));
    auto missing_party_alpha = rep_alpha_shares.get(repetition, missing_party);
    std::array<GF_CHECK, 2> missing_alpha = {missing_party_alpha[0],
                                             missing_party_alpha[1]};
    std::array<std::vector<uint8_t>, 2> p_deltas = {rep_p_deltas1[repetition],
                                                    rep_p_deltas2[repetition]};
    repetition_proof_t<GF_CHECK> proof{
        seeds[repetition],        commitment, rep_key_deltas[repetition],
        rep_t_deltas[repetition], p_deltas,   missing_alpha,
        rep_c_deltas[repetition],
    };
    proofs.push_back(proof);
  }

  signature_t<GF_CHECK> signature{salt, h_1, h_3, proofs};

  return signature;
}

template <typename GF_CHECK>
bool helium_verify_template(const signature_instance_t &instance,
                           const std::vector<uint8_t> &pk,
                           const signature_t<GF_CHECK> &signature,
                           const uint8_t *message, size_t message_len) {

  // so we dont have to type the long variant everytime
  const size_t L = instance.num_multiplications;

  std::vector<uint8_t> pt(instance.block_cipher_params.block_size),
      ct(instance.block_cipher_params.block_size);
  memcpy(pt.data(), pk.data(), pt.size());
  memcpy(ct.data(), pk.data() + pt.size(), ct.size());

  // do parallel repetitions
  // create seed trees and random tapes
  std::vector<SeedTree> seed_trees;
  // key share + L*z_share + (L-2)*p_share + 2*a_share + c_share
  const size_t random_tape_size =
      instance.block_cipher_params.key_size + instance.num_multiplications +
      (instance.num_multiplications - 2) + 3 * GF_CHECK::BYTE_SIZE;

  RandomTapes random_tapes(instance.num_repetitions, instance.num_MPC_parties,
                           random_tape_size);
  RepByteContainer party_seed_commitments(
      instance.num_repetitions, instance.num_MPC_parties, instance.digest_size);

  // h1 expansion
  std::vector<GF_CHECK> r_values =
      phase_1_expand<GF_CHECK>(instance, signature.h_1);
  // h2 expansion
  std::vector<GF_CHECK> c_deltas;
  c_deltas.reserve(instance.num_repetitions);
  for (const repetition_proof_t<GF_CHECK> &proof : signature.proofs) {
    c_deltas.push_back(proof.c_delta);
  }
  std::vector<uint8_t> h_2 =
      phase_2_commitment(instance, signature.salt, signature.h_1, c_deltas);
  std::vector<std::array<GF_CHECK, 2>> epsilons =
      phase_2_expand<GF_CHECK>(instance, h_2);

  // h3 expansion already happened in deserialize to get missing parties
  std::vector<uint16_t> missing_parties =
      phase_3_expand(instance, signature.h_3);

  // rebuild SeedTrees
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF_CHECK> &proof = signature.proofs[repetition];
    // regenerate generate seed tree for the N parties (except the
    // missing one)
    if (missing_parties[repetition] != proof.reveallist.second)
      throw std::runtime_error(
          "modified signature between deserialization and verify");
    seed_trees.push_back(SeedTree(proof.reveallist, instance.num_MPC_parties,
                                  signature.salt, repetition));
    // commit to each party's seed, fill up missing one with data from
    // proof
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
  RepContainer<uint8_t> rep_shared_s(instance.num_repetitions,
                                     instance.num_MPC_parties, L);
  RepContainer<uint8_t> rep_shared_t(instance.num_repetitions,
                                     instance.num_MPC_parties, L);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF_CHECK> &proof = signature.proofs[repetition];

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
          repetition, party, instance.block_cipher_params.key_size, L);
      for (size_t ell = 0; ell < L; ell++) {
        shared_t[ell] = random_t_shares[ell];
      }
    }
    // fix first share
    auto first_shared_t = rep_shared_t.get(repetition, 0);
    std::transform(std::begin(proof.t_delta), std::end(proof.t_delta),
                   std::begin(first_shared_t), std::begin(first_shared_t),
                   std::bit_xor<uint8_t>());

    // get shares of sbox inputs by executing MPC AES
    auto shared_s = rep_shared_s.get_repetition(repetition);
    auto shared_t = rep_shared_t.get_repetition(repetition);

    if (instance.block_cipher_params.key_size == 16)
      AES128::aes_128_s_shares(rep_shared_keys.get_repetition(repetition),
                               shared_t, pt, ct, shared_s);
    else
      throw std::runtime_error("invalid parameters");
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute sharing of P
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<field::GF2_8> rep_shared_p_points1(
      instance.num_repetitions, instance.num_MPC_parties, L - 1);
  RepContainer<field::GF2_8> rep_shared_p_points2(
      instance.num_repetitions, instance.num_MPC_parties, L - 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_parties[repetition]) {
        auto shared_p1 = rep_shared_p_points1.get(repetition, party);
        auto shared_p2 = rep_shared_p_points2.get(repetition, party);

        // first L/2 points are set to 1
        for (size_t ell = 0; ell < L / 2; ell++) {
          if (party == 0) {
            shared_p1[ell] = field::GF2_8(1);
            shared_p2[ell] = field::GF2_8(1);
          } else {
            shared_p1[ell] = field::GF2_8(0);
            shared_p2[ell] = field::GF2_8(0);
          }
        }
        // next L/2-1 points are read from the random tape
        auto random_P_shares = random_tapes.get_bytes(
            repetition, party, instance.block_cipher_params.key_size + L,
            L - 2);
        for (size_t ell = L / 2; ell < L - 1; ell++) {
          shared_p1[ell] = random_P_shares[(ell - L / 2) * 2];
          shared_p2[ell] = random_P_shares[(ell - L / 2) * 2 + 1];
        }
      }
    }

    // fix party 0's share
    if (0 != missing_parties[repetition]) {
      const repetition_proof_t<GF_CHECK> &proof = signature.proofs[repetition];
      auto first_shared_p1 = rep_shared_p_points1.get(repetition, 0);
      auto first_shared_p2 = rep_shared_p_points2.get(repetition, 0);
      for (size_t ell = L / 2; ell < L - 1; ell++) {
        first_shared_p1[ell] += proof.p_delta[0][ell - L / 2];
        first_shared_p2[ell] += proof.p_delta[1][ell - L / 2];
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute dot-product triple, except missing shares
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF_CHECK> rep_shared_a(instance.num_repetitions,
                                      instance.num_MPC_parties, 2);
  RepContainer<GF_CHECK> rep_shared_c(instance.num_repetitions,
                                      instance.num_MPC_parties, 1);
  // also generate valid dot triple a,y,c
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_parties[repetition]) {
        auto random_a_c_bytes = random_tapes.get_bytes(
            repetition, party,
            instance.block_cipher_params.key_size + L + (L - 2),
            3 * GF_CHECK::BYTE_SIZE);

        auto shared_a = rep_shared_a.get(repetition, party);
        auto shared_c = rep_shared_c.get(repetition, party);
        shared_a[0].from_bytes(random_a_c_bytes.data());
        shared_a[1].from_bytes(random_a_c_bytes.data() + GF_CHECK::BYTE_SIZE);
        shared_c[0].from_bytes(random_a_c_bytes.data() +
                               2 * GF_CHECK::BYTE_SIZE);
      }
    }
    // fix party 0's share
    if (0 != missing_parties[repetition]) {
      const repetition_proof_t<GF_CHECK> &proof = signature.proofs[repetition];
      rep_shared_c.get(repetition, 0)[0] += proof.c_delta;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute views of sacrificing checks
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF_CHECK> powers_of_R(1, instance.num_repetitions, L - 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    auto powers = powers_of_R.get(0, repetition);

    powers[0] = GF_CHECK(1);
    for (size_t i = 1; i < L - 1; i++) {
      powers[i] = powers[i - 1] * r_values[repetition];
    }
  }

  RepContainer<GF_CHECK> rep_alpha_shares(instance.num_repetitions,
                                          instance.num_MPC_parties, 2);
  RepContainer<GF_CHECK> rep_v_shares(instance.num_repetitions,
                                      instance.num_MPC_parties, 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {

    const repetition_proof_t<GF_CHECK> &proof = signature.proofs[repetition];
    size_t missing_party = missing_parties[repetition];

    auto r_powers = powers_of_R.get(0, repetition);
    auto r_powers_L = r_powers.subspan(0, L / 2);
    // since we did not yet calculate the individual shares of S,T,P for the
    // parties, we do so now
    std::vector<GF_CHECK> lag_L_eval_at_R;
    lag_L_eval_at_R.reserve(L / 2);
    for (const auto &lag_poly :
         precomputation::precomputed_lagrange_polys_100) {
      lag_L_eval_at_R.push_back(lifted_dot_product(r_powers_L, lag_poly));
    }
    std::vector<GF_CHECK> lag_2L_min_1_eval_at_R;
    lag_2L_min_1_eval_at_R.reserve(L - 1);
    for (const auto &lag_poly :
         precomputation::precomputed_lagrange_polys_199) {
      lag_2L_min_1_eval_at_R.push_back(lifted_dot_product(r_powers, lag_poly));
    }

    // execute sacrificing check protocol
    // alpha^i = eps * x^i + a^i
    GF_CHECK alpha1, alpha2;
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alpha_shares = rep_alpha_shares.get(repetition, party);
      if (party != missing_party) {
        auto s_shares = rep_shared_s.get(repetition, party);
        auto a_shares = rep_shared_a.get(repetition, party);
        // calculate S(R) via dot product of lag polys evaluated at R and
        // lifted S
        GF_CHECK S1_at_R_share = lifted_dot_product_uint8(
            lag_L_eval_at_R, s_shares.subspan(0, L / 2));
        GF_CHECK S2_at_R_share = lifted_dot_product_uint8(
            lag_L_eval_at_R, s_shares.subspan(L / 2, L / 2));
        alpha_shares[0] = S1_at_R_share * epsilons[repetition][0] + a_shares[0];
        alpha_shares[1] = S2_at_R_share * epsilons[repetition][1] + a_shares[1];
        alpha1 += alpha_shares[0];
        alpha2 += alpha_shares[1];
      } else {
        // fill missing shares
        alpha_shares[0] = proof.missing_alpha[0];
        alpha1 += alpha_shares[0];
        alpha_shares[1] = proof.missing_alpha[1];
        alpha2 += alpha_shares[1];
      }
    }
    // v^i = dot(eps, z^i) - c^i - dot(alpha, y^i)
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_party) {
        auto t_shares = rep_shared_t.get(repetition, party);
        auto p_points1 = rep_shared_p_points1.get(repetition, party);
        auto p_points2 = rep_shared_p_points2.get(repetition, party);

        // calculate T(R) via dot product of lag polys evaluated at R and
        // lifted T
        GF_CHECK T1_at_R_share = lifted_dot_product_uint8(
            lag_L_eval_at_R, t_shares.subspan(0, L / 2));
        GF_CHECK T2_at_R_share = lifted_dot_product_uint8(
            lag_L_eval_at_R, t_shares.subspan(L / 2, L / 2));

        GF_CHECK P1_at_R_share =
            lifted_dot_product(lag_2L_min_1_eval_at_R, p_points1);
        GF_CHECK P2_at_R_share =
            lifted_dot_product(lag_2L_min_1_eval_at_R, p_points2);
        auto v_shares = rep_v_shares.get(repetition, party);
        v_shares[0] -= rep_shared_c.get(repetition, party)[0];
        v_shares[0] +=
            epsilons[repetition][0] * P1_at_R_share - alpha1 * T1_at_R_share;
        v_shares[0] +=
            epsilons[repetition][1] * P2_at_R_share - alpha2 * T2_at_R_share;
      }
    }

    // calculate missing shares as 0 - sum_{i!=missing} v^i
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_party) {
        auto v_shares = rep_v_shares.get(repetition, party);
        rep_v_shares.get(repetition, missing_party)[0] -= v_shares[0];
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  // recompute h_1 and h_3
  /////////////////////////////////////////////////////////////////////////////
  std::vector<std::vector<uint8_t>> sk_deltas;
  std::vector<std::vector<uint8_t>> t_deltas;
  std::vector<std::vector<uint8_t>> p_deltas1;
  std::vector<std::vector<uint8_t>> p_deltas2;
  sk_deltas.reserve(instance.num_repetitions);
  t_deltas.reserve(instance.num_repetitions);
  p_deltas1.reserve(instance.num_repetitions);
  p_deltas2.reserve(instance.num_repetitions);
  for (const repetition_proof_t<GF_CHECK> &proof : signature.proofs) {
    sk_deltas.push_back(proof.sk_delta);
    t_deltas.push_back(proof.t_delta);
    p_deltas1.push_back(proof.p_delta[0]);
    p_deltas2.push_back(proof.p_delta[1]);
  }
  std::vector<uint8_t> h_1 = phase_1_commitment(
      instance, signature.salt, pk, message, message_len,
      party_seed_commitments, sk_deltas, t_deltas, p_deltas1, p_deltas2);

  std::vector<uint8_t> h_3 = phase_3_commitment(instance, signature.salt, h_2,
                                                rep_alpha_shares, rep_v_shares);
  // do checks
  if (memcmp(h_1.data(), signature.h_1.data(), h_1.size()) != 0) {
    return false;
  }
  if (memcmp(h_3.data(), signature.h_3.data(), h_3.size()) != 0) {
    return false;
  }

  return true;
}

template <typename GF_CHECK>
std::vector<uint8_t>
helium_serialize_signature(const signature_instance_t &instance,
                          const signature_t<GF_CHECK> &signature) {
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
           instance.block_cipher_params.num_sboxes -
           16 +                                            // delta z
           (instance.block_cipher_params.num_sboxes - 2) + // delta P
           GF_CHECK::BYTE_SIZE * 2 +                       // missing alpha
           GF_CHECK::BYTE_SIZE);                           // delta c
  serialized.reserve(signature_size);

  serialized.insert(serialized.end(), signature.salt.begin(),
                    signature.salt.end());
  serialized.insert(serialized.end(), signature.h_1.begin(),
                    signature.h_1.end());
  serialized.insert(serialized.end(), signature.h_3.begin(),
                    signature.h_3.end());

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF_CHECK> &proof = signature.proofs[repetition];
    for (const std::vector<uint8_t> &seed : proof.reveallist.first) {
      serialized.insert(serialized.end(), seed.begin(), seed.end());
    }
    serialized.insert(serialized.end(), proof.Com_e.begin(), proof.Com_e.end());
    serialized.insert(serialized.end(), proof.sk_delta.begin(),
                      proof.sk_delta.end());
    serialized.insert(serialized.end(), proof.t_delta.begin(),
                      proof.t_delta.end());
    serialized.insert(serialized.end(), proof.p_delta[0].begin(),
                      proof.p_delta[0].end());
    serialized.insert(serialized.end(), proof.p_delta[1].begin(),
                      proof.p_delta[1].end());
    std::array<uint8_t, GF_CHECK::BYTE_SIZE> buf{};
    proof.missing_alpha[0].to_bytes(buf.data());
    serialized.insert(serialized.end(), buf.begin(), buf.end());
    proof.missing_alpha[1].to_bytes(buf.data());
    serialized.insert(serialized.end(), buf.begin(), buf.end());
    proof.c_delta.to_bytes(buf.data());
    serialized.insert(serialized.end(), buf.begin(), buf.end());
  }

  // calculation as expected
  assert(signature_size == serialized.size());
  return serialized;
}

template <typename GF_CHECK>
signature_t<GF_CHECK>
helium_deserialize_signature(const signature_instance_t &instance,
                            const std::vector<uint8_t> &serialized) {

  size_t current_offset = 0;
  salt_t salt{};
  memcpy(salt.data(), serialized.data() + current_offset, salt.size());
  current_offset += salt.size();
  std::vector<uint8_t> h_1(instance.digest_size), h_3(instance.digest_size);
  memcpy(h_1.data(), serialized.data() + current_offset, h_1.size());
  current_offset += h_1.size();
  memcpy(h_3.data(), serialized.data() + current_offset, h_3.size());
  current_offset += h_3.size();
  std::vector<repetition_proof_t<GF_CHECK>> proofs;
  proofs.reserve(instance.num_repetitions);

  std::vector<uint16_t> missing_parties = phase_3_expand(instance, h_3);
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

    std::vector<uint8_t> t_delta(instance.num_multiplications - 16);
    memcpy(t_delta.data(), serialized.data() + current_offset, t_delta.size());
    current_offset += t_delta.size();

    std::array<std::vector<uint8_t>, 2> p_delta;
    p_delta[0].resize(instance.block_cipher_params.num_sboxes / 2 - 1);
    memcpy(p_delta[0].data(), serialized.data() + current_offset,
           p_delta[0].size());
    current_offset += p_delta[0].size();
    p_delta[1].resize(instance.block_cipher_params.num_sboxes / 2 - 1);
    memcpy(p_delta[1].data(), serialized.data() + current_offset,
           p_delta[1].size());
    current_offset += p_delta[1].size();

    std::array<GF_CHECK, 2> missing_alpha;
    GF_CHECK c_delta;
    missing_alpha[0].from_bytes(serialized.data() + current_offset);
    current_offset += GF_CHECK::BYTE_SIZE;
    missing_alpha[1].from_bytes(serialized.data() + current_offset);
    current_offset += GF_CHECK::BYTE_SIZE;
    c_delta.from_bytes(serialized.data() + current_offset);
    current_offset += GF_CHECK::BYTE_SIZE;

    proofs.emplace_back(repetition_proof_t<GF_CHECK>{
        reveallist, Com_e, sk_delta, t_delta, p_delta, missing_alpha, c_delta});
  }

  assert(current_offset == serialized.size());
  signature_t<GF_CHECK> signature{salt, h_1, h_3, proofs};
  return signature;
}

#define USED_CHECK_FIELD field::GF2_144

std::vector<uint8_t> helium_sign(const signature_instance_t &instance,
                                const keypair_t &keypair,
                                const uint8_t *message, size_t message_len) {
  if (instance.block_cipher_params.block_size == 16) {
    auto sig = helium_sign_template<USED_CHECK_FIELD>(instance, keypair, message,
                                                     message_len);
    return helium_serialize_signature(instance, sig);
  }
  throw std::runtime_error("parameter set not implemented");
}

bool helium_verify(const signature_instance_t &instance,
                  const std::vector<uint8_t> &pk,
                  const std::vector<uint8_t> &signature, const uint8_t *message,
                  size_t message_len) {
  if (instance.block_cipher_params.block_size == 16) {
    auto sig =
        helium_deserialize_signature<USED_CHECK_FIELD>(instance, signature);
    return helium_verify_template<USED_CHECK_FIELD>(instance, pk, sig, message,
                                                   message_len);
  }
  throw std::runtime_error("parameter set not implemented");
}