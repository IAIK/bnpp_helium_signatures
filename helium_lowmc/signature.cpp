#include "signature.h"

#include "field.h"
#include "lowmc.h"
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

template <typename GF>
std::vector<uint8_t>
phase_1_commitment(const signature_instance_t &instance, const salt_t &salt,
                   const std::vector<uint8_t> &pk, const uint8_t *message,
                   size_t message_len, const RepByteContainer &commitments,
                   const RepByteContainer &output_broadcasts,
                   const std::vector<std::vector<uint8_t>> &key_deltas,
                   const std::vector<std::vector<GF>> &z_deltas,
                   const std::vector<std::vector<GF>> &p_deltas) {

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
    for (size_t ell = 0; ell < instance.num_lifted_multiplications - 1; ell++) {
      hash_update_GF2E(&ctx, p_deltas[repetition][ell]);
    }
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
std::vector<GF> phase_2_expand(const signature_instance_t &instance,
                               const std::vector<uint8_t> &h_2) {
  hash_context ctx;
  hash_init(&ctx, instance.digest_size);
  hash_update(&ctx, h_2.data(), h_2.size());
  hash_final(&ctx);

  std::array<uint8_t, GF::BYTE_SIZE> buffer;
  std::vector<GF> eps_values;
  eps_values.reserve(instance.num_repetitions);
  for (size_t e = 0; e < instance.num_repetitions; e++) {
    hash_squeeze(&ctx, buffer.data(), buffer.size());
    GF eps;
    // eps does not have restrictions
    eps.from_bytes(buffer.data());
    eps_values.push_back(eps);
  }
  return eps_values;
}

template <typename GF>
std::vector<uint8_t>
phase_3_commitment(const signature_instance_t &instance, const salt_t &salt,
                   const std::vector<uint8_t> &h_2,
                   const RepContainer<GF> &alpha_shares,
                   const std::vector<std::vector<GF>> &v_shares) {

  hash_context ctx;
  hash_init_prefix(&ctx, instance.digest_size, HASH_PREFIX_3);
  hash_update(&ctx, salt.data(), salt.size());
  hash_update(&ctx, h_2.data(), h_2.size());

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alphas = alpha_shares.get(repetition, party);
      hash_update_GF2E(&ctx, alphas[0]);
      hash_update_GF2E(&ctx, v_shares[repetition][party]);
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

template <typename GF, typename GF_CHECK>
signature_t<GF, GF_CHECK>
helium_sign_template(const signature_instance_t &instance,
                     const keypair_t &keypair, const uint8_t *message,
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
  // key share + L*z_share + (L-1)*p_share + a_share + c_share
  const size_t random_tape_size =
      instance.block_cipher_params.key_size +
      instance.num_lifted_multiplications * GF::BYTE_SIZE +
      (instance.num_lifted_multiplications - 1) * GF::BYTE_SIZE +
      2 * GF_CHECK::BYTE_SIZE;

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
  RepContainer<GF> rep_shared_p_points(instance.num_repetitions,
                                       instance.num_MPC_parties, 2 * L - 1);
  std::vector<std::vector<uint8_t>> rep_key_deltas;
  rep_key_deltas.reserve(instance.num_repetitions);
  std::vector<std::vector<GF>> rep_z_deltas;
  rep_z_deltas.reserve(instance.num_repetitions);
  std::vector<std::vector<GF>> rep_p_deltas;
  rep_p_deltas.reserve(instance.num_repetitions);

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

  // interpolate S and T
  std::vector<GF> y_values_S(L);
  std::vector<GF> y_values_T(L);
  for (size_t i = 0; i < L; i++) {
    y_values_S[i] = mult_triples[i][0];
    y_values_T[i] = mult_triples[i][1];
  }
  std::vector<GF> S_poly, T_poly;
  std::vector<GF> first_L_field_elements =
      field::get_first_n_field_elements<GF>(L);
  std::vector<GF> first_2L_min_1_field_elements =
      field::get_first_n_field_elements<GF>(2 * L - 1);
  if (L == 86) {
    S_poly = field::interpolate_with_precomputation(
        precomputation::precomputed_lagrange_polys_86, y_values_S);
    T_poly = field::interpolate_with_precomputation(
        precomputation::precomputed_lagrange_polys_86, y_values_T);
  } else if (L == 100) {
    std ::vector<std::vector<GF>> precomputed_lagrange_polys_L =
        field::precompute_lagrange_polynomials(first_L_field_elements);

    std::vector<std::vector<GF>> precomputed_lagrange_polys_2L_min_1 =
        field::precompute_lagrange_polynomials(first_2L_min_1_field_elements);

    S_poly = field::interpolate_with_precomputation(
        precomputed_lagrange_polys_L, y_values_S);
    T_poly = field::interpolate_with_precomputation(
        precomputed_lagrange_polys_L, y_values_T);
  } else
    throw std::runtime_error("not implemented");

  std::vector<GF> P_poly = S_poly * T_poly;
  std::vector<GF> P_at_k(L - 1);
  for (size_t ell = L; ell < 2 * L - 1; ell++) {
    P_at_k[ell - L] = field::eval(P_poly, first_2L_min_1_field_elements[ell]);
  }

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    std::vector<GF> p_delta = P_at_k;
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto shared_p = rep_shared_p_points.get(repetition, party);
      auto shared_z = rep_shared_z.get(repetition, party);

      // first L points are set to z
      for (size_t ell = 0; ell < L; ell++) {
        shared_p[ell] = shared_z[ell];
      }
      // next L-1 points are read from the random tape
      auto random_P_shares = random_tapes.get_bytes(
          repetition, party,
          instance.block_cipher_params.key_size + L * GF::BYTE_SIZE,
          (L - 1) * GF::BYTE_SIZE);
      for (size_t ell = L; ell < 2 * L - 1; ell++) {
        shared_p[ell] = random_P_shares[ell - L];
        p_delta[ell - L] -= random_P_shares[ell - L];
      }
    }

    auto first_shared_p = rep_shared_p_points.get(repetition, 0);
    for (size_t ell = L; ell < 2 * L - 1; ell++) {
      first_shared_p[ell] += p_delta[ell - L];
    }
    rep_p_deltas.push_back(p_delta);
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 2: challenge for the polynomial evaluation
  /////////////////////////////////////////////////////////////////////////////

  // commit to salt, (all commitments of parties seeds,
  // output_broadcasts, key_delta, t_delta, P_delta) for all
  // repetitions
  std::vector<uint8_t> h_1 =
      phase_1_commitment(instance, salt, keypair.second, message, message_len,
                         party_seed_commitments, rep_output_broadcasts,
                         rep_key_deltas, rep_z_deltas, rep_p_deltas);

  // expand challenge hash to R values
  std::vector<GF_CHECK> r_values = phase_1_expand<GF_CHECK>(instance, h_1);

  /////////////////////////////////////////////////////////////////////////////
  // phase 3: evaluate the polynomial and calculate/commit to a checking
  // triple
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF_CHECK> powers_of_R(1, instance.num_repetitions, 2 * L - 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    auto powers = powers_of_R.get(0, repetition);

    powers[0] = GF_CHECK(1);
    for (size_t i = 1; i < 2 * L - 1; i++) {
      powers[i] = powers[i - 1] * r_values[repetition];
    }
  }

  std::vector<GF_CHECK> S_at_R;
  S_at_R.reserve(instance.num_repetitions);
  std::vector<GF_CHECK> T_at_R;
  T_at_R.reserve(instance.num_repetitions);
  std::vector<GF_CHECK> P_at_R;
  P_at_R.reserve(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    auto r_powers = powers_of_R.get(0, repetition);
    auto r_powers_L = r_powers.subspan(0, L);
    S_at_R.push_back(lifted_dot_product(r_powers_L, S_poly));
    T_at_R.push_back(lifted_dot_product(r_powers_L, T_poly));
    P_at_R.push_back(lifted_dot_product(r_powers, P_poly));
#ifndef NDEBUG
    assert(S_at_R[repetition] * T_at_R[repetition] == P_at_R[repetition]);
#endif
  }

  // build a triple a*b = c, with b = T(R)

  RepContainer<GF_CHECK> rep_shared_a(instance.num_repetitions,
                                      instance.num_MPC_parties, 1);
  RepContainer<GF_CHECK> rep_shared_c(instance.num_repetitions,
                                      instance.num_MPC_parties, 1);

  std::vector<GF_CHECK> rep_c_deltas;
  rep_c_deltas.reserve(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    GF_CHECK a(0), c(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {

      auto random_a_c_bytes = random_tapes.get_bytes(
          repetition, party,
          instance.block_cipher_params.key_size + L * GF::BYTE_SIZE +
              (L - 1) * GF::BYTE_SIZE,
          2 * GF_CHECK::BYTE_SIZE);

      auto shared_a = rep_shared_a.get(repetition, party);
      auto shared_c = rep_shared_c.get(repetition, party);
      shared_a[0].from_bytes(random_a_c_bytes.data());
      shared_c[0].from_bytes(random_a_c_bytes.data() + GF_CHECK::BYTE_SIZE);
      a += shared_a[0];
      c += shared_c[0];
    }
    // calc c_delta and fix first parties share
    GF_CHECK c_delta = a * T_at_R[repetition] - c;
    rep_shared_c.get(repetition, 0)[0] += c_delta;
    rep_c_deltas.push_back(c_delta);
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 4: challenge for the multiplication check
  /////////////////////////////////////////////////////////////////////////////

  std::vector<uint8_t> h_2 =
      phase_2_commitment(instance, salt, h_1, rep_c_deltas);

  // expand challenge hash to epsilon values
  std::vector<GF_CHECK> epsilons = phase_2_expand<GF_CHECK>(instance, h_2);

  /////////////////////////////////////////////////////////////////////////////
  // phase 5: commit to the views of the checking protocol
  /////////////////////////////////////////////////////////////////////////////

  RepContainer<GF_CHECK> rep_alpha_shares(instance.num_repetitions,
                                          instance.num_MPC_parties, 1);
  std::vector<std::vector<GF_CHECK>> v_shares(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    v_shares[repetition].resize(instance.num_MPC_parties);

#ifndef NDEBUG
    // sanity check: a * T(R) =c
    GF_CHECK acc(0);
    GF_CHECK ai(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      ai += rep_shared_a.get(repetition, party)[0];
    }
    acc += ai * T_at_R[repetition];
    GF_CHECK c(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      c += rep_shared_c.get(repetition, party)[0];
    }
    assert(acc == c);
#endif

    auto r_powers = powers_of_R.get(0, repetition);
    auto r_powers_L = r_powers.subspan(0, L);
    // since we did not yet calculate the individual shares of S,T,P for the
    // parties, we do so now
    std::vector<GF_CHECK> lag_L_eval_at_R;
    lag_L_eval_at_R.reserve(L);
    std::vector<GF_CHECK> lag_2L_min_1_eval_at_R;
    lag_2L_min_1_eval_at_R.reserve(2 * L - 1);
    if (L == 86) {
      for (const auto &lag_poly :
           precomputation::precomputed_lagrange_polys_86) {
        lag_L_eval_at_R.push_back(lifted_dot_product(r_powers_L, lag_poly));
      }
      for (const auto &lag_poly :
           precomputation::precomputed_lagrange_polys_171) {
        lag_2L_min_1_eval_at_R.push_back(
            lifted_dot_product(r_powers, lag_poly));
      }
    } else if (L == 100) {
      for (const auto &lag_poly :
           precomputation::precomputed_lagrange_polys_100) {
        lag_L_eval_at_R.push_back(lifted_dot_product(r_powers_L, lag_poly));
      }
      for (const auto &lag_poly :
           precomputation::precomputed_lagrange_polys_199) {
        lag_2L_min_1_eval_at_R.push_back(
            lifted_dot_product(r_powers, lag_poly));
      }
    }

    GF_CHECK alpha;
#ifndef NDEBUG
    GF_CHECK S_check, T_check, P_check;
#endif
    // execute sacrificing check protocol
    // alpha^i = eps * x^i + a^i
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alpha_shares = rep_alpha_shares.get(repetition, party);
      auto x_shares = rep_shared_x.get(repetition, party);
      auto a_shares = rep_shared_a.get(repetition, party);

      // calculate S(R) via dot product of lag polys evaluated at R and lifted
      // X
      GF_CHECK S_at_R_share = lifted_dot_product(lag_L_eval_at_R, x_shares);
#ifndef NDEBUG
      S_check += S_at_R_share;
#endif

      alpha_shares[0] = S_at_R_share * epsilons[repetition] + a_shares[0];
      alpha += alpha_shares[0];
    }
    // v^i = dot(eps, z^i) - c^i - dot(alpha, y^i)
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto y_shares = rep_shared_y.get(repetition, party);
      auto p_points = rep_shared_p_points.get(repetition, party);

      // calculate S(R) via dot product of lag polys evaluated at R and lifted
      // X
      GF_CHECK T_at_R_share = lifted_dot_product(lag_L_eval_at_R, y_shares);
      GF_CHECK P_at_R_share =
          lifted_dot_product(lag_2L_min_1_eval_at_R, p_points);
#ifndef NDEBUG
      T_check += T_at_R_share;
      P_check += P_at_R_share;
#endif
      v_shares[repetition][party] -= rep_shared_c.get(repetition, party)[0];
      v_shares[repetition][party] +=
          epsilons[repetition] * P_at_R_share - alpha * T_at_R_share;
    }
#ifndef NDEBUG
    // sanity check, shares of S(R),T(R),P(R) are correct
    assert(S_check == S_at_R[repetition]);
    assert(T_check == T_at_R[repetition]);
    assert(P_check == P_at_R[repetition]);
    // sanity check: vs are zero
    GF_CHECK v(0);
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      v += v_shares[repetition][party];
    }
    assert(v == GF_CHECK(0));
#endif
  }

  /////////////////////////////////////////////////////////////////////////////
  // phase 4: challenge the views of the checking protocol
  /////////////////////////////////////////////////////////////////////////////

  std::vector<uint8_t> h_3 =
      phase_3_commitment(instance, salt, h_2, rep_alpha_shares, v_shares);

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
  std::vector<repetition_proof_t<GF, GF_CHECK>> proofs;
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    size_t missing_party = missing_parties[repetition];
    std::vector<uint8_t> commitment(instance.digest_size);
    auto missing_commitment =
        party_seed_commitments.get(repetition, missing_party);
    std::copy(std::begin(missing_commitment), std::end(missing_commitment),
              std::begin(commitment));
    auto missing_party_alpha = rep_alpha_shares.get(repetition, missing_party);
    GF_CHECK missing_alpha(missing_party_alpha[0]);
    repetition_proof_t<GF, GF_CHECK> proof{
        seeds[repetition],          commitment,
        rep_key_deltas[repetition], rep_z_deltas[repetition],
        rep_p_deltas[repetition],   missing_alpha,
        rep_c_deltas[repetition],
    };
    proofs.push_back(proof);
  }

  signature_t<GF, GF_CHECK> signature{salt, h_1, h_3, proofs};

  return signature;
}

template <typename GF, typename GF_CHECK>
bool helium_verify_template(const signature_instance_t &instance,
                            const std::vector<uint8_t> &pk,
                            const signature_t<GF, GF_CHECK> &signature,
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
  // key share + L*z_share + (L-1)*p_share + a_share + c_share
  const size_t random_tape_size =
      instance.block_cipher_params.key_size +
      instance.num_lifted_multiplications * GF::BYTE_SIZE +
      (instance.num_lifted_multiplications - 1) * GF::BYTE_SIZE +
      2 * GF_CHECK::BYTE_SIZE;

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
  for (const repetition_proof_t<GF, GF_CHECK> &proof : signature.proofs) {
    c_deltas.push_back(proof.c_delta);
  }
  std::vector<uint8_t> h_2 =
      phase_2_commitment(instance, signature.salt, signature.h_1, c_deltas);
  std::vector<GF_CHECK> epsilons = phase_2_expand<GF_CHECK>(instance, h_2);

  // h3 expansion already happened in deserialize to get missing parties
  std::vector<uint16_t> missing_parties =
      phase_3_expand(instance, signature.h_3);

  // rebuild SeedTrees
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF, GF_CHECK> &proof =
        signature.proofs[repetition];
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
    const repetition_proof_t<GF, GF_CHECK> &proof =
        signature.proofs[repetition];

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
  // recompute sharing of P
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF> rep_shared_p_points(instance.num_repetitions,
                                       instance.num_MPC_parties, 2 * L - 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_parties[repetition]) {
        auto shared_p = rep_shared_p_points.get(repetition, party);
        auto shared_z = rep_shared_z.get(repetition, party);

        // first L points are set to z
        for (size_t ell = 0; ell < L; ell++) {
          shared_p[ell] = shared_z[ell];
        }
        // next L-1 points are read from the random tape
        auto random_P_shares = random_tapes.get_bytes(
            repetition, party,
            instance.block_cipher_params.key_size + L * GF::BYTE_SIZE,
            (L - 1) * GF::BYTE_SIZE);
        for (size_t ell = L; ell < 2 * L - 1; ell++) {
          shared_p[ell] = random_P_shares[ell - L];
        }
      }
    }

    // fix party 0's share
    if (0 != missing_parties[repetition]) {
      const repetition_proof_t<GF, GF_CHECK> &proof =
          signature.proofs[repetition];
      auto first_shared_p = rep_shared_p_points.get(repetition, 0);
      for (size_t ell = L; ell < 2 * L - 1; ell++) {
        first_shared_p[ell] += proof.p_delta[ell - L];
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute dot-product triple, except missing shares
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF_CHECK> rep_shared_a(instance.num_repetitions,
                                      instance.num_MPC_parties, 1);
  RepContainer<GF_CHECK> rep_shared_c(instance.num_repetitions,
                                      instance.num_MPC_parties, 1);
  // also generate valid dot triple a,y,c and save c_delta
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_parties[repetition]) {
        auto random_a_c_bytes = random_tapes.get_bytes(
            repetition, party,
            instance.block_cipher_params.key_size + L * GF::BYTE_SIZE +
                (L - 1) * GF::BYTE_SIZE,
            2 * GF_CHECK::BYTE_SIZE);

        auto shared_a = rep_shared_a.get(repetition, party);
        auto shared_c = rep_shared_c.get(repetition, party);
        shared_a[0].from_bytes(random_a_c_bytes.data());
        shared_c[0].from_bytes(random_a_c_bytes.data() + GF_CHECK::BYTE_SIZE);
      }
    }
    // fix party 0's share
    if (0 != missing_parties[repetition]) {
      const repetition_proof_t<GF, GF_CHECK> &proof =
          signature.proofs[repetition];
      rep_shared_c.get(repetition, 0)[0] += proof.c_delta;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // recompute views of sacrificing checks
  /////////////////////////////////////////////////////////////////////////////
  RepContainer<GF_CHECK> powers_of_R(1, instance.num_repetitions, 2 * L - 1);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    auto powers = powers_of_R.get(0, repetition);

    powers[0] = GF_CHECK(1);
    for (size_t i = 1; i < 2 * L - 1; i++) {
      powers[i] = powers[i - 1] * r_values[repetition];
    }
  }

  RepContainer<GF_CHECK> rep_alpha_shares(instance.num_repetitions,
                                          instance.num_MPC_parties, 1);
  std::vector<std::vector<GF_CHECK>> v_shares(instance.num_repetitions);

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {

    const repetition_proof_t<GF, GF_CHECK> &proof =
        signature.proofs[repetition];
    size_t missing_party = missing_parties[repetition];

    auto r_powers = powers_of_R.get(0, repetition);
    auto r_powers_L = r_powers.subspan(0, L);
    // since we did not yet calculate the individual shares of S,T,P for the
    // parties, we do so now
    std::vector<GF_CHECK> lag_L_eval_at_R;
    lag_L_eval_at_R.reserve(L);
    std::vector<GF_CHECK> lag_2L_min_1_eval_at_R;
    lag_2L_min_1_eval_at_R.reserve(2 * L - 1);
    if (L == 86) {
      for (const auto &lag_poly :
           precomputation::precomputed_lagrange_polys_86) {
        lag_L_eval_at_R.push_back(lifted_dot_product(r_powers_L, lag_poly));
      }
      for (const auto &lag_poly :
           precomputation::precomputed_lagrange_polys_171) {
        lag_2L_min_1_eval_at_R.push_back(
            lifted_dot_product(r_powers, lag_poly));
      }
    } else if (L == 100) {
      for (const auto &lag_poly :
           precomputation::precomputed_lagrange_polys_100) {
        lag_L_eval_at_R.push_back(lifted_dot_product(r_powers_L, lag_poly));
      }
      for (const auto &lag_poly :
           precomputation::precomputed_lagrange_polys_199) {
        lag_2L_min_1_eval_at_R.push_back(
            lifted_dot_product(r_powers, lag_poly));
      }
    }

    // execute sacrificing check protocol
    // alpha^i = eps * x^i + a^i
    GF_CHECK alpha;
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      auto alpha_shares = rep_alpha_shares.get(repetition, party);
      if (party != missing_party) {
        auto x_shares = rep_shared_x.get(repetition, party);
        auto a_shares = rep_shared_a.get(repetition, party);
        // calculate S(R) via dot product of lag polys evaluated at R and
        // lifted X
        GF_CHECK S_at_R_share = lifted_dot_product(lag_L_eval_at_R, x_shares);
        alpha_shares[0] = S_at_R_share * epsilons[repetition] + a_shares[0];
        alpha += alpha_shares[0];
      } else {
        // fill missing shares
        alpha_shares[0] = proof.missing_alpha;
        alpha += alpha_shares[0];
      }
    }
    v_shares[repetition].resize(instance.num_MPC_parties);
    // v^i = dot(eps, z^i) - c^i - dot(alpha, y^i)
    for (size_t party = 0; party < instance.num_MPC_parties; party++) {
      if (party != missing_party) {
        v_shares[repetition][party] -= rep_shared_c.get(repetition, party)[0];
        auto y_shares = rep_shared_y.get(repetition, party);
        auto p_points = rep_shared_p_points.get(repetition, party);

        // calculate S(R) via dot product of lag polys evaluated at R and
        // lifted X
        GF_CHECK T_at_R_share = lifted_dot_product(lag_L_eval_at_R, y_shares);
        GF_CHECK P_at_R_share =
            lifted_dot_product(lag_2L_min_1_eval_at_R, p_points);
        v_shares[repetition][party] +=
            epsilons[repetition] * P_at_R_share - alpha * T_at_R_share;
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
  // recompute h_1 and h_3
  /////////////////////////////////////////////////////////////////////////////
  std::vector<std::vector<uint8_t>> sk_deltas;
  std::vector<std::vector<GF>> z_deltas;
  std::vector<std::vector<GF>> p_deltas;
  sk_deltas.reserve(instance.num_repetitions);
  z_deltas.reserve(instance.num_repetitions);
  p_deltas.reserve(instance.num_repetitions);
  for (const repetition_proof_t<GF, GF_CHECK> &proof : signature.proofs) {
    sk_deltas.push_back(proof.sk_delta);
    z_deltas.push_back(proof.z_delta);
    p_deltas.push_back(proof.p_delta);
  }
  std::vector<uint8_t> h_1 =
      phase_1_commitment(instance, signature.salt, pk, message, message_len,
                         party_seed_commitments, rep_output_broadcasts,
                         sk_deltas, z_deltas, p_deltas);

  std::vector<uint8_t> h_3 = phase_3_commitment(instance, signature.salt, h_2,
                                                rep_alpha_shares, v_shares);
  // do checks
  if (memcmp(h_1.data(), signature.h_1.data(), h_1.size()) != 0) {
    return false;
  }
  if (memcmp(h_3.data(), signature.h_3.data(), h_3.size()) != 0) {
    return false;
  }

  return true;
}

template <typename GF, typename GF_CHECK>
std::vector<uint8_t>
helium_serialize_signature(const signature_instance_t &instance,
                           const signature_t<GF, GF_CHECK> &signature) {
  std::vector<uint8_t> serialized;

  // prealloc signature size
  const size_t packed_GF_elements_size =
      (instance.num_repetitions *                                // tau times
           (instance.num_lifted_multiplications * GF::BIT_SIZE + // (delta zs +
            (instance.num_lifted_multiplications - 1) *
                GF::BIT_SIZE) + // delta P +
       7)                       // round up to next byte
      / 8;
  const size_t signature_size =
      signature.salt.size() +    // salt
      instance.digest_size * 2 + // h_1, h_2
      instance.num_repetitions * // tau x
          (ceil_log2(instance.num_MPC_parties) *
               instance.seed_size +                // merkle tree path
           instance.digest_size +                  // Com_e
           instance.block_cipher_params.key_size + // delta sk
           GF_CHECK::BYTE_SIZE +                   // missing alpha
           GF_CHECK::BYTE_SIZE) +                  // delta c
      packed_GF_elements_size;
  serialized.reserve(signature_size);

  serialized.insert(serialized.end(), signature.salt.begin(),
                    signature.salt.end());
  serialized.insert(serialized.end(), signature.h_1.begin(),
                    signature.h_1.end());
  serialized.insert(serialized.end(), signature.h_3.begin(),
                    signature.h_3.end());

  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    const repetition_proof_t<GF, GF_CHECK> &proof =
        signature.proofs[repetition];
    for (const std::vector<uint8_t> &seed : proof.reveallist.first) {
      serialized.insert(serialized.end(), seed.begin(), seed.end());
    }
    serialized.insert(serialized.end(), proof.Com_e.begin(), proof.Com_e.end());
    serialized.insert(serialized.end(), proof.sk_delta.begin(),
                      proof.sk_delta.end());
    std::array<uint8_t, GF_CHECK::BYTE_SIZE> buf{};
    proof.missing_alpha.to_bytes(buf.data());
    serialized.insert(serialized.end(), buf.begin(), buf.end());
    proof.c_delta.to_bytes(buf.data());
    serialized.insert(serialized.end(), buf.begin(), buf.end());
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
    const repetition_proof_t<GF, GF_CHECK> &proof =
        signature.proofs[repetition];
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
    // p_deltas
    for (const GF &ele : proof.p_delta) {
      uint64_t tmp = ele.data;
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

template <typename GF, typename GF_CHECK>
signature_t<GF, GF_CHECK>
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
  std::vector<repetition_proof_t<GF, GF_CHECK>> proofs;
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

    GF_CHECK missing_alpha, c_delta;
    missing_alpha.from_bytes(serialized.data() + current_offset);
    current_offset += GF_CHECK::BYTE_SIZE;
    c_delta.from_bytes(serialized.data() + current_offset);
    current_offset += GF_CHECK::BYTE_SIZE;

    std::vector<GF> z_delta(instance.num_lifted_multiplications);
    std::vector<GF> p_delta(instance.num_lifted_multiplications - 1);

    proofs.emplace_back(repetition_proof_t<GF, GF_CHECK>{
        reveallist, Com_e, sk_delta, z_delta, p_delta, missing_alpha, c_delta});
  }
  // unpack tightly packed GF elements
  std::vector<uint8_t> packed_GF_elements(serialized.begin() + current_offset,
                                          serialized.end());
  // buffer so we can always read uint64_t values without invalid memory
  // access
  packed_GF_elements.resize(packed_GF_elements.size() + 7);
  // current implementation only works for this constraint
  static_assert(GF::BIT_SIZE <= 64);
  size_t total_bits_read = 0;
  size_t current_shift = 0;
  const size_t uneven_bits = (GF::BIT_SIZE % 8);
  for (size_t repetition = 0; repetition < instance.num_repetitions;
       repetition++) {
    repetition_proof_t<GF, GF_CHECK> &proof = proofs[repetition];
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
    // p_delta
    for (GF &ele : proof.p_delta) {
      size_t current_byte_index = (total_bits_read) / 8;
      uint64_t tmp =
          *((uint64_t *)(packed_GF_elements.data() + current_byte_index));
      tmp = le64toh(tmp) >> current_shift;
      current_shift = (current_shift + uneven_bits) % 8;
      ele.data = tmp & GF::ELEMENT_MASK;
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
  signature_t<GF, GF_CHECK> signature{salt, h_1, h_3, proofs};
  return signature;
}

std::vector<uint8_t> helium_sign(const signature_instance_t &instance,
                                 const keypair_t &keypair,
                                 const uint8_t *message, size_t message_len) {
  if (instance.block_cipher_params.block_size == 16 ||
      instance.block_cipher_params.block_size == 17) {
    auto sig = helium_sign_template<field::GF2_9, field::GF2_144>(
        instance, keypair, message, message_len);
    return helium_serialize_signature(instance, sig);
  }
  throw std::runtime_error("parameter set not implemented");
}

bool helium_verify(const signature_instance_t &instance,
                   const std::vector<uint8_t> &pk,
                   const std::vector<uint8_t> &signature,
                   const uint8_t *message, size_t message_len) {
  if (instance.block_cipher_params.block_size == 16 ||
      instance.block_cipher_params.block_size == 17) {
    auto sig = helium_deserialize_signature<field::GF2_9, field::GF2_144>(
        instance, signature);
    return helium_verify_template<field::GF2_9, field::GF2_144>(
        instance, pk, sig, message, message_len);
  }
  throw std::runtime_error("parameter set not implemented");
}