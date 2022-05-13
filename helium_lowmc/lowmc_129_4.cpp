#include "lowmc_129_4.h"
#include "field/matmul.h"
#include "field/rmfe.h"
#include "macros.h"
#include <cstring>

#include "lowmc_129_4_constants.h"

namespace {
void u64_array_to_char_array(uint8_t *dst, const std::array<uint64_t, 3> &data,
                             size_t len) {
  const size_t word_count = (len + 7) / sizeof(uint64_t);

  for (size_t i = word_count; i;
       --i, dst += sizeof(uint64_t), len -= sizeof(uint64_t)) {
    const uint64_t tmp = htobe64(data[i - 1]);
    memcpy(dst, &tmp, MIN(sizeof(tmp), len));
  }
}

void u64_array_from_char_array(std::array<uint64_t, 3> &result,
                               const uint8_t *data, size_t len) {
  const size_t word_count = (len + 7) / sizeof(uint64_t);

  for (size_t i = word_count; i;
       --i, data += sizeof(uint64_t), len -= sizeof(uint64_t)) {
    uint64_t tmp = 0;
    memcpy(&tmp, data, MIN(sizeof(tmp), len));
    result[i - 1] = be64toh(tmp);
  }
}
inline void xor_arrays(std::array<uint64_t, 3> &out,
                       const std::array<uint64_t, 3> &in1,
                       const std::array<uint64_t, 3> &in2) {
  out[0] = in1[0] ^ in2[0];
  out[1] = in1[1] ^ in2[1];
  out[2] = in1[2] ^ in2[2];
}

inline std::array<uint64_t, 3>
sbox(const std::array<uint64_t, 3> &state,
     std::vector<std::array<field::GF2_3, 3>> &sbox_values) {
  std::array<uint64_t, 3> result{};
  for (size_t i = 0; i < 43; i++) {
    // get input of sbox
    size_t idx = 192 - 1 - i * 3;
    uint64_t c = (state[idx / 64] >> (idx % 64)) & 1;
    idx--;
    uint64_t b = (state[idx / 64] >> (idx % 64)) & 1;
    idx--;
    uint64_t a = (state[idx / 64] >> (idx % 64)) & 1;

    uint64_t d = a ^ (b & c);
    uint64_t e = a ^ b ^ (a & c);
    uint64_t f = a ^ b ^ c ^ (a & b);
    field::GF2_3 in1((a << 2) | (b << 1) | c);
    field::GF2_3 in2(((a ^ b) << 2) | (a << 1) | c);
    field::GF2_3 out(((d) << 2) | ((d ^ e) << 1) | f);
    assert(in1 * in2 == out);
    sbox_values.push_back({in1, in2, out});

    idx = 192 - 1 - i * 3;
    result[idx / 64] |= f << (idx % 64);
    idx--;
    result[idx / 64] |= e << (idx % 64);
    idx--;
    result[idx / 64] |= d << (idx % 64);
  }
  return result;
}

std::array<uint64_t, 3> mpc_sbox_9(const std::array<uint64_t, 3> &state,
                                   size_t round, field::GF2_9 *x_shares,
                                   field::GF2_9 *y_shares,
                                   const field::GF2_9 *z_shares) {
  std::array<uint64_t, 3> result{};
  for (size_t i = 0; i < 43; i++) {
    const size_t sbox_idx = round * 43 + i;
    const size_t rmfe_idx = sbox_idx / 2;
    const size_t bit_idx = (sbox_idx % 2) * 3;
    // get input of sbox
    size_t idx = 192 - 1 - i * 3;
    const uint64_t c = (state[idx / 64] >> (idx % 64)) & 1;
    idx--;
    const uint64_t b = (state[idx / 64] >> (idx % 64)) & 1;
    idx--;
    const uint64_t a = (state[idx / 64] >> (idx % 64)) & 1;

    const uint64_t c_rmfe = rmfe::phi_2_3_single_bit(c, bit_idx + 0);
    x_shares[rmfe_idx].data ^= rmfe::phi_2_3_single_bit(a, bit_idx + 2);
    x_shares[rmfe_idx].data ^= rmfe::phi_2_3_single_bit(b, bit_idx + 1);
    x_shares[rmfe_idx].data ^= c_rmfe;
    y_shares[rmfe_idx].data ^= rmfe::phi_2_3_single_bit(a ^ b, bit_idx + 2);
    y_shares[rmfe_idx].data ^= rmfe::phi_2_3_single_bit(a, bit_idx + 1);
    y_shares[rmfe_idx].data ^= c_rmfe;

    const uint64_t z = z_shares[rmfe_idx].data;
    const uint64_t d = rmfe::psi_2_3_single_bit(z, bit_idx + 2);
    const uint64_t e = rmfe::psi_2_3_single_bit(z, bit_idx + 1);
    const uint64_t f = rmfe::psi_2_3_single_bit(z, bit_idx + 0);

    idx = 192 - 1 - i * 3;
    result[idx / 64] |= f << (idx % 64);
    idx--;
    result[idx / 64] |= (e ^ d) << (idx % 64);
    idx--;
    result[idx / 64] |= d << (idx % 64);
  }
  return result;
}

std::array<uint64_t, 3> mpc_sbox_45(const std::array<uint64_t, 3> &state,
                                    size_t round, field::GF2_45 *x_shares,
                                    field::GF2_45 *y_shares,
                                    const field::GF2_45 *z_shares) {
  std::array<uint64_t, 3> result{};
  for (size_t i = 0; i < 43; i++) {
    const size_t sbox_idx = round * 43 + i;
    const size_t rmfe_idx = sbox_idx / 8;
    const size_t bit_idx = (sbox_idx % 8) * 3;
    // get input of sbox
    size_t idx = 192 - 1 - i * 3;
    const uint64_t c = (state[idx / 64] >> (idx % 64)) & 1;
    idx--;
    const uint64_t b = (state[idx / 64] >> (idx % 64)) & 1;
    idx--;
    const uint64_t a = (state[idx / 64] >> (idx % 64)) & 1;

    const uint64_t c_rmfe = rmfe::phi_8_15_single_bit(c, bit_idx + 0);
    x_shares[rmfe_idx].data ^= rmfe::phi_8_15_single_bit(a, bit_idx + 2);
    x_shares[rmfe_idx].data ^= rmfe::phi_8_15_single_bit(b, bit_idx + 1);
    x_shares[rmfe_idx].data ^= c_rmfe;
    y_shares[rmfe_idx].data ^= rmfe::phi_8_15_single_bit(a ^ b, bit_idx + 2);
    y_shares[rmfe_idx].data ^= rmfe::phi_8_15_single_bit(a, bit_idx + 1);
    y_shares[rmfe_idx].data ^= c_rmfe;

    const uint64_t z = z_shares[rmfe_idx].data;
    const uint64_t d = rmfe::psi_8_15_single_bit(z, bit_idx + 2);
    const uint64_t e = rmfe::psi_8_15_single_bit(z, bit_idx + 1);
    const uint64_t f = rmfe::psi_8_15_single_bit(z, bit_idx + 0);

    idx = 192 - 1 - i * 3;
    result[idx / 64] |= f << (idx % 64);
    idx--;
    result[idx / 64] |= (e ^ d) << (idx % 64);
    idx--;
    result[idx / 64] |= d << (idx % 64);
  }
  return result;
}

std::array<uint64_t, 3> mpc_sbox_51(const std::array<uint64_t, 3> &state,
                                    size_t round, field::GF2_51 *x_shares,
                                    field::GF2_51 *y_shares,
                                    const field::GF2_51 *z_shares) {
  std::array<uint64_t, 3> result{};
  for (size_t i = 0; i < 43; i++) {
    const size_t sbox_idx = round * 43 + i;
    const size_t rmfe_idx = sbox_idx / 9;
    const size_t bit_idx = (sbox_idx % 9) * 3;
    // get input of sbox
    size_t idx = 192 - 1 - i * 3;
    const uint64_t c = (state[idx / 64] >> (idx % 64)) & 1;
    idx--;
    const uint64_t b = (state[idx / 64] >> (idx % 64)) & 1;
    idx--;
    const uint64_t a = (state[idx / 64] >> (idx % 64)) & 1;

    const uint64_t c_rmfe = rmfe::phi_9_17_single_bit(c, bit_idx + 0);
    x_shares[rmfe_idx].data ^= rmfe::phi_9_17_single_bit(a, bit_idx + 2);
    x_shares[rmfe_idx].data ^= rmfe::phi_9_17_single_bit(b, bit_idx + 1);
    x_shares[rmfe_idx].data ^= c_rmfe;
    y_shares[rmfe_idx].data ^= rmfe::phi_9_17_single_bit(a ^ b, bit_idx + 2);
    y_shares[rmfe_idx].data ^= rmfe::phi_9_17_single_bit(a, bit_idx + 1);
    y_shares[rmfe_idx].data ^= c_rmfe;

    const uint64_t z = z_shares[rmfe_idx].data;
    const uint64_t d = rmfe::psi_9_17_single_bit(z, bit_idx + 2);
    const uint64_t e = rmfe::psi_9_17_single_bit(z, bit_idx + 1);
    const uint64_t f = rmfe::psi_9_17_single_bit(z, bit_idx + 0);

    idx = 192 - 1 - i * 3;
    result[idx / 64] |= f << (idx % 64);
    idx--;
    result[idx / 64] |= (e ^ d) << (idx % 64);
    idx--;
    result[idx / 64] |= d << (idx % 64);
  }
  return result;
}

} // namespace

namespace LOWMC_129_4 {
template <>
std::vector<std::array<field::GF2_9, 3>>
lowmc_with_sbox_output<field::GF2_9>(const std::vector<uint8_t> &key_in,
                                     const std::vector<uint8_t> &plaintext_in,
                                     std::vector<uint8_t> &ciphertext_out) {

  ciphertext_out.resize(plaintext_in.size());

  std::array<uint64_t, 3> key{}, pt{}, state{};
  u64_array_from_char_array(key, key_in.data(), key_in.size());
  u64_array_from_char_array(pt, plaintext_in.data(), plaintext_in.size());

  std::array<uint64_t, 3> k0 =
      matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[0]);
  xor_arrays(state, pt, k0);

  std::vector<std::array<field::GF2_3, 3>> sbox_values;
  sbox_values.reserve(172);
  for (size_t round = 0; round < 4; round++) {
    state = sbox(state, sbox_values);

    std::array<uint64_t, 3> tmp =
        matmul::multiply_with_transposed_GF2_matrix_129(state, L_mats[round]);
    std::array<uint64_t, 3> ki =
        matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[round + 1]);
    xor_arrays(state, tmp, C_mats[round]);
    xor_arrays(state, state, ki);
  }

  u64_array_to_char_array(ciphertext_out.data(), state, ciphertext_out.size());

  // pack with rmfe
  std::vector<std::array<field::GF2_9, 3>> res(86);
  for (size_t x = 0; x < 86; x++) {
    std::array<field::GF2_3, 2> tmp;
    for (size_t i = 0; i < 2; i++) {
      tmp[i] = sbox_values[2 * x + i][0];
    }
    res[x][0] = rmfe::phi_2_3(tmp);
    for (size_t i = 0; i < 2; i++) {
      tmp[i] = sbox_values[2 * x + i][1];
    }
    res[x][1] = rmfe::phi_2_3(tmp);
    res[x][2] = res[x][0] * res[x][1];
#ifndef NDEBUG
    tmp = rmfe::psi_2_3_transpose(res[x][2]);
    for (size_t i = 0; i < 2; i++) {
      assert(tmp[i] == sbox_values[2 * x + i][2]);
    }
#endif
  }

  return res;
}

template <>
void lowmc_mpc<field::GF2_9>(
    const std::vector<gsl::span<uint8_t>> &key_in,
    const std::vector<gsl::span<field::GF2_9>> &z_shares,
    const std::vector<uint8_t> &plaintext_in,
    std::vector<gsl::span<uint8_t>> &ciphertext_out,
    std::vector<gsl::span<field::GF2_9>> &x_shares,
    std::vector<gsl::span<field::GF2_9>> &y_shares) {

  const size_t num_parties = key_in.size();
  std::array<uint64_t, 3> key{}, pt{}, state{};
  u64_array_from_char_array(pt, plaintext_in.data(), plaintext_in.size());

  for (size_t party = 0; party < num_parties; party++) {
    // lowmc eval
    u64_array_from_char_array(key, key_in[party].data(), key_in[party].size());
    std::array<uint64_t, 3> k0 =
        matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[0]);
    if (party == 0) {
      xor_arrays(state, pt, k0);
    } else {
      state = k0;
    }

    for (size_t round = 0; round < 4; round++) {
      state = mpc_sbox_9(state, round, x_shares[party].data(),
                         y_shares[party].data(), z_shares[party].data());
      std::array<uint64_t, 3> tmp =
          matmul::multiply_with_transposed_GF2_matrix_129(state, L_mats[round]);
      std::array<uint64_t, 3> ki =
          matmul::multiply_with_transposed_GF2_matrix_129(key,
                                                          K_mats[round + 1]);
      xor_arrays(state, tmp, ki);
      if (party == 0) {
        xor_arrays(state, state, C_mats[round]);
      }
    }

    u64_array_to_char_array(ciphertext_out[party].data(), state,
                            ciphertext_out[party].size());
  }
}

template <>
std::vector<std::array<field::GF2_45, 3>>
lowmc_with_sbox_output<field::GF2_45>(const std::vector<uint8_t> &key_in,
                                      const std::vector<uint8_t> &plaintext_in,
                                      std::vector<uint8_t> &ciphertext_out) {

  ciphertext_out.resize(plaintext_in.size());

  std::array<uint64_t, 3> key{}, pt{}, state{};
  u64_array_from_char_array(key, key_in.data(), key_in.size());
  u64_array_from_char_array(pt, plaintext_in.data(), plaintext_in.size());

  std::array<uint64_t, 3> k0 =
      matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[0]);
  xor_arrays(state, pt, k0);

  std::vector<std::array<field::GF2_3, 3>> sbox_values;
  sbox_values.reserve(172 + 4);
  for (size_t round = 0; round < 4; round++) {
    state = sbox(state, sbox_values);

    std::array<uint64_t, 3> tmp =
        matmul::multiply_with_transposed_GF2_matrix_129(state, L_mats[round]);
    std::array<uint64_t, 3> ki =
        matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[round + 1]);
    xor_arrays(state, tmp, C_mats[round]);
    xor_arrays(state, state, ki);
  }
  // pad to multiple of rmfe size
  sbox_values.resize(172 + 4);

  u64_array_to_char_array(ciphertext_out.data(), state, ciphertext_out.size());

  // pack with rmfe
  std::vector<std::array<field::GF2_45, 3>> res(22);
  for (size_t x = 0; x < 22; x++) {
    std::array<field::GF2_3, 8> tmp;
    for (size_t i = 0; i < 8; i++) {
      tmp[i] = sbox_values[8 * x + i][0];
    }
    res[x][0] = rmfe::phi_8_15(tmp);
    for (size_t i = 0; i < 8; i++) {
      tmp[i] = sbox_values[8 * x + i][1];
    }
    res[x][1] = rmfe::phi_8_15(tmp);
    res[x][2] = res[x][0] * res[x][1];
#ifndef NDEBUG
    tmp = rmfe::psi_8_15_transpose(res[x][2]);
    for (size_t i = 0; i < 8; i++) {
      assert(tmp[i] == sbox_values[8 * x + i][2]);
    }
#endif
  }

  return res;
}

template <>
void lowmc_mpc<field::GF2_45>(
    const std::vector<gsl::span<uint8_t>> &key_in,
    const std::vector<gsl::span<field::GF2_45>> &z_shares,
    const std::vector<uint8_t> &plaintext_in,
    std::vector<gsl::span<uint8_t>> &ciphertext_out,
    std::vector<gsl::span<field::GF2_45>> &x_shares,
    std::vector<gsl::span<field::GF2_45>> &y_shares) {

  const size_t num_parties = key_in.size();
  std::array<uint64_t, 3> key{}, pt{}, state{};
  u64_array_from_char_array(pt, plaintext_in.data(), plaintext_in.size());

  for (size_t party = 0; party < num_parties; party++) {
    // lowmc eval
    u64_array_from_char_array(key, key_in[party].data(), key_in[party].size());
    std::array<uint64_t, 3> k0 =
        matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[0]);
    if (party == 0) {
      xor_arrays(state, pt, k0);
    } else {
      state = k0;
    }

    for (size_t round = 0; round < 4; round++) {
      state = mpc_sbox_45(state, round, x_shares[party].data(),
                          y_shares[party].data(), z_shares[party].data());
      std::array<uint64_t, 3> tmp =
          matmul::multiply_with_transposed_GF2_matrix_129(state, L_mats[round]);
      std::array<uint64_t, 3> ki =
          matmul::multiply_with_transposed_GF2_matrix_129(key,
                                                          K_mats[round + 1]);
      xor_arrays(state, tmp, ki);
      if (party == 0) {
        xor_arrays(state, state, C_mats[round]);
      }
    }

    u64_array_to_char_array(ciphertext_out[party].data(), state,
                            ciphertext_out[party].size());
  }
}

template <>
std::vector<std::array<field::GF2_51, 3>>
lowmc_with_sbox_output<field::GF2_51>(const std::vector<uint8_t> &key_in,
                                      const std::vector<uint8_t> &plaintext_in,
                                      std::vector<uint8_t> &ciphertext_out) {

  ciphertext_out.resize(plaintext_in.size());

  std::array<uint64_t, 3> key{}, pt{}, state{};
  u64_array_from_char_array(key, key_in.data(), key_in.size());
  u64_array_from_char_array(pt, plaintext_in.data(), plaintext_in.size());

  std::array<uint64_t, 3> k0 =
      matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[0]);
  xor_arrays(state, pt, k0);

  std::vector<std::array<field::GF2_3, 3>> sbox_values;
  sbox_values.reserve(172 + 8);
  for (size_t round = 0; round < 4; round++) {
    state = sbox(state, sbox_values);

    std::array<uint64_t, 3> tmp =
        matmul::multiply_with_transposed_GF2_matrix_129(state, L_mats[round]);
    std::array<uint64_t, 3> ki =
        matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[round + 1]);
    xor_arrays(state, tmp, C_mats[round]);
    xor_arrays(state, state, ki);
  }
  // pad to multiple of rmfe size
  sbox_values.resize(172 + 8);

  u64_array_to_char_array(ciphertext_out.data(), state, ciphertext_out.size());

  // pack with rmfe
  std::vector<std::array<field::GF2_51, 3>> res(20);
  for (size_t x = 0; x < 20; x++) {
    std::array<field::GF2_3, 9> tmp;
    for (size_t i = 0; i < 9; i++) {
      tmp[i] = sbox_values[9 * x + i][0];
    }
    res[x][0] = rmfe::phi_9_17(tmp);
    for (size_t i = 0; i < 9; i++) {
      tmp[i] = sbox_values[9 * x + i][1];
    }
    res[x][1] = rmfe::phi_9_17(tmp);
    res[x][2] = res[x][0] * res[x][1];
#ifndef NDEBUG
    tmp = rmfe::psi_9_17_transpose(res[x][2]);
    for (size_t i = 0; i < 9; i++) {
      assert(tmp[i] == sbox_values[9 * x + i][2]);
    }
#endif
  }

  return res;
}

template <>
void lowmc_mpc<field::GF2_51>(
    const std::vector<gsl::span<uint8_t>> &key_in,
    const std::vector<gsl::span<field::GF2_51>> &z_shares,
    const std::vector<uint8_t> &plaintext_in,
    std::vector<gsl::span<uint8_t>> &ciphertext_out,
    std::vector<gsl::span<field::GF2_51>> &x_shares,
    std::vector<gsl::span<field::GF2_51>> &y_shares) {

  const size_t num_parties = key_in.size();
  std::array<uint64_t, 3> key{}, pt{}, state{};
  u64_array_from_char_array(pt, plaintext_in.data(), plaintext_in.size());

  for (size_t party = 0; party < num_parties; party++) {
    // lowmc eval
    u64_array_from_char_array(key, key_in[party].data(), key_in[party].size());
    std::array<uint64_t, 3> k0 =
        matmul::multiply_with_transposed_GF2_matrix_129(key, K_mats[0]);
    if (party == 0) {
      xor_arrays(state, pt, k0);
    } else {
      state = k0;
    }

    for (size_t round = 0; round < 4; round++) {
      state = mpc_sbox_51(state, round, x_shares[party].data(),
                          y_shares[party].data(), z_shares[party].data());
      std::array<uint64_t, 3> tmp =
          matmul::multiply_with_transposed_GF2_matrix_129(state, L_mats[round]);
      std::array<uint64_t, 3> ki =
          matmul::multiply_with_transposed_GF2_matrix_129(key,
                                                          K_mats[round + 1]);
      xor_arrays(state, tmp, ki);
      if (party == 0) {
        xor_arrays(state, state, C_mats[round]);
      }
    }

    u64_array_to_char_array(ciphertext_out[party].data(), state,
                            ciphertext_out[party].size());
  }
}
} // namespace LOWMC_129_4