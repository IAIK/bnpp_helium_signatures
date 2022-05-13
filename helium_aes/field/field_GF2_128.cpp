#if __INCLUDE_LEVEL__ != 0
#error "Don't include this file"
#endif
#include "field.h"

#include <cstring>
#include <iomanip>
#include <iostream>

extern "C" {
#include "portable_endian.h"
#include <emmintrin.h>
#include <immintrin.h>
#include <wmmintrin.h>
#include <xmmintrin.h>
}

namespace {

inline void clmul_karatsuba(__m128i out[2], const __m128i a, const __m128i b) {
  __m128i tmp[4];
  out[0] = _mm_clmulepi64_si128(a, b, 0x00);
  out[1] = _mm_clmulepi64_si128(a, b, 0x11);

  tmp[0] = _mm_srli_si128(a, 8);
  tmp[1] = _mm_xor_si128(a, tmp[0]);
  tmp[2] = _mm_srli_si128(b, 8);
  tmp[3] = _mm_xor_si128(b, tmp[2]);

  tmp[0] = _mm_clmulepi64_si128(tmp[1], tmp[3], 0x00);
  tmp[1] = _mm_xor_si128(out[1], out[0]);
  tmp[0] = _mm_xor_si128(tmp[0], tmp[1]);

  tmp[1] = _mm_slli_si128(tmp[0], 8);
  tmp[2] = _mm_srli_si128(tmp[0], 8);

  out[0] = _mm_xor_si128(out[0], tmp[1]);
  out[1] = _mm_xor_si128(out[1], tmp[2]);
}
inline void clmul_schoolbook(__m128i out[2], const __m128i a, const __m128i b) {
  __m128i tmp[3];
  out[0] = _mm_clmulepi64_si128(a, b, 0x00);
  out[1] = _mm_clmulepi64_si128(a, b, 0x11);

  tmp[0] = _mm_clmulepi64_si128(a, b, 0x01);
  tmp[1] = _mm_clmulepi64_si128(a, b, 0x10);

  tmp[0] = _mm_xor_si128(tmp[0], tmp[1]);
  tmp[1] = _mm_slli_si128(tmp[0], 8);
  tmp[2] = _mm_srli_si128(tmp[0], 8);

  out[0] = _mm_xor_si128(out[0], tmp[1]);
  out[1] = _mm_xor_si128(out[1], tmp[2]);
}

inline void sqr(__m128i out[2], const __m128i a) {
  __m128i tmp[2];
  __m128i sqrT = _mm_set_epi64x(0x5554515045444140, 0x1514111005040100);
  __m128i mask = _mm_set_epi64x(0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F);
  tmp[0] = _mm_and_si128(a, mask);
  tmp[1] = _mm_srli_epi64(a, 4);
  tmp[1] = _mm_and_si128(tmp[1], mask);
  tmp[0] = _mm_shuffle_epi8(sqrT, tmp[0]);
  tmp[1] = _mm_shuffle_epi8(sqrT, tmp[1]);
  out[0] = _mm_unpacklo_epi8(tmp[0], tmp[1]);
  out[1] = _mm_unpackhi_epi8(tmp[0], tmp[1]);
}

inline void reduce_u64(__m128i out[1], const __m128i in[2]) {
  uint64_t tmp[4];
  uint64_t t0, t1, t2;
  _mm_store_si128((__m128i *)&tmp[0], in[0]);
  _mm_store_si128((__m128i *)&tmp[2], in[1]);

  // printf("%16llx %16llx %16llx %16llx\n", tmp[3], tmp[2], tmp[1], tmp[0]);
  // modulus = x^128 + x^7 + x^2 + x^1 + 1
  t2 = (tmp[3] >> 57) ^ (tmp[3] >> 62) ^ (tmp[3] >> 63);
  tmp[2] ^= t2;
  t1 = (tmp[3] << 7) | (tmp[2] >> 57);
  t1 ^= (tmp[3] << 2) | (tmp[2] >> 62);
  t1 ^= (tmp[3] << 1) | (tmp[2] >> 63);
  t1 ^= tmp[3];
  tmp[1] ^= t1;
  t0 = (tmp[2] << 7);
  t0 ^= (tmp[2] << 2);
  t0 ^= (tmp[2] << 1);
  t0 ^= tmp[2];
  tmp[0] ^= t0;
  // printf("%16llx %16llx %16llx %16llx\n", tmp[3], tmp[2], tmp[1], tmp[0]);
  out[0] = _mm_load_si128((__m128i *)&tmp[0]);
}

inline void reduce_clmul(__m128i out[1], const __m128i in[2]) {
  __m128i p = _mm_set_epi64x(0x0, 0x87);
  __m128i t0, t1, t2;
  t0 = _mm_clmulepi64_si128(in[1], p, 0x01); // in[1]_high * p
  t1 = _mm_slli_si128(t0, 8);    // low 64bit of result, shifted to high
  t2 = _mm_srli_si128(t0, 8);    // high 64bit of result, shifted to high
  t2 = _mm_xor_si128(t2, in[1]); // update in[1]_low with high64 of result

  t0 = _mm_clmulepi64_si128(t2, p, 0x00); // updated in[1]_low * p
  out[0] = _mm_xor_si128(t0, in[0]);      // add in[1]_low * p to result
  out[0] = _mm_xor_si128(out[0], t1); // also add the low part of in[1]_high * p
}

inline void reduce_m128i(__m128i out[1], const __m128i in[2]) {
  __m128i t0, t1, t2, t3, t4, t5;

  t0 = _mm_srli_epi64(in[1], 57);
  t1 = _mm_srli_epi64(in[1], 62);
  t2 = _mm_srli_epi64(in[1], 63);
  t1 = _mm_xor_si128(t1, t2);
  t0 = _mm_xor_si128(t0, t1);

  t1 = _mm_slli_si128(t0, 8);
  t2 = _mm_srli_si128(t0, 8);

  t0 = _mm_xor_si128(in[1], t2);

  t3 = _mm_slli_epi64(t0, 7);
  t4 = _mm_slli_epi64(t0, 2);
  t5 = _mm_slli_epi64(t0, 1);
  t3 = _mm_xor_si128(t3, t4);
  t0 = _mm_xor_si128(t5, t0);
  t0 = _mm_xor_si128(t0, t3);
  t0 = _mm_xor_si128(t0, t1);

  out[0] = _mm_xor_si128(t0, in[0]);
}

inline void reduce_intel(__m128i out[1], const __m128i in[2]) {
  __m128i XMMMASK = _mm_setr_epi32(0xffffffff, 0x0, 0x0, 0x0);
  __m128i tmp3, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;

  tmp7 = _mm_srli_epi32(in[1], 31);
  tmp8 = _mm_srli_epi32(in[1], 30);
  tmp9 = _mm_srli_epi32(in[1], 25);
  tmp7 = _mm_xor_si128(tmp7, tmp8);
  tmp7 = _mm_xor_si128(tmp7, tmp9);
  tmp8 = _mm_shuffle_epi32(tmp7, 147);

  tmp7 = _mm_and_si128(XMMMASK, tmp8);
  tmp8 = _mm_andnot_si128(XMMMASK, tmp8);
  tmp3 = _mm_xor_si128(in[0], tmp8);
  tmp6 = _mm_xor_si128(in[1], tmp7);
  tmp10 = _mm_slli_epi32(tmp6, 1);
  tmp3 = _mm_xor_si128(tmp3, tmp10);
  tmp11 = _mm_slli_epi32(tmp6, 2);
  tmp3 = _mm_xor_si128(tmp3, tmp11);
  tmp12 = _mm_slli_epi32(tmp6, 7);
  tmp3 = _mm_xor_si128(tmp3, tmp12);

  out[0] = _mm_xor_si128(tmp3, tmp6);
}

inline void gf128mul(__m128i *out, const __m128i *in1, const __m128i *in2) {
  __m128i tmp[2];
  clmul_schoolbook(tmp, *in1, *in2);
  reduce_clmul(out, tmp);
}
inline void gf128sqr(__m128i *out, const __m128i *in) {
  __m128i tmp[2];
  sqr(tmp, *in);
  reduce_clmul(out, tmp);
}

inline void gf128add(__m128i *out, const __m128i *in1, const __m128i *in2) {
  *out = _mm_xor_si128(*in1, *in2);
}

static inline __m256i mm256_compute_mask_2(const uint64_t idx,
                                           const size_t bit) {
  const uint64_t m1 = -((idx >> bit) & 1);
  const uint64_t m2 = -((idx >> (bit + 1)) & 1);
  return _mm256_set_epi64x(m2, m2, m1, m1);
}

} // namespace

namespace field {

GF2_128::GF2_128(std::string hex_string) {
  // check if hex_string start with 0x or 0X
  if (hex_string.rfind("0x", 0) == 0 || hex_string.rfind("0X", 0) == 0) {
    hex_string = hex_string.substr(2);
  } else {
    throw std::runtime_error("input needs to be a hex number");
  }
  constexpr size_t num_hex_chars = 128 / 4;
  if (hex_string.length() > num_hex_chars)
    throw std::runtime_error("input hex is too large");
  // pad to 128 bit
  hex_string.insert(hex_string.begin(), num_hex_chars - hex_string.length(),
                    '0');
  // high 64 bit
  uint64_t high = std::stoull(hex_string.substr(0, 64 / 4), nullptr, 16);
  // low 64 bit
  uint64_t low = std::stoull(hex_string.substr(64 / 4, 64 / 4), nullptr, 16);
  data[0] = low;
  data[1] = high;
}

GF2_128 GF2_128::operator+(const GF2_128 &other) const {
  GF2_128 result;
  gf128add(result.as_m128i(), this->as_const_m128i(), other.as_const_m128i());
  return result;
}
GF2_128 &GF2_128::operator+=(const GF2_128 &other) {
  gf128add(this->as_m128i(), this->as_const_m128i(), other.as_const_m128i());
  return *this;
}
GF2_128 GF2_128::operator-(const GF2_128 &other) const {
  GF2_128 result;
  gf128add(result.as_m128i(), this->as_const_m128i(), other.as_const_m128i());
  return result;
}
GF2_128 &GF2_128::operator-=(const GF2_128 &other) {
  gf128add(this->as_m128i(), this->as_const_m128i(), other.as_const_m128i());
  return *this;
}
GF2_128 GF2_128::operator*(const GF2_128 &other) const {
  GF2_128 result;
  gf128mul(result.as_m128i(), this->as_const_m128i(), other.as_const_m128i());
  return result;
}
GF2_128 &GF2_128::operator*=(const GF2_128 &other) {
  gf128mul(this->as_m128i(), this->as_const_m128i(), other.as_const_m128i());
  return *this;
}
bool GF2_128::operator==(const GF2_128 &other) const {
  return std::memcmp(this->data.data(), other.data.data(), BYTE_SIZE) == 0;
}
bool GF2_128::operator!=(const GF2_128 &other) const {
  return std::memcmp(this->data.data(), other.data.data(), BYTE_SIZE) != 0;
}

GF2_128 GF2_128::inverse() const {
  constexpr size_t u[11] = {1, 2, 3, 6, 12, 24, 48, 51, 63, 126, 127};
  constexpr size_t u_len = sizeof(u) / sizeof(u[0]);
  // q = u[i] - u[i - 1] should give us the corresponding values
  // (1, 1, 3, 6, 12, 24, 3, 12, 63, 1), which will have corresponding indexes
  constexpr size_t q_index[u_len - 1] = {0, 0, 2, 3, 4, 5, 2, 4, 8, 0};
  __m128i b[u_len];

  b[0] = *(this->as_const_m128i());

  for (size_t i = 1; i < u_len; ++i) {

    __m128i b_p = b[i - 1];
    __m128i b_q = b[q_index[i - 1]];

    for (size_t m = u[q_index[i - 1]]; m; --m) {
      gf128sqr(&b_p, &b_p);
    }

    gf128mul(&b[i], &b_p, &b_q);
  }

  GF2_128 out;
  gf128sqr(out.as_m128i(), &b[u_len - 1]);

  return out;
}

GF2_128 GF2_128::inverse_slow() const {
  // Fixed-op square-multiply
  // 2^n - 2 in binary is 0b1111..10
  __m128i t1 = *this->as_const_m128i(), t2;
  gf128sqr(&t2, &t1);

  // First 127 one-bits (start from second)
  for (size_t i = 0; i < 126; i++) {
    gf128mul(&t1, &t1, &t2);
    gf128sqr(&t2, &t2);
  }

  // Final zero-bit
  gf128sqr(&t1, &t1);
  GF2_128 result;
  *result.as_m128i() = t1;
  return result;
}

void GF2_128::to_bytes(uint8_t *out) const {
  uint64_t le_data = htole64(data[0]);
  memcpy(out, (uint8_t *)(&le_data), sizeof(uint64_t));
  le_data = htole64(data[1]);
  memcpy(out + sizeof(uint64_t), (uint8_t *)(&le_data), sizeof(uint64_t));
}

void GF2_128::from_bytes(const uint8_t *in) {
  uint64_t tmp;
  memcpy((uint8_t *)(&tmp), in, sizeof(uint64_t));
  data[0] = le64toh(tmp);
  memcpy((uint8_t *)(&tmp), in + sizeof(uint64_t), sizeof(uint64_t));
  data[1] = le64toh(tmp);
}

} // namespace field

namespace {
constexpr std::array<field::GF2_128, 256> lifting_lut = {{
    std::array<uint64_t, 2>{0x0000000000000000, 0x0000000000000000},
    std::array<uint64_t, 2>{0x0000000000000001, 0x0000000000000000},
    std::array<uint64_t, 2>{0xa13fe8ac5560ce0d, 0x053d8555a9979a1c},
    std::array<uint64_t, 2>{0xa13fe8ac5560ce0c, 0x053d8555a9979a1c},
    std::array<uint64_t, 2>{0xec7759ca3488aee1, 0x4cf4b7439cbfbb84},
    std::array<uint64_t, 2>{0xec7759ca3488aee0, 0x4cf4b7439cbfbb84},
    std::array<uint64_t, 2>{0x4d48b16661e860ec, 0x49c9321635282198},
    std::array<uint64_t, 2>{0x4d48b16661e860ed, 0x49c9321635282198},
    std::array<uint64_t, 2>{0xbfcf02ae363946a8, 0x35ad604f7d51d2c6},
    std::array<uint64_t, 2>{0xbfcf02ae363946a9, 0x35ad604f7d51d2c6},
    std::array<uint64_t, 2>{0x1ef0ea02635988a5, 0x3090e51ad4c648da},
    std::array<uint64_t, 2>{0x1ef0ea02635988a4, 0x3090e51ad4c648da},
    std::array<uint64_t, 2>{0x53b85b6402b1e849, 0x7959d70ce1ee6942},
    std::array<uint64_t, 2>{0x53b85b6402b1e848, 0x7959d70ce1ee6942},
    std::array<uint64_t, 2>{0xf287b3c857d12644, 0x7c6452594879f35e},
    std::array<uint64_t, 2>{0xf287b3c857d12645, 0x7c6452594879f35e},
    std::array<uint64_t, 2>{0x6b8330483c2e9849, 0x0dcb364640a222fe},
    std::array<uint64_t, 2>{0x6b8330483c2e9848, 0x0dcb364640a222fe},
    std::array<uint64_t, 2>{0xcabcd8e4694e5644, 0x08f6b313e935b8e2},
    std::array<uint64_t, 2>{0xcabcd8e4694e5645, 0x08f6b313e935b8e2},
    std::array<uint64_t, 2>{0x87f4698208a636a8, 0x413f8105dc1d997a},
    std::array<uint64_t, 2>{0x87f4698208a636a9, 0x413f8105dc1d997a},
    std::array<uint64_t, 2>{0x26cb812e5dc6f8a5, 0x44020450758a0366},
    std::array<uint64_t, 2>{0x26cb812e5dc6f8a4, 0x44020450758a0366},
    std::array<uint64_t, 2>{0xd44c32e60a17dee1, 0x386656093df3f038},
    std::array<uint64_t, 2>{0xd44c32e60a17dee0, 0x386656093df3f038},
    std::array<uint64_t, 2>{0x7573da4a5f7710ec, 0x3d5bd35c94646a24},
    std::array<uint64_t, 2>{0x7573da4a5f7710ed, 0x3d5bd35c94646a24},
    std::array<uint64_t, 2>{0x383b6b2c3e9f7000, 0x7492e14aa14c4bbc},
    std::array<uint64_t, 2>{0x383b6b2c3e9f7001, 0x7492e14aa14c4bbc},
    std::array<uint64_t, 2>{0x990483806bffbe0d, 0x71af641f08dbd1a0},
    std::array<uint64_t, 2>{0x990483806bffbe0c, 0x71af641f08dbd1a0},
    std::array<uint64_t, 2>{0x252b49277b1b82b4, 0x549810e11a88dea5},
    std::array<uint64_t, 2>{0x252b49277b1b82b5, 0x549810e11a88dea5},
    std::array<uint64_t, 2>{0x8414a18b2e7b4cb9, 0x51a595b4b31f44b9},
    std::array<uint64_t, 2>{0x8414a18b2e7b4cb8, 0x51a595b4b31f44b9},
    std::array<uint64_t, 2>{0xc95c10ed4f932c55, 0x186ca7a286376521},
    std::array<uint64_t, 2>{0xc95c10ed4f932c54, 0x186ca7a286376521},
    std::array<uint64_t, 2>{0x6863f8411af3e258, 0x1d5122f72fa0ff3d},
    std::array<uint64_t, 2>{0x6863f8411af3e259, 0x1d5122f72fa0ff3d},
    std::array<uint64_t, 2>{0x9ae44b894d22c41c, 0x613570ae67d90c63},
    std::array<uint64_t, 2>{0x9ae44b894d22c41d, 0x613570ae67d90c63},
    std::array<uint64_t, 2>{0x3bdba32518420a11, 0x6408f5fbce4e967f},
    std::array<uint64_t, 2>{0x3bdba32518420a10, 0x6408f5fbce4e967f},
    std::array<uint64_t, 2>{0x7693124379aa6afd, 0x2dc1c7edfb66b7e7},
    std::array<uint64_t, 2>{0x7693124379aa6afc, 0x2dc1c7edfb66b7e7},
    std::array<uint64_t, 2>{0xd7acfaef2ccaa4f0, 0x28fc42b852f12dfb},
    std::array<uint64_t, 2>{0xd7acfaef2ccaa4f1, 0x28fc42b852f12dfb},
    std::array<uint64_t, 2>{0x4ea8796f47351afd, 0x595326a75a2afc5b},
    std::array<uint64_t, 2>{0x4ea8796f47351afc, 0x595326a75a2afc5b},
    std::array<uint64_t, 2>{0xef9791c31255d4f0, 0x5c6ea3f2f3bd6647},
    std::array<uint64_t, 2>{0xef9791c31255d4f1, 0x5c6ea3f2f3bd6647},
    std::array<uint64_t, 2>{0xa2df20a573bdb41c, 0x15a791e4c69547df},
    std::array<uint64_t, 2>{0xa2df20a573bdb41d, 0x15a791e4c69547df},
    std::array<uint64_t, 2>{0x03e0c80926dd7a11, 0x109a14b16f02ddc3},
    std::array<uint64_t, 2>{0x03e0c80926dd7a10, 0x109a14b16f02ddc3},
    std::array<uint64_t, 2>{0xf1677bc1710c5c55, 0x6cfe46e8277b2e9d},
    std::array<uint64_t, 2>{0xf1677bc1710c5c54, 0x6cfe46e8277b2e9d},
    std::array<uint64_t, 2>{0x5058936d246c9258, 0x69c3c3bd8eecb481},
    std::array<uint64_t, 2>{0x5058936d246c9259, 0x69c3c3bd8eecb481},
    std::array<uint64_t, 2>{0x1d10220b4584f2b4, 0x200af1abbbc49519},
    std::array<uint64_t, 2>{0x1d10220b4584f2b5, 0x200af1abbbc49519},
    std::array<uint64_t, 2>{0xbc2fcaa710e43cb9, 0x253774fe12530f05},
    std::array<uint64_t, 2>{0xbc2fcaa710e43cb8, 0x253774fe12530f05},
    std::array<uint64_t, 2>{0xc72bf2ef2521ff22, 0xd681a5686c0c1f75},
    std::array<uint64_t, 2>{0xc72bf2ef2521ff23, 0xd681a5686c0c1f75},
    std::array<uint64_t, 2>{0x66141a437041312f, 0xd3bc203dc59b8569},
    std::array<uint64_t, 2>{0x66141a437041312e, 0xd3bc203dc59b8569},
    std::array<uint64_t, 2>{0x2b5cab2511a951c3, 0x9a75122bf0b3a4f1},
    std::array<uint64_t, 2>{0x2b5cab2511a951c2, 0x9a75122bf0b3a4f1},
    std::array<uint64_t, 2>{0x8a63438944c99fce, 0x9f48977e59243eed},
    std::array<uint64_t, 2>{0x8a63438944c99fcf, 0x9f48977e59243eed},
    std::array<uint64_t, 2>{0x78e4f0411318b98a, 0xe32cc527115dcdb3},
    std::array<uint64_t, 2>{0x78e4f0411318b98b, 0xe32cc527115dcdb3},
    std::array<uint64_t, 2>{0xd9db18ed46787787, 0xe6114072b8ca57af},
    std::array<uint64_t, 2>{0xd9db18ed46787786, 0xe6114072b8ca57af},
    std::array<uint64_t, 2>{0x9493a98b2790176b, 0xafd872648de27637},
    std::array<uint64_t, 2>{0x9493a98b2790176a, 0xafd872648de27637},
    std::array<uint64_t, 2>{0x35ac412772f0d966, 0xaae5f7312475ec2b},
    std::array<uint64_t, 2>{0x35ac412772f0d967, 0xaae5f7312475ec2b},
    std::array<uint64_t, 2>{0xaca8c2a7190f676b, 0xdb4a932e2cae3d8b},
    std::array<uint64_t, 2>{0xaca8c2a7190f676a, 0xdb4a932e2cae3d8b},
    std::array<uint64_t, 2>{0x0d972a0b4c6fa966, 0xde77167b8539a797},
    std::array<uint64_t, 2>{0x0d972a0b4c6fa967, 0xde77167b8539a797},
    std::array<uint64_t, 2>{0x40df9b6d2d87c98a, 0x97be246db011860f},
    std::array<uint64_t, 2>{0x40df9b6d2d87c98b, 0x97be246db011860f},
    std::array<uint64_t, 2>{0xe1e073c178e70787, 0x9283a13819861c13},
    std::array<uint64_t, 2>{0xe1e073c178e70786, 0x9283a13819861c13},
    std::array<uint64_t, 2>{0x1367c0092f3621c3, 0xeee7f36151ffef4d},
    std::array<uint64_t, 2>{0x1367c0092f3621c2, 0xeee7f36151ffef4d},
    std::array<uint64_t, 2>{0xb25828a57a56efce, 0xebda7634f8687551},
    std::array<uint64_t, 2>{0xb25828a57a56efcf, 0xebda7634f8687551},
    std::array<uint64_t, 2>{0xff1099c31bbe8f22, 0xa2134422cd4054c9},
    std::array<uint64_t, 2>{0xff1099c31bbe8f23, 0xa2134422cd4054c9},
    std::array<uint64_t, 2>{0x5e2f716f4ede412f, 0xa72ec17764d7ced5},
    std::array<uint64_t, 2>{0x5e2f716f4ede412e, 0xa72ec17764d7ced5},
    std::array<uint64_t, 2>{0xe200bbc85e3a7d96, 0x8219b5897684c1d0},
    std::array<uint64_t, 2>{0xe200bbc85e3a7d97, 0x8219b5897684c1d0},
    std::array<uint64_t, 2>{0x433f53640b5ab39b, 0x872430dcdf135bcc},
    std::array<uint64_t, 2>{0x433f53640b5ab39a, 0x872430dcdf135bcc},
    std::array<uint64_t, 2>{0x0e77e2026ab2d377, 0xceed02caea3b7a54},
    std::array<uint64_t, 2>{0x0e77e2026ab2d376, 0xceed02caea3b7a54},
    std::array<uint64_t, 2>{0xaf480aae3fd21d7a, 0xcbd0879f43ace048},
    std::array<uint64_t, 2>{0xaf480aae3fd21d7b, 0xcbd0879f43ace048},
    std::array<uint64_t, 2>{0x5dcfb96668033b3e, 0xb7b4d5c60bd51316},
    std::array<uint64_t, 2>{0x5dcfb96668033b3f, 0xb7b4d5c60bd51316},
    std::array<uint64_t, 2>{0xfcf051ca3d63f533, 0xb2895093a242890a},
    std::array<uint64_t, 2>{0xfcf051ca3d63f532, 0xb2895093a242890a},
    std::array<uint64_t, 2>{0xb1b8e0ac5c8b95df, 0xfb406285976aa892},
    std::array<uint64_t, 2>{0xb1b8e0ac5c8b95de, 0xfb406285976aa892},
    std::array<uint64_t, 2>{0x1087080009eb5bd2, 0xfe7de7d03efd328e},
    std::array<uint64_t, 2>{0x1087080009eb5bd3, 0xfe7de7d03efd328e},
    std::array<uint64_t, 2>{0x89838b806214e5df, 0x8fd283cf3626e32e},
    std::array<uint64_t, 2>{0x89838b806214e5de, 0x8fd283cf3626e32e},
    std::array<uint64_t, 2>{0x28bc632c37742bd2, 0x8aef069a9fb17932},
    std::array<uint64_t, 2>{0x28bc632c37742bd3, 0x8aef069a9fb17932},
    std::array<uint64_t, 2>{0x65f4d24a569c4b3e, 0xc326348caa9958aa},
    std::array<uint64_t, 2>{0x65f4d24a569c4b3f, 0xc326348caa9958aa},
    std::array<uint64_t, 2>{0xc4cb3ae603fc8533, 0xc61bb1d9030ec2b6},
    std::array<uint64_t, 2>{0xc4cb3ae603fc8532, 0xc61bb1d9030ec2b6},
    std::array<uint64_t, 2>{0x364c892e542da377, 0xba7fe3804b7731e8},
    std::array<uint64_t, 2>{0x364c892e542da376, 0xba7fe3804b7731e8},
    std::array<uint64_t, 2>{0x97736182014d6d7a, 0xbf4266d5e2e0abf4},
    std::array<uint64_t, 2>{0x97736182014d6d7b, 0xbf4266d5e2e0abf4},
    std::array<uint64_t, 2>{0xda3bd0e460a50d96, 0xf68b54c3d7c88a6c},
    std::array<uint64_t, 2>{0xda3bd0e460a50d97, 0xf68b54c3d7c88a6c},
    std::array<uint64_t, 2>{0x7b04384835c5c39b, 0xf3b6d1967e5f1070},
    std::array<uint64_t, 2>{0x7b04384835c5c39a, 0xf3b6d1967e5f1070},
    std::array<uint64_t, 2>{0x7a7a8e94e136f9bc, 0x0950311a4fb78fe0},
    std::array<uint64_t, 2>{0x7a7a8e94e136f9bd, 0x0950311a4fb78fe0},
    std::array<uint64_t, 2>{0xdb456638b45637b1, 0x0c6db44fe62015fc},
    std::array<uint64_t, 2>{0xdb456638b45637b0, 0x0c6db44fe62015fc},
    std::array<uint64_t, 2>{0x960dd75ed5be575d, 0x45a48659d3083464},
    std::array<uint64_t, 2>{0x960dd75ed5be575c, 0x45a48659d3083464},
    std::array<uint64_t, 2>{0x37323ff280de9950, 0x4099030c7a9fae78},
    std::array<uint64_t, 2>{0x37323ff280de9951, 0x4099030c7a9fae78},
    std::array<uint64_t, 2>{0xc5b58c3ad70fbf14, 0x3cfd515532e65d26},
    std::array<uint64_t, 2>{0xc5b58c3ad70fbf15, 0x3cfd515532e65d26},
    std::array<uint64_t, 2>{0x648a6496826f7119, 0x39c0d4009b71c73a},
    std::array<uint64_t, 2>{0x648a6496826f7118, 0x39c0d4009b71c73a},
    std::array<uint64_t, 2>{0x29c2d5f0e38711f5, 0x7009e616ae59e6a2},
    std::array<uint64_t, 2>{0x29c2d5f0e38711f4, 0x7009e616ae59e6a2},
    std::array<uint64_t, 2>{0x88fd3d5cb6e7dff8, 0x7534634307ce7cbe},
    std::array<uint64_t, 2>{0x88fd3d5cb6e7dff9, 0x7534634307ce7cbe},
    std::array<uint64_t, 2>{0x11f9bedcdd1861f5, 0x049b075c0f15ad1e},
    std::array<uint64_t, 2>{0x11f9bedcdd1861f4, 0x049b075c0f15ad1e},
    std::array<uint64_t, 2>{0xb0c656708878aff8, 0x01a68209a6823702},
    std::array<uint64_t, 2>{0xb0c656708878aff9, 0x01a68209a6823702},
    std::array<uint64_t, 2>{0xfd8ee716e990cf14, 0x486fb01f93aa169a},
    std::array<uint64_t, 2>{0xfd8ee716e990cf15, 0x486fb01f93aa169a},
    std::array<uint64_t, 2>{0x5cb10fbabcf00119, 0x4d52354a3a3d8c86},
    std::array<uint64_t, 2>{0x5cb10fbabcf00118, 0x4d52354a3a3d8c86},
    std::array<uint64_t, 2>{0xae36bc72eb21275d, 0x3136671372447fd8},
    std::array<uint64_t, 2>{0xae36bc72eb21275c, 0x3136671372447fd8},
    std::array<uint64_t, 2>{0x0f0954debe41e950, 0x340be246dbd3e5c4},
    std::array<uint64_t, 2>{0x0f0954debe41e951, 0x340be246dbd3e5c4},
    std::array<uint64_t, 2>{0x4241e5b8dfa989bc, 0x7dc2d050eefbc45c},
    std::array<uint64_t, 2>{0x4241e5b8dfa989bd, 0x7dc2d050eefbc45c},
    std::array<uint64_t, 2>{0xe37e0d148ac947b1, 0x78ff5505476c5e40},
    std::array<uint64_t, 2>{0xe37e0d148ac947b0, 0x78ff5505476c5e40},
    std::array<uint64_t, 2>{0x5f51c7b39a2d7b08, 0x5dc821fb553f5145},
    std::array<uint64_t, 2>{0x5f51c7b39a2d7b09, 0x5dc821fb553f5145},
    std::array<uint64_t, 2>{0xfe6e2f1fcf4db505, 0x58f5a4aefca8cb59},
    std::array<uint64_t, 2>{0xfe6e2f1fcf4db504, 0x58f5a4aefca8cb59},
    std::array<uint64_t, 2>{0xb3269e79aea5d5e9, 0x113c96b8c980eac1},
    std::array<uint64_t, 2>{0xb3269e79aea5d5e8, 0x113c96b8c980eac1},
    std::array<uint64_t, 2>{0x121976d5fbc51be4, 0x140113ed601770dd},
    std::array<uint64_t, 2>{0x121976d5fbc51be5, 0x140113ed601770dd},
    std::array<uint64_t, 2>{0xe09ec51dac143da0, 0x686541b4286e8383},
    std::array<uint64_t, 2>{0xe09ec51dac143da1, 0x686541b4286e8383},
    std::array<uint64_t, 2>{0x41a12db1f974f3ad, 0x6d58c4e181f9199f},
    std::array<uint64_t, 2>{0x41a12db1f974f3ac, 0x6d58c4e181f9199f},
    std::array<uint64_t, 2>{0x0ce99cd7989c9341, 0x2491f6f7b4d13807},
    std::array<uint64_t, 2>{0x0ce99cd7989c9340, 0x2491f6f7b4d13807},
    std::array<uint64_t, 2>{0xadd6747bcdfc5d4c, 0x21ac73a21d46a21b},
    std::array<uint64_t, 2>{0xadd6747bcdfc5d4d, 0x21ac73a21d46a21b},
    std::array<uint64_t, 2>{0x34d2f7fba603e341, 0x500317bd159d73bb},
    std::array<uint64_t, 2>{0x34d2f7fba603e340, 0x500317bd159d73bb},
    std::array<uint64_t, 2>{0x95ed1f57f3632d4c, 0x553e92e8bc0ae9a7},
    std::array<uint64_t, 2>{0x95ed1f57f3632d4d, 0x553e92e8bc0ae9a7},
    std::array<uint64_t, 2>{0xd8a5ae31928b4da0, 0x1cf7a0fe8922c83f},
    std::array<uint64_t, 2>{0xd8a5ae31928b4da1, 0x1cf7a0fe8922c83f},
    std::array<uint64_t, 2>{0x799a469dc7eb83ad, 0x19ca25ab20b55223},
    std::array<uint64_t, 2>{0x799a469dc7eb83ac, 0x19ca25ab20b55223},
    std::array<uint64_t, 2>{0x8b1df555903aa5e9, 0x65ae77f268cca17d},
    std::array<uint64_t, 2>{0x8b1df555903aa5e8, 0x65ae77f268cca17d},
    std::array<uint64_t, 2>{0x2a221df9c55a6be4, 0x6093f2a7c15b3b61},
    std::array<uint64_t, 2>{0x2a221df9c55a6be5, 0x6093f2a7c15b3b61},
    std::array<uint64_t, 2>{0x676aac9fa4b20b08, 0x295ac0b1f4731af9},
    std::array<uint64_t, 2>{0x676aac9fa4b20b09, 0x295ac0b1f4731af9},
    std::array<uint64_t, 2>{0xc6554433f1d2c505, 0x2c6745e45de480e5},
    std::array<uint64_t, 2>{0xc6554433f1d2c504, 0x2c6745e45de480e5},
    std::array<uint64_t, 2>{0xbd517c7bc417069e, 0xdfd1947223bb9095},
    std::array<uint64_t, 2>{0xbd517c7bc417069f, 0xdfd1947223bb9095},
    std::array<uint64_t, 2>{0x1c6e94d79177c893, 0xdaec11278a2c0a89},
    std::array<uint64_t, 2>{0x1c6e94d79177c892, 0xdaec11278a2c0a89},
    std::array<uint64_t, 2>{0x512625b1f09fa87f, 0x93252331bf042b11},
    std::array<uint64_t, 2>{0x512625b1f09fa87e, 0x93252331bf042b11},
    std::array<uint64_t, 2>{0xf019cd1da5ff6672, 0x9618a6641693b10d},
    std::array<uint64_t, 2>{0xf019cd1da5ff6673, 0x9618a6641693b10d},
    std::array<uint64_t, 2>{0x029e7ed5f22e4036, 0xea7cf43d5eea4253},
    std::array<uint64_t, 2>{0x029e7ed5f22e4037, 0xea7cf43d5eea4253},
    std::array<uint64_t, 2>{0xa3a19679a74e8e3b, 0xef417168f77dd84f},
    std::array<uint64_t, 2>{0xa3a19679a74e8e3a, 0xef417168f77dd84f},
    std::array<uint64_t, 2>{0xeee9271fc6a6eed7, 0xa688437ec255f9d7},
    std::array<uint64_t, 2>{0xeee9271fc6a6eed6, 0xa688437ec255f9d7},
    std::array<uint64_t, 2>{0x4fd6cfb393c620da, 0xa3b5c62b6bc263cb},
    std::array<uint64_t, 2>{0x4fd6cfb393c620db, 0xa3b5c62b6bc263cb},
    std::array<uint64_t, 2>{0xd6d24c33f8399ed7, 0xd21aa2346319b26b},
    std::array<uint64_t, 2>{0xd6d24c33f8399ed6, 0xd21aa2346319b26b},
    std::array<uint64_t, 2>{0x77eda49fad5950da, 0xd7272761ca8e2877},
    std::array<uint64_t, 2>{0x77eda49fad5950db, 0xd7272761ca8e2877},
    std::array<uint64_t, 2>{0x3aa515f9ccb13036, 0x9eee1577ffa609ef},
    std::array<uint64_t, 2>{0x3aa515f9ccb13037, 0x9eee1577ffa609ef},
    std::array<uint64_t, 2>{0x9b9afd5599d1fe3b, 0x9bd39022563193f3},
    std::array<uint64_t, 2>{0x9b9afd5599d1fe3a, 0x9bd39022563193f3},
    std::array<uint64_t, 2>{0x691d4e9dce00d87f, 0xe7b7c27b1e4860ad},
    std::array<uint64_t, 2>{0x691d4e9dce00d87e, 0xe7b7c27b1e4860ad},
    std::array<uint64_t, 2>{0xc822a6319b601672, 0xe28a472eb7dffab1},
    std::array<uint64_t, 2>{0xc822a6319b601673, 0xe28a472eb7dffab1},
    std::array<uint64_t, 2>{0x856a1757fa88769e, 0xab43753882f7db29},
    std::array<uint64_t, 2>{0x856a1757fa88769f, 0xab43753882f7db29},
    std::array<uint64_t, 2>{0x2455fffbafe8b893, 0xae7ef06d2b604135},
    std::array<uint64_t, 2>{0x2455fffbafe8b892, 0xae7ef06d2b604135},
    std::array<uint64_t, 2>{0x987a355cbf0c842a, 0x8b49849339334e30},
    std::array<uint64_t, 2>{0x987a355cbf0c842b, 0x8b49849339334e30},
    std::array<uint64_t, 2>{0x3945ddf0ea6c4a27, 0x8e7401c690a4d42c},
    std::array<uint64_t, 2>{0x3945ddf0ea6c4a26, 0x8e7401c690a4d42c},
    std::array<uint64_t, 2>{0x740d6c968b842acb, 0xc7bd33d0a58cf5b4},
    std::array<uint64_t, 2>{0x740d6c968b842aca, 0xc7bd33d0a58cf5b4},
    std::array<uint64_t, 2>{0xd532843adee4e4c6, 0xc280b6850c1b6fa8},
    std::array<uint64_t, 2>{0xd532843adee4e4c7, 0xc280b6850c1b6fa8},
    std::array<uint64_t, 2>{0x27b537f28935c282, 0xbee4e4dc44629cf6},
    std::array<uint64_t, 2>{0x27b537f28935c283, 0xbee4e4dc44629cf6},
    std::array<uint64_t, 2>{0x868adf5edc550c8f, 0xbbd96189edf506ea},
    std::array<uint64_t, 2>{0x868adf5edc550c8e, 0xbbd96189edf506ea},
    std::array<uint64_t, 2>{0xcbc26e38bdbd6c63, 0xf210539fd8dd2772},
    std::array<uint64_t, 2>{0xcbc26e38bdbd6c62, 0xf210539fd8dd2772},
    std::array<uint64_t, 2>{0x6afd8694e8dda26e, 0xf72dd6ca714abd6e},
    std::array<uint64_t, 2>{0x6afd8694e8dda26f, 0xf72dd6ca714abd6e},
    std::array<uint64_t, 2>{0xf3f9051483221c63, 0x8682b2d579916cce},
    std::array<uint64_t, 2>{0xf3f9051483221c62, 0x8682b2d579916cce},
    std::array<uint64_t, 2>{0x52c6edb8d642d26e, 0x83bf3780d006f6d2},
    std::array<uint64_t, 2>{0x52c6edb8d642d26f, 0x83bf3780d006f6d2},
    std::array<uint64_t, 2>{0x1f8e5cdeb7aab282, 0xca760596e52ed74a},
    std::array<uint64_t, 2>{0x1f8e5cdeb7aab283, 0xca760596e52ed74a},
    std::array<uint64_t, 2>{0xbeb1b472e2ca7c8f, 0xcf4b80c34cb94d56},
    std::array<uint64_t, 2>{0xbeb1b472e2ca7c8e, 0xcf4b80c34cb94d56},
    std::array<uint64_t, 2>{0x4c3607bab51b5acb, 0xb32fd29a04c0be08},
    std::array<uint64_t, 2>{0x4c3607bab51b5aca, 0xb32fd29a04c0be08},
    std::array<uint64_t, 2>{0xed09ef16e07b94c6, 0xb61257cfad572414},
    std::array<uint64_t, 2>{0xed09ef16e07b94c7, 0xb61257cfad572414},
    std::array<uint64_t, 2>{0xa0415e708193f42a, 0xffdb65d9987f058c},
    std::array<uint64_t, 2>{0xa0415e708193f42b, 0xffdb65d9987f058c},
    std::array<uint64_t, 2>{0x017eb6dcd4f33a27, 0xfae6e08c31e89f90},
    std::array<uint64_t, 2>{0x017eb6dcd4f33a26, 0xfae6e08c31e89f90},
}};
}

// somewhat optimized inner product, only do one lazy reduction
field::GF2_128 dot_product(const std::vector<field::GF2_128> &lhs,
                           const std::vector<field::GF2_128> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  __m128i accum[2] = {_mm_setzero_si128(), _mm_setzero_si128()};
  __m128i tmp[3];
  tmp[0] = _mm_setzero_si128();
  tmp[1] = _mm_setzero_si128();
  for (size_t i = 0; i < lhs.size(); i++) {
    const __m128i *left = lhs[i].as_const_m128i();
    const __m128i *right = rhs[i].as_const_m128i();
    accum[0] =
        _mm_xor_si128(accum[0], _mm_clmulepi64_si128(*left, *right, 0x00));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(*left, *right, 0x11));
    tmp[0] = _mm_xor_si128(tmp[0], _mm_clmulepi64_si128(*left, *right, 0x10));
    tmp[1] = _mm_xor_si128(tmp[1], _mm_clmulepi64_si128(*left, *right, 0x01));
  }
  tmp[0] = _mm_xor_si128(tmp[0], tmp[1]);
  tmp[1] = _mm_slli_si128(tmp[0], 8);
  tmp[2] = _mm_srli_si128(tmp[0], 8);

  accum[0] = _mm_xor_si128(accum[0], tmp[1]);
  accum[1] = _mm_xor_si128(accum[1], tmp[2]);
  // combined reduction

  field::GF2_128 result;
  reduce_clmul(result.as_m128i(), accum);
  return result;
}

// somewhat optimized inner product, only do one lazy reduction
field::GF2_128
lifted_dot_product_uint8(const gsl::span<const field::GF2_128> &lhs,
                         const gsl::span<const uint8_t> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  __m128i accum[2] = {_mm_setzero_si128(), _mm_setzero_si128()};
  __m128i tmp[3];
  tmp[0] = _mm_setzero_si128();
  tmp[1] = _mm_setzero_si128();
  for (size_t i = 0; i < lhs.size(); i++) {
    const __m128i *left = lhs[i].as_const_m128i();
    const __m128i *right = lifting_lut[rhs[i]].as_const_m128i();
    accum[0] =
        _mm_xor_si128(accum[0], _mm_clmulepi64_si128(*left, *right, 0x00));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(*left, *right, 0x11));
    tmp[0] = _mm_xor_si128(tmp[0], _mm_clmulepi64_si128(*left, *right, 0x10));
    tmp[1] = _mm_xor_si128(tmp[1], _mm_clmulepi64_si128(*left, *right, 0x01));
  }
  tmp[0] = _mm_xor_si128(tmp[0], tmp[1]);
  tmp[1] = _mm_slli_si128(tmp[0], 8);
  tmp[2] = _mm_srli_si128(tmp[0], 8);

  accum[0] = _mm_xor_si128(accum[0], tmp[1]);
  accum[1] = _mm_xor_si128(accum[1], tmp[2]);
  // combined reduction

  field::GF2_128 result;
  reduce_clmul(result.as_m128i(), accum);
  return result;
}

// somewhat optimized inner product, only do one lazy reduction
field::GF2_128 lifted_dot_product(const gsl::span<const field::GF2_128> &lhs,
                                  const gsl::span<const field::GF2_8> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  __m128i accum[2] = {_mm_setzero_si128(), _mm_setzero_si128()};
  __m128i tmp[3];
  tmp[0] = _mm_setzero_si128();
  tmp[1] = _mm_setzero_si128();
  for (size_t i = 0; i < lhs.size(); i++) {
    const __m128i *left = lhs[i].as_const_m128i();
    const __m128i *right = lifting_lut[rhs[i].data].as_const_m128i();
    accum[0] =
        _mm_xor_si128(accum[0], _mm_clmulepi64_si128(*left, *right, 0x00));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(*left, *right, 0x11));
    tmp[0] = _mm_xor_si128(tmp[0], _mm_clmulepi64_si128(*left, *right, 0x10));
    tmp[1] = _mm_xor_si128(tmp[1], _mm_clmulepi64_si128(*left, *right, 0x01));
  }
  tmp[0] = _mm_xor_si128(tmp[0], tmp[1]);
  tmp[1] = _mm_slli_si128(tmp[0], 8);
  tmp[2] = _mm_srli_si128(tmp[0], 8);

  accum[0] = _mm_xor_si128(accum[0], tmp[1]);
  accum[1] = _mm_xor_si128(accum[1], tmp[2]);
  // combined reduction

  field::GF2_128 result;
  reduce_clmul(result.as_m128i(), accum);
  return result;
}

// naive horner eval
// field::GF2_128 field::eval_lifted(const gsl::span<const field::GF2_8> &poly,
//                                   const field::GF2_128 &point) {
//   field::GF2_128 acc;
//   long i;

//   for (i = poly.size() - 1; i >= 0; i--) {
//     acc *= point;
//     acc += lifting_lut[poly[i].data];
//   }

//   return acc;
// }

// more optimized horner eval by splitting the poly in four terms
//   x^3 * (c_3 + x^4*c_7 + x^8*c_11 +...)
// + x^2 * (c_2 + x^4*c_6 + x^8*c_10 + ...)
// + x   * (c_1 + x^4*c_5 + x^8*c_9 +...)
// +       (c_0 + x^4*c_4 + x^8*c_8 + ...)
// and evaluating both with horner. While this is actually a bit more work
// than the trivial horner eval, it is much more friendly to the CPU, since
// the four polys can be evaluated in a more interleaved manner
field::GF2_128 field::eval_lifted(const gsl::span<const field::GF2_8> &poly,
                                  const field::GF2_128 &point) {
  field::GF2_128 acc_0, acc_1, acc_2, acc_3;
  long i = poly.size() - 1;

  field::GF2_128 p2, p4;
  gf128sqr(p2.as_m128i(), point.as_const_m128i());
  gf128sqr(p4.as_m128i(), p2.as_const_m128i());

  // i is the degree, if it is not a multiple of 4 - 1, then we have more coeff
  // that we already need to handle here
  if (i % 4 == 2) {
    acc_2 = lifting_lut[poly[i].data];
    i--;
  }
  if (i % 4 == 1) {
    acc_1 = lifting_lut[poly[i].data];
    i--;
  }
  if (i % 4 == 0) {
    acc_0 = lifting_lut[poly[i].data];
    i--;
  }

  // now the degree is a (multiple of 4) - 1 for sure, we have an multiple of 4
  // number of coefficients and can handle 4 simultaneous to improve data
  // dependency problems
  for (; i >= 0; i -= 4) {
    acc_0 *= p4;
    acc_1 *= p4;
    acc_2 *= p4;
    acc_3 *= p4;
    acc_0 += lifting_lut[poly[i - 3].data];
    acc_1 += lifting_lut[poly[i - 2].data];
    acc_2 += lifting_lut[poly[i - 1].data];
    acc_3 += lifting_lut[poly[i].data];
  }

  return acc_0 + point * acc_1 + (acc_2 + point * acc_3) * p2;
}
// TEMPLATE INSTANTIATIONS for GF2_128

// yes we include the cpp file with the template stuff
#include "field_templates.cpp"

INSTANTIATE_TEMPLATES_FOR(field::GF2_128)
// END TEMPLATE INSTANTIATIONS for GF2_128