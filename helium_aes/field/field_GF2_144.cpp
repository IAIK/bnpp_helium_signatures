#if __INCLUDE_LEVEL__ != 0
#error "Don't include this file"
#endif
#include "field.h"

#include "../gsl-lite.hpp"
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

inline void clmul_schoolbook(__m128i out[3], const __m128i a[2],
                             const __m128i b[2]) {
  __m128i tmp[3];
  out[0] = _mm_clmulepi64_si128(a[0], b[0], 0x00);
  out[1] = _mm_clmulepi64_si128(a[0], b[0], 0x11);
  out[2] = _mm_clmulepi64_si128(a[1], b[1], 0x00);
  out[1] = _mm_xor_si128(out[1], _mm_clmulepi64_si128(a[0], b[1], 0x00));
  out[1] = _mm_xor_si128(out[1], _mm_clmulepi64_si128(a[1], b[0], 0x00));

  tmp[0] = _mm_clmulepi64_si128(a[0], b[0], 0x01);
  tmp[1] = _mm_clmulepi64_si128(a[0], b[0], 0x10);

  tmp[0] = _mm_xor_si128(tmp[0], tmp[1]);
  tmp[1] = _mm_slli_si128(tmp[0], 8);
  tmp[2] = _mm_srli_si128(tmp[0], 8);

  out[0] = _mm_xor_si128(out[0], tmp[1]);
  out[1] = _mm_xor_si128(out[1], tmp[2]);

  tmp[0] = _mm_clmulepi64_si128(a[1], b[0], 0x10);
  tmp[1] = _mm_clmulepi64_si128(a[0], b[1], 0x01);

  tmp[0] = _mm_xor_si128(tmp[0], tmp[1]);
  tmp[1] = _mm_slli_si128(tmp[0], 8);
  tmp[2] = _mm_srli_si128(tmp[0], 8);

  out[1] = _mm_xor_si128(out[1], tmp[1]);
  out[2] = _mm_xor_si128(out[2], tmp[2]);
}

inline void sqr(__m128i out[3], const __m128i a[2]) {
  __m128i tmp[2];
  __m128i sqrT = _mm_set_epi64x(0x5554515045444140, 0x1514111005040100);
  __m128i mask = _mm_set_epi64x(0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F);
  tmp[0] = _mm_and_si128(a[0], mask);
  tmp[1] = _mm_srli_epi64(a[0], 4);
  tmp[1] = _mm_and_si128(tmp[1], mask);
  tmp[0] = _mm_shuffle_epi8(sqrT, tmp[0]);
  tmp[1] = _mm_shuffle_epi8(sqrT, tmp[1]);
  out[0] = _mm_unpacklo_epi8(tmp[0], tmp[1]);
  out[1] = _mm_unpackhi_epi8(tmp[0], tmp[1]);

  tmp[0] = _mm_and_si128(a[1], mask);
  tmp[1] = _mm_srli_epi64(a[1], 4);
  tmp[1] = _mm_and_si128(tmp[1], mask);
  tmp[0] = _mm_shuffle_epi8(sqrT, tmp[0]);
  tmp[1] = _mm_shuffle_epi8(sqrT, tmp[1]);
  out[2] = _mm_unpacklo_epi8(tmp[0], tmp[1]);
}

inline void reduce_clmul(__m128i out[2], const __m128i in[3]) {
  // modulus = x^144 + x^7 + x^4 + x^2 + 1
  __m128i p = _mm_set_epi64x(0x0, 0x95);
  __m128i t0, t1, t2, t3;

  t0 = _mm_clmulepi64_si128(in[2], p, 0x00); // in[2]_low * p
  t1 = _mm_slli_si128(t0, 14);   // low 16bit of result, shifted to high
  t2 = _mm_srli_si128(t0, 2);    // high 112bit of result, shifted to low
  t3 = _mm_xor_si128(in[1], t2); // update in[1]_low
  out[0] = _mm_xor_si128(in[0], t1);

  t0 = _mm_clmulepi64_si128(in[1], p, 0x01); // in[1]_high * p
  t1 = _mm_slli_si128(t0, 6); // low 80bit of result, shifted to high
  out[0] = _mm_xor_si128(out[0], t1);

  t1 = _mm_srli_si128(t3, 2); // only part that is > 144
  t1 = _mm_and_si128(t1, _mm_set_epi64x(0x0, 0xFFFF'FFFF'FFFF));
  t0 = _mm_clmulepi64_si128(t1, p, 0x00); // in[1]_low * p
  out[0] = _mm_xor_si128(out[0], t0);
  out[1] = _mm_and_si128(t3, _mm_set_epi64x(0x0, 0xFFFF));
}

inline void gf2_144mul(__m128i *out, const __m128i *in1, const __m128i *in2) {
  __m128i tmp[3];
  clmul_schoolbook(tmp, in1, in2);
  reduce_clmul(out, tmp);
}
inline void gf2_144sqr(__m128i *out, const __m128i *in) {
  __m128i tmp[3];
  sqr(tmp, in);
  reduce_clmul(out, tmp);
}

inline void gf2_144add(__m128i *out, const __m128i *in1, const __m128i *in2) {
  out[0] = _mm_xor_si128(in1[0], in2[0]);
  out[1] = _mm_xor_si128(in1[1], in2[1]);
}
inline void gf2_144add(__m256i *out, const __m256i *in1, const __m256i *in2) {
  out[0] = _mm256_xor_si256(in1[0], in2[0]);
}

} // namespace

namespace field {

GF2_144::GF2_144(std::string hex_string) {
  // check if hex_string start with 0x or 0X
  if (hex_string.rfind("0x", 0) == 0 || hex_string.rfind("0X", 0) == 0) {
    hex_string = hex_string.substr(2);
  } else {
    throw std::runtime_error("input needs to be a hex number");
  }
  constexpr size_t num_hex_chars = 144 / 4;
  if (hex_string.length() > num_hex_chars)
    throw std::runtime_error("input hex is too large");
  // pad to 144 bit
  hex_string.insert(hex_string.begin(), num_hex_chars - hex_string.length(),
                    '0');
  // high 64 bit
  uint64_t high = std::stoull(hex_string.substr(0, 16 / 4), nullptr, 16);
  // middle 64 bit
  uint64_t mid = std::stoull(hex_string.substr(16 / 4, 64 / 4), nullptr, 16);
  // low 64 bit
  uint64_t low = std::stoull(hex_string.substr(80 / 4, 64 / 4), nullptr, 16);
  data[0] = low;
  data[1] = mid;
  data[2] = high;
  data[3] = 0;
}

GF2_144 GF2_144::operator+(const GF2_144 &other) const {
  GF2_144 result;
  gf2_144add(result.as_m256i(), this->as_const_m256i(), other.as_const_m256i());
  return result;
}
GF2_144 &GF2_144::operator+=(const GF2_144 &other) {
  gf2_144add(this->as_m256i(), this->as_const_m256i(), other.as_const_m256i());
  return *this;
}
GF2_144 GF2_144::operator-(const GF2_144 &other) const {
  GF2_144 result;
  gf2_144add(result.as_m256i(), this->as_const_m256i(), other.as_const_m256i());
  return result;
}
GF2_144 &GF2_144::operator-=(const GF2_144 &other) {
  gf2_144add(this->as_m256i(), this->as_const_m256i(), other.as_const_m256i());
  return *this;
}
GF2_144 GF2_144::operator*(const GF2_144 &other) const {
  GF2_144 result;
  gf2_144mul(result.as_m128i(), this->as_const_m128i(), other.as_const_m128i());
  return result;
}
GF2_144 &GF2_144::operator*=(const GF2_144 &other) {
  gf2_144mul(this->as_m128i(), this->as_const_m128i(), other.as_const_m128i());
  return *this;
}
bool GF2_144::operator==(const GF2_144 &other) const {
  return this->data == other.data;
}
bool GF2_144::operator!=(const GF2_144 &other) const {
  return this->data != other.data;
}

GF2_144 GF2_144::inverse() const {
  // Fixed-op square-multiply
  // 2^n - 2 in binary is 0b1111..10
  __m128i t1[2];
  __m128i t2[2];
  t1[0] = this->as_const_m128i()[0];
  t1[1] = this->as_const_m128i()[1];
  gf2_144mul(t2, t1, t1);

  // First 142 one-bits (start from second)
  for (size_t i = 0; i < 142; i++) {
    gf2_144mul(t1, t1, t2);
    gf2_144sqr(t2, t2);
  }

  GF2_144 out;
  // Final zero-bit
  gf2_144sqr(out.as_m128i(), t1);

  return out;
}

void GF2_144::to_bytes(uint8_t *out) const {
  uint64_t le_data = htole64(data[0]);
  memcpy(out, (uint8_t *)(&le_data), sizeof(uint64_t));
  le_data = htole64(data[1]);
  memcpy(out + sizeof(uint64_t), (uint8_t *)(&le_data), sizeof(uint64_t));
  le_data = htole64(data[2]);
  memcpy(out + 2 * sizeof(uint64_t), (uint8_t *)(&le_data), 2);
}

void GF2_144::from_bytes(const uint8_t *in) {
  uint64_t tmp;
  memcpy((uint8_t *)(&tmp), in, sizeof(uint64_t));
  data[0] = le64toh(tmp);
  memcpy((uint8_t *)(&tmp), in + sizeof(uint64_t), sizeof(uint64_t));
  data[1] = le64toh(tmp);
  tmp = 0;
  memcpy((uint8_t *)(&tmp), in + 2 * sizeof(uint64_t), 2);
  data[2] = le64toh(tmp);
}

} // namespace field

// somewhat optimized inner product, only do one lazy reduction
field::GF2_144 dot_product(const std::vector<field::GF2_144> &lhs,
                           const std::vector<field::GF2_144> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  __m128i accum[3] = {_mm_setzero_si128(), _mm_setzero_si128(),
                      _mm_setzero_si128()};
  __m128i tmp[4];
  tmp[0] = _mm_setzero_si128();
  tmp[1] = _mm_setzero_si128();
  for (size_t i = 0; i < lhs.size(); i++) {
    accum[0] = _mm_xor_si128(
        accum[0], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[0],
                                       rhs[i].as_const_m128i()[0], 0x00));
    accum[1] = _mm_xor_si128(
        accum[1], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[0],
                                       rhs[i].as_const_m128i()[0], 0x11));
    accum[1] = _mm_xor_si128(
        accum[1], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[1],
                                       rhs[i].as_const_m128i()[0], 0x00));
    accum[1] = _mm_xor_si128(
        accum[1], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[0],
                                       rhs[i].as_const_m128i()[1], 0x00));
    accum[2] = _mm_xor_si128(
        accum[2], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[1],
                                       rhs[i].as_const_m128i()[1], 0x00));
    tmp[0] = _mm_xor_si128(
        tmp[0], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[0],
                                     rhs[i].as_const_m128i()[0], 0x10));
    tmp[0] = _mm_xor_si128(
        tmp[0], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[0],
                                     rhs[i].as_const_m128i()[0], 0x01));

    tmp[1] = _mm_xor_si128(
        tmp[1], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[1],
                                     rhs[i].as_const_m128i()[0], 0x10));
    tmp[1] = _mm_xor_si128(
        tmp[1], _mm_clmulepi64_si128(lhs[i].as_const_m128i()[0],
                                     rhs[i].as_const_m128i()[1], 0x01));
  }
  tmp[2] = _mm_slli_si128(tmp[0], 8);
  tmp[3] = _mm_srli_si128(tmp[0], 8);

  accum[0] = _mm_xor_si128(accum[0], tmp[2]);
  accum[1] = _mm_xor_si128(accum[1], tmp[3]);

  tmp[2] = _mm_slli_si128(tmp[1], 8);
  tmp[3] = _mm_srli_si128(tmp[1], 8);

  accum[1] = _mm_xor_si128(accum[1], tmp[2]);
  accum[2] = _mm_xor_si128(accum[2], tmp[3]);
  // combined reduction
  field::GF2_144 result;
  reduce_clmul(result.as_m128i(), accum);
  return result;
}

namespace {
static constexpr std::array<field::GF2_144, 256> lifting_lut = {
    std::array<uint64_t, 4>{0x0000000000000000, 0x0000000000000000,
                            0x0000000000000000, 0x0},
    std::array<uint64_t, 4>{0x0000000000000001, 0x0000000000000000,
                            0x0000000000000000, 0x0},
    std::array<uint64_t, 4>{0x52e17b8c53faf5ae, 0xe400c113ace8d64b,
                            0x0000000000000dd2, 0x0},
    std::array<uint64_t, 4>{0x52e17b8c53faf5af, 0xe400c113ace8d64b,
                            0x0000000000000dd2, 0x0},
    std::array<uint64_t, 4>{0x86e36d7a347dca50, 0x967e1951156950c5,
                            0x0000000000003cec, 0x0},
    std::array<uint64_t, 4>{0x86e36d7a347dca51, 0x967e1951156950c5,
                            0x0000000000003cec, 0x0},
    std::array<uint64_t, 4>{0xd40216f667873ffe, 0x727ed842b981868e,
                            0x000000000000313e, 0x0},
    std::array<uint64_t, 4>{0xd40216f667873fff, 0x727ed842b981868e,
                            0x000000000000313e, 0x0},
    std::array<uint64_t, 4>{0xa05680d2ef91284f, 0xe6f27a4a130c5592,
                            0x0000000000005c3a, 0x0},
    std::array<uint64_t, 4>{0xa05680d2ef91284e, 0xe6f27a4a130c5592,
                            0x0000000000005c3a, 0x0},
    std::array<uint64_t, 4>{0xf2b7fb5ebc6bdde1, 0x02f2bb59bfe483d9,
                            0x00000000000051e8, 0x0},
    std::array<uint64_t, 4>{0xf2b7fb5ebc6bdde0, 0x02f2bb59bfe483d9,
                            0x00000000000051e8, 0x0},
    std::array<uint64_t, 4>{0x26b5eda8dbece21f, 0x708c631b06650557,
                            0x00000000000060d6, 0x0},
    std::array<uint64_t, 4>{0x26b5eda8dbece21e, 0x708c631b06650557,
                            0x00000000000060d6, 0x0},
    std::array<uint64_t, 4>{0x74549624881617b1, 0x948ca208aa8dd31c,
                            0x0000000000006d04, 0x0},
    std::array<uint64_t, 4>{0x74549624881617b0, 0x948ca208aa8dd31c,
                            0x0000000000006d04, 0x0},
    std::array<uint64_t, 4>{0xc085899f7198d52a, 0x28218a0aab15a598,
                            0x000000000000bd2f, 0x0},
    std::array<uint64_t, 4>{0xc085899f7198d52b, 0x28218a0aab15a598,
                            0x000000000000bd2f, 0x0},
    std::array<uint64_t, 4>{0x9264f21322622084, 0xcc214b1907fd73d3,
                            0x000000000000b0fd, 0x0},
    std::array<uint64_t, 4>{0x9264f21322622085, 0xcc214b1907fd73d3,
                            0x000000000000b0fd, 0x0},
    std::array<uint64_t, 4>{0x4666e4e545e51f7a, 0xbe5f935bbe7cf55d,
                            0x00000000000081c3, 0x0},
    std::array<uint64_t, 4>{0x4666e4e545e51f7b, 0xbe5f935bbe7cf55d,
                            0x00000000000081c3, 0x0},
    std::array<uint64_t, 4>{0x14879f69161fead4, 0x5a5f524812942316,
                            0x0000000000008c11, 0x0},
    std::array<uint64_t, 4>{0x14879f69161fead5, 0x5a5f524812942316,
                            0x0000000000008c11, 0x0},
    std::array<uint64_t, 4>{0x60d3094d9e09fd65, 0xced3f040b819f00a,
                            0x000000000000e115, 0x0},
    std::array<uint64_t, 4>{0x60d3094d9e09fd64, 0xced3f040b819f00a,
                            0x000000000000e115, 0x0},
    std::array<uint64_t, 4>{0x323272c1cdf308cb, 0x2ad3315314f12641,
                            0x000000000000ecc7, 0x0},
    std::array<uint64_t, 4>{0x323272c1cdf308ca, 0x2ad3315314f12641,
                            0x000000000000ecc7, 0x0},
    std::array<uint64_t, 4>{0xe6306437aa743735, 0x58ade911ad70a0cf,
                            0x000000000000ddf9, 0x0},
    std::array<uint64_t, 4>{0xe6306437aa743734, 0x58ade911ad70a0cf,
                            0x000000000000ddf9, 0x0},
    std::array<uint64_t, 4>{0xb4d11fbbf98ec29b, 0xbcad280201987684,
                            0x000000000000d02b, 0x0},
    std::array<uint64_t, 4>{0xb4d11fbbf98ec29a, 0xbcad280201987684,
                            0x000000000000d02b, 0x0},
    std::array<uint64_t, 4>{0x05c98752e572c25f, 0xae055720dce33dd9,
                            0x000000000000554d, 0x0},
    std::array<uint64_t, 4>{0x05c98752e572c25e, 0xae055720dce33dd9,
                            0x000000000000554d, 0x0},
    std::array<uint64_t, 4>{0x5728fcdeb68837f1, 0x4a059633700beb92,
                            0x000000000000589f, 0x0},
    std::array<uint64_t, 4>{0x5728fcdeb68837f0, 0x4a059633700beb92,
                            0x000000000000589f, 0x0},
    std::array<uint64_t, 4>{0x832aea28d10f080f, 0x387b4e71c98a6d1c,
                            0x00000000000069a1, 0x0},
    std::array<uint64_t, 4>{0x832aea28d10f080e, 0x387b4e71c98a6d1c,
                            0x00000000000069a1, 0x0},
    std::array<uint64_t, 4>{0xd1cb91a482f5fda1, 0xdc7b8f626562bb57,
                            0x0000000000006473, 0x0},
    std::array<uint64_t, 4>{0xd1cb91a482f5fda0, 0xdc7b8f626562bb57,
                            0x0000000000006473, 0x0},
    std::array<uint64_t, 4>{0xa59f07800ae3ea10, 0x48f72d6acfef684b,
                            0x0000000000000977, 0x0},
    std::array<uint64_t, 4>{0xa59f07800ae3ea11, 0x48f72d6acfef684b,
                            0x0000000000000977, 0x0},
    std::array<uint64_t, 4>{0xf77e7c0c59191fbe, 0xacf7ec796307be00,
                            0x00000000000004a5, 0x0},
    std::array<uint64_t, 4>{0xf77e7c0c59191fbf, 0xacf7ec796307be00,
                            0x00000000000004a5, 0x0},
    std::array<uint64_t, 4>{0x237c6afa3e9e2040, 0xde89343bda86388e,
                            0x000000000000359b, 0x0},
    std::array<uint64_t, 4>{0x237c6afa3e9e2041, 0xde89343bda86388e,
                            0x000000000000359b, 0x0},
    std::array<uint64_t, 4>{0x719d11766d64d5ee, 0x3a89f528766eeec5,
                            0x0000000000003849, 0x0},
    std::array<uint64_t, 4>{0x719d11766d64d5ef, 0x3a89f528766eeec5,
                            0x0000000000003849, 0x0},
    std::array<uint64_t, 4>{0xc54c0ecd94ea1775, 0x8624dd2a77f69841,
                            0x000000000000e862, 0x0},
    std::array<uint64_t, 4>{0xc54c0ecd94ea1774, 0x8624dd2a77f69841,
                            0x000000000000e862, 0x0},
    std::array<uint64_t, 4>{0x97ad7541c710e2db, 0x62241c39db1e4e0a,
                            0x000000000000e5b0, 0x0},
    std::array<uint64_t, 4>{0x97ad7541c710e2da, 0x62241c39db1e4e0a,
                            0x000000000000e5b0, 0x0},
    std::array<uint64_t, 4>{0x43af63b7a097dd25, 0x105ac47b629fc884,
                            0x000000000000d48e, 0x0},
    std::array<uint64_t, 4>{0x43af63b7a097dd24, 0x105ac47b629fc884,
                            0x000000000000d48e, 0x0},
    std::array<uint64_t, 4>{0x114e183bf36d288b, 0xf45a0568ce771ecf,
                            0x000000000000d95c, 0x0},
    std::array<uint64_t, 4>{0x114e183bf36d288a, 0xf45a0568ce771ecf,
                            0x000000000000d95c, 0x0},
    std::array<uint64_t, 4>{0x651a8e1f7b7b3f3a, 0x60d6a76064facdd3,
                            0x000000000000b458, 0x0},
    std::array<uint64_t, 4>{0x651a8e1f7b7b3f3b, 0x60d6a76064facdd3,
                            0x000000000000b458, 0x0},
    std::array<uint64_t, 4>{0x37fbf5932881ca94, 0x84d66673c8121b98,
                            0x000000000000b98a, 0x0},
    std::array<uint64_t, 4>{0x37fbf5932881ca95, 0x84d66673c8121b98,
                            0x000000000000b98a, 0x0},
    std::array<uint64_t, 4>{0xe3f9e3654f06f56a, 0xf6a8be3171939d16,
                            0x00000000000088b4, 0x0},
    std::array<uint64_t, 4>{0xe3f9e3654f06f56b, 0xf6a8be3171939d16,
                            0x00000000000088b4, 0x0},
    std::array<uint64_t, 4>{0xb11898e91cfc00c4, 0x12a87f22dd7b4b5d,
                            0x0000000000008566, 0x0},
    std::array<uint64_t, 4>{0xb11898e91cfc00c5, 0x12a87f22dd7b4b5d,
                            0x0000000000008566, 0x0},
    std::array<uint64_t, 4>{0x2341d6c02859d8ad, 0xa23a5e3e935fe719,
                            0x000000000000a816, 0x0},
    std::array<uint64_t, 4>{0x2341d6c02859d8ac, 0xa23a5e3e935fe719,
                            0x000000000000a816, 0x0},
    std::array<uint64_t, 4>{0x71a0ad4c7ba32d03, 0x463a9f2d3fb73152,
                            0x000000000000a5c4, 0x0},
    std::array<uint64_t, 4>{0x71a0ad4c7ba32d02, 0x463a9f2d3fb73152,
                            0x000000000000a5c4, 0x0},
    std::array<uint64_t, 4>{0xa5a2bbba1c2412fd, 0x3444476f8636b7dc,
                            0x00000000000094fa, 0x0},
    std::array<uint64_t, 4>{0xa5a2bbba1c2412fc, 0x3444476f8636b7dc,
                            0x00000000000094fa, 0x0},
    std::array<uint64_t, 4>{0xf743c0364fdee753, 0xd044867c2ade6197,
                            0x0000000000009928, 0x0},
    std::array<uint64_t, 4>{0xf743c0364fdee752, 0xd044867c2ade6197,
                            0x0000000000009928, 0x0},
    std::array<uint64_t, 4>{0x83175612c7c8f0e2, 0x44c824748053b28b,
                            0x000000000000f42c, 0x0},
    std::array<uint64_t, 4>{0x83175612c7c8f0e3, 0x44c824748053b28b,
                            0x000000000000f42c, 0x0},
    std::array<uint64_t, 4>{0xd1f62d9e9432054c, 0xa0c8e5672cbb64c0,
                            0x000000000000f9fe, 0x0},
    std::array<uint64_t, 4>{0xd1f62d9e9432054d, 0xa0c8e5672cbb64c0,
                            0x000000000000f9fe, 0x0},
    std::array<uint64_t, 4>{0x05f43b68f3b53ab2, 0xd2b63d25953ae24e,
                            0x000000000000c8c0, 0x0},
    std::array<uint64_t, 4>{0x05f43b68f3b53ab3, 0xd2b63d25953ae24e,
                            0x000000000000c8c0, 0x0},
    std::array<uint64_t, 4>{0x571540e4a04fcf1c, 0x36b6fc3639d23405,
                            0x000000000000c512, 0x0},
    std::array<uint64_t, 4>{0x571540e4a04fcf1d, 0x36b6fc3639d23405,
                            0x000000000000c512, 0x0},
    std::array<uint64_t, 4>{0xe3c45f5f59c10d87, 0x8a1bd434384a4281,
                            0x0000000000001539, 0x0},
    std::array<uint64_t, 4>{0xe3c45f5f59c10d86, 0x8a1bd434384a4281,
                            0x0000000000001539, 0x0},
    std::array<uint64_t, 4>{0xb12524d30a3bf829, 0x6e1b152794a294ca,
                            0x00000000000018eb, 0x0},
    std::array<uint64_t, 4>{0xb12524d30a3bf828, 0x6e1b152794a294ca,
                            0x00000000000018eb, 0x0},
    std::array<uint64_t, 4>{0x652732256dbcc7d7, 0x1c65cd652d231244,
                            0x00000000000029d5, 0x0},
    std::array<uint64_t, 4>{0x652732256dbcc7d6, 0x1c65cd652d231244,
                            0x00000000000029d5, 0x0},
    std::array<uint64_t, 4>{0x37c649a93e463279, 0xf8650c7681cbc40f,
                            0x0000000000002407, 0x0},
    std::array<uint64_t, 4>{0x37c649a93e463278, 0xf8650c7681cbc40f,
                            0x0000000000002407, 0x0},
    std::array<uint64_t, 4>{0x4392df8db65025c8, 0x6ce9ae7e2b461713,
                            0x0000000000004903, 0x0},
    std::array<uint64_t, 4>{0x4392df8db65025c9, 0x6ce9ae7e2b461713,
                            0x0000000000004903, 0x0},
    std::array<uint64_t, 4>{0x1173a401e5aad066, 0x88e96f6d87aec158,
                            0x00000000000044d1, 0x0},
    std::array<uint64_t, 4>{0x1173a401e5aad067, 0x88e96f6d87aec158,
                            0x00000000000044d1, 0x0},
    std::array<uint64_t, 4>{0xc571b2f7822def98, 0xfa97b72f3e2f47d6,
                            0x00000000000075ef, 0x0},
    std::array<uint64_t, 4>{0xc571b2f7822def99, 0xfa97b72f3e2f47d6,
                            0x00000000000075ef, 0x0},
    std::array<uint64_t, 4>{0x9790c97bd1d71a36, 0x1e97763c92c7919d,
                            0x000000000000783d, 0x0},
    std::array<uint64_t, 4>{0x9790c97bd1d71a37, 0x1e97763c92c7919d,
                            0x000000000000783d, 0x0},
    std::array<uint64_t, 4>{0x26885192cd2b1af2, 0x0c3f091e4fbcdac0,
                            0x000000000000fd5b, 0x0},
    std::array<uint64_t, 4>{0x26885192cd2b1af3, 0x0c3f091e4fbcdac0,
                            0x000000000000fd5b, 0x0},
    std::array<uint64_t, 4>{0x74692a1e9ed1ef5c, 0xe83fc80de3540c8b,
                            0x000000000000f089, 0x0},
    std::array<uint64_t, 4>{0x74692a1e9ed1ef5d, 0xe83fc80de3540c8b,
                            0x000000000000f089, 0x0},
    std::array<uint64_t, 4>{0xa06b3ce8f956d0a2, 0x9a41104f5ad58a05,
                            0x000000000000c1b7, 0x0},
    std::array<uint64_t, 4>{0xa06b3ce8f956d0a3, 0x9a41104f5ad58a05,
                            0x000000000000c1b7, 0x0},
    std::array<uint64_t, 4>{0xf28a4764aaac250c, 0x7e41d15cf63d5c4e,
                            0x000000000000cc65, 0x0},
    std::array<uint64_t, 4>{0xf28a4764aaac250d, 0x7e41d15cf63d5c4e,
                            0x000000000000cc65, 0x0},
    std::array<uint64_t, 4>{0x86ded14022ba32bd, 0xeacd73545cb08f52,
                            0x000000000000a161, 0x0},
    std::array<uint64_t, 4>{0x86ded14022ba32bc, 0xeacd73545cb08f52,
                            0x000000000000a161, 0x0},
    std::array<uint64_t, 4>{0xd43faacc7140c713, 0x0ecdb247f0585919,
                            0x000000000000acb3, 0x0},
    std::array<uint64_t, 4>{0xd43faacc7140c712, 0x0ecdb247f0585919,
                            0x000000000000acb3, 0x0},
    std::array<uint64_t, 4>{0x003dbc3a16c7f8ed, 0x7cb36a0549d9df97,
                            0x0000000000009d8d, 0x0},
    std::array<uint64_t, 4>{0x003dbc3a16c7f8ec, 0x7cb36a0549d9df97,
                            0x0000000000009d8d, 0x0},
    std::array<uint64_t, 4>{0x52dcc7b6453d0d43, 0x98b3ab16e53109dc,
                            0x000000000000905f, 0x0},
    std::array<uint64_t, 4>{0x52dcc7b6453d0d42, 0x98b3ab16e53109dc,
                            0x000000000000905f, 0x0},
    std::array<uint64_t, 4>{0xe60dd80dbcb3cfd8, 0x241e8314e4a97f58,
                            0x0000000000004074, 0x0},
    std::array<uint64_t, 4>{0xe60dd80dbcb3cfd9, 0x241e8314e4a97f58,
                            0x0000000000004074, 0x0},
    std::array<uint64_t, 4>{0xb4eca381ef493a76, 0xc01e42074841a913,
                            0x0000000000004da6, 0x0},
    std::array<uint64_t, 4>{0xb4eca381ef493a77, 0xc01e42074841a913,
                            0x0000000000004da6, 0x0},
    std::array<uint64_t, 4>{0x60eeb57788ce0588, 0xb2609a45f1c02f9d,
                            0x0000000000007c98, 0x0},
    std::array<uint64_t, 4>{0x60eeb57788ce0589, 0xb2609a45f1c02f9d,
                            0x0000000000007c98, 0x0},
    std::array<uint64_t, 4>{0x320fcefbdb34f026, 0x56605b565d28f9d6,
                            0x000000000000714a, 0x0},
    std::array<uint64_t, 4>{0x320fcefbdb34f027, 0x56605b565d28f9d6,
                            0x000000000000714a, 0x0},
    std::array<uint64_t, 4>{0x465b58df5322e797, 0xc2ecf95ef7a52aca,
                            0x0000000000001c4e, 0x0},
    std::array<uint64_t, 4>{0x465b58df5322e796, 0xc2ecf95ef7a52aca,
                            0x0000000000001c4e, 0x0},
    std::array<uint64_t, 4>{0x14ba235300d81239, 0x26ec384d5b4dfc81,
                            0x000000000000119c, 0x0},
    std::array<uint64_t, 4>{0x14ba235300d81238, 0x26ec384d5b4dfc81,
                            0x000000000000119c, 0x0},
    std::array<uint64_t, 4>{0xc0b835a5675f2dc7, 0x5492e00fe2cc7a0f,
                            0x00000000000020a2, 0x0},
    std::array<uint64_t, 4>{0xc0b835a5675f2dc6, 0x5492e00fe2cc7a0f,
                            0x00000000000020a2, 0x0},
    std::array<uint64_t, 4>{0x92594e2934a5d869, 0xb092211c4e24ac44,
                            0x0000000000002d70, 0x0},
    std::array<uint64_t, 4>{0x92594e2934a5d868, 0xb092211c4e24ac44,
                            0x0000000000002d70, 0x0},
    std::array<uint64_t, 4>{0x1c5fba85e3a60534, 0x36d1ea2ed40e0546,
                            0x00000000000034a4, 0x0},
    std::array<uint64_t, 4>{0x1c5fba85e3a60535, 0x36d1ea2ed40e0546,
                            0x00000000000034a4, 0x0},
    std::array<uint64_t, 4>{0x4ebec109b05cf09a, 0xd2d12b3d78e6d30d,
                            0x0000000000003976, 0x0},
    std::array<uint64_t, 4>{0x4ebec109b05cf09b, 0xd2d12b3d78e6d30d,
                            0x0000000000003976, 0x0},
    std::array<uint64_t, 4>{0x9abcd7ffd7dbcf64, 0xa0aff37fc1675583,
                            0x0000000000000848, 0x0},
    std::array<uint64_t, 4>{0x9abcd7ffd7dbcf65, 0xa0aff37fc1675583,
                            0x0000000000000848, 0x0},
    std::array<uint64_t, 4>{0xc85dac7384213aca, 0x44af326c6d8f83c8,
                            0x000000000000059a, 0x0},
    std::array<uint64_t, 4>{0xc85dac7384213acb, 0x44af326c6d8f83c8,
                            0x000000000000059a, 0x0},
    std::array<uint64_t, 4>{0xbc093a570c372d7b, 0xd0239064c70250d4,
                            0x000000000000689e, 0x0},
    std::array<uint64_t, 4>{0xbc093a570c372d7a, 0xd0239064c70250d4,
                            0x000000000000689e, 0x0},
    std::array<uint64_t, 4>{0xeee841db5fcdd8d5, 0x342351776bea869f,
                            0x000000000000654c, 0x0},
    std::array<uint64_t, 4>{0xeee841db5fcdd8d4, 0x342351776bea869f,
                            0x000000000000654c, 0x0},
    std::array<uint64_t, 4>{0x3aea572d384ae72b, 0x465d8935d26b0011,
                            0x0000000000005472, 0x0},
    std::array<uint64_t, 4>{0x3aea572d384ae72a, 0x465d8935d26b0011,
                            0x0000000000005472, 0x0},
    std::array<uint64_t, 4>{0x680b2ca16bb01285, 0xa25d48267e83d65a,
                            0x00000000000059a0, 0x0},
    std::array<uint64_t, 4>{0x680b2ca16bb01284, 0xa25d48267e83d65a,
                            0x00000000000059a0, 0x0},
    std::array<uint64_t, 4>{0xdcda331a923ed01e, 0x1ef060247f1ba0de,
                            0x000000000000898b, 0x0},
    std::array<uint64_t, 4>{0xdcda331a923ed01f, 0x1ef060247f1ba0de,
                            0x000000000000898b, 0x0},
    std::array<uint64_t, 4>{0x8e3b4896c1c425b0, 0xfaf0a137d3f37695,
                            0x0000000000008459, 0x0},
    std::array<uint64_t, 4>{0x8e3b4896c1c425b1, 0xfaf0a137d3f37695,
                            0x0000000000008459, 0x0},
    std::array<uint64_t, 4>{0x5a395e60a6431a4e, 0x888e79756a72f01b,
                            0x000000000000b567, 0x0},
    std::array<uint64_t, 4>{0x5a395e60a6431a4f, 0x888e79756a72f01b,
                            0x000000000000b567, 0x0},
    std::array<uint64_t, 4>{0x08d825ecf5b9efe0, 0x6c8eb866c69a2650,
                            0x000000000000b8b5, 0x0},
    std::array<uint64_t, 4>{0x08d825ecf5b9efe1, 0x6c8eb866c69a2650,
                            0x000000000000b8b5, 0x0},
    std::array<uint64_t, 4>{0x7c8cb3c87daff851, 0xf8021a6e6c17f54c,
                            0x000000000000d5b1, 0x0},
    std::array<uint64_t, 4>{0x7c8cb3c87daff850, 0xf8021a6e6c17f54c,
                            0x000000000000d5b1, 0x0},
    std::array<uint64_t, 4>{0x2e6dc8442e550dff, 0x1c02db7dc0ff2307,
                            0x000000000000d863, 0x0},
    std::array<uint64_t, 4>{0x2e6dc8442e550dfe, 0x1c02db7dc0ff2307,
                            0x000000000000d863, 0x0},
    std::array<uint64_t, 4>{0xfa6fdeb249d23201, 0x6e7c033f797ea589,
                            0x000000000000e95d, 0x0},
    std::array<uint64_t, 4>{0xfa6fdeb249d23200, 0x6e7c033f797ea589,
                            0x000000000000e95d, 0x0},
    std::array<uint64_t, 4>{0xa88ea53e1a28c7af, 0x8a7cc22cd59673c2,
                            0x000000000000e48f, 0x0},
    std::array<uint64_t, 4>{0xa88ea53e1a28c7ae, 0x8a7cc22cd59673c2,
                            0x000000000000e48f, 0x0},
    std::array<uint64_t, 4>{0x19963dd706d4c76b, 0x98d4bd0e08ed389f,
                            0x00000000000061e9, 0x0},
    std::array<uint64_t, 4>{0x19963dd706d4c76a, 0x98d4bd0e08ed389f,
                            0x00000000000061e9, 0x0},
    std::array<uint64_t, 4>{0x4b77465b552e32c5, 0x7cd47c1da405eed4,
                            0x0000000000006c3b, 0x0},
    std::array<uint64_t, 4>{0x4b77465b552e32c4, 0x7cd47c1da405eed4,
                            0x0000000000006c3b, 0x0},
    std::array<uint64_t, 4>{0x9f7550ad32a90d3b, 0x0eaaa45f1d84685a,
                            0x0000000000005d05, 0x0},
    std::array<uint64_t, 4>{0x9f7550ad32a90d3a, 0x0eaaa45f1d84685a,
                            0x0000000000005d05, 0x0},
    std::array<uint64_t, 4>{0xcd942b216153f895, 0xeaaa654cb16cbe11,
                            0x00000000000050d7, 0x0},
    std::array<uint64_t, 4>{0xcd942b216153f894, 0xeaaa654cb16cbe11,
                            0x00000000000050d7, 0x0},
    std::array<uint64_t, 4>{0xb9c0bd05e945ef24, 0x7e26c7441be16d0d,
                            0x0000000000003dd3, 0x0},
    std::array<uint64_t, 4>{0xb9c0bd05e945ef25, 0x7e26c7441be16d0d,
                            0x0000000000003dd3, 0x0},
    std::array<uint64_t, 4>{0xeb21c689babf1a8a, 0x9a260657b709bb46,
                            0x0000000000003001, 0x0},
    std::array<uint64_t, 4>{0xeb21c689babf1a8b, 0x9a260657b709bb46,
                            0x0000000000003001, 0x0},
    std::array<uint64_t, 4>{0x3f23d07fdd382574, 0xe858de150e883dc8,
                            0x000000000000013f, 0x0},
    std::array<uint64_t, 4>{0x3f23d07fdd382575, 0xe858de150e883dc8,
                            0x000000000000013f, 0x0},
    std::array<uint64_t, 4>{0x6dc2abf38ec2d0da, 0x0c581f06a260eb83,
                            0x0000000000000ced, 0x0},
    std::array<uint64_t, 4>{0x6dc2abf38ec2d0db, 0x0c581f06a260eb83,
                            0x0000000000000ced, 0x0},
    std::array<uint64_t, 4>{0xd913b448774c1241, 0xb0f53704a3f89d07,
                            0x000000000000dcc6, 0x0},
    std::array<uint64_t, 4>{0xd913b448774c1240, 0xb0f53704a3f89d07,
                            0x000000000000dcc6, 0x0},
    std::array<uint64_t, 4>{0x8bf2cfc424b6e7ef, 0x54f5f6170f104b4c,
                            0x000000000000d114, 0x0},
    std::array<uint64_t, 4>{0x8bf2cfc424b6e7ee, 0x54f5f6170f104b4c,
                            0x000000000000d114, 0x0},
    std::array<uint64_t, 4>{0x5ff0d9324331d811, 0x268b2e55b691cdc2,
                            0x000000000000e02a, 0x0},
    std::array<uint64_t, 4>{0x5ff0d9324331d810, 0x268b2e55b691cdc2,
                            0x000000000000e02a, 0x0},
    std::array<uint64_t, 4>{0x0d11a2be10cb2dbf, 0xc28bef461a791b89,
                            0x000000000000edf8, 0x0},
    std::array<uint64_t, 4>{0x0d11a2be10cb2dbe, 0xc28bef461a791b89,
                            0x000000000000edf8, 0x0},
    std::array<uint64_t, 4>{0x7945349a98dd3a0e, 0x56074d4eb0f4c895,
                            0x00000000000080fc, 0x0},
    std::array<uint64_t, 4>{0x7945349a98dd3a0f, 0x56074d4eb0f4c895,
                            0x00000000000080fc, 0x0},
    std::array<uint64_t, 4>{0x2ba44f16cb27cfa0, 0xb2078c5d1c1c1ede,
                            0x0000000000008d2e, 0x0},
    std::array<uint64_t, 4>{0x2ba44f16cb27cfa1, 0xb2078c5d1c1c1ede,
                            0x0000000000008d2e, 0x0},
    std::array<uint64_t, 4>{0xffa659e0aca0f05e, 0xc079541fa59d9850,
                            0x000000000000bc10, 0x0},
    std::array<uint64_t, 4>{0xffa659e0aca0f05f, 0xc079541fa59d9850,
                            0x000000000000bc10, 0x0},
    std::array<uint64_t, 4>{0xad47226cff5a05f0, 0x2479950c09754e1b,
                            0x000000000000b1c2, 0x0},
    std::array<uint64_t, 4>{0xad47226cff5a05f1, 0x2479950c09754e1b,
                            0x000000000000b1c2, 0x0},
    std::array<uint64_t, 4>{0x3f1e6c45cbffdd99, 0x94ebb4104751e25f,
                            0x0000000000009cb2, 0x0},
    std::array<uint64_t, 4>{0x3f1e6c45cbffdd98, 0x94ebb4104751e25f,
                            0x0000000000009cb2, 0x0},
    std::array<uint64_t, 4>{0x6dff17c998052837, 0x70eb7503ebb93414,
                            0x0000000000009160, 0x0},
    std::array<uint64_t, 4>{0x6dff17c998052836, 0x70eb7503ebb93414,
                            0x0000000000009160, 0x0},
    std::array<uint64_t, 4>{0xb9fd013fff8217c9, 0x0295ad415238b29a,
                            0x000000000000a05e, 0x0},
    std::array<uint64_t, 4>{0xb9fd013fff8217c8, 0x0295ad415238b29a,
                            0x000000000000a05e, 0x0},
    std::array<uint64_t, 4>{0xeb1c7ab3ac78e267, 0xe6956c52fed064d1,
                            0x000000000000ad8c, 0x0},
    std::array<uint64_t, 4>{0xeb1c7ab3ac78e266, 0xe6956c52fed064d1,
                            0x000000000000ad8c, 0x0},
    std::array<uint64_t, 4>{0x9f48ec97246ef5d6, 0x7219ce5a545db7cd,
                            0x000000000000c088, 0x0},
    std::array<uint64_t, 4>{0x9f48ec97246ef5d7, 0x7219ce5a545db7cd,
                            0x000000000000c088, 0x0},
    std::array<uint64_t, 4>{0xcda9971b77940078, 0x96190f49f8b56186,
                            0x000000000000cd5a, 0x0},
    std::array<uint64_t, 4>{0xcda9971b77940079, 0x96190f49f8b56186,
                            0x000000000000cd5a, 0x0},
    std::array<uint64_t, 4>{0x19ab81ed10133f86, 0xe467d70b4134e708,
                            0x000000000000fc64, 0x0},
    std::array<uint64_t, 4>{0x19ab81ed10133f87, 0xe467d70b4134e708,
                            0x000000000000fc64, 0x0},
    std::array<uint64_t, 4>{0x4b4afa6143e9ca28, 0x00671618eddc3143,
                            0x000000000000f1b6, 0x0},
    std::array<uint64_t, 4>{0x4b4afa6143e9ca29, 0x00671618eddc3143,
                            0x000000000000f1b6, 0x0},
    std::array<uint64_t, 4>{0xff9be5daba6708b3, 0xbcca3e1aec4447c7,
                            0x000000000000219d, 0x0},
    std::array<uint64_t, 4>{0xff9be5daba6708b2, 0xbcca3e1aec4447c7,
                            0x000000000000219d, 0x0},
    std::array<uint64_t, 4>{0xad7a9e56e99dfd1d, 0x58caff0940ac918c,
                            0x0000000000002c4f, 0x0},
    std::array<uint64_t, 4>{0xad7a9e56e99dfd1c, 0x58caff0940ac918c,
                            0x0000000000002c4f, 0x0},
    std::array<uint64_t, 4>{0x797888a08e1ac2e3, 0x2ab4274bf92d1702,
                            0x0000000000001d71, 0x0},
    std::array<uint64_t, 4>{0x797888a08e1ac2e2, 0x2ab4274bf92d1702,
                            0x0000000000001d71, 0x0},
    std::array<uint64_t, 4>{0x2b99f32cdde0374d, 0xceb4e65855c5c149,
                            0x00000000000010a3, 0x0},
    std::array<uint64_t, 4>{0x2b99f32cdde0374c, 0xceb4e65855c5c149,
                            0x00000000000010a3, 0x0},
    std::array<uint64_t, 4>{0x5fcd650855f620fc, 0x5a384450ff481255,
                            0x0000000000007da7, 0x0},
    std::array<uint64_t, 4>{0x5fcd650855f620fd, 0x5a384450ff481255,
                            0x0000000000007da7, 0x0},
    std::array<uint64_t, 4>{0x0d2c1e84060cd552, 0xbe38854353a0c41e,
                            0x0000000000007075, 0x0},
    std::array<uint64_t, 4>{0x0d2c1e84060cd553, 0xbe38854353a0c41e,
                            0x0000000000007075, 0x0},
    std::array<uint64_t, 4>{0xd92e0872618beaac, 0xcc465d01ea214290,
                            0x000000000000414b, 0x0},
    std::array<uint64_t, 4>{0xd92e0872618beaad, 0xcc465d01ea214290,
                            0x000000000000414b, 0x0},
    std::array<uint64_t, 4>{0x8bcf73fe32711f02, 0x28469c1246c994db,
                            0x0000000000004c99, 0x0},
    std::array<uint64_t, 4>{0x8bcf73fe32711f03, 0x28469c1246c994db,
                            0x0000000000004c99, 0x0},
    std::array<uint64_t, 4>{0x3ad7eb172e8d1fc6, 0x3aeee3309bb2df86,
                            0x000000000000c9ff, 0x0},
    std::array<uint64_t, 4>{0x3ad7eb172e8d1fc7, 0x3aeee3309bb2df86,
                            0x000000000000c9ff, 0x0},
    std::array<uint64_t, 4>{0x6836909b7d77ea68, 0xdeee2223375a09cd,
                            0x000000000000c42d, 0x0},
    std::array<uint64_t, 4>{0x6836909b7d77ea69, 0xdeee2223375a09cd,
                            0x000000000000c42d, 0x0},
    std::array<uint64_t, 4>{0xbc34866d1af0d596, 0xac90fa618edb8f43,
                            0x000000000000f513, 0x0},
    std::array<uint64_t, 4>{0xbc34866d1af0d597, 0xac90fa618edb8f43,
                            0x000000000000f513, 0x0},
    std::array<uint64_t, 4>{0xeed5fde1490a2038, 0x48903b7222335908,
                            0x000000000000f8c1, 0x0},
    std::array<uint64_t, 4>{0xeed5fde1490a2039, 0x48903b7222335908,
                            0x000000000000f8c1, 0x0},
    std::array<uint64_t, 4>{0x9a816bc5c11c3789, 0xdc1c997a88be8a14,
                            0x00000000000095c5, 0x0},
    std::array<uint64_t, 4>{0x9a816bc5c11c3788, 0xdc1c997a88be8a14,
                            0x00000000000095c5, 0x0},
    std::array<uint64_t, 4>{0xc860104992e6c227, 0x381c586924565c5f,
                            0x0000000000009817, 0x0},
    std::array<uint64_t, 4>{0xc860104992e6c226, 0x381c586924565c5f,
                            0x0000000000009817, 0x0},
    std::array<uint64_t, 4>{0x1c6206bff561fdd9, 0x4a62802b9dd7dad1,
                            0x000000000000a929, 0x0},
    std::array<uint64_t, 4>{0x1c6206bff561fdd8, 0x4a62802b9dd7dad1,
                            0x000000000000a929, 0x0},
    std::array<uint64_t, 4>{0x4e837d33a69b0877, 0xae624138313f0c9a,
                            0x000000000000a4fb, 0x0},
    std::array<uint64_t, 4>{0x4e837d33a69b0876, 0xae624138313f0c9a,
                            0x000000000000a4fb, 0x0},
    std::array<uint64_t, 4>{0xfa5262885f15caec, 0x12cf693a30a77a1e,
                            0x00000000000074d0, 0x0},
    std::array<uint64_t, 4>{0xfa5262885f15caed, 0x12cf693a30a77a1e,
                            0x00000000000074d0, 0x0},
    std::array<uint64_t, 4>{0xa8b319040cef3f42, 0xf6cfa8299c4fac55,
                            0x0000000000007902, 0x0},
    std::array<uint64_t, 4>{0xa8b319040cef3f43, 0xf6cfa8299c4fac55,
                            0x0000000000007902, 0x0},
    std::array<uint64_t, 4>{0x7cb10ff26b6800bc, 0x84b1706b25ce2adb,
                            0x000000000000483c, 0x0},
    std::array<uint64_t, 4>{0x7cb10ff26b6800bd, 0x84b1706b25ce2adb,
                            0x000000000000483c, 0x0},
    std::array<uint64_t, 4>{0x2e50747e3892f512, 0x60b1b1788926fc90,
                            0x00000000000045ee, 0x0},
    std::array<uint64_t, 4>{0x2e50747e3892f513, 0x60b1b1788926fc90,
                            0x00000000000045ee, 0x0},
    std::array<uint64_t, 4>{0x5a04e25ab084e2a3, 0xf43d137023ab2f8c,
                            0x00000000000028ea, 0x0},
    std::array<uint64_t, 4>{0x5a04e25ab084e2a2, 0xf43d137023ab2f8c,
                            0x00000000000028ea, 0x0},
    std::array<uint64_t, 4>{0x08e599d6e37e170d, 0x103dd2638f43f9c7,
                            0x0000000000002538, 0x0},
    std::array<uint64_t, 4>{0x08e599d6e37e170c, 0x103dd2638f43f9c7,
                            0x0000000000002538, 0x0},
    std::array<uint64_t, 4>{0xdce78f2084f928f3, 0x62430a2136c27f49,
                            0x0000000000001406, 0x0},
    std::array<uint64_t, 4>{0xdce78f2084f928f2, 0x62430a2136c27f49,
                            0x0000000000001406, 0x0},
    std::array<uint64_t, 4>{0x8e06f4acd703dd5d, 0x8643cb329a2aa902,
                            0x00000000000019d4, 0x0},
    std::array<uint64_t, 4>{0x8e06f4acd703dd5c, 0x8643cb329a2aa902,
                            0x00000000000019d4, 0x0},
};

} // namespace

// somewhat optimized inner product, only do one lazy reduction
field::GF2_144
lifted_dot_product_uint8(const gsl::span<const field::GF2_144> &lhs,
                         const gsl::span<const uint8_t> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  __m128i accum[3] = {_mm_setzero_si128(), _mm_setzero_si128(),
                      _mm_setzero_si128()};
  __m128i tmp[4];
  tmp[0] = _mm_setzero_si128();
  tmp[1] = _mm_setzero_si128();
  for (size_t i = 0; i < lhs.size(); i++) {
    const __m128i *left = lhs[i].as_const_m128i();
    const __m128i *right = lifting_lut[rhs[i]].as_const_m128i();
    accum[0] =
        _mm_xor_si128(accum[0], _mm_clmulepi64_si128(left[0], right[0], 0x00));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(left[0], right[0], 0x11));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(left[1], right[0], 0x00));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(left[0], right[1], 0x00));
    accum[2] =
        _mm_xor_si128(accum[2], _mm_clmulepi64_si128(left[1], right[1], 0x00));
    tmp[0] =
        _mm_xor_si128(tmp[0], _mm_clmulepi64_si128(left[0], right[0], 0x10));
    tmp[0] =
        _mm_xor_si128(tmp[0], _mm_clmulepi64_si128(left[0], right[0], 0x01));

    tmp[1] =
        _mm_xor_si128(tmp[1], _mm_clmulepi64_si128(left[1], right[0], 0x10));
    tmp[1] =
        _mm_xor_si128(tmp[1], _mm_clmulepi64_si128(left[0], right[1], 0x01));
  }
  tmp[2] = _mm_slli_si128(tmp[0], 8);
  tmp[3] = _mm_srli_si128(tmp[0], 8);

  accum[0] = _mm_xor_si128(accum[0], tmp[2]);
  accum[1] = _mm_xor_si128(accum[1], tmp[3]);

  tmp[2] = _mm_slli_si128(tmp[1], 8);
  tmp[3] = _mm_srli_si128(tmp[1], 8);

  accum[1] = _mm_xor_si128(accum[1], tmp[2]);
  accum[2] = _mm_xor_si128(accum[2], tmp[3]);
  // combined reduction
  field::GF2_144 result;
  reduce_clmul(result.as_m128i(), accum);
  return result;
}
// somewhat optimized inner product, only do one lazy reduction
field::GF2_144 lifted_dot_product(const gsl::span<const field::GF2_144> &lhs,
                                  const gsl::span<const field::GF2_8> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  __m128i accum[3] = {_mm_setzero_si128(), _mm_setzero_si128(),
                      _mm_setzero_si128()};
  __m128i tmp[4];
  tmp[0] = _mm_setzero_si128();
  tmp[1] = _mm_setzero_si128();
  for (size_t i = 0; i < lhs.size(); i++) {
    const __m128i *left = lhs[i].as_const_m128i();
    const __m128i *right = lifting_lut[rhs[i].data].as_const_m128i();
    accum[0] =
        _mm_xor_si128(accum[0], _mm_clmulepi64_si128(left[0], right[0], 0x00));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(left[0], right[0], 0x11));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(left[1], right[0], 0x00));
    accum[1] =
        _mm_xor_si128(accum[1], _mm_clmulepi64_si128(left[0], right[1], 0x00));
    accum[2] =
        _mm_xor_si128(accum[2], _mm_clmulepi64_si128(left[1], right[1], 0x00));
    tmp[0] =
        _mm_xor_si128(tmp[0], _mm_clmulepi64_si128(left[0], right[0], 0x10));
    tmp[0] =
        _mm_xor_si128(tmp[0], _mm_clmulepi64_si128(left[0], right[0], 0x01));

    tmp[1] =
        _mm_xor_si128(tmp[1], _mm_clmulepi64_si128(left[1], right[0], 0x10));
    tmp[1] =
        _mm_xor_si128(tmp[1], _mm_clmulepi64_si128(left[0], right[1], 0x01));
  }
  tmp[2] = _mm_slli_si128(tmp[0], 8);
  tmp[3] = _mm_srli_si128(tmp[0], 8);

  accum[0] = _mm_xor_si128(accum[0], tmp[2]);
  accum[1] = _mm_xor_si128(accum[1], tmp[3]);

  tmp[2] = _mm_slli_si128(tmp[1], 8);
  tmp[3] = _mm_srli_si128(tmp[1], 8);

  accum[1] = _mm_xor_si128(accum[1], tmp[2]);
  accum[2] = _mm_xor_si128(accum[2], tmp[3]);
  // combined reduction
  field::GF2_144 result;
  reduce_clmul(result.as_m128i(), accum);
  return result;
}

// horner eval, slow
// field::GF2_144 field::eval_lifted(const gsl::span<const field::GF2_8> &poly,
//                                   const field::GF2_144 &point) {
//   field::GF2_144 acc;
//   long i;

//   for (i = poly.size() - 1; i >= 0; i--) {
//     acc *= point;
//     acc += lifting_lut[poly[i].data];
//   }

//   return acc;
// }

// more optimized horner eval by splitting the poly in two terms
// x * (c_1 + x^2*c_3 + x^4*c_5 +...)
//   + (c_0 + x^2*c_2 + x^4*c_4 + ...)
// and evaluating both with horner. While this is actually a bit more work
// than the trivial horner eval, it is much more friendly to the CPU, since
// the two polys can be evaluated in a more interleaved manner
// field::GF2_144 field::eval_lifted(const gsl::span<const field::GF2_8> &poly,
//                                   const field::GF2_144 &point) {
//   field::GF2_144 acc_even, acc_odd;
//   long i = poly.size() - 1;

//   field::GF2_144 p2;
//   gf2_144sqr(p2.as_m128i(), point.as_const_m128i());

//   // i is the degree, if it is even, then we have one more even coeff that we
//   // already need to handle here
//   if (i % 2 == 0) {
//     acc_even = lifting_lut[poly[i].data];
//     i--;
//   }

//   // now the degree is odd for sure, we have an even number of coefficients
//   and
//   // can handle two simultaneous to improve data dependency problems
//   for (; i >= 0; i -= 2) {
//     acc_even *= p2;
//     acc_odd *= p2;
//     acc_even += lifting_lut[poly[i - 1].data];
//     acc_odd += lifting_lut[poly[i].data];
//   }

//   return acc_even + acc_odd * point;
// }

// more optimized horner eval by splitting the poly in four terms
//   x^3 * (c_3 + x^4*c_7 + x^8*c_11 +...)
// + x^2 * (c_2 + x^4*c_6 + x^8*c_10 + ...)
// + x   * (c_1 + x^4*c_5 + x^8*c_9 +...)
// +       (c_0 + x^4*c_4 + x^8*c_8 + ...)
// and evaluating all with horner. While this is actually a bit more work
// than the trivial horner eval, it is much more friendly to the CPU, since
// the four polys can be evaluated in a more interleaved manner
field::GF2_144 field::eval_lifted(const gsl::span<const field::GF2_8> &poly,
                                  const field::GF2_144 &point) {
  field::GF2_144 acc_0, acc_1, acc_2, acc_3;
  long i = poly.size() - 1;

  field::GF2_144 p2, p4;
  gf2_144sqr(p2.as_m128i(), point.as_const_m128i());
  gf2_144sqr(p4.as_m128i(), p2.as_const_m128i());

  // i is the degree, if it is not a multiple of 4 - 1, then we have more
  // coeff that we already need to handle here
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

  // now the degree is a multiple of 4 - 1 for sure, we have an multiple of 4
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

// eight terms
// field::GF2_144 field::eval_lifted(const gsl::span<const field::GF2_8> &poly,
//                                   const field::GF2_144 &point) {
//   std::array<field::GF2_144, 8> acc{};
//   long i = poly.size() - 1;

//   field::GF2_144 p2, p4, p8;
//   gf2_144sqr(p2.as_m128i(), point.as_const_m128i());
//   gf2_144sqr(p4.as_m128i(), p2.as_const_m128i());
//   gf2_144sqr(p8.as_m128i(), p4.as_const_m128i());

//   // i is the degree, if it is not a multiple of 8 - 1, then we have more
//   coeff
//   // that we already need to handle here
//   if (i % 8 == 6) {
//     acc[6] = lifting_lut[poly[i].data];
//     i--;
//   }
//   if (i % 8 == 5) {
//     acc[5] = lifting_lut[poly[i].data];
//     i--;
//   }
//   if (i % 8 == 4) {
//     acc[4] = lifting_lut[poly[i].data];
//     i--;
//   }
//   if (i % 8 == 3) {
//     acc[3] = lifting_lut[poly[i].data];
//     i--;
//   }
//   if (i % 8 == 2) {
//     acc[2] = lifting_lut[poly[i].data];
//     i--;
//   }
//   if (i % 8 == 1) {
//     acc[1] = lifting_lut[poly[i].data];
//     i--;
//   }
//   if (i % 8 == 0) {
//     acc[0] = lifting_lut[poly[i].data];
//     i--;
//   }

//   // now the degree is a multiple of 8 - 1 for sure, we have an multiple of 8
//   // number of coefficients and can handle 8 simultaneous to improve data
//   // dependency problems
//   for (; i >= 0; i -= 8) {
//     acc[0] *= p8;
//     acc[1] *= p8;
//     acc[2] *= p8;
//     acc[3] *= p8;
//     acc[4] *= p8;
//     acc[5] *= p8;
//     acc[6] *= p8;
//     acc[7] *= p8;
//     acc[0] += lifting_lut[poly[i - 7].data];
//     acc[1] += lifting_lut[poly[i - 6].data];
//     acc[2] += lifting_lut[poly[i - 5].data];
//     acc[3] += lifting_lut[poly[i - 4].data];
//     acc[4] += lifting_lut[poly[i - 3].data];
//     acc[5] += lifting_lut[poly[i - 2].data];
//     acc[6] += lifting_lut[poly[i - 1].data];
//     acc[7] += lifting_lut[poly[i].data];
//   }

//   return acc[0] + point * acc[1] + (acc[2] + point * acc[3]) * p2 +
//          (acc[4] + point * acc[5] + (acc[6] + point * acc[7]) * p2) * p4;
// }

// TEMPLATE INSTANTIATIONS for GF2_144

// yes we include the cpp file with the template stuff
#include "field_templates.cpp"

INSTANTIATE_TEMPLATES_FOR(field::GF2_144)
// END TEMPLATE INSTANTIATIONS for GF2_144