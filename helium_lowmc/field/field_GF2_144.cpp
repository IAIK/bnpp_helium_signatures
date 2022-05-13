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

// lifting lut
namespace {
static constexpr std::array<field::GF2_144, 512> lifting_lut = {
    std::array<uint64_t, 4>{0x0000000000000000, 0x0000000000000000,
                            0x0000000000000000, 0x0},
    std::array<uint64_t, 4>{0x0000000000000001, 0x0000000000000000,
                            0x0000000000000000, 0x0},
    std::array<uint64_t, 4>{0xad108d4ca1aa14d9, 0xc629a25bc9b02366,
                            0x0000000000000149, 0x0},
    std::array<uint64_t, 4>{0xad108d4ca1aa14d8, 0xc629a25bc9b02366,
                            0x0000000000000149, 0x0},
    std::array<uint64_t, 4>{0xa6ac71b680120780, 0x31a81a0631a3060d,
                            0x0000000000001488, 0x0},
    std::array<uint64_t, 4>{0xa6ac71b680120781, 0x31a81a0631a3060d,
                            0x0000000000001488, 0x0},
    std::array<uint64_t, 4>{0x0bbcfcfa21b81359, 0xf781b85df813256b,
                            0x00000000000015c1, 0x0},
    std::array<uint64_t, 4>{0x0bbcfcfa21b81358, 0xf781b85df813256b,
                            0x00000000000015c1, 0x0},
    std::array<uint64_t, 4>{0x61cc52e1e9d29592, 0x87eaeaf1615004ef,
                            0x00000000000008f6, 0x0},
    std::array<uint64_t, 4>{0x61cc52e1e9d29593, 0x87eaeaf1615004ef,
                            0x00000000000008f6, 0x0},
    std::array<uint64_t, 4>{0xccdcdfad4878814b, 0x41c348aaa8e02789,
                            0x00000000000009bf, 0x0},
    std::array<uint64_t, 4>{0xccdcdfad4878814a, 0x41c348aaa8e02789,
                            0x00000000000009bf, 0x0},
    std::array<uint64_t, 4>{0xc760235769c09212, 0xb642f0f750f302e2,
                            0x0000000000001c7e, 0x0},
    std::array<uint64_t, 4>{0xc760235769c09213, 0xb642f0f750f302e2,
                            0x0000000000001c7e, 0x0},
    std::array<uint64_t, 4>{0x6a70ae1bc86a86cb, 0x706b52ac99432184,
                            0x0000000000001d37, 0x0},
    std::array<uint64_t, 4>{0x6a70ae1bc86a86ca, 0x706b52ac99432184,
                            0x0000000000001d37, 0x0},
    std::array<uint64_t, 4>{0x4b06c0b616d44b04, 0x215685e22441f700,
                            0x0000000000009c24, 0x0},
    std::array<uint64_t, 4>{0x4b06c0b616d44b05, 0x215685e22441f700,
                            0x0000000000009c24, 0x0},
    std::array<uint64_t, 4>{0xe6164dfab77e5fdd, 0xe77f27b9edf1d466,
                            0x0000000000009d6d, 0x0},
    std::array<uint64_t, 4>{0xe6164dfab77e5fdc, 0xe77f27b9edf1d466,
                            0x0000000000009d6d, 0x0},
    std::array<uint64_t, 4>{0xedaab10096c64c84, 0x10fe9fe415e2f10d,
                            0x00000000000088ac, 0x0},
    std::array<uint64_t, 4>{0xedaab10096c64c85, 0x10fe9fe415e2f10d,
                            0x00000000000088ac, 0x0},
    std::array<uint64_t, 4>{0x40ba3c4c376c585d, 0xd6d73dbfdc52d26b,
                            0x00000000000089e5, 0x0},
    std::array<uint64_t, 4>{0x40ba3c4c376c585c, 0xd6d73dbfdc52d26b,
                            0x00000000000089e5, 0x0},
    std::array<uint64_t, 4>{0x2aca9257ff06de96, 0xa6bc6f134511f3ef,
                            0x00000000000094d2, 0x0},
    std::array<uint64_t, 4>{0x2aca9257ff06de97, 0xa6bc6f134511f3ef,
                            0x00000000000094d2, 0x0},
    std::array<uint64_t, 4>{0x87da1f1b5eacca4f, 0x6095cd488ca1d089,
                            0x000000000000959b, 0x0},
    std::array<uint64_t, 4>{0x87da1f1b5eacca4e, 0x6095cd488ca1d089,
                            0x000000000000959b, 0x0},
    std::array<uint64_t, 4>{0x8c66e3e17f14d916, 0x9714751574b2f5e2,
                            0x000000000000805a, 0x0},
    std::array<uint64_t, 4>{0x8c66e3e17f14d917, 0x9714751574b2f5e2,
                            0x000000000000805a, 0x0},
    std::array<uint64_t, 4>{0x21766eaddebecdcf, 0x513dd74ebd02d684,
                            0x0000000000008113, 0x0},
    std::array<uint64_t, 4>{0x21766eaddebecdce, 0x513dd74ebd02d684,
                            0x0000000000008113, 0x0},
    std::array<uint64_t, 4>{0xb01a51e3a1fd897d, 0x1c5c28e7dde9c23e,
                            0x000000000000c11b, 0x0},
    std::array<uint64_t, 4>{0xb01a51e3a1fd897c, 0x1c5c28e7dde9c23e,
                            0x000000000000c11b, 0x0},
    std::array<uint64_t, 4>{0x1d0adcaf00579da4, 0xda758abc1459e158,
                            0x000000000000c052, 0x0},
    std::array<uint64_t, 4>{0x1d0adcaf00579da5, 0xda758abc1459e158,
                            0x000000000000c052, 0x0},
    std::array<uint64_t, 4>{0x16b6205521ef8efd, 0x2df432e1ec4ac433,
                            0x000000000000d593, 0x0},
    std::array<uint64_t, 4>{0x16b6205521ef8efc, 0x2df432e1ec4ac433,
                            0x000000000000d593, 0x0},
    std::array<uint64_t, 4>{0xbba6ad1980459a24, 0xebdd90ba25fae755,
                            0x000000000000d4da, 0x0},
    std::array<uint64_t, 4>{0xbba6ad1980459a25, 0xebdd90ba25fae755,
                            0x000000000000d4da, 0x0},
    std::array<uint64_t, 4>{0xd1d60302482f1cef, 0x9bb6c216bcb9c6d1,
                            0x000000000000c9ed, 0x0},
    std::array<uint64_t, 4>{0xd1d60302482f1cee, 0x9bb6c216bcb9c6d1,
                            0x000000000000c9ed, 0x0},
    std::array<uint64_t, 4>{0x7cc68e4ee9850836, 0x5d9f604d7509e5b7,
                            0x000000000000c8a4, 0x0},
    std::array<uint64_t, 4>{0x7cc68e4ee9850837, 0x5d9f604d7509e5b7,
                            0x000000000000c8a4, 0x0},
    std::array<uint64_t, 4>{0x777a72b4c83d1b6f, 0xaa1ed8108d1ac0dc,
                            0x000000000000dd65, 0x0},
    std::array<uint64_t, 4>{0x777a72b4c83d1b6e, 0xaa1ed8108d1ac0dc,
                            0x000000000000dd65, 0x0},
    std::array<uint64_t, 4>{0xda6afff869970fb6, 0x6c377a4b44aae3ba,
                            0x000000000000dc2c, 0x0},
    std::array<uint64_t, 4>{0xda6afff869970fb7, 0x6c377a4b44aae3ba,
                            0x000000000000dc2c, 0x0},
    std::array<uint64_t, 4>{0xfb1c9155b729c279, 0x3d0aad05f9a8353e,
                            0x0000000000005d3f, 0x0},
    std::array<uint64_t, 4>{0xfb1c9155b729c278, 0x3d0aad05f9a8353e,
                            0x0000000000005d3f, 0x0},
    std::array<uint64_t, 4>{0x560c1c191683d6a0, 0xfb230f5e30181658,
                            0x0000000000005c76, 0x0},
    std::array<uint64_t, 4>{0x560c1c191683d6a1, 0xfb230f5e30181658,
                            0x0000000000005c76, 0x0},
    std::array<uint64_t, 4>{0x5db0e0e3373bc5f9, 0x0ca2b703c80b3333,
                            0x00000000000049b7, 0x0},
    std::array<uint64_t, 4>{0x5db0e0e3373bc5f8, 0x0ca2b703c80b3333,
                            0x00000000000049b7, 0x0},
    std::array<uint64_t, 4>{0xf0a06daf9691d120, 0xca8b155801bb1055,
                            0x00000000000048fe, 0x0},
    std::array<uint64_t, 4>{0xf0a06daf9691d121, 0xca8b155801bb1055,
                            0x00000000000048fe, 0x0},
    std::array<uint64_t, 4>{0x9ad0c3b45efb57eb, 0xbae047f498f831d1,
                            0x00000000000055c9, 0x0},
    std::array<uint64_t, 4>{0x9ad0c3b45efb57ea, 0xbae047f498f831d1,
                            0x00000000000055c9, 0x0},
    std::array<uint64_t, 4>{0x37c04ef8ff514332, 0x7cc9e5af514812b7,
                            0x0000000000005480, 0x0},
    std::array<uint64_t, 4>{0x37c04ef8ff514333, 0x7cc9e5af514812b7,
                            0x0000000000005480, 0x0},
    std::array<uint64_t, 4>{0x3c7cb202dee9506b, 0x8b485df2a95b37dc,
                            0x0000000000004141, 0x0},
    std::array<uint64_t, 4>{0x3c7cb202dee9506a, 0x8b485df2a95b37dc,
                            0x0000000000004141, 0x0},
    std::array<uint64_t, 4>{0x916c3f4e7f4344b2, 0x4d61ffa960eb14ba,
                            0x0000000000004008, 0x0},
    std::array<uint64_t, 4>{0x916c3f4e7f4344b3, 0x4d61ffa960eb14ba,
                            0x0000000000004008, 0x0},
    std::array<uint64_t, 4>{0x85df559884114854, 0xce201bef723e373b,
                            0x000000000000713b, 0x0},
    std::array<uint64_t, 4>{0x85df559884114855, 0xce201bef723e373b,
                            0x000000000000713b, 0x0},
    std::array<uint64_t, 4>{0x28cfd8d425bb5c8d, 0x0809b9b4bb8e145d,
                            0x0000000000007072, 0x0},
    std::array<uint64_t, 4>{0x28cfd8d425bb5c8c, 0x0809b9b4bb8e145d,
                            0x0000000000007072, 0x0},
    std::array<uint64_t, 4>{0x2373242e04034fd4, 0xff8801e9439d3136,
                            0x00000000000065b3, 0x0},
    std::array<uint64_t, 4>{0x2373242e04034fd5, 0xff8801e9439d3136,
                            0x00000000000065b3, 0x0},
    std::array<uint64_t, 4>{0x8e63a962a5a95b0d, 0x39a1a3b28a2d1250,
                            0x00000000000064fa, 0x0},
    std::array<uint64_t, 4>{0x8e63a962a5a95b0c, 0x39a1a3b28a2d1250,
                            0x00000000000064fa, 0x0},
    std::array<uint64_t, 4>{0xe41307796dc3ddc6, 0x49caf11e136e33d4,
                            0x00000000000079cd, 0x0},
    std::array<uint64_t, 4>{0xe41307796dc3ddc7, 0x49caf11e136e33d4,
                            0x00000000000079cd, 0x0},
    std::array<uint64_t, 4>{0x49038a35cc69c91f, 0x8fe35345dade10b2,
                            0x0000000000007884, 0x0},
    std::array<uint64_t, 4>{0x49038a35cc69c91e, 0x8fe35345dade10b2,
                            0x0000000000007884, 0x0},
    std::array<uint64_t, 4>{0x42bf76cfedd1da46, 0x7862eb1822cd35d9,
                            0x0000000000006d45, 0x0},
    std::array<uint64_t, 4>{0x42bf76cfedd1da47, 0x7862eb1822cd35d9,
                            0x0000000000006d45, 0x0},
    std::array<uint64_t, 4>{0xefaffb834c7bce9f, 0xbe4b4943eb7d16bf,
                            0x0000000000006c0c, 0x0},
    std::array<uint64_t, 4>{0xefaffb834c7bce9e, 0xbe4b4943eb7d16bf,
                            0x0000000000006c0c, 0x0},
    std::array<uint64_t, 4>{0xced9952e92c50350, 0xef769e0d567fc03b,
                            0x000000000000ed1f, 0x0},
    std::array<uint64_t, 4>{0xced9952e92c50351, 0xef769e0d567fc03b,
                            0x000000000000ed1f, 0x0},
    std::array<uint64_t, 4>{0x63c91862336f1789, 0x295f3c569fcfe35d,
                            0x000000000000ec56, 0x0},
    std::array<uint64_t, 4>{0x63c91862336f1788, 0x295f3c569fcfe35d,
                            0x000000000000ec56, 0x0},
    std::array<uint64_t, 4>{0x6875e49812d704d0, 0xdede840b67dcc636,
                            0x000000000000f997, 0x0},
    std::array<uint64_t, 4>{0x6875e49812d704d1, 0xdede840b67dcc636,
                            0x000000000000f997, 0x0},
    std::array<uint64_t, 4>{0xc56569d4b37d1009, 0x18f72650ae6ce550,
                            0x000000000000f8de, 0x0},
    std::array<uint64_t, 4>{0xc56569d4b37d1008, 0x18f72650ae6ce550,
                            0x000000000000f8de, 0x0},
    std::array<uint64_t, 4>{0xaf15c7cf7b1796c2, 0x689c74fc372fc4d4,
                            0x000000000000e5e9, 0x0},
    std::array<uint64_t, 4>{0xaf15c7cf7b1796c3, 0x689c74fc372fc4d4,
                            0x000000000000e5e9, 0x0},
    std::array<uint64_t, 4>{0x02054a83dabd821b, 0xaeb5d6a7fe9fe7b2,
                            0x000000000000e4a0, 0x0},
    std::array<uint64_t, 4>{0x02054a83dabd821a, 0xaeb5d6a7fe9fe7b2,
                            0x000000000000e4a0, 0x0},
    std::array<uint64_t, 4>{0x09b9b679fb059142, 0x59346efa068cc2d9,
                            0x000000000000f161, 0x0},
    std::array<uint64_t, 4>{0x09b9b679fb059143, 0x59346efa068cc2d9,
                            0x000000000000f161, 0x0},
    std::array<uint64_t, 4>{0xa4a93b355aaf859b, 0x9f1dcca1cf3ce1bf,
                            0x000000000000f028, 0x0},
    std::array<uint64_t, 4>{0xa4a93b355aaf859a, 0x9f1dcca1cf3ce1bf,
                            0x000000000000f028, 0x0},
    std::array<uint64_t, 4>{0x35c5047b25ecc129, 0xd27c3308afd7f505,
                            0x000000000000b020, 0x0},
    std::array<uint64_t, 4>{0x35c5047b25ecc128, 0xd27c3308afd7f505,
                            0x000000000000b020, 0x0},
    std::array<uint64_t, 4>{0x98d589378446d5f0, 0x145591536667d663,
                            0x000000000000b169, 0x0},
    std::array<uint64_t, 4>{0x98d589378446d5f1, 0x145591536667d663,
                            0x000000000000b169, 0x0},
    std::array<uint64_t, 4>{0x936975cda5fec6a9, 0xe3d4290e9e74f308,
                            0x000000000000a4a8, 0x0},
    std::array<uint64_t, 4>{0x936975cda5fec6a8, 0xe3d4290e9e74f308,
                            0x000000000000a4a8, 0x0},
    std::array<uint64_t, 4>{0x3e79f8810454d270, 0x25fd8b5557c4d06e,
                            0x000000000000a5e1, 0x0},
    std::array<uint64_t, 4>{0x3e79f8810454d271, 0x25fd8b5557c4d06e,
                            0x000000000000a5e1, 0x0},
    std::array<uint64_t, 4>{0x5409569acc3e54bb, 0x5596d9f9ce87f1ea,
                            0x000000000000b8d6, 0x0},
    std::array<uint64_t, 4>{0x5409569acc3e54ba, 0x5596d9f9ce87f1ea,
                            0x000000000000b8d6, 0x0},
    std::array<uint64_t, 4>{0xf919dbd66d944062, 0x93bf7ba20737d28c,
                            0x000000000000b99f, 0x0},
    std::array<uint64_t, 4>{0xf919dbd66d944063, 0x93bf7ba20737d28c,
                            0x000000000000b99f, 0x0},
    std::array<uint64_t, 4>{0xf2a5272c4c2c533b, 0x643ec3ffff24f7e7,
                            0x000000000000ac5e, 0x0},
    std::array<uint64_t, 4>{0xf2a5272c4c2c533a, 0x643ec3ffff24f7e7,
                            0x000000000000ac5e, 0x0},
    std::array<uint64_t, 4>{0x5fb5aa60ed8647e2, 0xa21761a43694d481,
                            0x000000000000ad17, 0x0},
    std::array<uint64_t, 4>{0x5fb5aa60ed8647e3, 0xa21761a43694d481,
                            0x000000000000ad17, 0x0},
    std::array<uint64_t, 4>{0x7ec3c4cd33388a2d, 0xf32ab6ea8b960205,
                            0x0000000000002c04, 0x0},
    std::array<uint64_t, 4>{0x7ec3c4cd33388a2c, 0xf32ab6ea8b960205,
                            0x0000000000002c04, 0x0},
    std::array<uint64_t, 4>{0xd3d3498192929ef4, 0x350314b142262163,
                            0x0000000000002d4d, 0x0},
    std::array<uint64_t, 4>{0xd3d3498192929ef5, 0x350314b142262163,
                            0x0000000000002d4d, 0x0},
    std::array<uint64_t, 4>{0xd86fb57bb32a8dad, 0xc282acecba350408,
                            0x000000000000388c, 0x0},
    std::array<uint64_t, 4>{0xd86fb57bb32a8dac, 0xc282acecba350408,
                            0x000000000000388c, 0x0},
    std::array<uint64_t, 4>{0x757f383712809974, 0x04ab0eb77385276e,
                            0x00000000000039c5, 0x0},
    std::array<uint64_t, 4>{0x757f383712809975, 0x04ab0eb77385276e,
                            0x00000000000039c5, 0x0},
    std::array<uint64_t, 4>{0x1f0f962cdaea1fbf, 0x74c05c1beac606ea,
                            0x00000000000024f2, 0x0},
    std::array<uint64_t, 4>{0x1f0f962cdaea1fbe, 0x74c05c1beac606ea,
                            0x00000000000024f2, 0x0},
    std::array<uint64_t, 4>{0xb21f1b607b400b66, 0xb2e9fe402376258c,
                            0x00000000000025bb, 0x0},
    std::array<uint64_t, 4>{0xb21f1b607b400b67, 0xb2e9fe402376258c,
                            0x00000000000025bb, 0x0},
    std::array<uint64_t, 4>{0xb9a3e79a5af8183f, 0x4568461ddb6500e7,
                            0x000000000000307a, 0x0},
    std::array<uint64_t, 4>{0xb9a3e79a5af8183e, 0x4568461ddb6500e7,
                            0x000000000000307a, 0x0},
    std::array<uint64_t, 4>{0x14b36ad6fb520ce6, 0x8341e44612d52381,
                            0x0000000000003133, 0x0},
    std::array<uint64_t, 4>{0x14b36ad6fb520ce7, 0x8341e44612d52381,
                            0x0000000000003133, 0x0},
    std::array<uint64_t, 4>{0x3227a25b442fa786, 0xca5f380900e9d0cb,
                            0x0000000000005892, 0x0},
    std::array<uint64_t, 4>{0x3227a25b442fa787, 0xca5f380900e9d0cb,
                            0x0000000000005892, 0x0},
    std::array<uint64_t, 4>{0x9f372f17e585b35f, 0x0c769a52c959f3ad,
                            0x00000000000059db, 0x0},
    std::array<uint64_t, 4>{0x9f372f17e585b35e, 0x0c769a52c959f3ad,
                            0x00000000000059db, 0x0},
    std::array<uint64_t, 4>{0x948bd3edc43da006, 0xfbf7220f314ad6c6,
                            0x0000000000004c1a, 0x0},
    std::array<uint64_t, 4>{0x948bd3edc43da007, 0xfbf7220f314ad6c6,
                            0x0000000000004c1a, 0x0},
    std::array<uint64_t, 4>{0x399b5ea16597b4df, 0x3dde8054f8faf5a0,
                            0x0000000000004d53, 0x0},
    std::array<uint64_t, 4>{0x399b5ea16597b4de, 0x3dde8054f8faf5a0,
                            0x0000000000004d53, 0x0},
    std::array<uint64_t, 4>{0x53ebf0baadfd3214, 0x4db5d2f861b9d424,
                            0x0000000000005064, 0x0},
    std::array<uint64_t, 4>{0x53ebf0baadfd3215, 0x4db5d2f861b9d424,
                            0x0000000000005064, 0x0},
    std::array<uint64_t, 4>{0xfefb7df60c5726cd, 0x8b9c70a3a809f742,
                            0x000000000000512d, 0x0},
    std::array<uint64_t, 4>{0xfefb7df60c5726cc, 0x8b9c70a3a809f742,
                            0x000000000000512d, 0x0},
    std::array<uint64_t, 4>{0xf547810c2def3594, 0x7c1dc8fe501ad229,
                            0x00000000000044ec, 0x0},
    std::array<uint64_t, 4>{0xf547810c2def3595, 0x7c1dc8fe501ad229,
                            0x00000000000044ec, 0x0},
    std::array<uint64_t, 4>{0x58570c408c45214d, 0xba346aa599aaf14f,
                            0x00000000000045a5, 0x0},
    std::array<uint64_t, 4>{0x58570c408c45214c, 0xba346aa599aaf14f,
                            0x00000000000045a5, 0x0},
    std::array<uint64_t, 4>{0x792162ed52fbec82, 0xeb09bdeb24a827cb,
                            0x000000000000c4b6, 0x0},
    std::array<uint64_t, 4>{0x792162ed52fbec83, 0xeb09bdeb24a827cb,
                            0x000000000000c4b6, 0x0},
    std::array<uint64_t, 4>{0xd431efa1f351f85b, 0x2d201fb0ed1804ad,
                            0x000000000000c5ff, 0x0},
    std::array<uint64_t, 4>{0xd431efa1f351f85a, 0x2d201fb0ed1804ad,
                            0x000000000000c5ff, 0x0},
    std::array<uint64_t, 4>{0xdf8d135bd2e9eb02, 0xdaa1a7ed150b21c6,
                            0x000000000000d03e, 0x0},
    std::array<uint64_t, 4>{0xdf8d135bd2e9eb03, 0xdaa1a7ed150b21c6,
                            0x000000000000d03e, 0x0},
    std::array<uint64_t, 4>{0x729d9e177343ffdb, 0x1c8805b6dcbb02a0,
                            0x000000000000d177, 0x0},
    std::array<uint64_t, 4>{0x729d9e177343ffda, 0x1c8805b6dcbb02a0,
                            0x000000000000d177, 0x0},
    std::array<uint64_t, 4>{0x18ed300cbb297910, 0x6ce3571a45f82324,
                            0x000000000000cc40, 0x0},
    std::array<uint64_t, 4>{0x18ed300cbb297911, 0x6ce3571a45f82324,
                            0x000000000000cc40, 0x0},
    std::array<uint64_t, 4>{0xb5fdbd401a836dc9, 0xaacaf5418c480042,
                            0x000000000000cd09, 0x0},
    std::array<uint64_t, 4>{0xb5fdbd401a836dc8, 0xaacaf5418c480042,
                            0x000000000000cd09, 0x0},
    std::array<uint64_t, 4>{0xbe4141ba3b3b7e90, 0x5d4b4d1c745b2529,
                            0x000000000000d8c8, 0x0},
    std::array<uint64_t, 4>{0xbe4141ba3b3b7e91, 0x5d4b4d1c745b2529,
                            0x000000000000d8c8, 0x0},
    std::array<uint64_t, 4>{0x1351ccf69a916a49, 0x9b62ef47bdeb064f,
                            0x000000000000d981, 0x0},
    std::array<uint64_t, 4>{0x1351ccf69a916a48, 0x9b62ef47bdeb064f,
                            0x000000000000d981, 0x0},
    std::array<uint64_t, 4>{0x823df3b8e5d22efb, 0xd60310eedd0012f5,
                            0x0000000000009989, 0x0},
    std::array<uint64_t, 4>{0x823df3b8e5d22efa, 0xd60310eedd0012f5,
                            0x0000000000009989, 0x0},
    std::array<uint64_t, 4>{0x2f2d7ef444783a22, 0x102ab2b514b03193,
                            0x00000000000098c0, 0x0},
    std::array<uint64_t, 4>{0x2f2d7ef444783a23, 0x102ab2b514b03193,
                            0x00000000000098c0, 0x0},
    std::array<uint64_t, 4>{0x2491820e65c0297b, 0xe7ab0ae8eca314f8,
                            0x0000000000008d01, 0x0},
    std::array<uint64_t, 4>{0x2491820e65c0297a, 0xe7ab0ae8eca314f8,
                            0x0000000000008d01, 0x0},
    std::array<uint64_t, 4>{0x89810f42c46a3da2, 0x2182a8b32513379e,
                            0x0000000000008c48, 0x0},
    std::array<uint64_t, 4>{0x89810f42c46a3da3, 0x2182a8b32513379e,
                            0x0000000000008c48, 0x0},
    std::array<uint64_t, 4>{0xe3f1a1590c00bb69, 0x51e9fa1fbc50161a,
                            0x000000000000917f, 0x0},
    std::array<uint64_t, 4>{0xe3f1a1590c00bb68, 0x51e9fa1fbc50161a,
                            0x000000000000917f, 0x0},
    std::array<uint64_t, 4>{0x4ee12c15adaaafb0, 0x97c0584475e0357c,
                            0x0000000000009036, 0x0},
    std::array<uint64_t, 4>{0x4ee12c15adaaafb1, 0x97c0584475e0357c,
                            0x0000000000009036, 0x0},
    std::array<uint64_t, 4>{0x455dd0ef8c12bce9, 0x6041e0198df31017,
                            0x00000000000085f7, 0x0},
    std::array<uint64_t, 4>{0x455dd0ef8c12bce8, 0x6041e0198df31017,
                            0x00000000000085f7, 0x0},
    std::array<uint64_t, 4>{0xe84d5da32db8a830, 0xa668424244433371,
                            0x00000000000084be, 0x0},
    std::array<uint64_t, 4>{0xe84d5da32db8a831, 0xa668424244433371,
                            0x00000000000084be, 0x0},
    std::array<uint64_t, 4>{0xc93b330ef30665ff, 0xf755950cf941e5f5,
                            0x00000000000005ad, 0x0},
    std::array<uint64_t, 4>{0xc93b330ef30665fe, 0xf755950cf941e5f5,
                            0x00000000000005ad, 0x0},
    std::array<uint64_t, 4>{0x642bbe4252ac7126, 0x317c375730f1c693,
                            0x00000000000004e4, 0x0},
    std::array<uint64_t, 4>{0x642bbe4252ac7127, 0x317c375730f1c693,
                            0x00000000000004e4, 0x0},
    std::array<uint64_t, 4>{0x6f9742b87314627f, 0xc6fd8f0ac8e2e3f8,
                            0x0000000000001125, 0x0},
    std::array<uint64_t, 4>{0x6f9742b87314627e, 0xc6fd8f0ac8e2e3f8,
                            0x0000000000001125, 0x0},
    std::array<uint64_t, 4>{0xc287cff4d2be76a6, 0x00d42d510152c09e,
                            0x000000000000106c, 0x0},
    std::array<uint64_t, 4>{0xc287cff4d2be76a7, 0x00d42d510152c09e,
                            0x000000000000106c, 0x0},
    std::array<uint64_t, 4>{0xa8f761ef1ad4f06d, 0x70bf7ffd9811e11a,
                            0x0000000000000d5b, 0x0},
    std::array<uint64_t, 4>{0xa8f761ef1ad4f06c, 0x70bf7ffd9811e11a,
                            0x0000000000000d5b, 0x0},
    std::array<uint64_t, 4>{0x05e7eca3bb7ee4b4, 0xb696dda651a1c27c,
                            0x0000000000000c12, 0x0},
    std::array<uint64_t, 4>{0x05e7eca3bb7ee4b5, 0xb696dda651a1c27c,
                            0x0000000000000c12, 0x0},
    std::array<uint64_t, 4>{0x0e5b10599ac6f7ed, 0x411765fba9b2e717,
                            0x00000000000019d3, 0x0},
    std::array<uint64_t, 4>{0x0e5b10599ac6f7ec, 0x411765fba9b2e717,
                            0x00000000000019d3, 0x0},
    std::array<uint64_t, 4>{0xa34b9d153b6ce334, 0x873ec7a06002c471,
                            0x000000000000189a, 0x0},
    std::array<uint64_t, 4>{0xa34b9d153b6ce335, 0x873ec7a06002c471,
                            0x000000000000189a, 0x0},
    std::array<uint64_t, 4>{0xb7f8f7c3c03eefd2, 0x047f23e672d7e7f0,
                            0x00000000000029a9, 0x0},
    std::array<uint64_t, 4>{0xb7f8f7c3c03eefd3, 0x047f23e672d7e7f0,
                            0x00000000000029a9, 0x0},
    std::array<uint64_t, 4>{0x1ae87a8f6194fb0b, 0xc25681bdbb67c496,
                            0x00000000000028e0, 0x0},
    std::array<uint64_t, 4>{0x1ae87a8f6194fb0a, 0xc25681bdbb67c496,
                            0x00000000000028e0, 0x0},
    std::array<uint64_t, 4>{0x11548675402ce852, 0x35d739e04374e1fd,
                            0x0000000000003d21, 0x0},
    std::array<uint64_t, 4>{0x11548675402ce853, 0x35d739e04374e1fd,
                            0x0000000000003d21, 0x0},
    std::array<uint64_t, 4>{0xbc440b39e186fc8b, 0xf3fe9bbb8ac4c29b,
                            0x0000000000003c68, 0x0},
    std::array<uint64_t, 4>{0xbc440b39e186fc8a, 0xf3fe9bbb8ac4c29b,
                            0x0000000000003c68, 0x0},
    std::array<uint64_t, 4>{0xd634a52229ec7a40, 0x8395c9171387e31f,
                            0x000000000000215f, 0x0},
    std::array<uint64_t, 4>{0xd634a52229ec7a41, 0x8395c9171387e31f,
                            0x000000000000215f, 0x0},
    std::array<uint64_t, 4>{0x7b24286e88466e99, 0x45bc6b4cda37c079,
                            0x0000000000002016, 0x0},
    std::array<uint64_t, 4>{0x7b24286e88466e98, 0x45bc6b4cda37c079,
                            0x0000000000002016, 0x0},
    std::array<uint64_t, 4>{0x7098d494a9fe7dc0, 0xb23dd3112224e512,
                            0x00000000000035d7, 0x0},
    std::array<uint64_t, 4>{0x7098d494a9fe7dc1, 0xb23dd3112224e512,
                            0x00000000000035d7, 0x0},
    std::array<uint64_t, 4>{0xdd8859d808546919, 0x7414714aeb94c674,
                            0x000000000000349e, 0x0},
    std::array<uint64_t, 4>{0xdd8859d808546918, 0x7414714aeb94c674,
                            0x000000000000349e, 0x0},
    std::array<uint64_t, 4>{0xfcfe3775d6eaa4d6, 0x2529a604569610f0,
                            0x000000000000b58d, 0x0},
    std::array<uint64_t, 4>{0xfcfe3775d6eaa4d7, 0x2529a604569610f0,
                            0x000000000000b58d, 0x0},
    std::array<uint64_t, 4>{0x51eeba397740b00f, 0xe300045f9f263396,
                            0x000000000000b4c4, 0x0},
    std::array<uint64_t, 4>{0x51eeba397740b00e, 0xe300045f9f263396,
                            0x000000000000b4c4, 0x0},
    std::array<uint64_t, 4>{0x5a5246c356f8a356, 0x1481bc02673516fd,
                            0x000000000000a105, 0x0},
    std::array<uint64_t, 4>{0x5a5246c356f8a357, 0x1481bc02673516fd,
                            0x000000000000a105, 0x0},
    std::array<uint64_t, 4>{0xf742cb8ff752b78f, 0xd2a81e59ae85359b,
                            0x000000000000a04c, 0x0},
    std::array<uint64_t, 4>{0xf742cb8ff752b78e, 0xd2a81e59ae85359b,
                            0x000000000000a04c, 0x0},
    std::array<uint64_t, 4>{0x9d3265943f383144, 0xa2c34cf537c6141f,
                            0x000000000000bd7b, 0x0},
    std::array<uint64_t, 4>{0x9d3265943f383145, 0xa2c34cf537c6141f,
                            0x000000000000bd7b, 0x0},
    std::array<uint64_t, 4>{0x3022e8d89e92259d, 0x64eaeeaefe763779,
                            0x000000000000bc32, 0x0},
    std::array<uint64_t, 4>{0x3022e8d89e92259c, 0x64eaeeaefe763779,
                            0x000000000000bc32, 0x0},
    std::array<uint64_t, 4>{0x3b9e1422bf2a36c4, 0x936b56f306651212,
                            0x000000000000a9f3, 0x0},
    std::array<uint64_t, 4>{0x3b9e1422bf2a36c5, 0x936b56f306651212,
                            0x000000000000a9f3, 0x0},
    std::array<uint64_t, 4>{0x968e996e1e80221d, 0x5542f4a8cfd53174,
                            0x000000000000a8ba, 0x0},
    std::array<uint64_t, 4>{0x968e996e1e80221c, 0x5542f4a8cfd53174,
                            0x000000000000a8ba, 0x0},
    std::array<uint64_t, 4>{0x07e2a62061c366af, 0x18230b01af3e25ce,
                            0x000000000000e8b2, 0x0},
    std::array<uint64_t, 4>{0x07e2a62061c366ae, 0x18230b01af3e25ce,
                            0x000000000000e8b2, 0x0},
    std::array<uint64_t, 4>{0xaaf22b6cc0697276, 0xde0aa95a668e06a8,
                            0x000000000000e9fb, 0x0},
    std::array<uint64_t, 4>{0xaaf22b6cc0697277, 0xde0aa95a668e06a8,
                            0x000000000000e9fb, 0x0},
    std::array<uint64_t, 4>{0xa14ed796e1d1612f, 0x298b11079e9d23c3,
                            0x000000000000fc3a, 0x0},
    std::array<uint64_t, 4>{0xa14ed796e1d1612e, 0x298b11079e9d23c3,
                            0x000000000000fc3a, 0x0},
    std::array<uint64_t, 4>{0x0c5e5ada407b75f6, 0xefa2b35c572d00a5,
                            0x000000000000fd73, 0x0},
    std::array<uint64_t, 4>{0x0c5e5ada407b75f7, 0xefa2b35c572d00a5,
                            0x000000000000fd73, 0x0},
    std::array<uint64_t, 4>{0x662ef4c18811f33d, 0x9fc9e1f0ce6e2121,
                            0x000000000000e044, 0x0},
    std::array<uint64_t, 4>{0x662ef4c18811f33c, 0x9fc9e1f0ce6e2121,
                            0x000000000000e044, 0x0},
    std::array<uint64_t, 4>{0xcb3e798d29bbe7e4, 0x59e043ab07de0247,
                            0x000000000000e10d, 0x0},
    std::array<uint64_t, 4>{0xcb3e798d29bbe7e5, 0x59e043ab07de0247,
                            0x000000000000e10d, 0x0},
    std::array<uint64_t, 4>{0xc08285770803f4bd, 0xae61fbf6ffcd272c,
                            0x000000000000f4cc, 0x0},
    std::array<uint64_t, 4>{0xc08285770803f4bc, 0xae61fbf6ffcd272c,
                            0x000000000000f4cc, 0x0},
    std::array<uint64_t, 4>{0x6d92083ba9a9e064, 0x684859ad367d044a,
                            0x000000000000f585, 0x0},
    std::array<uint64_t, 4>{0x6d92083ba9a9e065, 0x684859ad367d044a,
                            0x000000000000f585, 0x0},
    std::array<uint64_t, 4>{0x4ce4669677172dab, 0x39758ee38b7fd2ce,
                            0x0000000000007496, 0x0},
    std::array<uint64_t, 4>{0x4ce4669677172daa, 0x39758ee38b7fd2ce,
                            0x0000000000007496, 0x0},
    std::array<uint64_t, 4>{0xe1f4ebdad6bd3972, 0xff5c2cb842cff1a8,
                            0x00000000000075df, 0x0},
    std::array<uint64_t, 4>{0xe1f4ebdad6bd3973, 0xff5c2cb842cff1a8,
                            0x00000000000075df, 0x0},
    std::array<uint64_t, 4>{0xea481720f7052a2b, 0x08dd94e5badcd4c3,
                            0x000000000000601e, 0x0},
    std::array<uint64_t, 4>{0xea481720f7052a2a, 0x08dd94e5badcd4c3,
                            0x000000000000601e, 0x0},
    std::array<uint64_t, 4>{0x47589a6c56af3ef2, 0xcef436be736cf7a5,
                            0x0000000000006157, 0x0},
    std::array<uint64_t, 4>{0x47589a6c56af3ef3, 0xcef436be736cf7a5,
                            0x0000000000006157, 0x0},
    std::array<uint64_t, 4>{0x2d2834779ec5b839, 0xbe9f6412ea2fd621,
                            0x0000000000007c60, 0x0},
    std::array<uint64_t, 4>{0x2d2834779ec5b838, 0xbe9f6412ea2fd621,
                            0x0000000000007c60, 0x0},
    std::array<uint64_t, 4>{0x8038b93b3f6face0, 0x78b6c649239ff547,
                            0x0000000000007d29, 0x0},
    std::array<uint64_t, 4>{0x8038b93b3f6face1, 0x78b6c649239ff547,
                            0x0000000000007d29, 0x0},
    std::array<uint64_t, 4>{0x8b8445c11ed7bfb9, 0x8f377e14db8cd02c,
                            0x00000000000068e8, 0x0},
    std::array<uint64_t, 4>{0x8b8445c11ed7bfb8, 0x8f377e14db8cd02c,
                            0x00000000000068e8, 0x0},
    std::array<uint64_t, 4>{0x2694c88dbf7dab60, 0x491edc4f123cf34a,
                            0x00000000000069a1, 0x0},
    std::array<uint64_t, 4>{0x2694c88dbf7dab61, 0x491edc4f123cf34a,
                            0x00000000000069a1, 0x0},
    std::array<uint64_t, 4>{0x47420c4940fecae0, 0x4d1754889e210cff,
                            0x000000000000f912, 0x0},
    std::array<uint64_t, 4>{0x47420c4940fecae1, 0x4d1754889e210cff,
                            0x000000000000f912, 0x0},
    std::array<uint64_t, 4>{0xea528105e154de39, 0x8b3ef6d357912f99,
                            0x000000000000f85b, 0x0},
    std::array<uint64_t, 4>{0xea528105e154de38, 0x8b3ef6d357912f99,
                            0x000000000000f85b, 0x0},
    std::array<uint64_t, 4>{0xe1ee7dffc0eccd60, 0x7cbf4e8eaf820af2,
                            0x000000000000ed9a, 0x0},
    std::array<uint64_t, 4>{0xe1ee7dffc0eccd61, 0x7cbf4e8eaf820af2,
                            0x000000000000ed9a, 0x0},
    std::array<uint64_t, 4>{0x4cfef0b36146d9b9, 0xba96ecd566322994,
                            0x000000000000ecd3, 0x0},
    std::array<uint64_t, 4>{0x4cfef0b36146d9b8, 0xba96ecd566322994,
                            0x000000000000ecd3, 0x0},
    std::array<uint64_t, 4>{0x268e5ea8a92c5f72, 0xcafdbe79ff710810,
                            0x000000000000f1e4, 0x0},
    std::array<uint64_t, 4>{0x268e5ea8a92c5f73, 0xcafdbe79ff710810,
                            0x000000000000f1e4, 0x0},
    std::array<uint64_t, 4>{0x8b9ed3e408864bab, 0x0cd41c2236c12b76,
                            0x000000000000f0ad, 0x0},
    std::array<uint64_t, 4>{0x8b9ed3e408864baa, 0x0cd41c2236c12b76,
                            0x000000000000f0ad, 0x0},
    std::array<uint64_t, 4>{0x80222f1e293e58f2, 0xfb55a47fced20e1d,
                            0x000000000000e56c, 0x0},
    std::array<uint64_t, 4>{0x80222f1e293e58f3, 0xfb55a47fced20e1d,
                            0x000000000000e56c, 0x0},
    std::array<uint64_t, 4>{0x2d32a25288944c2b, 0x3d7c062407622d7b,
                            0x000000000000e425, 0x0},
    std::array<uint64_t, 4>{0x2d32a25288944c2a, 0x3d7c062407622d7b,
                            0x000000000000e425, 0x0},
    std::array<uint64_t, 4>{0x0c44ccff562a81e4, 0x6c41d16aba60fbff,
                            0x0000000000006536, 0x0},
    std::array<uint64_t, 4>{0x0c44ccff562a81e5, 0x6c41d16aba60fbff,
                            0x0000000000006536, 0x0},
    std::array<uint64_t, 4>{0xa15441b3f780953d, 0xaa68733173d0d899,
                            0x000000000000647f, 0x0},
    std::array<uint64_t, 4>{0xa15441b3f780953c, 0xaa68733173d0d899,
                            0x000000000000647f, 0x0},
    std::array<uint64_t, 4>{0xaae8bd49d6388664, 0x5de9cb6c8bc3fdf2,
                            0x00000000000071be, 0x0},
    std::array<uint64_t, 4>{0xaae8bd49d6388665, 0x5de9cb6c8bc3fdf2,
                            0x00000000000071be, 0x0},
    std::array<uint64_t, 4>{0x07f83005779292bd, 0x9bc069374273de94,
                            0x00000000000070f7, 0x0},
    std::array<uint64_t, 4>{0x07f83005779292bc, 0x9bc069374273de94,
                            0x00000000000070f7, 0x0},
    std::array<uint64_t, 4>{0x6d889e1ebff81476, 0xebab3b9bdb30ff10,
                            0x0000000000006dc0, 0x0},
    std::array<uint64_t, 4>{0x6d889e1ebff81477, 0xebab3b9bdb30ff10,
                            0x0000000000006dc0, 0x0},
    std::array<uint64_t, 4>{0xc09813521e5200af, 0x2d8299c01280dc76,
                            0x0000000000006c89, 0x0},
    std::array<uint64_t, 4>{0xc09813521e5200ae, 0x2d8299c01280dc76,
                            0x0000000000006c89, 0x0},
    std::array<uint64_t, 4>{0xcb24efa83fea13f6, 0xda03219dea93f91d,
                            0x0000000000007948, 0x0},
    std::array<uint64_t, 4>{0xcb24efa83fea13f7, 0xda03219dea93f91d,
                            0x0000000000007948, 0x0},
    std::array<uint64_t, 4>{0x663462e49e40072f, 0x1c2a83c62323da7b,
                            0x0000000000007801, 0x0},
    std::array<uint64_t, 4>{0x663462e49e40072e, 0x1c2a83c62323da7b,
                            0x0000000000007801, 0x0},
    std::array<uint64_t, 4>{0xf7585daae103439d, 0x514b7c6f43c8cec1,
                            0x0000000000003809, 0x0},
    std::array<uint64_t, 4>{0xf7585daae103439c, 0x514b7c6f43c8cec1,
                            0x0000000000003809, 0x0},
    std::array<uint64_t, 4>{0x5a48d0e640a95744, 0x9762de348a78eda7,
                            0x0000000000003940, 0x0},
    std::array<uint64_t, 4>{0x5a48d0e640a95745, 0x9762de348a78eda7,
                            0x0000000000003940, 0x0},
    std::array<uint64_t, 4>{0x51f42c1c6111441d, 0x60e36669726bc8cc,
                            0x0000000000002c81, 0x0},
    std::array<uint64_t, 4>{0x51f42c1c6111441c, 0x60e36669726bc8cc,
                            0x0000000000002c81, 0x0},
    std::array<uint64_t, 4>{0xfce4a150c0bb50c4, 0xa6cac432bbdbebaa,
                            0x0000000000002dc8, 0x0},
    std::array<uint64_t, 4>{0xfce4a150c0bb50c5, 0xa6cac432bbdbebaa,
                            0x0000000000002dc8, 0x0},
    std::array<uint64_t, 4>{0x96940f4b08d1d60f, 0xd6a1969e2298ca2e,
                            0x00000000000030ff, 0x0},
    std::array<uint64_t, 4>{0x96940f4b08d1d60e, 0xd6a1969e2298ca2e,
                            0x00000000000030ff, 0x0},
    std::array<uint64_t, 4>{0x3b848207a97bc2d6, 0x108834c5eb28e948,
                            0x00000000000031b6, 0x0},
    std::array<uint64_t, 4>{0x3b848207a97bc2d7, 0x108834c5eb28e948,
                            0x00000000000031b6, 0x0},
    std::array<uint64_t, 4>{0x30387efd88c3d18f, 0xe7098c98133bcc23,
                            0x0000000000002477, 0x0},
    std::array<uint64_t, 4>{0x30387efd88c3d18e, 0xe7098c98133bcc23,
                            0x0000000000002477, 0x0},
    std::array<uint64_t, 4>{0x9d28f3b12969c556, 0x21202ec3da8bef45,
                            0x000000000000253e, 0x0},
    std::array<uint64_t, 4>{0x9d28f3b12969c557, 0x21202ec3da8bef45,
                            0x000000000000253e, 0x0},
    std::array<uint64_t, 4>{0xbc5e9d1cf7d70899, 0x701df98d678939c1,
                            0x000000000000a42d, 0x0},
    std::array<uint64_t, 4>{0xbc5e9d1cf7d70898, 0x701df98d678939c1,
                            0x000000000000a42d, 0x0},
    std::array<uint64_t, 4>{0x114e1050567d1c40, 0xb6345bd6ae391aa7,
                            0x000000000000a564, 0x0},
    std::array<uint64_t, 4>{0x114e1050567d1c41, 0xb6345bd6ae391aa7,
                            0x000000000000a564, 0x0},
    std::array<uint64_t, 4>{0x1af2ecaa77c50f19, 0x41b5e38b562a3fcc,
                            0x000000000000b0a5, 0x0},
    std::array<uint64_t, 4>{0x1af2ecaa77c50f18, 0x41b5e38b562a3fcc,
                            0x000000000000b0a5, 0x0},
    std::array<uint64_t, 4>{0xb7e261e6d66f1bc0, 0x879c41d09f9a1caa,
                            0x000000000000b1ec, 0x0},
    std::array<uint64_t, 4>{0xb7e261e6d66f1bc1, 0x879c41d09f9a1caa,
                            0x000000000000b1ec, 0x0},
    std::array<uint64_t, 4>{0xdd92cffd1e059d0b, 0xf7f7137c06d93d2e,
                            0x000000000000acdb, 0x0},
    std::array<uint64_t, 4>{0xdd92cffd1e059d0a, 0xf7f7137c06d93d2e,
                            0x000000000000acdb, 0x0},
    std::array<uint64_t, 4>{0x708242b1bfaf89d2, 0x31deb127cf691e48,
                            0x000000000000ad92, 0x0},
    std::array<uint64_t, 4>{0x708242b1bfaf89d3, 0x31deb127cf691e48,
                            0x000000000000ad92, 0x0},
    std::array<uint64_t, 4>{0x7b3ebe4b9e179a8b, 0xc65f097a377a3b23,
                            0x000000000000b853, 0x0},
    std::array<uint64_t, 4>{0x7b3ebe4b9e179a8a, 0xc65f097a377a3b23,
                            0x000000000000b853, 0x0},
    std::array<uint64_t, 4>{0xd62e33073fbd8e52, 0x0076ab21feca1845,
                            0x000000000000b91a, 0x0},
    std::array<uint64_t, 4>{0xd62e33073fbd8e53, 0x0076ab21feca1845,
                            0x000000000000b91a, 0x0},
    std::array<uint64_t, 4>{0xc29d59d1c4ef82b4, 0x83374f67ec1f3bc4,
                            0x0000000000008829, 0x0},
    std::array<uint64_t, 4>{0xc29d59d1c4ef82b5, 0x83374f67ec1f3bc4,
                            0x0000000000008829, 0x0},
    std::array<uint64_t, 4>{0x6f8dd49d6545966d, 0x451eed3c25af18a2,
                            0x0000000000008960, 0x0},
    std::array<uint64_t, 4>{0x6f8dd49d6545966c, 0x451eed3c25af18a2,
                            0x0000000000008960, 0x0},
    std::array<uint64_t, 4>{0x6431286744fd8534, 0xb29f5561ddbc3dc9,
                            0x0000000000009ca1, 0x0},
    std::array<uint64_t, 4>{0x6431286744fd8535, 0xb29f5561ddbc3dc9,
                            0x0000000000009ca1, 0x0},
    std::array<uint64_t, 4>{0xc921a52be55791ed, 0x74b6f73a140c1eaf,
                            0x0000000000009de8, 0x0},
    std::array<uint64_t, 4>{0xc921a52be55791ec, 0x74b6f73a140c1eaf,
                            0x0000000000009de8, 0x0},
    std::array<uint64_t, 4>{0xa3510b302d3d1726, 0x04dda5968d4f3f2b,
                            0x00000000000080df, 0x0},
    std::array<uint64_t, 4>{0xa3510b302d3d1727, 0x04dda5968d4f3f2b,
                            0x00000000000080df, 0x0},
    std::array<uint64_t, 4>{0x0e41867c8c9703ff, 0xc2f407cd44ff1c4d,
                            0x0000000000008196, 0x0},
    std::array<uint64_t, 4>{0x0e41867c8c9703fe, 0xc2f407cd44ff1c4d,
                            0x0000000000008196, 0x0},
    std::array<uint64_t, 4>{0x05fd7a86ad2f10a6, 0x3575bf90bcec3926,
                            0x0000000000009457, 0x0},
    std::array<uint64_t, 4>{0x05fd7a86ad2f10a7, 0x3575bf90bcec3926,
                            0x0000000000009457, 0x0},
    std::array<uint64_t, 4>{0xa8edf7ca0c85047f, 0xf35c1dcb755c1a40,
                            0x000000000000951e, 0x0},
    std::array<uint64_t, 4>{0xa8edf7ca0c85047e, 0xf35c1dcb755c1a40,
                            0x000000000000951e, 0x0},
    std::array<uint64_t, 4>{0x899b9967d23bc9b0, 0xa261ca85c85eccc4,
                            0x000000000000140d, 0x0},
    std::array<uint64_t, 4>{0x899b9967d23bc9b1, 0xa261ca85c85eccc4,
                            0x000000000000140d, 0x0},
    std::array<uint64_t, 4>{0x248b142b7391dd69, 0x644868de01eeefa2,
                            0x0000000000001544, 0x0},
    std::array<uint64_t, 4>{0x248b142b7391dd68, 0x644868de01eeefa2,
                            0x0000000000001544, 0x0},
    std::array<uint64_t, 4>{0x2f37e8d15229ce30, 0x93c9d083f9fdcac9,
                            0x0000000000000085, 0x0},
    std::array<uint64_t, 4>{0x2f37e8d15229ce31, 0x93c9d083f9fdcac9,
                            0x0000000000000085, 0x0},
    std::array<uint64_t, 4>{0x8227659df383dae9, 0x55e072d8304de9af,
                            0x00000000000001cc, 0x0},
    std::array<uint64_t, 4>{0x8227659df383dae8, 0x55e072d8304de9af,
                            0x00000000000001cc, 0x0},
    std::array<uint64_t, 4>{0xe857cb863be95c22, 0x258b2074a90ec82b,
                            0x0000000000001cfb, 0x0},
    std::array<uint64_t, 4>{0xe857cb863be95c23, 0x258b2074a90ec82b,
                            0x0000000000001cfb, 0x0},
    std::array<uint64_t, 4>{0x454746ca9a4348fb, 0xe3a2822f60beeb4d,
                            0x0000000000001db2, 0x0},
    std::array<uint64_t, 4>{0x454746ca9a4348fa, 0xe3a2822f60beeb4d,
                            0x0000000000001db2, 0x0},
    std::array<uint64_t, 4>{0x4efbba30bbfb5ba2, 0x14233a7298adce26,
                            0x0000000000000873, 0x0},
    std::array<uint64_t, 4>{0x4efbba30bbfb5ba3, 0x14233a7298adce26,
                            0x0000000000000873, 0x0},
    std::array<uint64_t, 4>{0xe3eb377c1a514f7b, 0xd20a9829511ded40,
                            0x000000000000093a, 0x0},
    std::array<uint64_t, 4>{0xe3eb377c1a514f7a, 0xd20a9829511ded40,
                            0x000000000000093a, 0x0},
    std::array<uint64_t, 4>{0x7287083265120bc9, 0x9f6b678031f6f9fa,
                            0x0000000000004932, 0x0},
    std::array<uint64_t, 4>{0x7287083265120bc8, 0x9f6b678031f6f9fa,
                            0x0000000000004932, 0x0},
    std::array<uint64_t, 4>{0xdf97857ec4b81f10, 0x5942c5dbf846da9c,
                            0x000000000000487b, 0x0},
    std::array<uint64_t, 4>{0xdf97857ec4b81f11, 0x5942c5dbf846da9c,
                            0x000000000000487b, 0x0},
    std::array<uint64_t, 4>{0xd42b7984e5000c49, 0xaec37d860055fff7,
                            0x0000000000005dba, 0x0},
    std::array<uint64_t, 4>{0xd42b7984e5000c48, 0xaec37d860055fff7,
                            0x0000000000005dba, 0x0},
    std::array<uint64_t, 4>{0x793bf4c844aa1890, 0x68eadfddc9e5dc91,
                            0x0000000000005cf3, 0x0},
    std::array<uint64_t, 4>{0x793bf4c844aa1891, 0x68eadfddc9e5dc91,
                            0x0000000000005cf3, 0x0},
    std::array<uint64_t, 4>{0x134b5ad38cc09e5b, 0x18818d7150a6fd15,
                            0x00000000000041c4, 0x0},
    std::array<uint64_t, 4>{0x134b5ad38cc09e5a, 0x18818d7150a6fd15,
                            0x00000000000041c4, 0x0},
    std::array<uint64_t, 4>{0xbe5bd79f2d6a8a82, 0xdea82f2a9916de73,
                            0x000000000000408d, 0x0},
    std::array<uint64_t, 4>{0xbe5bd79f2d6a8a83, 0xdea82f2a9916de73,
                            0x000000000000408d, 0x0},
    std::array<uint64_t, 4>{0xb5e72b650cd299db, 0x292997776105fb18,
                            0x000000000000554c, 0x0},
    std::array<uint64_t, 4>{0xb5e72b650cd299da, 0x292997776105fb18,
                            0x000000000000554c, 0x0},
    std::array<uint64_t, 4>{0x18f7a629ad788d02, 0xef00352ca8b5d87e,
                            0x0000000000005405, 0x0},
    std::array<uint64_t, 4>{0x18f7a629ad788d03, 0xef00352ca8b5d87e,
                            0x0000000000005405, 0x0},
    std::array<uint64_t, 4>{0x3981c88473c640cd, 0xbe3de26215b70efa,
                            0x000000000000d516, 0x0},
    std::array<uint64_t, 4>{0x3981c88473c640cc, 0xbe3de26215b70efa,
                            0x000000000000d516, 0x0},
    std::array<uint64_t, 4>{0x949145c8d26c5414, 0x78144039dc072d9c,
                            0x000000000000d45f, 0x0},
    std::array<uint64_t, 4>{0x949145c8d26c5415, 0x78144039dc072d9c,
                            0x000000000000d45f, 0x0},
    std::array<uint64_t, 4>{0x9f2db932f3d4474d, 0x8f95f864241408f7,
                            0x000000000000c19e, 0x0},
    std::array<uint64_t, 4>{0x9f2db932f3d4474c, 0x8f95f864241408f7,
                            0x000000000000c19e, 0x0},
    std::array<uint64_t, 4>{0x323d347e527e5394, 0x49bc5a3feda42b91,
                            0x000000000000c0d7, 0x0},
    std::array<uint64_t, 4>{0x323d347e527e5395, 0x49bc5a3feda42b91,
                            0x000000000000c0d7, 0x0},
    std::array<uint64_t, 4>{0x584d9a659a14d55f, 0x39d7089374e70a15,
                            0x000000000000dde0, 0x0},
    std::array<uint64_t, 4>{0x584d9a659a14d55e, 0x39d7089374e70a15,
                            0x000000000000dde0, 0x0},
    std::array<uint64_t, 4>{0xf55d17293bbec186, 0xfffeaac8bd572973,
                            0x000000000000dca9, 0x0},
    std::array<uint64_t, 4>{0xf55d17293bbec187, 0xfffeaac8bd572973,
                            0x000000000000dca9, 0x0},
    std::array<uint64_t, 4>{0xfee1ebd31a06d2df, 0x087f129545440c18,
                            0x000000000000c968, 0x0},
    std::array<uint64_t, 4>{0xfee1ebd31a06d2de, 0x087f129545440c18,
                            0x000000000000c968, 0x0},
    std::array<uint64_t, 4>{0x53f1669fbbacc606, 0xce56b0ce8cf42f7e,
                            0x000000000000c821, 0x0},
    std::array<uint64_t, 4>{0x53f1669fbbacc607, 0xce56b0ce8cf42f7e,
                            0x000000000000c821, 0x0},
    std::array<uint64_t, 4>{0x7565ae1204d16d66, 0x87486c819ec8dc34,
                            0x000000000000a180, 0x0},
    std::array<uint64_t, 4>{0x7565ae1204d16d67, 0x87486c819ec8dc34,
                            0x000000000000a180, 0x0},
    std::array<uint64_t, 4>{0xd875235ea57b79bf, 0x4161ceda5778ff52,
                            0x000000000000a0c9, 0x0},
    std::array<uint64_t, 4>{0xd875235ea57b79be, 0x4161ceda5778ff52,
                            0x000000000000a0c9, 0x0},
    std::array<uint64_t, 4>{0xd3c9dfa484c36ae6, 0xb6e07687af6bda39,
                            0x000000000000b508, 0x0},
    std::array<uint64_t, 4>{0xd3c9dfa484c36ae7, 0xb6e07687af6bda39,
                            0x000000000000b508, 0x0},
    std::array<uint64_t, 4>{0x7ed952e825697e3f, 0x70c9d4dc66dbf95f,
                            0x000000000000b441, 0x0},
    std::array<uint64_t, 4>{0x7ed952e825697e3e, 0x70c9d4dc66dbf95f,
                            0x000000000000b441, 0x0},
    std::array<uint64_t, 4>{0x14a9fcf3ed03f8f4, 0x00a28670ff98d8db,
                            0x000000000000a976, 0x0},
    std::array<uint64_t, 4>{0x14a9fcf3ed03f8f5, 0x00a28670ff98d8db,
                            0x000000000000a976, 0x0},
    std::array<uint64_t, 4>{0xb9b971bf4ca9ec2d, 0xc68b242b3628fbbd,
                            0x000000000000a83f, 0x0},
    std::array<uint64_t, 4>{0xb9b971bf4ca9ec2c, 0xc68b242b3628fbbd,
                            0x000000000000a83f, 0x0},
    std::array<uint64_t, 4>{0xb2058d456d11ff74, 0x310a9c76ce3bded6,
                            0x000000000000bdfe, 0x0},
    std::array<uint64_t, 4>{0xb2058d456d11ff75, 0x310a9c76ce3bded6,
                            0x000000000000bdfe, 0x0},
    std::array<uint64_t, 4>{0x1f150009ccbbebad, 0xf7233e2d078bfdb0,
                            0x000000000000bcb7, 0x0},
    std::array<uint64_t, 4>{0x1f150009ccbbebac, 0xf7233e2d078bfdb0,
                            0x000000000000bcb7, 0x0},
    std::array<uint64_t, 4>{0x3e636ea412052662, 0xa61ee963ba892b34,
                            0x0000000000003da4, 0x0},
    std::array<uint64_t, 4>{0x3e636ea412052663, 0xa61ee963ba892b34,
                            0x0000000000003da4, 0x0},
    std::array<uint64_t, 4>{0x9373e3e8b3af32bb, 0x60374b3873390852,
                            0x0000000000003ced, 0x0},
    std::array<uint64_t, 4>{0x9373e3e8b3af32ba, 0x60374b3873390852,
                            0x0000000000003ced, 0x0},
    std::array<uint64_t, 4>{0x98cf1f12921721e2, 0x97b6f3658b2a2d39,
                            0x000000000000292c, 0x0},
    std::array<uint64_t, 4>{0x98cf1f12921721e3, 0x97b6f3658b2a2d39,
                            0x000000000000292c, 0x0},
    std::array<uint64_t, 4>{0x35df925e33bd353b, 0x519f513e429a0e5f,
                            0x0000000000002865, 0x0},
    std::array<uint64_t, 4>{0x35df925e33bd353a, 0x519f513e429a0e5f,
                            0x0000000000002865, 0x0},
    std::array<uint64_t, 4>{0x5faf3c45fbd7b3f0, 0x21f40392dbd92fdb,
                            0x0000000000003552, 0x0},
    std::array<uint64_t, 4>{0x5faf3c45fbd7b3f1, 0x21f40392dbd92fdb,
                            0x0000000000003552, 0x0},
    std::array<uint64_t, 4>{0xf2bfb1095a7da729, 0xe7dda1c912690cbd,
                            0x000000000000341b, 0x0},
    std::array<uint64_t, 4>{0xf2bfb1095a7da728, 0xe7dda1c912690cbd,
                            0x000000000000341b, 0x0},
    std::array<uint64_t, 4>{0xf9034df37bc5b470, 0x105c1994ea7a29d6,
                            0x00000000000021da, 0x0},
    std::array<uint64_t, 4>{0xf9034df37bc5b471, 0x105c1994ea7a29d6,
                            0x00000000000021da, 0x0},
    std::array<uint64_t, 4>{0x5413c0bfda6fa0a9, 0xd675bbcf23ca0ab0,
                            0x0000000000002093, 0x0},
    std::array<uint64_t, 4>{0x5413c0bfda6fa0a8, 0xd675bbcf23ca0ab0,
                            0x0000000000002093, 0x0},
    std::array<uint64_t, 4>{0xc57ffff1a52ce41b, 0x9b14446643211e0a,
                            0x000000000000609b, 0x0},
    std::array<uint64_t, 4>{0xc57ffff1a52ce41a, 0x9b14446643211e0a,
                            0x000000000000609b, 0x0},
    std::array<uint64_t, 4>{0x686f72bd0486f0c2, 0x5d3de63d8a913d6c,
                            0x00000000000061d2, 0x0},
    std::array<uint64_t, 4>{0x686f72bd0486f0c3, 0x5d3de63d8a913d6c,
                            0x00000000000061d2, 0x0},
    std::array<uint64_t, 4>{0x63d38e47253ee39b, 0xaabc5e6072821807,
                            0x0000000000007413, 0x0},
    std::array<uint64_t, 4>{0x63d38e47253ee39a, 0xaabc5e6072821807,
                            0x0000000000007413, 0x0},
    std::array<uint64_t, 4>{0xcec3030b8494f742, 0x6c95fc3bbb323b61,
                            0x000000000000755a, 0x0},
    std::array<uint64_t, 4>{0xcec3030b8494f743, 0x6c95fc3bbb323b61,
                            0x000000000000755a, 0x0},
    std::array<uint64_t, 4>{0xa4b3ad104cfe7189, 0x1cfeae9722711ae5,
                            0x000000000000686d, 0x0},
    std::array<uint64_t, 4>{0xa4b3ad104cfe7188, 0x1cfeae9722711ae5,
                            0x000000000000686d, 0x0},
    std::array<uint64_t, 4>{0x09a3205ced546550, 0xdad70cccebc13983,
                            0x0000000000006924, 0x0},
    std::array<uint64_t, 4>{0x09a3205ced546551, 0xdad70cccebc13983,
                            0x0000000000006924, 0x0},
    std::array<uint64_t, 4>{0x021fdca6ccec7609, 0x2d56b49113d21ce8,
                            0x0000000000007ce5, 0x0},
    std::array<uint64_t, 4>{0x021fdca6ccec7608, 0x2d56b49113d21ce8,
                            0x0000000000007ce5, 0x0},
    std::array<uint64_t, 4>{0xaf0f51ea6d4662d0, 0xeb7f16cada623f8e,
                            0x0000000000007dac, 0x0},
    std::array<uint64_t, 4>{0xaf0f51ea6d4662d1, 0xeb7f16cada623f8e,
                            0x0000000000007dac, 0x0},
    std::array<uint64_t, 4>{0x8e793f47b3f8af1f, 0xba42c1846760e90a,
                            0x000000000000fcbf, 0x0},
    std::array<uint64_t, 4>{0x8e793f47b3f8af1e, 0xba42c1846760e90a,
                            0x000000000000fcbf, 0x0},
    std::array<uint64_t, 4>{0x2369b20b1252bbc6, 0x7c6b63dfaed0ca6c,
                            0x000000000000fdf6, 0x0},
    std::array<uint64_t, 4>{0x2369b20b1252bbc7, 0x7c6b63dfaed0ca6c,
                            0x000000000000fdf6, 0x0},
    std::array<uint64_t, 4>{0x28d54ef133eaa89f, 0x8beadb8256c3ef07,
                            0x000000000000e837, 0x0},
    std::array<uint64_t, 4>{0x28d54ef133eaa89e, 0x8beadb8256c3ef07,
                            0x000000000000e837, 0x0},
    std::array<uint64_t, 4>{0x85c5c3bd9240bc46, 0x4dc379d99f73cc61,
                            0x000000000000e97e, 0x0},
    std::array<uint64_t, 4>{0x85c5c3bd9240bc47, 0x4dc379d99f73cc61,
                            0x000000000000e97e, 0x0},
    std::array<uint64_t, 4>{0xefb56da65a2a3a8d, 0x3da82b750630ede5,
                            0x000000000000f449, 0x0},
    std::array<uint64_t, 4>{0xefb56da65a2a3a8c, 0x3da82b750630ede5,
                            0x000000000000f449, 0x0},
    std::array<uint64_t, 4>{0x42a5e0eafb802e54, 0xfb81892ecf80ce83,
                            0x000000000000f500, 0x0},
    std::array<uint64_t, 4>{0x42a5e0eafb802e55, 0xfb81892ecf80ce83,
                            0x000000000000f500, 0x0},
    std::array<uint64_t, 4>{0x49191c10da383d0d, 0x0c0031733793ebe8,
                            0x000000000000e0c1, 0x0},
    std::array<uint64_t, 4>{0x49191c10da383d0c, 0x0c0031733793ebe8,
                            0x000000000000e0c1, 0x0},
    std::array<uint64_t, 4>{0xe409915c7b9229d4, 0xca299328fe23c88e,
                            0x000000000000e188, 0x0},
    std::array<uint64_t, 4>{0xe409915c7b9229d5, 0xca299328fe23c88e,
                            0x000000000000e188, 0x0},
    std::array<uint64_t, 4>{0xf0bafb8a80c02532, 0x4968776eecf6eb0f,
                            0x000000000000d0bb, 0x0},
    std::array<uint64_t, 4>{0xf0bafb8a80c02533, 0x4968776eecf6eb0f,
                            0x000000000000d0bb, 0x0},
    std::array<uint64_t, 4>{0x5daa76c6216a31eb, 0x8f41d5352546c869,
                            0x000000000000d1f2, 0x0},
    std::array<uint64_t, 4>{0x5daa76c6216a31ea, 0x8f41d5352546c869,
                            0x000000000000d1f2, 0x0},
    std::array<uint64_t, 4>{0x56168a3c00d222b2, 0x78c06d68dd55ed02,
                            0x000000000000c433, 0x0},
    std::array<uint64_t, 4>{0x56168a3c00d222b3, 0x78c06d68dd55ed02,
                            0x000000000000c433, 0x0},
    std::array<uint64_t, 4>{0xfb060770a178366b, 0xbee9cf3314e5ce64,
                            0x000000000000c57a, 0x0},
    std::array<uint64_t, 4>{0xfb060770a178366a, 0xbee9cf3314e5ce64,
                            0x000000000000c57a, 0x0},
    std::array<uint64_t, 4>{0x9176a96b6912b0a0, 0xce829d9f8da6efe0,
                            0x000000000000d84d, 0x0},
    std::array<uint64_t, 4>{0x9176a96b6912b0a1, 0xce829d9f8da6efe0,
                            0x000000000000d84d, 0x0},
    std::array<uint64_t, 4>{0x3c662427c8b8a479, 0x08ab3fc44416cc86,
                            0x000000000000d904, 0x0},
    std::array<uint64_t, 4>{0x3c662427c8b8a478, 0x08ab3fc44416cc86,
                            0x000000000000d904, 0x0},
    std::array<uint64_t, 4>{0x37dad8dde900b720, 0xff2a8799bc05e9ed,
                            0x000000000000ccc5, 0x0},
    std::array<uint64_t, 4>{0x37dad8dde900b721, 0xff2a8799bc05e9ed,
                            0x000000000000ccc5, 0x0},
    std::array<uint64_t, 4>{0x9aca559148aaa3f9, 0x390325c275b5ca8b,
                            0x000000000000cd8c, 0x0},
    std::array<uint64_t, 4>{0x9aca559148aaa3f8, 0x390325c275b5ca8b,
                            0x000000000000cd8c, 0x0},
    std::array<uint64_t, 4>{0xbbbc3b3c96146e36, 0x683ef28cc8b71c0f,
                            0x0000000000004c9f, 0x0},
    std::array<uint64_t, 4>{0xbbbc3b3c96146e37, 0x683ef28cc8b71c0f,
                            0x0000000000004c9f, 0x0},
    std::array<uint64_t, 4>{0x16acb67037be7aef, 0xae1750d701073f69,
                            0x0000000000004dd6, 0x0},
    std::array<uint64_t, 4>{0x16acb67037be7aee, 0xae1750d701073f69,
                            0x0000000000004dd6, 0x0},
    std::array<uint64_t, 4>{0x1d104a8a160669b6, 0x5996e88af9141a02,
                            0x0000000000005817, 0x0},
    std::array<uint64_t, 4>{0x1d104a8a160669b7, 0x5996e88af9141a02,
                            0x0000000000005817, 0x0},
    std::array<uint64_t, 4>{0xb000c7c6b7ac7d6f, 0x9fbf4ad130a43964,
                            0x000000000000595e, 0x0},
    std::array<uint64_t, 4>{0xb000c7c6b7ac7d6e, 0x9fbf4ad130a43964,
                            0x000000000000595e, 0x0},
    std::array<uint64_t, 4>{0xda7069dd7fc6fba4, 0xefd4187da9e718e0,
                            0x0000000000004469, 0x0},
    std::array<uint64_t, 4>{0xda7069dd7fc6fba5, 0xefd4187da9e718e0,
                            0x0000000000004469, 0x0},
    std::array<uint64_t, 4>{0x7760e491de6cef7d, 0x29fdba2660573b86,
                            0x0000000000004520, 0x0},
    std::array<uint64_t, 4>{0x7760e491de6cef7c, 0x29fdba2660573b86,
                            0x0000000000004520, 0x0},
    std::array<uint64_t, 4>{0x7cdc186bffd4fc24, 0xde7c027b98441eed,
                            0x00000000000050e1, 0x0},
    std::array<uint64_t, 4>{0x7cdc186bffd4fc25, 0xde7c027b98441eed,
                            0x00000000000050e1, 0x0},
    std::array<uint64_t, 4>{0xd1cc95275e7ee8fd, 0x1855a02051f43d8b,
                            0x00000000000051a8, 0x0},
    std::array<uint64_t, 4>{0xd1cc95275e7ee8fc, 0x1855a02051f43d8b,
                            0x00000000000051a8, 0x0},
    std::array<uint64_t, 4>{0x40a0aa69213dac4f, 0x55345f89311f2931,
                            0x00000000000011a0, 0x0},
    std::array<uint64_t, 4>{0x40a0aa69213dac4e, 0x55345f89311f2931,
                            0x00000000000011a0, 0x0},
    std::array<uint64_t, 4>{0xedb027258097b896, 0x931dfdd2f8af0a57,
                            0x00000000000010e9, 0x0},
    std::array<uint64_t, 4>{0xedb027258097b897, 0x931dfdd2f8af0a57,
                            0x00000000000010e9, 0x0},
    std::array<uint64_t, 4>{0xe60cdbdfa12fabcf, 0x649c458f00bc2f3c,
                            0x0000000000000528, 0x0},
    std::array<uint64_t, 4>{0xe60cdbdfa12fabce, 0x649c458f00bc2f3c,
                            0x0000000000000528, 0x0},
    std::array<uint64_t, 4>{0x4b1c56930085bf16, 0xa2b5e7d4c90c0c5a,
                            0x0000000000000461, 0x0},
    std::array<uint64_t, 4>{0x4b1c56930085bf17, 0xa2b5e7d4c90c0c5a,
                            0x0000000000000461, 0x0},
    std::array<uint64_t, 4>{0x216cf888c8ef39dd, 0xd2deb578504f2dde,
                            0x0000000000001956, 0x0},
    std::array<uint64_t, 4>{0x216cf888c8ef39dc, 0xd2deb578504f2dde,
                            0x0000000000001956, 0x0},
    std::array<uint64_t, 4>{0x8c7c75c469452d04, 0x14f7172399ff0eb8,
                            0x000000000000181f, 0x0},
    std::array<uint64_t, 4>{0x8c7c75c469452d05, 0x14f7172399ff0eb8,
                            0x000000000000181f, 0x0},
    std::array<uint64_t, 4>{0x87c0893e48fd3e5d, 0xe376af7e61ec2bd3,
                            0x0000000000000dde, 0x0},
    std::array<uint64_t, 4>{0x87c0893e48fd3e5c, 0xe376af7e61ec2bd3,
                            0x0000000000000dde, 0x0},
    std::array<uint64_t, 4>{0x2ad00472e9572a84, 0x255f0d25a85c08b5,
                            0x0000000000000c97, 0x0},
    std::array<uint64_t, 4>{0x2ad00472e9572a85, 0x255f0d25a85c08b5,
                            0x0000000000000c97, 0x0},
    std::array<uint64_t, 4>{0x0ba66adf37e9e74b, 0x7462da6b155ede31,
                            0x0000000000008d84, 0x0},
    std::array<uint64_t, 4>{0x0ba66adf37e9e74a, 0x7462da6b155ede31,
                            0x0000000000008d84, 0x0},
    std::array<uint64_t, 4>{0xa6b6e7939643f392, 0xb24b7830dceefd57,
                            0x0000000000008ccd, 0x0},
    std::array<uint64_t, 4>{0xa6b6e7939643f393, 0xb24b7830dceefd57,
                            0x0000000000008ccd, 0x0},
    std::array<uint64_t, 4>{0xad0a1b69b7fbe0cb, 0x45cac06d24fdd83c,
                            0x000000000000990c, 0x0},
    std::array<uint64_t, 4>{0xad0a1b69b7fbe0ca, 0x45cac06d24fdd83c,
                            0x000000000000990c, 0x0},
    std::array<uint64_t, 4>{0x001a96251651f412, 0x83e36236ed4dfb5a,
                            0x0000000000009845, 0x0},
    std::array<uint64_t, 4>{0x001a96251651f413, 0x83e36236ed4dfb5a,
                            0x0000000000009845, 0x0},
    std::array<uint64_t, 4>{0x6a6a383ede3b72d9, 0xf388309a740edade,
                            0x0000000000008572, 0x0},
    std::array<uint64_t, 4>{0x6a6a383ede3b72d8, 0xf388309a740edade,
                            0x0000000000008572, 0x0},
    std::array<uint64_t, 4>{0xc77ab5727f916600, 0x35a192c1bdbef9b8,
                            0x000000000000843b, 0x0},
    std::array<uint64_t, 4>{0xc77ab5727f916601, 0x35a192c1bdbef9b8,
                            0x000000000000843b, 0x0},
    std::array<uint64_t, 4>{0xccc649885e297559, 0xc2202a9c45addcd3,
                            0x00000000000091fa, 0x0},
    std::array<uint64_t, 4>{0xccc649885e297558, 0xc2202a9c45addcd3,
                            0x00000000000091fa, 0x0},
    std::array<uint64_t, 4>{0x61d6c4c4ff836180, 0x040988c78c1dffb5,
                            0x00000000000090b3, 0x0},
    std::array<uint64_t, 4>{0x61d6c4c4ff836181, 0x040988c78c1dffb5,
                            0x00000000000090b3, 0x0},
};

} // namespace

namespace field {
// field::GF2_144 eval_lifted(const gsl::span<const field::GF2_9> &poly,
//                            const field::GF2_144 &point) {
//   GF2_144 acc;
//   long i;

//   for (i = poly.size() - 1; i >= 0; i--) {
//     acc *= point;
//     acc += lift(poly[i].data);
//   }

//   return acc;
// }
} // namespace field

// more optimized horner eval by splitting the poly in four terms
//   x^3 * (c_3 + x^4*c_7 + x^8*c_11 +...)
// + x^2 * (c_2 + x^4*c_6 + x^8*c_10 + ...)
// + x   * (c_1 + x^4*c_5 + x^8*c_9 +...)
// +       (c_0 + x^4*c_4 + x^8*c_8 + ...)
// and evaluating both with horner. While this is actually a bit more work
// than the trivial horner eval, it is much more friendly to the CPU, since
// the four polys can be evaluated in a more interleaved manner
field::GF2_144 field::eval_lifted(const gsl::span<const field::GF2_9> &poly,
                                  const field::GF2_144 &point) {
  field::GF2_144 acc_0, acc_1, acc_2, acc_3;
  long i = poly.size() - 1;

  field::GF2_144 p2, p4;
  gf2_144sqr(p2.as_m128i(), point.as_const_m128i());
  gf2_144sqr(p4.as_m128i(), p2.as_const_m128i());

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

// somewhat optimized inner product, only do one lazy reduction
field::GF2_144 lifted_dot_product(const gsl::span<const field::GF2_144> &lhs,
                                  const gsl::span<const field::GF2_9> &rhs) {

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

// TEMPLATE INSTANTIATIONS for GF2_144

// yes we include the cpp file with the template stuff
#include "field_templates.cpp"

INSTANTIATE_TEMPLATES_FOR(field::GF2_144)

// END TEMPLATE INSTANTIATIONS for GF2_144