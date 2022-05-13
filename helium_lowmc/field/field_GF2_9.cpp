#if __INCLUDE_LEVEL__ != 0
#error "Don't include this file"
#endif
#include "field.h"

#include <cstring>
#include <iomanip>
#include <iostream>

extern "C" {
#include "portable_endian.h"
}

namespace {

inline uint16_t gf2_9mul_naive(uint16_t a, uint16_t b) {
  // should be const time, but probably slower than other variants
  // maybe just do a lookup table inside an uint64_t?
  uint16_t result = 0;
  for (int i = 0; i < 9; ++i) {
    uint16_t mask = -(b & 1);
    result ^= (a & mask);
    uint16_t mask2 = -((a >> 8) & 1);
    a <<= 1;
    // reduce by x^9 + x + 1 if needed
    a ^= (0x203 & mask2);
    b >>= 1;
  }
  return result;
}

inline uint16_t gf2_9mul(uint16_t a, uint16_t b) {
  uint64_t mul = _mm_extract_epi64(
      _mm_clmulepi64_si128(_mm_set1_epi64x(a), _mm_set1_epi64x(b), 0x0), 0);
  // reduce by x^9 + x + 1
  uint64_t upper = (mul >> 9) << 1;
  uint64_t upper2 = (mul >> 9);
  uint64_t lower = mul & 0x1FF;
  return (upper ^ upper2 ^ lower);
}

} // namespace

namespace field {

GF2_9 GF2_9::operator+(const GF2_9 &other) const {
  return GF2_9(this->data ^ other.data);
}
GF2_9 &GF2_9::operator+=(const GF2_9 &other) {
  this->data ^= other.data;
  return *this;
}
GF2_9 GF2_9::operator-(const GF2_9 &other) const {
  return GF2_9(this->data ^ other.data);
}
GF2_9 &GF2_9::operator-=(const GF2_9 &other) {
  this->data ^= other.data;
  return *this;
}
GF2_9 GF2_9::operator*(const GF2_9 &other) const {
  return GF2_9(gf2_9mul(this->data, other.data));
}
GF2_9 &GF2_9::operator*=(const GF2_9 &other) {
  this->data = gf2_9mul(this->data, other.data);
  return *this;
}
bool GF2_9::operator==(const GF2_9 &other) const {
  return this->data == other.data;
}
bool GF2_9::operator!=(const GF2_9 &other) const {
  return this->data != other.data;
}

void GF2_9::to_bytes(uint8_t *out) const {
  uint16_t tmp = htole16(this->data);
  memcpy(out, (uint8_t *)&tmp, BYTE_SIZE);
}

void GF2_9::from_bytes(const uint8_t *in) {
  uint16_t tmp;
  memcpy((uint8_t *)&tmp, in, BYTE_SIZE);
  this->data = le16toh(tmp) & ELEMENT_MASK;
}

GF2_9 GF2_9::inverse() const {
  // Fixed-op square-multiply
  // 2^n - 2 in binary is 0b1111..10
  uint16_t t1 = this->data;
  uint16_t t2 = t1;

  // First 7 one-bits (start from second)
  for (size_t i = 0; i < 7; i++) {
    t2 = gf2_9mul(t2, t2);
    t1 = gf2_9mul(t1, t2);
  }

  // Final zero-bit
  t1 = gf2_9mul(t1, t1);
  return GF2_9(t1);
}

} // namespace field

field::GF2_9 dot_product(const std::vector<field::GF2_9> &lhs,
                         const std::vector<field::GF2_9> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  uint16_t accum = 0;
  for (size_t i = 0; i < lhs.size(); i++) {
    accum ^= gf2_9mul(lhs[i].data, rhs[i].data);
  }

  // combined reduction
  return field::GF2_9(accum);
}

// TEMPLATE INSTANTIATIONS for GF2_9

// yes we include the cpp file with the template stuff
#include "field_templates.cpp"

INSTANTIATE_TEMPLATES_FOR(field::GF2_9)
// END TEMPLATE INSTANTIATIONS for GF2_9
template std::vector<field::GF2_9> field::interpolate_with_precomputation(
    const std::array<std::array<field::GF2_9, 86>, 86>
        &precomputed_lagrange_polynomials,
    const std::vector<field::GF2_9> &y_values);