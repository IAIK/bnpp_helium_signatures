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

inline uint8_t gf2_8mul_naive(uint16_t a, uint16_t b) {
  uint8_t result = 0;

  for (int i = 0; i < 8; ++i) {
    uint8_t mask = -(b & 1);
    result ^= (a & mask);
    uint16_t mask2 = -((uint16_t)(a >> 7) & 1);
    a <<= 1;
    // reduce by x^8 + x^4 + x^3 + x + 1
    a ^= (0x11b & mask2);
    b >>= 1;
  }
  return result;
}

inline uint8_t gf2_8mul(uint16_t a, uint16_t b) {
  uint64_t mul = _mm_extract_epi64(
      _mm_clmulepi64_si128(_mm_set1_epi64x(a), _mm_set1_epi64x(b), 0x0), 0);
  // reduce by x^8 + x^4 + x^3 + x + 1
  uint64_t upper = (mul >> 8);
  uint64_t lower = mul & 0xFF;
  upper = upper ^ (upper >> 4) ^ (upper >> 5) ^ (upper >> 7);
  lower = lower ^ (upper << 4) ^ (upper << 3) ^ (upper << 1) ^ upper;
  return lower & 0xFF;
}

} // namespace

namespace field {

GF2_8 GF2_8::operator+(const GF2_8 &other) const {
  return GF2_8(this->data ^ other.data);
}
GF2_8 &GF2_8::operator+=(const GF2_8 &other) {
  this->data ^= other.data;
  return *this;
}
GF2_8 GF2_8::operator-(const GF2_8 &other) const {
  return GF2_8(this->data ^ other.data);
}
GF2_8 &GF2_8::operator-=(const GF2_8 &other) {
  this->data ^= other.data;
  return *this;
}
GF2_8 GF2_8::operator*(const GF2_8 &other) const {
  return GF2_8(gf2_8mul_naive(this->data, other.data));
}
GF2_8 &GF2_8::operator*=(const GF2_8 &other) {
  this->data = gf2_8mul_naive(this->data, other.data);
  return *this;
}
bool GF2_8::operator==(const GF2_8 &other) const {
  return this->data == other.data;
}
bool GF2_8::operator!=(const GF2_8 &other) const {
  return this->data != other.data;
}

void GF2_8::to_bytes(uint8_t *out) const { *out = this->data; }

void GF2_8::from_bytes(const uint8_t *in) { this->data = *in; }

GF2_8 GF2_8::inverse() const {
  // Fixed-op square-multiply
  // 2^n - 2 in binary is 0b1111..10
  uint16_t t1 = this->data;
  uint16_t t2 = t1;

  // First 6 one-bits (start from second)
  for (size_t i = 0; i < 6; i++) {
    t2 = gf2_8mul(t2, t2);
    t1 = gf2_8mul(t1, t2);
  }

  // Final zero-bit
  t1 = gf2_8mul(t1, t1);
  return GF2_8(t1);
}

} // namespace field

field::GF2_8 dot_product(const std::vector<field::GF2_8> &lhs,
                         const std::vector<field::GF2_8> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  uint8_t accum = 0;
  for (size_t i = 0; i < lhs.size(); i++) {
    accum ^= gf2_8mul(lhs[i].data, rhs[i].data);
  }

  // combined reduction
  return field::GF2_8(accum);
}

std::vector<field::GF2_8> field::interpolate_with_precomputation(
    const std::array<std::array<field::GF2_8, 100>, 100>
        &precomputed_lagrange_polynomials,
    const std::vector<field::GF2_8> &y_values) {
  if (precomputed_lagrange_polynomials.size() != y_values.size() ||
      y_values.empty())
    throw std::runtime_error("invalid sizes for interpolation");

  std::vector<field::GF2_8> res(precomputed_lagrange_polynomials[0].size());
  size_t m = y_values.size();
  for (size_t k = 0; k < m; k++) {
    for (size_t j = 0; j < m; j++) {
      res[j] += precomputed_lagrange_polynomials[k][j] * y_values[k];
    }
  }
  return res;
}

// TEMPLATE INSTANTIATIONS for GF2_8

// yes we include the cpp file with the template stuff
#include "field_templates.cpp"

INSTANTIATE_TEMPLATES_FOR(field::GF2_8)
// END TEMPLATE INSTANTIATIONS for GF2_8