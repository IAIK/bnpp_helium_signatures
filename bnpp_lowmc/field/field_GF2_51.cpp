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

inline __m128i clmul(uint64_t a, uint64_t b) {
  return _mm_clmulepi64_si128(_mm_set_epi64x(0, a), _mm_set_epi64x(0, b), 0);
}

uint64_t reduce_GF2_51(__m128i in) {
  // modulus = x^51 + x^6 + x^3 + x + 1
  constexpr uint64_t upper_mask =
      0x3FFFFFFFFFULL; // mask for getting the relevant bits of the upper 64 bit
                       // word
  constexpr uint64_t lower_mask =
      0x7FFFFFFFFFFFFULL; // mask for getting the relevant bits of the lower 64
                          // bit word
  uint64_t R_lower = _mm_extract_epi64(in, 0);
  uint64_t R_upper = ((_mm_extract_epi64(in, 1) & upper_mask) << 13) |
                     (R_lower >> 51); // build the part to reduce from the upper
                                      // word + remaining from lower word
  // reduce
  uint64_t T = R_upper;
  R_upper = R_upper ^ (T >> 45) ^ (T >> 48) ^ (T >> 50);
  R_lower = R_lower ^ (R_upper << 6) ^ (R_upper << 3) ^ (R_upper << 1) ^
            (R_upper << 0);
  return lower_mask & R_lower;
}

inline uint64_t gf51mul(const uint64_t in1, const uint64_t in2) {
  __m128i tmp = clmul(in1, in2);
  return reduce_GF2_51(tmp);
}

} // namespace

namespace field {

GF2_51::GF2_51(std::string hex_string) {
  // check if hex_string start with 0x or 0X
  if (hex_string.rfind("0x", 0) == 0 || hex_string.rfind("0X", 0) == 0) {
    hex_string = hex_string.substr(2);
  } else {
    throw std::runtime_error("input needs to be a hex number");
  }
  constexpr size_t num_hex_chars = (51 + 3) / 4;
  if (hex_string.length() > num_hex_chars)
    throw std::runtime_error("input hex is too large");
  // pad to 52 bit
  hex_string.insert(hex_string.begin(), num_hex_chars - hex_string.length(),
                    '0');
  uint64_t value = std::stoull(hex_string, nullptr, 16);
  data = value;
}

GF2_51 GF2_51::operator+(const GF2_51 &other) const {
  return GF2_51(this->data ^ other.data);
}
GF2_51 &GF2_51::operator+=(const GF2_51 &other) {
  this->data ^= other.data;
  return *this;
}
GF2_51 GF2_51::operator-(const GF2_51 &other) const {
  return GF2_51(this->data ^ other.data);
}
GF2_51 &GF2_51::operator-=(const GF2_51 &other) {
  this->data ^= other.data;
  return *this;
}
GF2_51 GF2_51::operator*(const GF2_51 &other) const {
  return GF2_51(gf51mul(this->data, other.data));
}
GF2_51 &GF2_51::operator*=(const GF2_51 &other) {
  this->data = gf51mul(this->data, other.data);
  return *this;
}
bool GF2_51::operator==(const GF2_51 &other) const {
  return this->data == other.data;
}
bool GF2_51::operator!=(const GF2_51 &other) const {
  return this->data != other.data;
}

GF2_51 GF2_51::inverse() const {
  // Fixed-op square-multiply
  // 2^n - 2 in binary is 0b1111..10
  uint64_t t1 = this->data;
  uint64_t t2 = gf51mul(t1, t1);

  // First 49 one-bits (start from second)
  for (size_t i = 0; i < 49; i++) {
    t1 = gf51mul(t1, t2);
    t2 = gf51mul(t2, t2);
  }

  // Final zero-bit
  t1 = gf51mul(t1, t1);
  return GF2_51(t1);
}

void GF2_51::to_bytes(uint8_t *out) const {
  uint64_t le_data = htole64(data);
  memcpy(out, (uint8_t *)(&le_data), BYTE_SIZE);
}

void GF2_51::from_bytes(const uint8_t *in) {
  uint64_t tmp;
  memcpy((uint8_t *)(&tmp), in, BYTE_SIZE);
  data = le64toh(tmp) & ELEMENT_MASK;
}

} // namespace field

// somewhat optimized inner product, only do one lazy reduction
field::GF2_51 dot_product(const std::vector<field::GF2_51> &lhs,
                          const std::vector<field::GF2_51> &rhs) {

  if (lhs.size() != rhs.size())
    throw std::runtime_error("adding vectors of different sizes");

  __m128i accum = _mm_setzero_si128();
  for (size_t i = 0; i < lhs.size(); i++) {
    accum = _mm_xor_si128(accum, clmul(lhs[i].data, rhs[i].data));
  }

  // combined reduction
  return field::GF2_51(reduce_GF2_51(accum));
}

std::ostream &operator<<(std::ostream &os, const field::GF2_51 &ele) {
  os << "0x" << std::setfill('0') << std::hex << std::setw(16) << ele.data;
  return os;
}

// TEMPLATE INSTANTIATIONS for GF2_51

// yes we include the cpp file with the template stuff
#include "field_templates.cpp"

INSTANTIATE_TEMPLATES_FOR(field::GF2_51)
// END TEMPLATE INSTANTIATIONS for GF2_51