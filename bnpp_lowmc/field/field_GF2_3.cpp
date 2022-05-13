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

inline uint8_t gf3mul(uint8_t a, uint8_t b) {
  // should be const time, but probably slower than other variants
  // maybe just do a lookup table inside an uint64_t?
  unsigned char result = 0;
  for (int i = 0; i < 3; ++i) {
    uint8_t mask = -(b & 1);
    result ^= (a & mask);
    uint8_t mask2 = -((a >> 2) & 1);
    a <<= 1;
    // reduce by x^3 + x + 1 if needed
    a ^= (0xb & mask2);
    b >>= 1;
  }
  return result;
}

} // namespace

namespace field {

GF2_3 GF2_3::operator+(const GF2_3 &other) const {
  return GF2_3(this->data ^ other.data);
}
GF2_3 &GF2_3::operator+=(const GF2_3 &other) {
  this->data ^= other.data;
  return *this;
}
GF2_3 GF2_3::operator-(const GF2_3 &other) const {
  return GF2_3(this->data ^ other.data);
}
GF2_3 &GF2_3::operator-=(const GF2_3 &other) {
  this->data ^= other.data;
  return *this;
}
GF2_3 GF2_3::operator*(const GF2_3 &other) const {
  return GF2_3(gf3mul(this->data, other.data));
}
GF2_3 &GF2_3::operator*=(const GF2_3 &other) {
  this->data = gf3mul(this->data, other.data);
  return *this;
}
bool GF2_3::operator==(const GF2_3 &other) const {
  return this->data == other.data;
}
bool GF2_3::operator!=(const GF2_3 &other) const {
  return this->data != other.data;
}

void GF2_3::to_bytes(uint8_t *out) const { *out = this->data; }

void GF2_3::from_bytes(const uint8_t *in) { this->data = (*in) & ELEMENT_MASK; }

} // namespace field
