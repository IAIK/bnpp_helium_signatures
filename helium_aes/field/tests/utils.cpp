#include "utils.h"

#include <NTL/GF2EX.h>
#include <NTL/GF2X.h>
#include <stdexcept>

using namespace NTL;

namespace utils {

static GF2X modulus;

void init_ntl_extension_field(NTL_INSTANCE instance) {
  switch (instance) {
  case GF2_8: {
    // modulus = x^8 + x^4 + x^3 + x^1 + 1
    clear(modulus);
    SetCoeff(modulus, 8);
    SetCoeff(modulus, 4);
    SetCoeff(modulus, 3);
    SetCoeff(modulus, 1);
    SetCoeff(modulus, 0);
    GF2E::init(modulus);
  } break;
  case GF2_128: {
    // modulus = x^128 + x^7 + x^2 + x^1 + 1
    clear(modulus);
    SetCoeff(modulus, 128);
    SetCoeff(modulus, 7);
    SetCoeff(modulus, 2);
    SetCoeff(modulus, 1);
    SetCoeff(modulus, 0);
    GF2E::init(modulus);
  } break;
  case GF2_144: {
    // modulus = x^144 + x^7 + x^4 + x^2 + 1
    clear(modulus);
    SetCoeff(modulus, 144);
    SetCoeff(modulus, 7);
    SetCoeff(modulus, 4);
    SetCoeff(modulus, 2);
    SetCoeff(modulus, 0);
    GF2E::init(modulus);
  } break;
  default:
    throw std::runtime_error("instance not implemented.");
  }
}

GF2E GF2E_from_bytes(const std::vector<uint8_t> &value) {
  // assumes value is already smaller than current modulus
  GF2X inner = GF2XFromBytes(value.data(), value.size());
  // GF2E result(INIT_NO_ALLOC);
  // result.LoopHole() = inner;
  // return result;
  return conv<GF2E>(inner);
}

vec_GF2E get_first_n_field_elements(size_t n) {
  vec_GF2E result;
  result.SetLength(n);
  for (size_t i = 0; i < n; i++) {
    size_t ii = i;
    GF2X tmp;
    for (size_t j = 0; ii != 0; j++, ii >>= 1) {
      if (ii & 1) {
        SetCoeff(tmp, j);
      }
    }
    result[i] = conv<GF2E>(tmp);
  }
  return result;
}
std::vector<GF2EX> precompute_lagrange_polynomials(const vec_GF2E &x_values) {
  size_t m = x_values.length();
  std::vector<GF2EX> precomputed_lagrange_polynomials;
  precomputed_lagrange_polynomials.reserve(m);

  GF2EX full_poly = BuildFromRoots(x_values);
  GF2EX lagrange_poly;
  GF2EX missing_term;
  SetX(missing_term);
  for (size_t k = 0; k < m; k++) {
    SetCoeff(missing_term, 0, -x_values[k]);
    lagrange_poly = full_poly / missing_term;
    lagrange_poly = lagrange_poly / eval(lagrange_poly, x_values[k]);
    precomputed_lagrange_polynomials.push_back(lagrange_poly);
  }

  return precomputed_lagrange_polynomials;
}

GF2EX interpolate_with_precomputation(
    const std::vector<GF2EX> &precomputed_lagrange_polynomials,
    const vec_GF2E &y_values) {
  if (precomputed_lagrange_polynomials.size() != (size_t)y_values.length())
    throw std::runtime_error("invalid sizes for interpolation");

  GF2EX res;
  size_t m = y_values.length();
  for (size_t k = 0; k < m; k++) {
    res += precomputed_lagrange_polynomials[k] * y_values[k];
  }
  return res;
}
} // namespace utils
