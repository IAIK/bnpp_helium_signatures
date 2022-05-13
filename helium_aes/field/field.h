#pragma once

#include "../gsl-lite.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

extern "C" {
#include <emmintrin.h>
#include <immintrin.h>
}

namespace field {
class GF2_8;
class GF2_128;
class GF2_144;
} // namespace field

field::GF2_8 dot_product(const std::vector<field::GF2_8> &lhs,
                         const std::vector<field::GF2_8> &rhs);
field::GF2_128 dot_product(const std::vector<field::GF2_128> &lhs,
                           const std::vector<field::GF2_128> &rhs);
field::GF2_128
lifted_dot_product_uint8(const gsl::span<const field::GF2_128> &lhs,
                         const gsl::span<const uint8_t> &rhs);
field::GF2_128 lifted_dot_product(const gsl::span<const field::GF2_128> &lhs,
                                  const gsl::span<const field::GF2_8> &rhs);
field::GF2_144 dot_product(const std::vector<field::GF2_144> &lhs,
                           const std::vector<field::GF2_144> &rhs);
field::GF2_144
lifted_dot_product_uint8(const gsl::span<const field::GF2_144> &lhs,
                         const gsl::span<const uint8_t> &rhs);
field::GF2_144 lifted_dot_product(const gsl::span<const field::GF2_144> &lhs,
                                  const gsl::span<const field::GF2_8> &rhs);
namespace field {

class GF2_8 {
public:
  uint8_t data;
  constexpr static size_t BYTE_SIZE = 1;
  constexpr static size_t BIT_SIZE = 8;
  constexpr GF2_8() : data{} {};
  constexpr GF2_8(uint8_t data) : data{data} {}
  GF2_8(const GF2_8 &other) = default;
  ~GF2_8() = default;
  GF2_8 &operator=(const GF2_8 &other) = default;

  inline void clear() { data = 0; }
  inline bool is_zero() const { return data == 0; }
  inline void set_coeff(size_t idx) { data |= (1ULL << (idx % 8)); }

  GF2_8 operator+(const GF2_8 &other) const;
  GF2_8 &operator+=(const GF2_8 &other);
  GF2_8 operator-(const GF2_8 &other) const;
  GF2_8 &operator-=(const GF2_8 &other);
  GF2_8 operator*(const GF2_8 &other) const;
  GF2_8 &operator*=(const GF2_8 &other);
  GF2_8 operator-() const { return *this; };
  bool operator==(const GF2_8 &other) const;
  bool operator!=(const GF2_8 &other) const;

  void to_bytes(uint8_t *out) const;
  void from_bytes(const uint8_t *in);

  GF2_8 inverse() const;
  friend GF2_8(::dot_product)(const std::vector<field::GF2_8> &lhs,
                              const std::vector<field::GF2_8> &rhs);
  friend GF2_128(::lifted_dot_product)(
      const gsl::span<const field::GF2_128> &lhs,
      const gsl::span<const field::GF2_8> &rhs);
  friend GF2_144(::lifted_dot_product)(
      const gsl::span<const field::GF2_144> &lhs,
      const gsl::span<const field::GF2_8> &rhs);
};

class GF2_128 {
  alignas(16) std::array<uint64_t, 2> data;

public:
  // helper functions for sse stuff
  inline __m128i *as_m128i() {
    return reinterpret_cast<__m128i *>(data.data());
  };
  inline const __m128i *as_const_m128i() const {
    return reinterpret_cast<const __m128i *>(data.data());
  };

  constexpr static size_t BYTE_SIZE = 16;
  constexpr GF2_128() : data{} {};
  constexpr GF2_128(uint64_t data) : data{data, 0} {}
  constexpr GF2_128(std::array<uint64_t, 2> data) : data(data) {}
  GF2_128(std::string hex_string);
  GF2_128(const GF2_128 &other) = default;
  ~GF2_128() = default;
  GF2_128 &operator=(const GF2_128 &other) = default;

  inline void clear() { data = {}; }
  inline bool is_zero() const { return data[0] == 0 && data[1] == 0; }
  inline void set_coeff(size_t idx) { data[idx / 64] |= (1ULL << (idx % 64)); }

  GF2_128 operator+(const GF2_128 &other) const;
  GF2_128 &operator+=(const GF2_128 &other);
  GF2_128 operator-(const GF2_128 &other) const;
  GF2_128 &operator-=(const GF2_128 &other);
  GF2_128 operator*(const GF2_128 &other) const;
  GF2_128 &operator*=(const GF2_128 &other);
  bool operator==(const GF2_128 &other) const;
  bool operator!=(const GF2_128 &other) const;

  GF2_128 inverse() const;
  GF2_128 inverse_slow() const;
  GF2_128 multiply_with_GF2_matrix(const std::array<uint64_t, 2> *matrix) const;
  GF2_128 multiply_with_transposed_GF2_matrix(
      const std::array<uint64_t, 2> *matrix) const;

  void to_bytes(uint8_t *out) const;
  void from_bytes(const uint8_t *in);

  friend GF2_128(::dot_product)(const std::vector<field::GF2_128> &lhs,
                                const std::vector<field::GF2_128> &rhs);
  friend GF2_128(::lifted_dot_product_uint8)(
      const gsl::span<const field::GF2_128> &lhs,
      const gsl::span<const uint8_t> &rhs);
  friend GF2_128(::lifted_dot_product)(
      const gsl::span<const field::GF2_128> &lhs,
      const gsl::span<const field::GF2_8> &rhs);
};

class GF2_144 {

public:
  alignas(32) std::array<uint64_t, 4> data; // waste a word for alignment

  // helper functions for sse stuff
  inline __m128i *as_m128i() {
    return reinterpret_cast<__m128i *>(data.data());
  };
  inline const __m128i *as_const_m128i() const {
    return reinterpret_cast<const __m128i *>(data.data());
  };
  // helper functions for avx2 stuff
  inline __m256i *as_m256i() {
    return reinterpret_cast<__m256i *>(data.data());
  };
  inline const __m256i *as_const_m256i() const {
    return reinterpret_cast<const __m256i *>(data.data());
  };

  constexpr static size_t BYTE_SIZE = 18;
  constexpr static size_t BIT_SIZE = 144;
  constexpr GF2_144() : data{} {};
  constexpr GF2_144(uint64_t data) : data{data, 0, 0, 0} {}
  constexpr GF2_144(std::array<uint64_t, 4> data) : data{data} {}
  GF2_144(std::string hex_string);
  GF2_144(const GF2_144 &other) = default;
  ~GF2_144() = default;
  GF2_144 &operator=(const GF2_144 &other) = default;

  inline void clear() { data = {0, 0, 0, 0}; }
  inline bool is_zero() const {
    return data[0] == 0 && data[1] == 0 && data[2] == 0;
  }
  inline void set_coeff(size_t idx) { data[idx / 64] |= (1ULL << (idx % 64)); }

  GF2_144 operator+(const GF2_144 &other) const;
  GF2_144 &operator+=(const GF2_144 &other);
  GF2_144 operator-(const GF2_144 &other) const;
  GF2_144 &operator-=(const GF2_144 &other);
  GF2_144 operator*(const GF2_144 &other) const;
  GF2_144 &operator*=(const GF2_144 &other);
  GF2_144 operator-() const { return *this; };
  bool operator==(const GF2_144 &other) const;
  bool operator!=(const GF2_144 &other) const;

  GF2_144 inverse() const;
  GF2_144 multiply_with_GF2_matrix(const std::array<uint64_t, 2> *matrix) const;
  GF2_144 multiply_with_transposed_GF2_matrix(
      const std::array<uint64_t, 2> *matrix) const;

  void to_bytes(uint8_t *out) const;
  void from_bytes(const uint8_t *in);

  friend GF2_144(::dot_product)(const std::vector<field::GF2_144> &lhs,
                                const std::vector<field::GF2_144> &rhs);
  friend GF2_144(::lifted_dot_product_uint8)(
      const gsl::span<const field::GF2_144> &lhs,
      const gsl::span<const uint8_t> &rhs);
  friend GF2_144(::lifted_dot_product)(
      const gsl::span<const field::GF2_144> &lhs,
      const gsl::span<const field::GF2_8> &rhs);
};

std::vector<field::GF2_8> interpolate_with_precomputation(
    const std::array<std::array<field::GF2_8, 100>, 100>
        &precomputed_lagrange_polynomials,
    const std::vector<field::GF2_8> &y_values);

field::GF2_128 eval_lifted(const gsl::span<const field::GF2_8> &poly,
                           const field::GF2_128 &point);
field::GF2_144 eval_lifted(const gsl::span<const field::GF2_8> &poly,
                           const field::GF2_144 &point);

template <typename GF> std::vector<GF> get_first_n_field_elements(size_t n);
template <typename GF>
std::vector<std::vector<GF>>
precompute_lagrange_polynomials(const std::vector<GF> &x_values);
template <typename GF>
std::vector<GF> interpolate_with_precomputation(
    const std::vector<std::vector<GF>> &precomputed_lagrange_polynomials,
    const std::vector<GF> &y_values);

template <typename GF>
std::vector<GF> build_from_roots(const std::vector<GF> &roots);

template <typename GF> GF eval(const std::vector<GF> &poly, const GF &point);
} // namespace field

template <typename GF>
std::vector<GF> operator+(const std::vector<GF> &lhs,
                          const std::vector<GF> &rhs);
template <typename GF>
std::vector<GF> &operator+=(std::vector<GF> &self, const std::vector<GF> &rhs);
template <typename GF>
std::vector<GF> operator*(const std::vector<GF> &lhs, const GF &rhs);
template <typename GF>
std::vector<GF> operator*(const GF &lhs, const std::vector<GF> &rhs);
template <typename GF>
std::vector<GF> operator*(const std::vector<GF> &lhs,
                          const std::vector<GF> &rhs);
