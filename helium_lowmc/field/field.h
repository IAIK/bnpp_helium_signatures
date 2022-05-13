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
class GF2_3;
class GF2_9;
class GF2_45;
class GF2_51;
class GF2_144;
} // namespace field

field::GF2_9 dot_product(const std::vector<field::GF2_9> &lhs,
                         const std::vector<field::GF2_9> &rhs);
field::GF2_45 dot_product(const std::vector<field::GF2_45> &lhs,
                          const std::vector<field::GF2_45> &rhs);
field::GF2_51 dot_product(const std::vector<field::GF2_51> &lhs,
                          const std::vector<field::GF2_51> &rhs);
field::GF2_144 dot_product(const std::vector<field::GF2_144> &lhs,
                           const std::vector<field::GF2_144> &rhs);
field::GF2_144 lifted_dot_product(const gsl::span<const field::GF2_144> &lhs,
                                  const gsl::span<const field::GF2_9> &rhs);
std::ostream &operator<<(std::ostream &os, const field::GF2_45 &ele);
std::ostream &operator<<(std::ostream &os, const field::GF2_51 &ele);

namespace field {

// trimmed-down version of the small field, only implements field ops
class GF2_3 {
public:
  uint8_t data;
  constexpr static size_t BYTE_SIZE = 1;
  constexpr static size_t BIT_SIZE = 3;
  constexpr static uint8_t ELEMENT_MASK = 0x7ULL;
  constexpr GF2_3() : data{} {};
  constexpr GF2_3(uint8_t data) : data{data} {}
  GF2_3(const GF2_3 &other) = default;
  ~GF2_3() = default;
  GF2_3 &operator=(const GF2_3 &other) = default;

  inline void clear() { data = 0; }
  inline bool is_zero() const { return data == 0; }
  inline void set_coeff(size_t idx) { data |= (1ULL << (idx % 8)); }

  GF2_3 operator+(const GF2_3 &other) const;
  GF2_3 &operator+=(const GF2_3 &other);
  GF2_3 operator-(const GF2_3 &other) const;
  GF2_3 &operator-=(const GF2_3 &other);
  GF2_3 operator*(const GF2_3 &other) const;
  GF2_3 &operator*=(const GF2_3 &other);
  GF2_3 operator-() const { return *this; };
  bool operator==(const GF2_3 &other) const;
  bool operator!=(const GF2_3 &other) const;

  void to_bytes(uint8_t *out) const;
  void from_bytes(const uint8_t *in);
};

class GF2_9 {
public:
  uint16_t data;
  constexpr static size_t BYTE_SIZE = 2;
  constexpr static size_t BIT_SIZE = 9;
  constexpr static uint16_t ELEMENT_MASK = 0x1FFULL;
  constexpr GF2_9() : data{} {};
  constexpr GF2_9(uint16_t data) : data{data} {}
  GF2_9(const GF2_9 &other) = default;
  ~GF2_9() = default;
  GF2_9 &operator=(const GF2_9 &other) = default;

  inline void clear() { data = 0; }
  inline bool is_zero() const { return data == 0; }
  inline void set_coeff(size_t idx) { data |= (1ULL << (idx % 16)); }

  GF2_9 operator+(const GF2_9 &other) const;
  GF2_9 &operator+=(const GF2_9 &other);
  GF2_9 operator-(const GF2_9 &other) const;
  GF2_9 &operator-=(const GF2_9 &other);
  GF2_9 operator*(const GF2_9 &other) const;
  GF2_9 &operator*=(const GF2_9 &other);
  GF2_9 operator-() const { return *this; };
  bool operator==(const GF2_9 &other) const;
  bool operator!=(const GF2_9 &other) const;

  void to_bytes(uint8_t *out) const;
  void from_bytes(const uint8_t *in);

  GF2_9 inverse() const;
  friend GF2_9(::dot_product)(const std::vector<field::GF2_9> &lhs,
                              const std::vector<field::GF2_9> &rhs);
  friend GF2_144(::lifted_dot_product)(
      const gsl::span<const field::GF2_144> &lhs,
      const gsl::span<const field::GF2_9> &rhs);
};

class GF2_45 {

public:
  uint64_t data;

  constexpr static size_t BYTE_SIZE = 6;
  constexpr static size_t BIT_SIZE = 45;
  constexpr static uint64_t ELEMENT_MASK = 0x1FFFFFFFFFFFULL;
  constexpr GF2_45() : data{} {};
  constexpr GF2_45(uint64_t data) : data{data} {}
  GF2_45(std::string hex_string);
  GF2_45(const GF2_45 &other) = default;
  ~GF2_45() = default;
  GF2_45 &operator=(const GF2_45 &other) = default;

  inline void clear() { data = 0; }
  inline bool is_zero() const { return data == 0; }
  inline void set_coeff(size_t idx) { data |= (1ULL << (idx % 64)); }

  GF2_45 operator+(const GF2_45 &other) const;
  GF2_45 &operator+=(const GF2_45 &other);
  GF2_45 operator-(const GF2_45 &other) const;
  GF2_45 &operator-=(const GF2_45 &other);
  GF2_45 operator*(const GF2_45 &other) const;
  GF2_45 &operator*=(const GF2_45 &other);
  GF2_45 operator-() const { return *this; };
  bool operator==(const GF2_45 &other) const;
  bool operator!=(const GF2_45 &other) const;

  GF2_45 inverse() const;
  GF2_45 multiply_with_GF2_matrix(const std::array<uint64_t, 2> *matrix) const;
  GF2_45 multiply_with_transposed_GF2_matrix(
      const std::array<uint64_t, 2> *matrix) const;

  void to_bytes(uint8_t *out) const;
  void from_bytes(const uint8_t *in);

  friend GF2_45(::dot_product)(const std::vector<field::GF2_45> &lhs,
                               const std::vector<field::GF2_45> &rhs);
  friend std::ostream &(::operator<<)(std::ostream &os,
                                      const field::GF2_45 &ele);
};

class GF2_51 {

public:
  uint64_t data;

  constexpr static size_t BYTE_SIZE = 7;
  constexpr static size_t BIT_SIZE = 51;
  constexpr static uint64_t ELEMENT_MASK = 0x7FFFFFFFFFFFFULL;
  constexpr GF2_51() : data{} {};
  constexpr GF2_51(uint64_t data) : data{data} {}
  GF2_51(std::string hex_string);
  GF2_51(const GF2_51 &other) = default;
  ~GF2_51() = default;
  GF2_51 &operator=(const GF2_51 &other) = default;

  inline void clear() { data = 0; }
  inline bool is_zero() const { return data == 0; }
  inline void set_coeff(size_t idx) { data |= (1ULL << (idx % 64)); }

  GF2_51 operator+(const GF2_51 &other) const;
  GF2_51 &operator+=(const GF2_51 &other);
  GF2_51 operator-(const GF2_51 &other) const;
  GF2_51 &operator-=(const GF2_51 &other);
  GF2_51 operator*(const GF2_51 &other) const;
  GF2_51 &operator*=(const GF2_51 &other);
  GF2_51 operator-() const { return *this; };
  bool operator==(const GF2_51 &other) const;
  bool operator!=(const GF2_51 &other) const;

  GF2_51 inverse() const;
  GF2_51 multiply_with_GF2_matrix(const std::array<uint64_t, 2> *matrix) const;
  GF2_51 multiply_with_transposed_GF2_matrix(
      const std::array<uint64_t, 2> *matrix) const;

  void to_bytes(uint8_t *out) const;
  void from_bytes(const uint8_t *in);

  friend GF2_51(::dot_product)(const std::vector<field::GF2_51> &lhs,
                               const std::vector<field::GF2_51> &rhs);
  friend std::ostream &(::operator<<)(std::ostream &os,
                                      const field::GF2_51 &ele);
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
  friend GF2_144(::lifted_dot_product)(
      const gsl::span<const field::GF2_144> &lhs,
      const gsl::span<const field::GF2_9> &rhs);
};

GF2_144 eval_lifted(const gsl::span<const GF2_9> &poly, const GF2_144 &point);

template <typename GF> std::vector<GF> get_first_n_field_elements(size_t n);
template <typename GF>
std::vector<std::vector<GF>>
precompute_lagrange_polynomials(const std::vector<GF> &x_values);
template <typename GF>
std::vector<GF> interpolate_with_precomputation(
    const std::vector<std::vector<GF>> &precomputed_lagrange_polynomials,
    const std::vector<GF> &y_values);

template <typename GF, size_t N>
std::vector<GF> interpolate_with_precomputation(
    const std::array<std::array<GF, N>, N> &precomputed_lagrange_polynomials,
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
template <typename GF, size_t N>
std::vector<GF> operator*(const std::array<GF, N> &lhs, const GF &rhs);
template <typename GF>
std::vector<GF> operator*(const GF &lhs, const std::vector<GF> &rhs);
template <typename GF>
std::vector<GF> operator*(const std::vector<GF> &lhs,
                          const std::vector<GF> &rhs);
