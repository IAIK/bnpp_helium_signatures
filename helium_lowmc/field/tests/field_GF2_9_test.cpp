#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include "../field.h"
#include "utils.h"

#include <NTL/GF2EX.h>

TEST_CASE("Constructors for GF(2^9)", "[GF2_9]") {
  field::GF2_9 a;
  a.set_coeff(2);
  a.set_coeff(0);

  field::GF2_9 a_int(0x5);
  REQUIRE(a == a_int);
}

TEST_CASE("Basic Arithmetic in GF(2^9)", "[GF2_9]") {
  field::GF2_9 zero;
  field::GF2_9 one(1);
  field::GF2_9 x(2);
  field::GF2_9 x_2(4);

  REQUIRE(one + zero == one);
  REQUIRE(zero + one == one);
  REQUIRE(zero + zero == zero);
  REQUIRE(one + one == zero);

  REQUIRE(one * one == one);
  REQUIRE(zero * one == zero);
  REQUIRE(one * zero == zero);
  REQUIRE(zero * zero == zero);

  REQUIRE(x * one == x);
  REQUIRE(x * x == x_2);
}
TEST_CASE("Modular Arithmetic KATs GF(2^9)", "[GF2_9]") {
  field::GF2_9 a, b;
  a.set_coeff(8);
  a.set_coeff(4);
  a.set_coeff(2);
  a.set_coeff(1);
  a.set_coeff(0);

  b.set_coeff(8);
  b.set_coeff(6);
  b.set_coeff(4);
  b.set_coeff(3);
  b.set_coeff(1);

  field::GF2_9 a_int(0x117), b_int(0x15a);
  REQUIRE(a == a_int);
  REQUIRE(b == b_int);

  field::GF2_9 ab(0xca);
  field::GF2_9 aplusb(0x4d);
  field::GF2_9 a_2(0x95);
  field::GF2_9 b_2(0xdc);
  field::GF2_9 ab_calc = a_int * b_int;
  REQUIRE(ab_calc == ab);
  REQUIRE(a * a == a_2);
  REQUIRE(b * b == b_2);
  REQUIRE(a + b == aplusb);

  BENCHMARK("GF Addition") { return a + b; };
  BENCHMARK("GF Multiplication") { return a * b; };
}

TEST_CASE("NTL to_bytes = custom to_bytes GF(2^9", "[field]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_9);
  field::GF2_9 a, b;
  a.set_coeff(8);
  a.set_coeff(4);
  a.set_coeff(2);
  a.set_coeff(1);
  a.set_coeff(0);

  b.set_coeff(8);
  b.set_coeff(6);
  b.set_coeff(4);
  b.set_coeff(3);
  b.set_coeff(1);
  GF2X c, d;
  SetCoeff(c, 8);
  SetCoeff(c, 4);
  SetCoeff(c, 2);
  SetCoeff(c, 1);
  SetCoeff(c, 0);
  GF2E c_e = conv<GF2E>(c);

  SetCoeff(d, 8);
  SetCoeff(d, 6);
  SetCoeff(d, 4);
  SetCoeff(d, 3);
  SetCoeff(d, 1);
  GF2E d_e = conv<GF2E>(d);

  const GF2X &poly_rep_c = rep(c_e);
  std::vector<uint8_t> buffer_c(field::GF2_9::BYTE_SIZE);
  BytesFromGF2X(buffer_c.data(), poly_rep_c, buffer_c.size());
  const GF2X &poly_rep_d = rep(d_e);
  std::vector<uint8_t> buffer_d(field::GF2_9::BYTE_SIZE);
  BytesFromGF2X(buffer_d.data(), poly_rep_d, buffer_d.size());

  std::vector<uint8_t> buffer_a(field::GF2_9::BYTE_SIZE);
  a.to_bytes(buffer_a.data());
  std::vector<uint8_t> buffer_b(field::GF2_9::BYTE_SIZE);
  b.to_bytes(buffer_b.data());
  REQUIRE(buffer_a == buffer_c);
  REQUIRE(buffer_b == buffer_d);
}
TEST_CASE("NTL to custom conversion GF(2^9)", "[GF2_9]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_9);
  field::GF2_9 a, b;
  a.set_coeff(8);
  a.set_coeff(4);
  a.set_coeff(2);
  a.set_coeff(1);
  a.set_coeff(0);

  b.set_coeff(8);
  b.set_coeff(6);
  b.set_coeff(4);
  b.set_coeff(3);
  b.set_coeff(1);

  field::GF2_9 ab = a * b;
  GF2E a_ntl = utils::custom_to_ntl(a);
  GF2E ab_ntl = utils::custom_to_ntl(ab);
  GF2E b_ntl = ab_ntl / a_ntl;
  field::GF2_9 b2 = utils::ntl_to_custom<field::GF2_9>(b_ntl);
  REQUIRE(b == b2);
}

TEST_CASE("NTL inverse == custom inverse GF(2^9)", "[GF2_9]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_9);
  field::GF2_9 a;
  a.set_coeff(8);
  a.set_coeff(4);
  a.set_coeff(2);
  a.set_coeff(1);
  a.set_coeff(0);

  field::GF2_9 b = a.inverse();
  field::GF2_9 c =
      utils::ntl_to_custom<field::GF2_9>(inv(utils::custom_to_ntl(a)));

  REQUIRE(b == c);
  REQUIRE(a * b == field::GF2_9(1));
  BENCHMARK("GF inverse") { return a.inverse(); };
}
TEST_CASE("NTL interpolation == custom interpolation GF(2^9)", "[GF2_9]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_9);

  std::vector<field::GF2_9> a =
      field::get_first_n_field_elements<field::GF2_9>(100);
  vec_GF2E b = utils::get_first_n_field_elements(100);
  for (size_t i = 0; i < 100; i++) {
    REQUIRE(a[i] == utils::ntl_to_custom<field::GF2_9>(b[i]));
  }
  std::vector<field::GF2_9> a_from_roots = field::build_from_roots(a);
  GF2EX b_from_roots = BuildFromRoots(b);
  REQUIRE(a_from_roots.size() == (size_t)b_from_roots.rep.length());
  for (size_t j = 0; j < a_from_roots.size(); j++) {
    REQUIRE(a_from_roots[j] ==
            utils::ntl_to_custom<field::GF2_9>(b_from_roots[j]));
  }

  std::vector<std::vector<field::GF2_9>> a_lag =
      field::precompute_lagrange_polynomials(a);
  std::vector<GF2EX> b_lag = utils::precompute_lagrange_polynomials(b);

  REQUIRE(a_lag.size() == b_lag.size());
  for (size_t i = 0; i < a_lag.size(); i++) {
    REQUIRE(a_lag[i].size() == (size_t)b_lag[i].rep.length());
    for (size_t j = 0; j < a_lag[i].size(); j++) {
      REQUIRE(a_lag[i][j] == utils::ntl_to_custom<field::GF2_9>(b_lag[i][j]));
    }
  }

  BENCHMARK("Lagrange Poly Precomputation") {
    return field::precompute_lagrange_polynomials(a);
  };
  BENCHMARK("Lagrange Poly Interpolation") {
    return field::interpolate_with_precomputation(a_lag, a);
  };
}

TEST_CASE("NTL dot product == custom GF(2^9)", "[GF2_9]") {

  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_9);
  std::vector<field::GF2_9> a =
      field::get_first_n_field_elements<field::GF2_9>(100);
  std::vector<field::GF2_9> b =
      field::get_first_n_field_elements<field::GF2_9>(200);
  b.erase(b.begin(), b.begin() + 100);

  vec_GF2E a_ntl = utils::get_first_n_field_elements(200);
  vec_GF2E b_ntl;
  b_ntl.SetLength(100);
  for (size_t i = 0; i < 100; i++)
    b_ntl[i] = a_ntl[100 + i];
  a_ntl.SetLength(100);

  field::GF2_9 result = dot_product(a, b);
  GF2E result_ntl;
  NTL::InnerProduct(result_ntl, a_ntl, b_ntl);
  REQUIRE(result == utils::ntl_to_custom<field::GF2_9>(result_ntl));
}