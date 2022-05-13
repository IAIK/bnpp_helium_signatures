#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include "../field.h"
#include "utils.h"

#include <NTL/GF2EX.h>

TEST_CASE("Constructors for GF(2^51)", "[GF2_3]") {
  field::GF2_3 a;
  a.set_coeff(2);
  a.set_coeff(0);

  field::GF2_3 a_int(0x5);
  REQUIRE(a == a_int);
}

TEST_CASE("Basic Arithmetic in GF(2^3)", "[GF2_3]") {
  field::GF2_3 zero;
  field::GF2_3 one(1);
  field::GF2_3 x(2);
  field::GF2_3 x_2(4);

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
TEST_CASE("Modular Arithmetic KATs GF(2^51)", "[GF2_3]") {
  field::GF2_3 a, b;
  a.set_coeff(2);
  a.set_coeff(0);

  b.set_coeff(2);
  b.set_coeff(1);

  field::GF2_3 a_int(0x5), b_int(0x6);
  REQUIRE(a == a_int);
  REQUIRE(b == b_int);

  field::GF2_3 ab(0x3);
  field::GF2_3 a_2(0x7);
  field::GF2_3 b_2(0x2);
  field::GF2_3 ab_calc = a_int * b_int;
  REQUIRE(ab_calc == ab);
  REQUIRE(a * a == a_2);
  REQUIRE(b * b == b_2);

  BENCHMARK("GF Addition") { return a + b; };
  BENCHMARK("GF Multiplication") { return a * b; };
}

TEST_CASE("NTL to_bytes = custom to_bytes GF(2^51)", "[field]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_3);
  field::GF2_3 a, b;
  a.set_coeff(2);
  a.set_coeff(0);

  b.set_coeff(2);
  b.set_coeff(1);
  GF2X c, d;
  SetCoeff(c, 2);
  SetCoeff(c, 0);
  GF2E c_e = conv<GF2E>(c);

  SetCoeff(d, 2);
  SetCoeff(d, 1);
  GF2E d_e = conv<GF2E>(d);

  const GF2X &poly_rep_c = rep(c_e);
  std::vector<uint8_t> buffer_c(field::GF2_3::BYTE_SIZE);
  BytesFromGF2X(buffer_c.data(), poly_rep_c, buffer_c.size());
  const GF2X &poly_rep_d = rep(d_e);
  std::vector<uint8_t> buffer_d(field::GF2_3::BYTE_SIZE);
  BytesFromGF2X(buffer_d.data(), poly_rep_d, buffer_d.size());

  std::vector<uint8_t> buffer_a(field::GF2_3::BYTE_SIZE);
  a.to_bytes(buffer_a.data());
  std::vector<uint8_t> buffer_b(field::GF2_3::BYTE_SIZE);
  b.to_bytes(buffer_b.data());
  REQUIRE(buffer_a == buffer_c);
  REQUIRE(buffer_b == buffer_d);
}
TEST_CASE("NTL to custom conversion GF(2^3)", "[GF2_3]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_3);
  field::GF2_3 a, b;
  a.set_coeff(2);
  a.set_coeff(0);

  b.set_coeff(2);
  b.set_coeff(1);

  field::GF2_3 ab = a * b;
  GF2E a_ntl = utils::custom_to_ntl(a);
  GF2E ab_ntl = utils::custom_to_ntl(ab);
  GF2E b_ntl = ab_ntl / a_ntl;
  field::GF2_3 b2 = utils::ntl_to_custom<field::GF2_3>(b_ntl);
  REQUIRE(b == b2);
}
