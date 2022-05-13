#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include "../field.h"
#include "utils.h"

#include <NTL/GF2EX.h>

TEST_CASE("Constructors for GF(2^128)", "[GF2_128]") {
  field::GF2_128 a;
  a.set_coeff(126);
  a.set_coeff(125);
  a.set_coeff(124);
  a.set_coeff(123);
  a.set_coeff(121);
  a.set_coeff(120);
  a.set_coeff(118);
  a.set_coeff(116);
  a.set_coeff(115);
  a.set_coeff(113);
  a.set_coeff(112);
  a.set_coeff(110);
  a.set_coeff(108);
  a.set_coeff(106);
  a.set_coeff(102);
  a.set_coeff(101);
  a.set_coeff(98);
  a.set_coeff(96);
  a.set_coeff(94);
  a.set_coeff(93);
  a.set_coeff(92);
  a.set_coeff(89);
  a.set_coeff(88);
  a.set_coeff(86);
  a.set_coeff(85);
  a.set_coeff(84);
  a.set_coeff(82);
  a.set_coeff(78);
  a.set_coeff(76);
  a.set_coeff(74);
  a.set_coeff(73);
  a.set_coeff(70);
  a.set_coeff(69);
  a.set_coeff(66);
  a.set_coeff(64);
  a.set_coeff(62);
  a.set_coeff(61);
  a.set_coeff(57);
  a.set_coeff(56);
  a.set_coeff(54);
  a.set_coeff(53);
  a.set_coeff(52);
  a.set_coeff(50);
  a.set_coeff(46);
  a.set_coeff(45);
  a.set_coeff(43);
  a.set_coeff(42);
  a.set_coeff(41);
  a.set_coeff(40);
  a.set_coeff(38);
  a.set_coeff(37);
  a.set_coeff(36);
  a.set_coeff(33);
  a.set_coeff(30);
  a.set_coeff(28);
  a.set_coeff(27);
  a.set_coeff(26);
  a.set_coeff(24);
  a.set_coeff(22);
  a.set_coeff(20);
  a.set_coeff(17);
  a.set_coeff(16);
  a.set_coeff(14);
  a.set_coeff(10);
  a.set_coeff(9);
  a.set_coeff(8);
  a.set_coeff(6);
  a.set_coeff(4);
  a.set_coeff(3);
  a.set_coeff(2);
  a.set_coeff(0);

  field::GF2_128 a_int(
      std::array<uint64_t, 2>{0x63746f725d53475d, 0x7b5b546573745665});
  field::GF2_128 a_str("0x7b5b54657374566563746f725d53475d");
  REQUIRE(a == a_int);
  REQUIRE(a == a_str);
}

TEST_CASE("Basic Arithmetic in GF(2^128)", "[GF2_128]") {
  field::GF2_128 zero;
  field::GF2_128 one(1);
  field::GF2_128 x(2);
  field::GF2_128 x_2(4);

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
TEST_CASE("Modular Arithmetic KATs GF(2^128)", "[GF2_128]") {
  field::GF2_128 a, b;
  a.set_coeff(126);
  a.set_coeff(125);
  a.set_coeff(124);
  a.set_coeff(123);
  a.set_coeff(121);
  a.set_coeff(120);
  a.set_coeff(118);
  a.set_coeff(116);
  a.set_coeff(115);
  a.set_coeff(113);
  a.set_coeff(112);
  a.set_coeff(110);
  a.set_coeff(108);
  a.set_coeff(106);
  a.set_coeff(102);
  a.set_coeff(101);
  a.set_coeff(98);
  a.set_coeff(96);
  a.set_coeff(94);
  a.set_coeff(93);
  a.set_coeff(92);
  a.set_coeff(89);
  a.set_coeff(88);
  a.set_coeff(86);
  a.set_coeff(85);
  a.set_coeff(84);
  a.set_coeff(82);
  a.set_coeff(78);
  a.set_coeff(76);
  a.set_coeff(74);
  a.set_coeff(73);
  a.set_coeff(70);
  a.set_coeff(69);
  a.set_coeff(66);
  a.set_coeff(64);
  a.set_coeff(62);
  a.set_coeff(61);
  a.set_coeff(57);
  a.set_coeff(56);
  a.set_coeff(54);
  a.set_coeff(53);
  a.set_coeff(52);
  a.set_coeff(50);
  a.set_coeff(46);
  a.set_coeff(45);
  a.set_coeff(43);
  a.set_coeff(42);
  a.set_coeff(41);
  a.set_coeff(40);
  a.set_coeff(38);
  a.set_coeff(37);
  a.set_coeff(36);
  a.set_coeff(33);
  a.set_coeff(30);
  a.set_coeff(28);
  a.set_coeff(27);
  a.set_coeff(26);
  a.set_coeff(24);
  a.set_coeff(22);
  a.set_coeff(20);
  a.set_coeff(17);
  a.set_coeff(16);
  a.set_coeff(14);
  a.set_coeff(10);
  a.set_coeff(9);
  a.set_coeff(8);
  a.set_coeff(6);
  a.set_coeff(4);
  a.set_coeff(3);
  a.set_coeff(2);
  a.set_coeff(0);

  b.set_coeff(126);
  b.set_coeff(123);
  b.set_coeff(118);
  b.set_coeff(117);
  b.set_coeff(115);
  b.set_coeff(112);
  b.set_coeff(109);
  b.set_coeff(107);
  b.set_coeff(102);
  b.set_coeff(100);
  b.set_coeff(97);
  b.set_coeff(96);
  b.set_coeff(94);
  b.set_coeff(93);
  b.set_coeff(91);
  b.set_coeff(86);
  b.set_coeff(85);
  b.set_coeff(80);
  b.set_coeff(78);
  b.set_coeff(77);
  b.set_coeff(76);
  b.set_coeff(75);
  b.set_coeff(72);
  b.set_coeff(69);
  b.set_coeff(67);
  b.set_coeff(64);
  b.set_coeff(62);
  b.set_coeff(60);
  b.set_coeff(59);
  b.set_coeff(57);
  b.set_coeff(56);
  b.set_coeff(54);
  b.set_coeff(50);
  b.set_coeff(49);
  b.set_coeff(48);
  b.set_coeff(46);
  b.set_coeff(45);
  b.set_coeff(44);
  b.set_coeff(42);
  b.set_coeff(40);
  b.set_coeff(38);
  b.set_coeff(37);
  b.set_coeff(34);
  b.set_coeff(32);
  b.set_coeff(30);
  b.set_coeff(29);
  b.set_coeff(28);
  b.set_coeff(25);
  b.set_coeff(22);
  b.set_coeff(21);
  b.set_coeff(19);
  b.set_coeff(18);
  b.set_coeff(17);
  b.set_coeff(16);
  b.set_coeff(14);
  b.set_coeff(13);
  b.set_coeff(11);
  b.set_coeff(10);
  b.set_coeff(9);
  b.set_coeff(6);
  b.set_coeff(4);
  b.set_coeff(3);
  b.set_coeff(2);
  b.set_coeff(0);
  // a = 0x7b5b54657374566563746f725d53475d
  // b = 0x48692853686179295b477565726f6e5d
  field::GF2_128 a_int("0x7b5b54657374566563746f725d53475d"),
      b_int("0x48692853686179295b477565726f6e5d");
  REQUIRE(a == a_int);
  REQUIRE(b == b_int);

  // a*b = 0x040229a09a5ed12e7e4e10da323506d2
  // a*a = 0xdc56c143eb2f71f9f8c0f27ded737090
  // b*b = 0x408f5d5028d9e18058ce78d8de11282e
  field::GF2_128 ab("0x040229a09a5ed12e7e4e10da323506d2");
  field::GF2_128 a_2("0xdc56c143eb2f71f9f8c0f27ded737090");
  field::GF2_128 b_2("0x408f5d5028d9e18058ce78d8de11282e");
  field::GF2_128 ab_calc = a_int * b_int;
  REQUIRE(ab_calc == ab);
  REQUIRE(a * a == a_2);
  REQUIRE(b * b == b_2);

  BENCHMARK("GF Addition") { return a + b; };
  BENCHMARK("GF Multiplication") { return a * b; };
}

TEST_CASE("NTL to_bytes = custom to_bytes GF(2^128)", "[field]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_128);
  field::GF2_128 a, b;
  a.set_coeff(103);
  a.set_coeff(99);
  a.set_coeff(70);
  a.set_coeff(40);
  a.set_coeff(31);
  a.set_coeff(29);
  a.set_coeff(28);
  a.set_coeff(24);
  a.set_coeff(23);
  a.set_coeff(21);
  a.set_coeff(19);
  a.set_coeff(15);
  a.set_coeff(14);
  a.set_coeff(9);
  a.set_coeff(8);
  a.set_coeff(0);

  b.set_coeff(29);
  b.set_coeff(27);
  b.set_coeff(26);
  b.set_coeff(25);
  b.set_coeff(20);
  b.set_coeff(17);
  b.set_coeff(14);
  b.set_coeff(11);
  b.set_coeff(10);
  b.set_coeff(5);
  b.set_coeff(3);
  b.set_coeff(2);
  GF2X c, d;
  SetCoeff(c, 103);
  SetCoeff(c, 99);
  SetCoeff(c, 70);
  SetCoeff(c, 40);
  SetCoeff(c, 31);
  SetCoeff(c, 29);
  SetCoeff(c, 28);
  SetCoeff(c, 24);
  SetCoeff(c, 23);
  SetCoeff(c, 21);
  SetCoeff(c, 19);
  SetCoeff(c, 15);
  SetCoeff(c, 14);
  SetCoeff(c, 9);
  SetCoeff(c, 8);
  SetCoeff(c, 0);
  GF2E c_e = conv<GF2E>(c);

  SetCoeff(d, 29);
  SetCoeff(d, 27);
  SetCoeff(d, 26);
  SetCoeff(d, 25);
  SetCoeff(d, 20);
  SetCoeff(d, 17);
  SetCoeff(d, 14);
  SetCoeff(d, 11);
  SetCoeff(d, 10);
  SetCoeff(d, 5);
  SetCoeff(d, 3);
  SetCoeff(d, 2);
  GF2E d_e = conv<GF2E>(d);

  const GF2X &poly_rep_c = rep(c_e);
  std::vector<uint8_t> buffer_c(field::GF2_128::BYTE_SIZE);
  BytesFromGF2X(buffer_c.data(), poly_rep_c, buffer_c.size());
  const GF2X &poly_rep_d = rep(d_e);
  std::vector<uint8_t> buffer_d(field::GF2_128::BYTE_SIZE);
  BytesFromGF2X(buffer_d.data(), poly_rep_d, buffer_d.size());

  std::vector<uint8_t> buffer_a(field::GF2_128::BYTE_SIZE);
  a.to_bytes(buffer_a.data());
  std::vector<uint8_t> buffer_b(field::GF2_128::BYTE_SIZE);
  b.to_bytes(buffer_b.data());
  REQUIRE(buffer_a == buffer_c);
  REQUIRE(buffer_b == buffer_d);
}
TEST_CASE("NTL to custom conversion GF(2^128)", "[GF2_128]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_128);
  field::GF2_128 a, b;
  a.set_coeff(31);
  a.set_coeff(29);
  a.set_coeff(28);
  a.set_coeff(24);
  a.set_coeff(23);
  a.set_coeff(21);
  a.set_coeff(19);
  a.set_coeff(15);
  a.set_coeff(14);
  a.set_coeff(9);
  a.set_coeff(8);
  a.set_coeff(0);

  b.set_coeff(29);
  b.set_coeff(27);
  b.set_coeff(26);
  b.set_coeff(25);
  b.set_coeff(20);
  b.set_coeff(17);
  b.set_coeff(14);
  b.set_coeff(11);
  b.set_coeff(10);
  b.set_coeff(5);
  b.set_coeff(3);
  b.set_coeff(2);

  field::GF2_128 ab = a * b;
  GF2E a_ntl = utils::custom_to_ntl(a);
  GF2E ab_ntl = utils::custom_to_ntl(ab);
  GF2E b_ntl = ab_ntl / a_ntl;
  field::GF2_128 b2 = utils::ntl_to_custom<field::GF2_128>(b_ntl);
  REQUIRE(b == b2);
}
TEST_CASE("Custom fast inverse GF(2^128)", "[GF2_128]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_128);
  field::GF2_128 a;
  a.set_coeff(31);
  a.set_coeff(29);
  a.set_coeff(28);
  a.set_coeff(24);
  a.set_coeff(23);
  a.set_coeff(21);
  a.set_coeff(19);
  a.set_coeff(15);
  a.set_coeff(14);
  a.set_coeff(9);
  a.set_coeff(8);
  a.set_coeff(0);

  for (size_t i = 0; i < 1; i++) {
    field::GF2_128 b = a.inverse();
    field::GF2_128 c = field::GF2_128("0x8368bf5aa4edd94e58692b13a40f3197");
    REQUIRE(b == c);
  }
  BENCHMARK("GF inverse") { return a.inverse(); };
}
TEST_CASE("Custom slow inverse GF(2^128)", "[GF2_128]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_128);
  field::GF2_128 a;
  a.set_coeff(31);
  a.set_coeff(29);
  a.set_coeff(28);
  a.set_coeff(24);
  a.set_coeff(23);
  a.set_coeff(21);
  a.set_coeff(19);
  a.set_coeff(15);
  a.set_coeff(14);
  a.set_coeff(9);
  a.set_coeff(8);
  a.set_coeff(0);

  for (size_t i = 0; i < 1; i++) {
    field::GF2_128 b = a.inverse_slow();
    field::GF2_128 c = field::GF2_128("0x8368bf5aa4edd94e58692b13a40f3197");
    REQUIRE(b == c);
  }
  BENCHMARK("GF inverse") { return a.inverse_slow(); };
}
TEST_CASE("NTL inverse == custom inverse GF(2^128)", "[GF2_128]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_128);
  field::GF2_128 a;
  a.set_coeff(31);
  a.set_coeff(29);
  a.set_coeff(28);
  a.set_coeff(24);
  a.set_coeff(23);
  a.set_coeff(21);
  a.set_coeff(19);
  a.set_coeff(15);
  a.set_coeff(14);
  a.set_coeff(9);
  a.set_coeff(8);
  a.set_coeff(0);

  field::GF2_128 b = a.inverse();
  field::GF2_128 c =
      utils::ntl_to_custom<field::GF2_128>(inv(utils::custom_to_ntl(a)));
  // std::cout << utils::custom_to_ntl(a) << ", " << utils::custom_to_ntl(b)
  //<< ", " << utils::custom_to_ntl(c) << "\n";
  // std::cout << utils::custom_to_ntl(a * b) << ", "
  //<< utils::custom_to_ntl(a * c) << ", "
  //<< utils::custom_to_ntl(a) * utils::custom_to_ntl(c) << "\n";
  REQUIRE(b == c);
  REQUIRE(a * b == field::GF2_128(1));
  BENCHMARK("GF inverse") { return a.inverse(); };
}
TEST_CASE("NTL interpolation == custom interpolation GF(2^128)", "[GF2_128]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_128);

  std::vector<field::GF2_128> a =
      field::get_first_n_field_elements<field::GF2_128>(100);
  vec_GF2E b = utils::get_first_n_field_elements(100);
  for (size_t i = 0; i < 100; i++) {
    REQUIRE(a[i] == utils::ntl_to_custom<field::GF2_128>(b[i]));
  }
  std::vector<field::GF2_128> a_from_roots = field::build_from_roots(a);
  GF2EX b_from_roots = BuildFromRoots(b);
  REQUIRE(a_from_roots.size() == (size_t)b_from_roots.rep.length());
  for (size_t j = 0; j < a_from_roots.size(); j++) {
    REQUIRE(a_from_roots[j] ==
            utils::ntl_to_custom<field::GF2_128>(b_from_roots[j]));
  }

  std::vector<std::vector<field::GF2_128>> a_lag =
      field::precompute_lagrange_polynomials(a);
  std::vector<GF2EX> b_lag = utils::precompute_lagrange_polynomials(b);

  REQUIRE(a_lag.size() == b_lag.size());
  for (size_t i = 0; i < a_lag.size(); i++) {
    REQUIRE(a_lag[i].size() == (size_t)b_lag[i].rep.length());
    for (size_t j = 0; j < a_lag[i].size(); j++) {
      REQUIRE(a_lag[i][j] == utils::ntl_to_custom<field::GF2_128>(b_lag[i][j]));
    }
  }

  BENCHMARK("Lagrange Poly Precomputation") {
    return field::precompute_lagrange_polynomials(a);
  };
  BENCHMARK("Lagrange Poly Interpolation") {
    return field::interpolate_with_precomputation(a_lag, a);
  };
}

TEST_CASE("NTL dot product == custom GF(2^128)", "[GF2_128]") {

  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_128);
  std::vector<field::GF2_128> a =
      field::get_first_n_field_elements<field::GF2_128>(100);
  std::vector<field::GF2_128> b =
      field::get_first_n_field_elements<field::GF2_128>(200);
  b.erase(b.begin(), b.begin() + 100);

  vec_GF2E a_ntl = utils::get_first_n_field_elements(200);
  vec_GF2E b_ntl;
  b_ntl.SetLength(100);
  for (size_t i = 0; i < 100; i++)
    b_ntl[i] = a_ntl[100 + i];
  a_ntl.SetLength(100);

  field::GF2_128 result = dot_product(a, b);
  GF2E result_ntl;
  NTL::InnerProduct(result_ntl, a_ntl, b_ntl);
  REQUIRE(result == utils::ntl_to_custom<field::GF2_128>(result_ntl));
}
