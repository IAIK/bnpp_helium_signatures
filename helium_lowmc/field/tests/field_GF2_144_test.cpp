#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include "../field.h"
#include "utils.h"

#include <NTL/GF2EX.h>

TEST_CASE("Constructors for GF(2^144)", "[GF2_144]") {
  field::GF2_144 a;
  a.set_coeff(2);
  a.set_coeff(3);
  a.set_coeff(8);
  a.set_coeff(9);
  a.set_coeff(10);
  a.set_coeff(15);
  a.set_coeff(17);
  a.set_coeff(18);
  a.set_coeff(22);
  a.set_coeff(24);
  a.set_coeff(25);
  a.set_coeff(28);
  a.set_coeff(32);
  a.set_coeff(33);
  a.set_coeff(34);
  a.set_coeff(36);
  a.set_coeff(37);
  a.set_coeff(38);
  a.set_coeff(40);
  a.set_coeff(42);
  a.set_coeff(45);
  a.set_coeff(46);
  a.set_coeff(50);
  a.set_coeff(52);
  a.set_coeff(53);
  a.set_coeff(56);
  a.set_coeff(58);
  a.set_coeff(60);
  a.set_coeff(61);
  a.set_coeff(68);
  a.set_coeff(71);
  a.set_coeff(74);
  a.set_coeff(75);
  a.set_coeff(76);
  a.set_coeff(77);
  a.set_coeff(80);
  a.set_coeff(83);
  a.set_coeff(84);
  a.set_coeff(87);
  a.set_coeff(88);
  a.set_coeff(89);
  a.set_coeff(93);
  a.set_coeff(94);
  a.set_coeff(95);
  a.set_coeff(96);
  a.set_coeff(97);
  a.set_coeff(98);
  a.set_coeff(99);
  a.set_coeff(100);
  a.set_coeff(103);
  a.set_coeff(106);
  a.set_coeff(107);
  a.set_coeff(109);
  a.set_coeff(112);
  a.set_coeff(113);
  a.set_coeff(114);
  a.set_coeff(116);
  a.set_coeff(118);
  a.set_coeff(119);
  a.set_coeff(120);
  a.set_coeff(121);
  a.set_coeff(123);
  a.set_coeff(125);
  a.set_coeff(131);
  a.set_coeff(132);
  a.set_coeff(133);
  a.set_coeff(134);
  a.set_coeff(135);
  a.set_coeff(136);
  a.set_coeff(137);
  a.set_coeff(138);
  a.set_coeff(140);
  a.set_coeff(141);
  a.set_coeff(142);

  field::GF2_144 a_int(
      std::array<uint64_t, 4>{0x353465771346870C, 0x2BD72C9FE3993C90,
                              0x00000000000077F8, 0x0000000000000000});
  field::GF2_144 a_str("0x77F82BD72C9FE3993C90353465771346870C");
  REQUIRE(a == a_int);
  REQUIRE(a == a_str);
}

TEST_CASE("Basic Arithmetic GF(2^144)", "[GF2_144]") {
  field::GF2_144 zero;
  field::GF2_144 one(1);
  field::GF2_144 x(2);
  field::GF2_144 x_2(4);

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
TEST_CASE("Modular Arithmetic KATs GF(2^144)", "[GF2_144]") {
  field::GF2_144 a("0x77F82BD72C9FE3993C90353465771346870C");
  field::GF2_144 b("0x73DC3335FA6A60AAF0CA0799EFB48110F7CE");
  field::GF2_144 ab("0xF88207477B4DFAFF6AD5D608F355E4D28501");

  field::GF2_144 ab_calc = a * b;
  REQUIRE(ab_calc == ab);
}

TEST_CASE("NTL to_bytes = custom to_bytes GF(2^144)", "[field]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_144);
  field::GF2_144 a, b;
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
  std::vector<uint8_t> buffer_c(field::GF2_144::BYTE_SIZE);
  BytesFromGF2X(buffer_c.data(), poly_rep_c, buffer_c.size());
  const GF2X &poly_rep_d = rep(d_e);
  std::vector<uint8_t> buffer_d(field::GF2_144::BYTE_SIZE);
  BytesFromGF2X(buffer_d.data(), poly_rep_d, buffer_d.size());

  std::vector<uint8_t> buffer_a(field::GF2_144::BYTE_SIZE);
  a.to_bytes(buffer_a.data());
  std::vector<uint8_t> buffer_b(field::GF2_144::BYTE_SIZE);
  b.to_bytes(buffer_b.data());
  REQUIRE(buffer_a == buffer_c);
  REQUIRE(buffer_b == buffer_d);
}
TEST_CASE("NTL to custom conversion GF(2^144)", "[GF2_144]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_144);
  field::GF2_144 a("0x77F82BD72C9FE3993C90353465771346870C");
  field::GF2_144 b("0x73DC3335FA6A60AAF0CA0799EFB48110F7CE");

  field::GF2_144 ab = a * b;
  GF2E a_ntl = utils::custom_to_ntl(a);
  GF2E ab_ntl = utils::custom_to_ntl(ab);
  GF2E b_ntl = ab_ntl / a_ntl;
  field::GF2_144 b2 = utils::ntl_to_custom<field::GF2_144>(b_ntl);
  REQUIRE(b == b2);
}
TEST_CASE("Custom fast inverse GF(2^144)", "[GF2_144]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_144);
  field::GF2_144 a("0x77F82BD72C9FE3993C90353465771346870C");

  field::GF2_144 b = a.inverse();
  field::GF2_144 c = field::GF2_144("0x530742B69D84D3B39CC04F437337270CE644");
  REQUIRE(b == c);
  BENCHMARK("GF inverse") { return a.inverse(); };
}

TEST_CASE("NTL inverse == custom GF(2^144)", "[GF2_144]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_144);
  field::GF2_144 a("0x77F82BD72C9FE3993C90353465771346870C");

  field::GF2_144 b = a.inverse();
  field::GF2_144 c =
      utils::ntl_to_custom<field::GF2_144>(inv(utils::custom_to_ntl(a)));
  // std::cout << utils::custom_to_ntl(a) << ", " << utils::custom_to_ntl(b)
  //<< ", " << utils::custom_to_ntl(c) << "\n";
  // std::cout << utils::custom_to_ntl(a * b) << ", "
  //<< utils::custom_to_ntl(a * c) << ", "
  //<< utils::custom_to_ntl(a) * utils::custom_to_ntl(c) << "\n";
  REQUIRE(b == c);
  REQUIRE(a * b == field::GF2_144(1));
}
TEST_CASE("NTL interpolation == custom GF(2^144)", "[GF2_144]") {
  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_144);

  std::vector<field::GF2_144> a =
      field::get_first_n_field_elements<field::GF2_144>(100);
  vec_GF2E b = utils::get_first_n_field_elements(100);
  for (size_t i = 0; i < 100; i++) {
    REQUIRE(a[i] == utils::ntl_to_custom<field::GF2_144>(b[i]));
  }
  std::vector<field::GF2_144> a_from_roots = field::build_from_roots(a);
  GF2EX b_from_roots = BuildFromRoots(b);
  REQUIRE(a_from_roots.size() == (size_t)b_from_roots.rep.length());
  for (size_t j = 0; j < a_from_roots.size(); j++) {
    REQUIRE(a_from_roots[j] ==
            utils::ntl_to_custom<field::GF2_144>(b_from_roots[j]));
  }

  std::vector<std::vector<field::GF2_144>> a_lag =
      field::precompute_lagrange_polynomials(a);
  std::vector<GF2EX> b_lag = utils::precompute_lagrange_polynomials(b);

  REQUIRE(a_lag.size() == b_lag.size());
  for (size_t i = 0; i < a_lag.size(); i++) {
    REQUIRE(a_lag[i].size() == (size_t)b_lag[i].rep.length());
    for (size_t j = 0; j < a_lag[i].size(); j++) {
      REQUIRE(a_lag[i][j] == utils::ntl_to_custom<field::GF2_144>(b_lag[i][j]));
    }
  }
}
TEST_CASE("NTL dot product == custom GF(2^144)", "[GF2_144]") {

  utils::init_ntl_extension_field(utils::NTL_INSTANCE::GF2_144);
  std::vector<field::GF2_144> a =
      field::get_first_n_field_elements<field::GF2_144>(100);
  std::vector<field::GF2_144> b =
      field::get_first_n_field_elements<field::GF2_144>(201);
  b.erase(b.begin(), b.begin() + 101);

  vec_GF2E a_ntl = utils::get_first_n_field_elements(201);
  vec_GF2E b_ntl;
  b_ntl.SetLength(100);
  for (size_t i = 0; i < 100; i++)
    b_ntl[i] = a_ntl[101 + i];
  a_ntl.SetLength(100);

  field::GF2_144 result = dot_product(a, b);
  GF2E result_ntl;
  NTL::InnerProduct(result_ntl, a_ntl, b_ntl);
  REQUIRE(result == utils::ntl_to_custom<field::GF2_144>(result_ntl));
}
