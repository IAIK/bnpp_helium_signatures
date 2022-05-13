#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include "../field.h"
#include "../rmfe.h"
#include "utils.h"

TEST_CASE("Basic RMFE-(2,3)_(2^3) test", "[RMFE]") {
  std::array<field::GF2_3, 2> a, b;
  for (size_t i = 0; i < 2; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((8 - i) & 0x7);
  }

  field::GF2_9 a_enc = rmfe::phi_2_3(a);
  field::GF2_9 b_enc = rmfe::phi_2_3(b);

  field::GF2_9 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 2> c_dec = rmfe::psi_2_3(c_enc);

  BENCHMARK("rmfe::phi_2_3") { return rmfe::phi_2_3(a); };
  BENCHMARK("rmfe::psi_2_3") { return rmfe::psi_2_3(c_enc); };

  for (size_t i = 0; i < 2; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(2,3)_(2^3) test, variant 2", "[RMFE]") {
  std::array<field::GF2_3, 2> a, b;
  for (size_t i = 0; i < 2; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((8 - i) & 0x7);
  }

  field::GF2_9 a_enc = rmfe::phi_2_3(a);
  field::GF2_9 b_enc = rmfe::phi_2_3(b);

  field::GF2_9 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 2> c_dec = rmfe::psi_2_3_transpose(c_enc);

  BENCHMARK("rmfe::psi_2_3_transpose") {
    return rmfe::psi_2_3_transpose(c_enc);
  };

  for (size_t i = 0; i < 2; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(8,15)_(2^3) test", "[RMFE]") {
  std::array<field::GF2_3, 8> a, b;
  for (size_t i = 0; i < 8; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((8 - i) & 0x7);
  }

  field::GF2_45 a_enc = rmfe::phi_8_15(a);
  field::GF2_45 b_enc = rmfe::phi_8_15(b);

  field::GF2_45 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 8> c_dec = rmfe::psi_8_15(c_enc);

  BENCHMARK("rmfe::phi_8_15") { return rmfe::phi_8_15(a); };
  BENCHMARK("rmfe::psi_8_15") { return rmfe::psi_8_15(c_enc); };

  for (size_t i = 0; i < 8; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(8,15)_(2^3) test, variant 2", "[RMFE]") {
  std::array<field::GF2_3, 8> a, b;
  for (size_t i = 0; i < 8; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((8 - i) & 0x7);
  }

  field::GF2_45 a_enc = rmfe::phi_8_15(a);
  field::GF2_45 b_enc = rmfe::phi_8_15(b);

  field::GF2_45 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 8> c_dec = rmfe::psi_8_15_transpose(c_enc);

  BENCHMARK("rmfe::psi_8_15_transpose") {
    return rmfe::psi_8_15_transpose(c_enc);
  };

  for (size_t i = 0; i < 8; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(8,15)_(2^3) test, ptr interface", "[RMFE]") {
  std::array<field::GF2_3, 8> a, b;
  for (size_t i = 0; i < 8; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((8 - i) & 0x7);
  }

  field::GF2_45 a_enc, b_enc;
  rmfe::phi_8_15(&a_enc, a.data());
  rmfe::phi_8_15(&b_enc, b.data());

  field::GF2_45 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 8> c_dec;
  rmfe::psi_8_15(c_dec.data(), &c_enc);

  BENCHMARK("rmfe::phi_8_15, ptr") { return rmfe::phi_8_15(&a_enc, a.data()); };
  BENCHMARK("rmfe::psi_8_15, ptr") {
    return rmfe::psi_8_15(c_dec.data(), &c_enc);
  };

  for (size_t i = 0; i < 8; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(8,15)_(2^3) test, ptr interface, variant 2", "[RMFE]") {
  std::array<field::GF2_3, 8> a, b;
  for (size_t i = 0; i < 8; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((8 - i) & 0x7);
  }

  field::GF2_45 a_enc, b_enc;
  rmfe::phi_8_15(&a_enc, a.data());
  rmfe::phi_8_15(&b_enc, b.data());

  field::GF2_45 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 8> c_dec;
  rmfe::psi_8_15_transpose(c_dec.data(), &c_enc);

  BENCHMARK("rmfe::psi_8_15_transpose, ptr") {
    return rmfe::psi_8_15_transpose(c_dec.data(), &c_enc);
  };

  for (size_t i = 0; i < 8; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(9,17)_(2^3) test", "[RMFE]") {
  std::array<field::GF2_3, 9> a, b;
  for (size_t i = 0; i < 9; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((9 - i) & 0x7);
  }

  field::GF2_51 a_enc = rmfe::phi_9_17(a);
  field::GF2_51 b_enc = rmfe::phi_9_17(b);

  field::GF2_51 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 9> c_dec = rmfe::psi_9_17(c_enc);

  BENCHMARK("rmfe::phi_9_17") { return rmfe::phi_9_17(a); };
  BENCHMARK("rmfe::psi_9_17") { return rmfe::psi_9_17(c_enc); };

  for (size_t i = 0; i < 9; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(9,17)_(2^3) test, variant 2", "[RMFE]") {
  std::array<field::GF2_3, 9> a, b;
  for (size_t i = 0; i < 9; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((9 - i) & 0x7);
  }

  field::GF2_51 a_enc = rmfe::phi_9_17(a);
  field::GF2_51 b_enc = rmfe::phi_9_17(b);

  field::GF2_51 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 9> c_dec = rmfe::psi_9_17_transpose(c_enc);

  BENCHMARK("rmfe::psi_9_17_transpose") {
    return rmfe::psi_9_17_transpose(c_enc);
  };

  for (size_t i = 0; i < 9; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(9,17)_(2^3) test, ptr interface", "[RMFE]") {
  std::array<field::GF2_3, 9> a, b;
  for (size_t i = 0; i < 9; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((9 - i) & 0x7);
  }

  field::GF2_51 a_enc, b_enc;
  rmfe::phi_9_17(&a_enc, a.data());
  rmfe::phi_9_17(&b_enc, b.data());

  field::GF2_51 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 9> c_dec;
  rmfe::psi_9_17(c_dec.data(), &c_enc);

  BENCHMARK("rmfe::phi_9_17, ptr") { return rmfe::phi_9_17(&a_enc, a.data()); };
  BENCHMARK("rmfe::psi_9_17, ptr") {
    return rmfe::psi_9_17(c_dec.data(), &c_enc);
  };

  for (size_t i = 0; i < 9; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}

TEST_CASE("Basic RMFE-(9,17)_(2^3) test, ptr interface, variant 2", "[RMFE]") {
  std::array<field::GF2_3, 9> a, b;
  for (size_t i = 0; i < 9; i++) {
    a[i] = field::GF2_3(i & 0x7);
    b[i] = field::GF2_3((9 - i) & 0x7);
  }

  field::GF2_51 a_enc, b_enc;
  rmfe::phi_9_17(&a_enc, a.data());
  rmfe::phi_9_17(&b_enc, b.data());

  field::GF2_51 c_enc = a_enc * b_enc;
  std::array<field::GF2_3, 9> c_dec;
  rmfe::psi_9_17_transpose(c_dec.data(), &c_enc);

  BENCHMARK("rmfe::psi_9_17_transpose, ptr") {
    return rmfe::psi_9_17_transpose(c_dec.data(), &c_enc);
  };

  for (size_t i = 0; i < 9; i++) {
    REQUIRE(a[i] * b[i] == c_dec[i]);
  }
}