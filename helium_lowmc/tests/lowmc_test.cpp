#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../lowmc.h"

TEST_CASE("LOWMC-129-4-KAT", "[cipher]") {

  // key, pt, ct
  std::array<std::vector<uint8_t>, 3> testvectors[] = {
      {{
          {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
          {0xab, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
          {0x2f, 0xd7, 0xd5, 0x42, 0x5e, 0xe3, 0x5e, 0x66, 0x7c, 0x97, 0x2f,
           0x12, 0xfb, 0x15, 0x3e, 0x9d, 0x80},
      }},
  };
  for (auto testcase : testvectors) {
    std::vector<uint8_t> key = testcase[0];
    std::vector<uint8_t> plaintext = testcase[1];
    std::vector<uint8_t> ciphertext_expected = testcase[2];

    std::vector<uint8_t> ciphertext;

    LOWMC_129_4::lowmc_with_sbox_output<field::GF2_9>(key, plaintext,
                                                      ciphertext);
    REQUIRE(ciphertext_expected == ciphertext);

    LOWMC_129_4::lowmc_with_sbox_output<field::GF2_45>(key, plaintext,
                                                       ciphertext);
    REQUIRE(ciphertext_expected == ciphertext);

    LOWMC_129_4::lowmc_with_sbox_output<field::GF2_51>(key, plaintext,
                                                       ciphertext);
    REQUIRE(ciphertext_expected == ciphertext);
  }
}

TEST_CASE("LowMC-128-20-KAT", "[cipher]") {

  // key, pt, ct
  std::array<std::vector<uint8_t>, 3> testvectors[] = {
      {{
          {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x00, 0x00},
          {0xff, 0xd5, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x00, 0x00},
          {0xb3, 0x2a, 0x88, 0xd2, 0xb0, 0x86, 0x40, 0xe8, 0xc8, 0xbf, 0x34,
           0xe3, 0x67, 0xab, 0xc2, 0x92},
      }},
  };
  for (auto testcase : testvectors) {
    std::vector<uint8_t> key = testcase[0];
    std::vector<uint8_t> plaintext = testcase[1];
    std::vector<uint8_t> ciphertext_expected = testcase[2];

    std::vector<uint8_t> ciphertext;

    LOWMC_128_20::lowmc_with_sbox_output<field::GF2_9>(key, plaintext,
                                                       ciphertext);
    REQUIRE(ciphertext_expected == ciphertext);

    LOWMC_128_20::lowmc_with_sbox_output<field::GF2_45>(key, plaintext,
                                                        ciphertext);
    REQUIRE(ciphertext_expected == ciphertext);

    LOWMC_128_20::lowmc_with_sbox_output<field::GF2_51>(key, plaintext,
                                                        ciphertext);
    REQUIRE(ciphertext_expected == ciphertext);
  }
}
