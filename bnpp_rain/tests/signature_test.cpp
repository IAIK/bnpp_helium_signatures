#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include "../signature.h"

TEST_CASE("Sign and verify a message", "[signature]") {
  const char *message = "TestMessage";
  const signature_instance_t &instance = instance_get(Rainier_3_L1_Param1);
  const std::vector<uint8_t> key = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                    0xff, 0xff, 0xff, 0xff};
  const std::vector<uint8_t> plaintext = {0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
                                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                          0x00, 0x00, 0x00, 0x00};
  const std::vector<uint8_t> ciphertext_expected = {
      0xae, 0x1e, 0xad, 0x61, 0x3b, 0xba, 0x98, 0x72,
      0xa0, 0xfc, 0x8b, 0x79, 0xa1, 0xda, 0x91, 0x23};

  keypair_t keypair;
  keypair.first = key;
  keypair.second = plaintext;
  keypair.second.insert(keypair.second.end(), ciphertext_expected.begin(),
                        ciphertext_expected.end());

  std::vector<uint8_t> serialized_signature = rainier_sign(
      instance, keypair, (const uint8_t *)message, strlen(message));
  std::cout << "signature length: " << serialized_signature.size()
            << " bytes\n";
  REQUIRE(rainier_verify(instance, keypair.second, serialized_signature,
                         (const uint8_t *)message, strlen(message)));
  BENCHMARK("signing") {
    return rainier_sign(instance, keypair, (const uint8_t *)message,
                        strlen(message));
  };
  BENCHMARK("verification") {
    return rainier_verify(instance, keypair.second, serialized_signature,
                          (const uint8_t *)message, strlen(message));
  };
}
