#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include "../signature.h"

TEST_CASE("Sign and verify a message", "[signature]") {
  const char *message = "TestMessage";
  const signature_instance_t &instance = instance_get(AES128_L1_Param1);
  const std::vector<uint8_t> key = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                    0xff, 0xff, 0xff, 0xff};
  const std::vector<uint8_t> plaintext = {0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
                                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                          0x00, 0x00, 0x00, 0x00};
  const std::vector<uint8_t> ciphertext_expected = {
      0x0b, 0x5a, 0x81, 0x4d, 0x95, 0x60, 0x1c, 0xc7,
      0xef, 0xe7, 0x12, 0x28, 0x3e, 0x05, 0xef, 0x8f};

  keypair_t keypair;
  keypair.first = key;
  keypair.second = plaintext;
  keypair.second.insert(keypair.second.end(), ciphertext_expected.begin(),
                        ciphertext_expected.end());

  std::vector<uint8_t> serialized_signature =
      helium_sign(instance, keypair, (const uint8_t *)message, strlen(message));
  std::cout << "signature length: " << serialized_signature.size()
            << " bytes\n";
  REQUIRE(helium_verify(instance, keypair.second, serialized_signature,
                        (const uint8_t *)message, strlen(message)));
  BENCHMARK("signing") {
    return helium_sign(instance, keypair, (const uint8_t *)message,
                       strlen(message));
  };
  BENCHMARK("verification") {
    return helium_verify(instance, keypair.second, serialized_signature,
                         (const uint8_t *)message, strlen(message));
  };
}
