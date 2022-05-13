#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include "../signature.h"

TEST_CASE("Sign and verify a message", "[signature]") {
  const char *message = "TestMessage";
  const signature_instance_t &instance = instance_get(LowMC_4_L1_Param1);
  const std::vector<uint8_t> key = {0x80, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00, 0x00};
  const std::vector<uint8_t> plaintext = {0xab, 0xff, 0x00, 0x00, 0x00, 0x00,
                                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                          0x00, 0x00, 0x00, 0x00, 0x00};
  const std::vector<uint8_t> ciphertext_expected = {
      0x2f, 0xd7, 0xd5, 0x42, 0x5e, 0xe3, 0x5e, 0x66, 0x7c,
      0x97, 0x2f, 0x12, 0xfb, 0x15, 0x3e, 0x9d, 0x80};

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
