/*
 *  This file is part of the optimized implementation of the Picnic signature
 * scheme. See the accompanying documentation for complete details.
 *
 *  The code is provided under the MIT license, see LICENSE for
 *  more details.
 *  SPDX-License-Identifier: MIT
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "../signature.h"
#include "bench_timing.h"
#include "bench_utils.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

struct timing_and_size_t {
  uint64_t keygen, sign, verify, size;
};

static void print_timings(const std::vector<timing_and_size_t> &timings) {
  timing_and_size_t average = {0, 0, 0, 0};
  printf("keygen,sign,verify,size\n");
  for (size_t i = 0; i < timings.size(); i++) {
    timing_and_size_t timing = timings[i];
    printf("%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", timing.keygen,
           timing.sign, timing.verify, timing.size);
    if (i > 0) {
      average.keygen += timing.keygen;
      average.sign += timing.sign;
      average.verify += timing.verify;
      average.size += timing.size;
    }
  }

  uint64_t n = timings.size() - 1;
  printf("average of last %" PRIu64 " rows:\n", n);
  printf("%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
         average.keygen / n, average.sign / n, average.verify / n,
         average.size / n);
}

static int bench_sign_and_verify_free(const bench_options_free_t *options) {
  static const uint8_t m[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

  std::vector<timing_and_size_t> timings(options->iter);

  timing_context_t ctx;
  if (!timing_init(&ctx)) {
    printf("Failed to initialize timing functionality.\n");
    return (-1);
  }
  signature_instance_t instance;
  instance.digest_size = 2 * options->kappa;
  instance.seed_size = options->kappa;
  if (options->sboxes != 172 && options->sboxes != 200) {
    printf("invalid num of lowmc sboxes, choose 172 or 200\n");
    return (-1);
  }
  if (options->kappa != 16) {
    printf("invalid kappa, choose 16\n");
    return (-1);
  }
  switch (options->sboxes) {
  case 172:
    instance.block_cipher_params = {17, 17, 172};
    instance.num_lifted_multiplications = 86;
    break;
  case 200:
    instance.block_cipher_params = {16, 16, 200};
    instance.num_lifted_multiplications = 100;
    break;
  }
  instance.num_MPC_parties = options->N;
  instance.num_repetitions = options->tau;
  instance.params = PARAMETER_SET_INVALID;
  printf("Instance: N=%d, tau=%d, Seclvl=%d\n", instance.num_MPC_parties,
         instance.num_repetitions, instance.block_cipher_params.key_size);

  for (unsigned int i = 0; i != options->iter; ++i) {
    timing_and_size_t &timing = timings[i];

    uint64_t start_time = timing_read(&ctx);
    keypair_t keypair = helium_keygen(instance);

    uint64_t tmp_time = timing_read(&ctx);
    timing.keygen = tmp_time - start_time;
    start_time = timing_read(&ctx);

    std::vector<uint8_t> signature =
        helium_sign(instance, keypair, m, sizeof(m));

    tmp_time = timing_read(&ctx);
    timing.sign = tmp_time - start_time;
    timing.size = signature.size();

    start_time = timing_read(&ctx);
    bool ok = helium_verify(instance, keypair.second, signature, m, sizeof(m));
    tmp_time = timing_read(&ctx);
    timing.verify = tmp_time - start_time;
    if (!ok)
      std::cerr << "failed to verify signature" << std::endl;
  }

  timing_close(&ctx);
  print_timings(timings);

  return (0);
}

int main(int argc, char **argv) {
  bench_options_free_t opts = {0, 0, 0, 0, 0};
  int ret = parse_args_free(&opts, argc, argv);

  if (!ret) {
    return (-1);
  }

  return bench_sign_and_verify_free(&opts);
}
