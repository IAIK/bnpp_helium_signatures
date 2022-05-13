/*
 *  This file is part of the optimized implementation of the Helium
 * signature scheme. See the accompanying documentation for complete details.
 *
 *  The code is provided under the MIT license, see LICENSE for
 *  more details.
 *  SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <cstdlib>

/** Parameter set names */
enum params_t {
  PARAMETER_SET_INVALID = 0,
  AES128_L1_Param1 = 1,
  AES128_L1_Param2 = 1,
  PARAMETER_SET_MAX_INDEX = 3
};

struct aes_params_t {
  uint32_t key_size;
  uint32_t block_size;
  uint32_t num_sboxes;
};

struct signature_instance_t {

  aes_params_t block_cipher_params;

  uint32_t digest_size;     /* bytes */
  uint32_t seed_size;       /* bytes */
  uint32_t num_repetitions; // tau
  uint32_t num_MPC_parties; // N

  uint32_t num_multiplications;

  params_t params;
};

const signature_instance_t &instance_get(params_t param);
