/*
 *  This file is part of the optimized implementation of the Picnic signature
 * scheme. See the accompanying documentation for complete details.
 *
 *  The code is provided under the MIT license, see LICENSE for
 *  more details.
 *  SPDX-License-Identifier: MIT
 */

#include "instances.h"

#include <stdexcept>

/* key_size, block_size, num_blocks, num_sboxes */
constexpr lowmc_params_t LOWMC_129_4_PARAMS = {17, 17, 172};
constexpr lowmc_params_t LOWMC_128_20_PARAMS = {16, 16, 200};

static const signature_instance_t instances[PARAMETER_SET_MAX_INDEX] = {
    {
        {0, 0, 0},
        0,
        0,
        0,
        0,
        0,
        PARAMETER_SET_INVALID,
    },
    /* some sample instances for ~64 and 256 parties */
    /* LowMC_params, digest size, seed size, T, N, L */
    {LOWMC_129_4_PARAMS, 32, 16, 22, 57, 86, LowMC_4_L1_Param1},
    {LOWMC_129_4_PARAMS, 32, 16, 16, 256, 86, LowMC_4_L1_Param1},
    {LOWMC_128_20_PARAMS, 32, 16, 22, 57, 100, LowMC_20_L1_Param2},
    {LOWMC_128_20_PARAMS, 32, 16, 16, 256, 100, LowMC_20_L1_Param2},
};

const signature_instance_t &instance_get(params_t param) {
  if (param <= PARAMETER_SET_INVALID || param >= PARAMETER_SET_MAX_INDEX) {
    throw std::runtime_error("invalid parameter set");
  }

  return instances[param];
}
