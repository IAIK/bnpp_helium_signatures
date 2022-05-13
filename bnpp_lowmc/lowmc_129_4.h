#pragma once

#include "field/field.h"
#include "gsl-lite.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

namespace LOWMC_129_4 {

template <typename GF>
std::vector<std::array<GF, 3>>
lowmc_with_sbox_output(const std::vector<uint8_t> &key_in,
                       const std::vector<uint8_t> &plaintext_in,
                       std::vector<uint8_t> &ciphertext_out);

template <typename GF>
void lowmc_mpc(const std::vector<gsl::span<uint8_t>> &key_in,
               const std::vector<gsl::span<GF>> &z_shares,
               const std::vector<uint8_t> &plaintext_in,
               std::vector<gsl::span<uint8_t>> &ciphertext_out,
               std::vector<gsl::span<GF>> &x_shares,
               std::vector<gsl::span<GF>> &y_shares);
} // namespace LOWMC_129_4