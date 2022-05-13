#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
extern "C" {
#include <emmintrin.h>
#include <immintrin.h>
}

#include "field.h"

namespace rmfe {
// (2,3)_2^3 RMFE
extern uint16_t phi_2_3_matrix[];
extern uint16_t psi_2_3_matrix_transpose[];

inline uint64_t phi_2_3_single_bit(uint64_t input, uint64_t bit) {
  return phi_2_3_matrix[bit] & ((uint64_t)(-input));
}
inline uint64_t psi_2_3_single_bit(uint64_t input, uint64_t bit) {
  return _mm_popcnt_u64(input & psi_2_3_matrix_transpose[bit]) & 1;
}

field::GF2_9 phi_2_3(std::array<field::GF2_3, 2> input);
std::array<field::GF2_3, 2> psi_2_3(field::GF2_9 input);
std::array<field::GF2_3, 2> psi_2_3_transpose(field::GF2_9 input);

void phi_8_15(field::GF2_45 *output, const field::GF2_3 *input);
void psi_8_15(field::GF2_3 *output, const field::GF2_45 *input);
void psi_8_15_transpose(field::GF2_3 *output, const field::GF2_45 *input);

// (8,15)_2^3 RMFE
extern uint64_t phi_8_15_matrix[];
extern uint64_t psi_8_15_matrix_transpose[];

inline uint64_t phi_8_15_single_bit(uint64_t input, uint64_t bit) {
  return phi_8_15_matrix[bit] & ((uint64_t)(-input));
}
inline uint64_t psi_8_15_single_bit(uint64_t input, uint64_t bit) {
  return _mm_popcnt_u64(input & psi_8_15_matrix_transpose[bit]) & 1;
}

field::GF2_45 phi_8_15(std::array<field::GF2_3, 8> input);
std::array<field::GF2_3, 8> psi_8_15(field::GF2_45 input);
std::array<field::GF2_3, 8> psi_8_15_transpose(field::GF2_45 input);

void phi_8_15(field::GF2_45 *output, const field::GF2_3 *input);
void psi_8_15(field::GF2_3 *output, const field::GF2_45 *input);
void psi_8_15_transpose(field::GF2_3 *output, const field::GF2_45 *input);

// (9,17)_2^3 RMFE
extern uint64_t phi_9_17_matrix[];
extern uint64_t psi_9_17_matrix_transpose[];

inline uint64_t phi_9_17_single_bit(uint64_t input, uint64_t bit) {
  return phi_9_17_matrix[bit] & ((uint64_t)(-input));
}
inline uint64_t psi_9_17_single_bit(uint64_t input, uint64_t bit) {
  return _mm_popcnt_u64(input & psi_9_17_matrix_transpose[bit]) & 1;
}

field::GF2_51 phi_9_17(std::array<field::GF2_3, 9> input);
std::array<field::GF2_3, 9> psi_9_17(field::GF2_51 input);
std::array<field::GF2_3, 9> psi_9_17_transpose(field::GF2_51 input);

void phi_9_17(field::GF2_51 *output, const field::GF2_3 *input);
void psi_9_17(field::GF2_3 *output, const field::GF2_51 *input);
void psi_9_17_transpose(field::GF2_3 *output, const field::GF2_51 *input);

} // namespace rmfe
