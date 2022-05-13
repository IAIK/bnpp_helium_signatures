#include "rmfe.h"

namespace rmfe {

uint16_t phi_2_3_matrix[] = {
    0x3, 0x104, 0x171, 0x2, 0x1f8, 0x5f,
};
uint16_t psi_2_3_matrix[] = {
    0x9, 0x8, 0x8, 0x1, 0x12, 0x2f, 0x3a, 0x1c, 0xa,
};
uint16_t psi_2_3_matrix_transpose[] = {
    0x29, 0x170, 0xa0, 0x1e7, 0xd0, 0x60,
};

uint16_t phi_3_5_matrix[] = {
    0x7fa1, 0x582, 0x34da, 0x74ae, 0x7252, 0x553e, 0xb0e, 0x74ac, 0x74aa,
};
uint16_t psi_3_5_matrix_transpose[] = {
    0x1381, 0x47a0, 0x9a0, 0x717f, 0x3d60, 0x760, 0x43a9, 0x10da, 0x2814,
};
uint64_t phi_8_15_matrix[] = {
    0x81,           0x178f73d7c137, 0x1db55c68bf0b, 0xfe,
    0x1af5a29a81da, 0x16d997b0d5f2, 0x5550c6451a2,  0xaaa18c8a246,
    0x1a4ad63ec6f9, 0x1bdccd3d1661, 0xc6557473bba,  0x1bb11e66cebb,
    0x1cf704432ecf, 0x1158a15b9b9f, 0x19ee08865c87, 0x16b56ae8d871,
    0xcd5c6a3ad02,  0x1bdfbf39698a, 0x19a208277ffb, 0x12fc7fe301bd,
    0x1b86fadff52,  0xd69a7d5cef8,  0xcbb67fc0a7d,  0x1b018f825877,
};

uint32_t psi_8_15_matrix[] = {
    0x249249, 0xfac688, 0x6beb08, 0x5f58c8, 0xb1af88, 0xd635c8, 0x8d7d48,
    0x249248, 0xfac688, 0x6beb08, 0x5f58c8, 0xb1af88, 0xd635c8, 0x8d7d48,
    0x249248, 0xbc63b6, 0x589f1f, 0x64a909, 0x1c1bf6, 0xe9b612, 0x49e8a3,
    0x7032f1, 0x9a1a61, 0x587f2d, 0x51eb8a, 0x81cb7b, 0x226878, 0x906935,
    0xbc961f, 0xd2ca18, 0x1f20f7, 0x262303, 0x71fe2f, 0x61869b, 0x7599ab,
    0xc68707, 0xdd2ffe, 0xd61046, 0x5886f,  0x2d9c1b, 0xa0e726, 0x847fde,
    0x381558, 0x997bd5, 0x289950,
};

uint64_t psi_8_15_matrix_transpose[] = {
    0x8cfdaf30001,  0x3ffd31d8000,  0xb7958858000,  0x6d737837fff,
    0x1e927e2d8000, 0x1554ef48000,  0x1e70466474e9, 0xa1641349d3a,
    0x1f1c8b87ba74, 0xb1bb3edcb97,  0x79b10893972,  0x1ad52fd72e5c,
    0x1ea510ed74e9, 0xb11cdbaba74,  0xb012f90a74e,  0x11cf331b4b97,
    0x8d743182e5c,  0x28e440172e,   0x2fcd006f4e9,  0x1c9050dda74e,
    0xc3579e59d3a,  0x1587942acb97, 0x3f21bb172e,   0xb383a48b972,
};

uint64_t phi_9_17_matrix[] = {
    0x81,
    0x736825b5404dd,
    0x55510a2f581bf,
    0xfe,
    0x5db039267f896,
    0x199e0c3590113,
    0x6b5cf2af4c9c5,
    0x56b9e55e992c3,
    0x4ba381d7b2c3c,
    0x3a5efb89ed117,
    0x4ee30c9a3723b,
    0x35f55ec4d9ff7,
    0x3beb5167b83c,
    0x2325aa2ba90bb,
    0x77d6a2cf717a,
    0x19834710b3bf8,
    0x7fa9863d19921,
    0x2a85c931d4d0a,
    0x68e247b93716f,
    0x6ffb41d0ecda1,
    0x3e3fcea282e36,
    0x23ddbc995ea07,
    0x60d911c814b92,
    0x4bfd463f7499,
    0x102,
    0x66d04b6a809f1,
    0x2aa2145eb0335,
};

uint32_t psi_9_17_matrix[] = {
    0x249249,  0xfac688,  0x6beb08,  0x5f58c8,  0xb1af88,  0xd635c8,  0x8d7d48,
    0x249248,  0xfac688,  0x6beb08,  0x5f58c8,  0xb1af88,  0xd635c8,  0x8d7d48,
    0x249248,  0xfac688,  0x16beb08, 0x28981b3, 0x5da979e, 0xf43e2c,  0x16c7428,
    0x292c553, 0xd37a3e,  0x3959838, 0x11c012d, 0x1c6d993, 0x64d76ab, 0x7e921c9,
    0x4e7d2d2, 0x4d20b1f, 0x63c2457, 0x1bd8671, 0x504d3cb, 0x76552bc, 0x6c3c4a2,
    0x6ced7c1, 0x5ca11a1, 0xe2b414,  0x163fc10, 0x195cb6b, 0xf8b113,  0x26af0d0,
    0xbb07fe,  0x7c98b78, 0x56be2,   0x618d960, 0x1471481, 0x26f147b, 0x2a15ad6,
    0x7243ffe, 0x66316e2,
};

uint64_t psi_9_17_matrix_transpose[] = {
    0xc199ef220001,  0x7958576660000, 0x30422614c0000, 0x28c832dddffff,
    0x38f62f2e60000, 0x6bc9685da0000, 0x7be89d82074e9, 0x7561f1e069d3a,
    0x23d992b273a74, 0x71c8bb44dcb97, 0x6c46cc43cb972, 0x338c022c92e5c,
    0x7e37b16dc74e9, 0x213604c593a74, 0x132cf1671a74e, 0x2bed92a7cb97,
    0x5dcc69cc32e5c, 0x4c67c3265972e, 0x2d08bd79874e9, 0xaf18cd17a74e,
    0x2580e1ec9d3a,  0x78762d819cb97, 0x4cb7e3e5d972e, 0x10dbcbaeeb972,
    0x248d38b950000, 0x7aa0e4ca20000, 0x6281f7c040000,
};
} // namespace rmfe

namespace rmfe {
field::GF2_9 phi_2_3(std::array<field::GF2_3, 2> input) {
  field::GF2_9 result(0);
  for (size_t i = 0; i < 3 * 2; i++) {
    uint16_t mask = -((uint16_t)((input[i / 3].data >> (i % 3)) & 1));
    result.data ^= mask & phi_2_3_matrix[i];
  }
  return result;
}
std::array<field::GF2_3, 2> psi_2_3(field::GF2_9 input) {
  uint16_t temp = 0;
  for (size_t i = 0; i < 9; i++) {
    uint16_t mask = -((uint16_t)((input.data >> i) & 1));
    temp ^= mask & psi_2_3_matrix[i];
  }
  std::array<field::GF2_3, 2> result = {};
  for (size_t i = 0; i < 2; i++) {
    result[i] = field::GF2_3(temp & 0x7);
    temp = temp >> 3;
  }
  return result;
}
std::array<field::GF2_3, 2> psi_2_3_transpose(field::GF2_9 input) {
  uint16_t temp = 0;
  for (size_t i = 0; i < 6; i++) {
    temp ^= (_mm_popcnt_u64(input.data & psi_2_3_matrix_transpose[i]) & 1) << i;
  }
  std::array<field::GF2_3, 2> result = {};
  for (size_t i = 0; i < 2; i++) {
    result[i] = field::GF2_3(temp & 0x7);
    temp = temp >> 3;
  }
  return result;
}

field::GF2_45 phi_8_15(std::array<field::GF2_3, 8> input) {
  field::GF2_45 result(0);
  for (size_t i = 0; i < 3 * 8; i++) {
    uint64_t mask = -((uint64_t)((input[i / 3].data >> (i % 3)) & 1));
    result.data ^= mask & phi_8_15_matrix[i];
  }
  return result;
}
std::array<field::GF2_3, 8> psi_8_15(field::GF2_45 input) {
  uint32_t temp = 0;
  for (size_t i = 0; i < 45; i++) {
    uint32_t mask = -((uint32_t)((input.data >> i) & 1));
    temp ^= mask & psi_8_15_matrix[i];
  }
  std::array<field::GF2_3, 8> result = {};
  for (size_t i = 0; i < 8; i++) {
    result[i] = field::GF2_3(temp & 0x7);
    temp = temp >> 3;
  }
  return result;
}
std::array<field::GF2_3, 8> psi_8_15_transpose(field::GF2_45 input) {
  uint32_t temp = 0;
  for (size_t i = 0; i < 24; i++) {
    temp ^= (_mm_popcnt_u64(input.data & psi_8_15_matrix_transpose[i]) & 1)
            << i;
  }
  std::array<field::GF2_3, 8> result = {};
  for (size_t i = 0; i < 8; i++) {
    result[i] = field::GF2_3(temp & 0x7);
    temp = temp >> 3;
  }
  return result;
}

void phi_8_15(field::GF2_45 *output, const field::GF2_3 *input) {
  output->data = 0;
  uint64_t *matrix = phi_8_15_matrix;
  for (size_t i = 0; i < 8; i++, input++) {
    uint64_t tmp = input->data;
    for (size_t j = 0; j < 3; j++, matrix++) {
      uint64_t mask = -((uint64_t)((tmp >> j) & 1));
      output->data ^= mask & *matrix;
    }
  }
}
void psi_8_15(field::GF2_3 *output, const field::GF2_45 *input) {
  uint32_t temp = 0;
  for (size_t i = 0; i < 45; i++) {
    uint32_t mask = -((uint32_t)((input->data >> i) & 1));
    temp ^= mask & psi_8_15_matrix[i];
  }
  for (size_t i = 0; i < 8; i++, output++) {
    output->data = temp & 0x7;
    temp = temp >> 3;
  }
}

void psi_8_15_transpose(field::GF2_3 *output, const field::GF2_45 *input) {
  for (size_t i = 0; i < 8; i++) {
    output[i].data =
        ((_mm_popcnt_u64(input->data & psi_8_15_matrix_transpose[i * 3 + 0]) &
          1)
         << 0) |
        ((_mm_popcnt_u64(input->data & psi_8_15_matrix_transpose[i * 3 + 1]) &
          1)
         << 1) |
        ((_mm_popcnt_u64(input->data & psi_8_15_matrix_transpose[i * 3 + 2]) &
          1)
         << 2);
  }
}
} // namespace rmfe

namespace rmfe {
field::GF2_51 phi_9_17(std::array<field::GF2_3, 9> input) {
  field::GF2_51 result(0);
  for (size_t i = 0; i < 3 * 9; i++) {
    uint64_t mask = -((uint64_t)((input[i / 3].data >> (i % 3)) & 1));
    result.data ^= mask & phi_9_17_matrix[i];
  }
  return result;
}
std::array<field::GF2_3, 9> psi_9_17(field::GF2_51 input) {
  uint32_t temp = 0;
  for (size_t i = 0; i < 51; i++) {
    uint32_t mask = -((uint32_t)((input.data >> i) & 1));
    temp ^= mask & psi_9_17_matrix[i];
  }
  std::array<field::GF2_3, 9> result = {};
  for (size_t i = 0; i < 9; i++) {
    result[i] = field::GF2_3(temp & 0x7);
    temp = temp >> 3;
  }
  return result;
}
std::array<field::GF2_3, 9> psi_9_17_transpose(field::GF2_51 input) {
  uint32_t temp = 0;
  for (size_t i = 0; i < 27; i++) {
    temp ^= (_mm_popcnt_u64(input.data & psi_9_17_matrix_transpose[i]) & 1)
            << i;
  }
  std::array<field::GF2_3, 9> result = {};
  for (size_t i = 0; i < 9; i++) {
    result[i] = field::GF2_3(temp & 0x7);
    temp = temp >> 3;
  }
  return result;
}

void phi_9_17(field::GF2_51 *output, const field::GF2_3 *input) {
  output->data = 0;
  uint64_t *matrix = phi_9_17_matrix;
  for (size_t i = 0; i < 9; i++, input++) {
    uint64_t tmp = input->data;
    for (size_t j = 0; j < 3; j++, matrix++) {
      uint64_t mask = -((uint64_t)((tmp >> j) & 1));
      output->data ^= mask & *matrix;
    }
  }
}
void psi_9_17(field::GF2_3 *output, const field::GF2_51 *input) {
  uint32_t temp = 0;
  for (size_t i = 0; i < 51; i++) {
    uint32_t mask = -((uint32_t)((input->data >> i) & 1));
    temp ^= mask & psi_9_17_matrix[i];
  }
  for (size_t i = 0; i < 9; i++, output++) {
    output->data = temp & 0x7;
    temp = temp >> 3;
  }
}

void psi_9_17_transpose(field::GF2_3 *output, const field::GF2_51 *input) {
  for (size_t i = 0; i < 9; i++) {
    output[i].data =
        ((_mm_popcnt_u64(input->data & psi_9_17_matrix_transpose[i * 3 + 0]) &
          1)
         << 0) |
        ((_mm_popcnt_u64(input->data & psi_9_17_matrix_transpose[i * 3 + 1]) &
          1)
         << 1) |
        ((_mm_popcnt_u64(input->data & psi_9_17_matrix_transpose[i * 3 + 2]) &
          1)
         << 2);
  }
}

} // namespace rmfe