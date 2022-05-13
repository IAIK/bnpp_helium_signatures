#
#include <array>
#include <cstdint>

namespace matmul {

std::array<uint64_t, 3> multiply_with_transposed_GF2_matrix_129(
    const std::array<uint64_t, 3> input,
    std::pair<const std::array<uint64_t, 4> *, const std::array<uint64_t, 4> *>
        matrix);
std::array<uint64_t, 4>
multiply_with_transposed_GF2_matrix_128(const std::array<uint64_t, 4> input,
                                        const std::array<uint64_t, 4> *matrix);
} // namespace matmul