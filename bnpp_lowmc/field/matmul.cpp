#include "matmul.h"

extern "C" {
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <wmmintrin.h>
}

namespace {

static inline __m256i mm256_compute_mask(const uint64_t idx, const size_t bit) {
  return _mm256_set1_epi64x(-((idx >> bit) & 1));
}

static inline __m256i mm256_compute_mask_2(const uint64_t idx,
                                           const size_t bit) {
  const uint64_t m1 = -((idx >> bit) & 1);
  const uint64_t m2 = -((idx >> (bit + 1)) & 1);
  return _mm256_set_epi64x(m2, m2, m1, m1);
}
} // namespace

namespace matmul {

std::array<uint64_t, 3> multiply_with_transposed_GF2_matrix_129(
    const std::array<uint64_t, 3> input,
    std::pair<const std::array<uint64_t, 4> *, const std::array<uint64_t, 4> *>
        matrix) {
  const uint64_t *vptr = input.data() + 1;
  const __m256i *Ablock =
      reinterpret_cast<const __m256i *>(matrix.first->data());

  __m256i cval[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
  cval[0] = _mm256_and_si256(Ablock[0], mm256_compute_mask_2(input[0], 62));
  Ablock++;

  for (unsigned int w = 2; w; --w, ++vptr) {
    uint64_t idx = *vptr;
    for (unsigned int i = sizeof(uint64_t) * 8; i;
         i -= 8, idx >>= 8, Ablock += 4) {
      cval[0] = _mm256_xor_si256(
          cval[0], _mm256_and_si256(Ablock[0], mm256_compute_mask_2(idx, 0)));
      cval[1] = _mm256_xor_si256(
          cval[1], _mm256_and_si256(Ablock[1], mm256_compute_mask_2(idx, 2)));
      cval[0] = _mm256_xor_si256(
          cval[0], _mm256_and_si256(Ablock[2], mm256_compute_mask_2(idx, 4)));
      cval[1] = _mm256_xor_si256(
          cval[1], _mm256_and_si256(Ablock[3], mm256_compute_mask_2(idx, 6)));
    }
  }
  alignas(32) std::array<uint64_t, 3> result{};
  cval[0] = _mm256_xor_si256(cval[0], cval[1]);
  __m128i res = _mm256_extracti128_si256(
      _mm256_xor_si256(
          cval[0], _mm256_permute4x64_epi64(cval[0], _MM_SHUFFLE(3, 2, 3, 2))),
      0);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(result.data() + 1), res);
  // we have handled the first 128 columns, handle the 129th one
  alignas(32) std::array<uint64_t, 3> tmp = input;
  tmp[0] &= matrix.second->data()[0];
  tmp[1] &= matrix.second->data()[1];
  tmp[2] &= matrix.second->data()[2];
  uint64_t bit = __builtin_parityll(tmp[0] ^ tmp[1] ^ tmp[2]);
  result[0] = bit << 63;

  return result;
}

std::array<uint64_t, 4>
multiply_with_transposed_GF2_matrix_128(const std::array<uint64_t, 4> input,
                                        const std::array<uint64_t, 4> *matrix) {
  const uint64_t *vptr = input.data();
  const __m256i *Ablock = reinterpret_cast<const __m256i *>(matrix->data());

  __m256i cval[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
  for (unsigned int w = 2; w; --w, ++vptr) {
    uint64_t idx = *vptr;
    for (unsigned int i = sizeof(uint64_t) * 8; i;
         i -= 8, idx >>= 8, Ablock += 4) {
      cval[0] = _mm256_xor_si256(
          cval[0], _mm256_and_si256(Ablock[0], mm256_compute_mask_2(idx, 0)));
      cval[1] = _mm256_xor_si256(
          cval[1], _mm256_and_si256(Ablock[1], mm256_compute_mask_2(idx, 2)));
      cval[0] = _mm256_xor_si256(
          cval[0], _mm256_and_si256(Ablock[2], mm256_compute_mask_2(idx, 4)));
      cval[1] = _mm256_xor_si256(
          cval[1], _mm256_and_si256(Ablock[3], mm256_compute_mask_2(idx, 6)));
    }
  }
  cval[0] = _mm256_xor_si256(cval[0], cval[1]);
  alignas(32) std::array<uint64_t, 4> result{};
  __m128i *resultBlock = reinterpret_cast<__m128i *>(result.data());
  *resultBlock = _mm256_extracti128_si256(
      _mm256_xor_si256(
          cval[0], _mm256_permute4x64_epi64(cval[0], _MM_SHUFFLE(3, 2, 3, 2))),
      0);
  return result;
}
} // namespace matmul