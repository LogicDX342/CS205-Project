#include "matmul.h"
#include <benchmark/benchmark.h>
#include <iostream>

// create matrix
struct Matrix a, b, c;

static void init_matrix(Matrix *mat, size_t size = 128)
{
    mat->rows = size;
    mat->cols = size;
    mat->data = (float *)aligned_alloc(64, size * size * sizeof(float));
    for (size_t i = 0; i < size * size; i++)
    {
        mat->data[i] = (float)rand() / (RAND_MAX);
    }
}

template <void (*matmul_func)(Matrix *, Matrix *, Matrix *)>
void BM_Matmul(benchmark::State &state)
{
    Matrix a, b, c;
    init_matrix(&a, state.range(0));
    init_matrix(&b, state.range(0));
    init_matrix(&c, state.range(0));

    for (auto _ : state)
    {
        matmul_func(&a, &b, &c);
    }
}
// void BM_Matmul_test(benchmark::State &state)
// {
//     Matrix a, b, c;
//     init_matrix(&a, state.range(0));
//     init_matrix(&b, state.range(0));
//     init_matrix(&c, state.range(0));
//     int N0 = state.range(1);
//     int K0 = state.range(2);
//     for (auto _ : state)
//     {
//         matmul_improved_gepb_test(&a, &b, &c, N0, K0);
//     }
// }

// BENCHMARK(BM_Matmul_test)->ArgsProduct({{1024}, benchmark::CreateDenseRange(16, 256, 16), benchmark::CreateDenseRange(16, 256, 16)});

#define BENCHMARK_ARGS() ->DenseRange(16, 1024, 16)
BENCHMARK_TEMPLATE(BM_Matmul, matmul_plain)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_plain)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_reordered)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_simd)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_batched)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_parallel)BENCHMARK_ARGS();

BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_gepb)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_gepb_packedB)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_gepb_packed)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_gepb_packed_simd)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_gepb_packed_parallel)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_improved_gepb_packed_avx512)BENCHMARK_ARGS();
BENCHMARK_TEMPLATE(BM_Matmul, matmul_openblas)BENCHMARK_ARGS();

BENCHMARK_MAIN();