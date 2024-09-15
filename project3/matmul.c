#include "matmul.h"
#include "cblas.h"
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>

void matmul_plain(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    for (size_t i = 0; i < a->rows; i++)
    {
        for (size_t j = 0; j < b->cols; j++)
        {
            for (size_t k = 0; k < a->cols; k++)
            {
                c->data[i * c->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
}

void matmul_improved_plain(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < K; j++)
        {
            for (size_t k = 0; k < N; k++)
            {
                c_data[i * N + j] += a_data[i * N + k] * b_data[k * K + j];
            }
        }
    }
}

void matmul_improved_reordered(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    for (size_t i = 0; i < M; i++)
    {
        for (size_t k = 0; k < N; k++)
        {
            for (size_t j = 0; j < K; j++)
            {
                c_data[i * K + j] += a_data[i * N + k] * b_data[k * K + j];
            }
        }
    }
}

void matmul_improved_aligned(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    float __attribute__((aligned(256))) *a_data = a->data;
    float __attribute__((aligned(256))) *b_data = b->data;
    float __attribute__((aligned(256))) *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    for (size_t i = 0; i < M; i++)
    {
        for (size_t k = 0; k < N; k++)
        {
            for (size_t j = 0; j < K; j++)
            {
                c_data[i * K + j] += a_data[i * N + k] * b_data[k * K + j];
            }
        }
    }
}

void matmul_improved_simd(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m256 vecA;
    __m256 vecB;
    __m256 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    for (size_t i = 0; i < M; i++)
    {
        vecC = _mm256_setzero_ps();
        for (size_t k = 0; k < N; k++)
        {
            vecA = _mm256_set1_ps(a_data[i * N + k]);
            for (size_t j = 0; j < K; j += 8)
            {
                vecB = _mm256_load_ps(&b_data[k * K + j]);
                vecC = _mm256_load_ps(&c_data[i * K + j]);
                vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                _mm256_store_ps(&c_data[i * K + j], vecC);
            }
        }
    }
}

void matmul_improved_batched(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m256 vecA;
    __m256 vecB;
    __m256 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    for (size_t ii = 0; ii < M; ii += BATCH_SIZE)
    {
        for (size_t jj = 0; jj < N; jj += BATCH_SIZE)
        {
            for (size_t i = ii; i < ii + BATCH_SIZE && i < M; i++)
            {
                vecC = _mm256_setzero_ps();
                for (size_t j = jj; j < jj + BATCH_SIZE && j < N; j++)
                {
                    vecA = _mm256_set1_ps(a_data[i * N + j]);
                    for (size_t k = 0; k < K; k += 8)
                    {
                        vecB = _mm256_load_ps(&b_data[j * K + k]);
                        vecC = _mm256_load_ps(&c_data[i * K + k]);
                        vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                        _mm256_store_ps(&c_data[i * K + k], vecC);
                    }
                }
            }
        }
    }
}

void matmul_improved_parallel(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m256 vecA;
    __m256 vecB;
    __m256 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

#pragma omp parallel for private(vecA, vecB, vecC)
    for (size_t ii = 0; ii < M; ii += BATCH_SIZE)
    {
        for (size_t jj = 0; jj < N; jj += BATCH_SIZE)
        {
            for (size_t i = ii; i < ii + BATCH_SIZE && i < M; i++)
            {
                vecC = _mm256_setzero_ps();
                for (size_t j = jj; j < jj + BATCH_SIZE && j < N; j++)
                {
                    vecA = _mm256_set1_ps(a_data[i * N + j]);
                    for (size_t k = 0; k < K; k += 8)
                    {
                        vecB = _mm256_load_ps(&b_data[j * K + k]);
                        vecC = _mm256_load_ps(&c_data[i * K + k]);
                        vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                        _mm256_store_ps(&c_data[i * K + k], vecC);
                    }
                }
            }
        }
    }
}

void matmul_improved_gepb(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    for (size_t n0 = 0; n0 < N; n0 += 16)
    {
        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = n0; n < n0 + 16; n++)
                {
                    for (size_t k = k0; k < k0 + 16; k++)
                    {
                        c_data[m * K + k] += a_data[m * N + n] * b_data[n * K + k];
                    }
                }
            }
        }
    }
}

void matmul_improved_gepb_packedB(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{

    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    float temp[16 * 16];

    for (size_t n0 = 0; n0 < N; n0 += 16)
    {
        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            for (size_t n = 0; n < 16; n++)
            {
                for (size_t k = 0; k < 16; k++)
                {
                    temp[n * 16 + k] = b_data[(n0 + n) * K + k0 + k];
                }
            }

            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = n0; n < n0 + 16; n++)
                {
                    for (size_t k = k0; k < k0 + 16; k++)
                    {
                        c_data[m * K + k] += a_data[m * N + n] * temp[(n - n0) * 16 + (k - k0)];
                    }
                }
            }
        }
    }
}

void matmul_improved_gepb_packedB_simd(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m256 vecA;
    __m256 vecB;
    __m256 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    float B_packed[16 * 16];

    for (size_t n0 = 0; n0 < N; n0 += 16)
    {
        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            for (size_t n = 0; n < 16; n++)
            {
                for (size_t k = 0; k < 16; k += 8)
                {
                    _mm256_store_ps(&B_packed[n * 16 + k], _mm256_load_ps(&b_data[(n0 + n) * K + k0 + k]));
                }
            }

            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = 0; n < 16; n++)
                {
                    vecA = _mm256_set1_ps(a_data[m * N + n0 + n]);
                    for (size_t k = 0; k < 16; k += 8)
                    {
                        vecC = _mm256_load_ps(&c_data[m * K + k0 + k]);
                        vecB = _mm256_load_ps(&B_packed[n * 16 + k]);
                        vecC = _mm256_add_ps(vecC, _mm256_mul_ps(vecA, vecB));
                        _mm256_store_ps(&c_data[m * K + k0 + k], vecC);
                    }
                }
            }
        }
    }
}
void matmul_improved_gepb_packedB_parallel(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m256 vecA;
    __m256 vecB;
    __m256 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    float B_packed[16 * 16];
#pragma omp parallel for collapse(2) private(vecA, vecB, vecC, B_packed)
    for (size_t n0 = 0; n0 < N; n0 += 16)
    {
        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            for (size_t n = 0; n < 16; n++)
            {
                for (size_t k = 0; k < 16; k += 8)
                {
                    _mm256_store_ps(&B_packed[n * 16 + k], _mm256_load_ps(&b_data[(n0 + n) * K + k0 + k]));
                }
            }
            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = 0; n < 16; n++)
                {
                    vecA = _mm256_set1_ps(a_data[m * N + n0 + n]);
                    for (size_t k = 0; k < 16; k += 8)
                    {
                        vecC = _mm256_load_ps(&c_data[m * K + k0 + k]);
                        vecB = _mm256_load_ps(&B_packed[n * 16 + k]);
                        vecC = _mm256_add_ps(vecC, _mm256_mul_ps(vecA, vecB));
                        _mm256_store_ps(&c_data[m * K + k0 + k], vecC);
                    }
                }
            }
        }
    }
}
void matmul_improved_gepb_packed(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{

    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    float B_packed[16 * 16];
    float *A_packed = aligned_alloc(32, M * 16 * sizeof(float));
    float *C_packed = aligned_alloc(32, M * K * sizeof(float));

    for (size_t n0 = 0; n0 < N; n0 += 16)
    {
        // pack A
        for (size_t m = 0; m < M; m++)
        {
            for (size_t n = 0; n < 16; n++)
            {
                A_packed[m * 16 + n] = a_data[m * N + n0 + n];
            }
        }

        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            // pack B
            for (size_t n = 0; n < 16; n++)
            {
                for (size_t k = 0; k < 16; k++)
                {
                    B_packed[n * 16 + k] = b_data[(n0 + n) * K + k0 + k];
                }
            }

            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = n0; n < n0 + 16; n++)
                {
                    for (size_t k = k0; k < k0 + 16; k++)
                    {
                        C_packed[k0 * M + m * 16 + k - k0] += A_packed[m * 16 + n - n0] * B_packed[(n - n0) * 16 + (k - k0)];
                    }
                }
            }
        }
    }

    float *ptr = C_packed;
    for (size_t k0 = 0; k0 < K; k0 += 16)
    {
        for (size_t m = 0; m < M; m++)
        {
            for (size_t k = 0; k < 16; k++)
            {
                c_data[m * K + k0 + k] = *ptr++;
            }
        }
    }
    free(A_packed);
    free(C_packed);
}

void matmul_improved_gepb_packed_simd(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m256 vecA;
    __m256 vecB;
    __m256 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    float B_packed[16 * 16] __attribute__((aligned(32)));
    float *A_packed = aligned_alloc(32, M * 16 * sizeof(float));
    float *C_packed = aligned_alloc(32, M * K * sizeof(float));

    for (size_t n0 = 0; n0 < N; n0 += 16)
    {
        // pack A
        for (size_t m = 0; m < M; m++)
        {
            for (size_t n = 0; n < 16; n += 8)
            {
                _mm256_store_ps(&A_packed[m * 16 + n], _mm256_load_ps(&a_data[m * N + n0 + n]));
            }
        }
        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            // pack B
            for (size_t n = 0; n < 16; n++)
            {
                for (size_t k = 0; k < 16; k += 8)
                {
                    _mm256_store_ps(&B_packed[n * 16 + k], _mm256_load_ps(&b_data[(n0 + n) * K + k0 + k]));
                }
            }
            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = 0; n < 16; n++)
                {
                    for (size_t k = 0; k < 16; k += 8)
                    {
                        vecA = _mm256_broadcast_ss(A_packed + m * 16 + n);
                        vecB = _mm256_load_ps(B_packed + n * 16 + k);
                        vecC = _mm256_load_ps(C_packed + k0 * M + m * 16 + k);
                        vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                        _mm256_store_ps(C_packed + k0 * M + m * 16 + k, vecC);
                    }
                }
            }
        }
    }

    float *ptr = C_packed;
    for (size_t k0 = 0; k0 < K; k0 += 16)
    {
        for (size_t m = 0; m < M; m++)
        {
            __m256 ymm0 = _mm256_load_ps(ptr);
            __m256 ymm1 = _mm256_load_ps(ptr + 8);
            _mm256_store_ps(c_data + m * K + k0, ymm0);
            _mm256_store_ps(c_data + m * K + k0 + 8, ymm1);
            ptr += 16;
        }
    }
    free(A_packed);
    free(C_packed);
}

void matmul_improved_gepb_packed_parallel(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m256 vecA;
    __m256 vecB;
    __m256 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    float B_packed[16 * 16] __attribute__((aligned(32)));
    float *A_packed = aligned_alloc(32, M * 16 * sizeof(float));
    float *C_packed = aligned_alloc(32, M * K * sizeof(float));

    for (size_t n0 = 0; n0 < N; n0 += 16)
    {
        // pack A
        for (size_t m = 0; m < M; m++)
        {
            for (size_t n = 0; n < 16; n += 8)
            {
                _mm256_store_ps(&A_packed[m * 16 + n], _mm256_load_ps(&a_data[m * N + n0 + n]));
            }
        }
#pragma omp parallel for private(vecA, vecB, vecC, B_packed)
        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            // pack B
            for (size_t n = 0; n < 16; n++)
            {
                for (size_t k = 0; k < 16; k += 8)
                {
                    _mm256_store_ps(&B_packed[n * 16 + k], _mm256_load_ps(&b_data[(n0 + n) * K + k0 + k]));
                }
            }
            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = 0; n < 16; n++)
                {
                    for (size_t k = 0; k < 16; k += 8)
                    {
                        vecA = _mm256_broadcast_ss(A_packed + m * 16 + n);
                        vecB = _mm256_load_ps(B_packed + n * 16 + k);
                        vecC = _mm256_load_ps(C_packed + k0 * M + m * 16 + k);
                        vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                        _mm256_store_ps(C_packed + k0 * M + m * 16 + k, vecC);
                    }
                }
            }
        }
    }

    float *ptr = C_packed;
    for (size_t k0 = 0; k0 < K; k0 += 16)
    {
        for (size_t m = 0; m < M; m++)
        {
            __m256 ymm0 = _mm256_load_ps(ptr);
            __m256 ymm1 = _mm256_load_ps(ptr + 8);
            _mm256_store_ps(c_data + m * K + k0, ymm0);
            _mm256_store_ps(c_data + m * K + k0 + 8, ymm1);
            ptr += 16;
        }
    }
    free(A_packed);
    free(C_packed);
}
void matmul_improved_gepb_packed_dynamic(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m256 vecA;
    __m256 vecB;
    __m256 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    size_t chunk_size_n=16;
    size_t chunk_size_k=16;
    chunk_size_n =(N<32)?N:32;
    chunk_size_k=(K<256)?K:256;

    float B_packed[chunk_size_n * chunk_size_k] __attribute__((aligned(32)));
    float *A_packed = aligned_alloc(32, M * chunk_size_n * sizeof(float));
    float *C_packed = aligned_alloc(32, M * K * sizeof(float));
    
    for (size_t n0 = 0; n0 < N; n0 += chunk_size_n)
    {
        // pack A
        for (size_t m = 0; m < M; m++)
        {
            for (size_t n = 0; n < chunk_size_n; n += 8)
            {
                _mm256_store_ps(&A_packed[m * chunk_size_n + n], _mm256_load_ps(&a_data[m * N + n0 + n]));
            }
        }
#pragma omp parallel for private(vecA, vecB, vecC, B_packed)
        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            // pack B
            for (size_t n = 0; n < chunk_size_n; n++)
            {
                for (size_t k = 0; k < chunk_size_k; k += 8)
                {
                    _mm256_store_ps(&B_packed[n * chunk_size_k + k], _mm256_load_ps(&b_data[(n0 + n) * K + k0 + k]));
                }
            }
            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = 0; n < chunk_size_n; n++)
                {
                    for (size_t k = 0; k < chunk_size_k; k += 8)
                    {
                        vecA = _mm256_broadcast_ss(A_packed + m * chunk_size_n + n);
                        vecB = _mm256_load_ps(B_packed + n * chunk_size_k + k);
                        vecC = _mm256_load_ps(C_packed + k0 * M + m * chunk_size_n + k);
                        vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                        // _mm256_store_ps(C_packed + k0 * M + m * chunk_size_n + k, vecC);
                    }
                }
            }
        }
    }

    float *ptr = C_packed;
    for (size_t k0 = 0; k0 < K; k0 += chunk_size_k)
    {
        for (size_t m = 0; m < M; m++)
        {
            for (size_t k = 0; k < chunk_size_k; k += 8)
            {
                __m256 ymm = _mm256_load_ps(ptr);
                _mm256_store_ps(c_data + m * K + k0 + k, ymm);
                ptr += 8;
            }
        }
    }
    free(A_packed);
    free(C_packed);
}

void matmul_improved_gepb_packed_avx512(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    __m512 vecA;
    __m512 vecB;
    __m512 vecC;
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    size_t chunk_size_n=16;
    size_t chunk_size_k=16;
    // chunk_size_n =(N<64)?N:64;
    // chunk_size_k=(K<512)?K:512;

    float B_packed[16 * 16] __attribute__((aligned(64)));
    float *A_packed = aligned_alloc(64, M * chunk_size_n * sizeof(float));
    float *C_packed = aligned_alloc(64, M * K * sizeof(float));
    

    
    for (size_t n0 = 0; n0 < N; n0 += chunk_size_n)
    {
        // pack A
        for (size_t m = 0; m < M; m++)
        {
            for (size_t n = 0; n < chunk_size_n; n += 16)
            {
                _mm512_store_ps(&A_packed[m * chunk_size_n + n], _mm512_load_ps(&a_data[m * N + n0 + n]));
            }
        }
// #pragma omp parallel for private(vecA, vecB, vecC, B_packed)
        for (size_t k0 = 0; k0 < K; k0 += 16)
        {
            // pack B
            for (size_t n = 0; n < 16; n++)
            {
                for (size_t k = 0; k < chunk_size_k; k += 16)
                {
                    _mm512_store_ps(&B_packed[n * 16 + k], _mm512_load_ps(&b_data[(n0 + n) * K + k0 + k]));
                }
            }
            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = 0; n < chunk_size_n; n++)
                {
                    for (size_t k = 0; k < chunk_size_k; k += 16)
                    {
                        vecA = _mm512_broadcastss_ps(_mm_load_ss(A_packed + m * chunk_size_n + n));
                        vecB = _mm512_load_ps(B_packed + n * 16 + k);
                        vecC = _mm512_load_ps(C_packed + k0 * M + m * chunk_size_n + k);
                        vecC = _mm512_fmadd_ps(vecA, vecB, vecC);
                        _mm512_store_ps(C_packed + k0 * M + m * chunk_size_n + k, vecC);
                    }
                }
            }
        }
    }

    float *ptr = C_packed;
    for (size_t k0 = 0; k0 < K; k0 += 16)
    {
        for (size_t m = 0; m < M; m++)
        {
            __m512 ymm0 = _mm512_load_ps(ptr);
            _mm512_store_ps(c_data + m * K + k0, ymm0);
            ptr += 16;
        }
    }
    free(A_packed);
    free(C_packed);
}

void matmul_improved_gepb_test(struct Matrix *a, struct Matrix *b, struct Matrix *c,int N0,int K0)
{
    float *a_data = a->data;
    float *b_data = b->data;
    float *c_data = c->data;
    size_t M = a->rows;
    size_t N = a->cols;
    size_t K = b->cols;

    for (size_t n0 = 0; n0 < N; n0 += N0)
    {
        for (size_t k0 = 0; k0 < K; k0 += K0)
        {
            for (size_t m = 0; m < M; m++)
            {
                for (size_t n = n0; n < n0 + N0; n++)
                {
                    for (size_t k = k0; k < k0 + K0; k++)
                    {
                        c_data[m * K + k] += a_data[m * N + n] * b_data[n * K + k];
                    }
                }
            }
        }
    }
}

void matmul_openblas(struct Matrix *a, struct Matrix *b, struct Matrix *c)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->rows, b->cols, a->cols, 1.0, a->data, a->cols, b->data, b->cols, 0.0, c->data, c->cols);
}

// void matmul(struct Matrix *a, struct Matrix *b, struct Matrix *c, void (*matmul_impl)(struct Matrix *, struct Matrix *, struct Matrix *))
// {
//     matmul_impl(a, b, c);
// }
// int main()
// {
//     struct Matrix a = {SIZE, SIZE, (float *)aligned_alloc(32, SIZE * SIZE * sizeof(float))};
//     struct Matrix b = {SIZE, SIZE, (float *)aligned_alloc(32, SIZE * SIZE * sizeof(float))};
//     struct Matrix c1 = {SIZE, SIZE, (float *)aligned_alloc(32, SIZE * SIZE * sizeof(float))};
//     struct Matrix c2 = {SIZE, SIZE, (float *)aligned_alloc(32, SIZE * SIZE * sizeof(float))};

//     for (size_t i = 0; i < SIZE * SIZE; i++)
//     {
//         a.data[i] = (float)i;
//         b.data[i] = (float)i;
//     }
//     struct timeval start, end;
//     long seconds, micros;

//     gettimeofday(&start, NULL);
//     matmul(&a, &b, &c1, matmul_improved_gepb_packedB_simd);
//     gettimeofday(&end, NULL);

//     seconds = (end.tv_sec - start.tv_sec);
//     micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

//     printf("Time Plain: %f\n", seconds + micros * 1e-6);

//     gettimeofday(&start, NULL);
//     matmul(&a, &b, &c2, matmul_improved_gepb_packedB_parallel);
//     gettimeofday(&end, NULL);

//     seconds = (end.tv_sec - start.tv_sec);
//     micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

//     printf("Time SIMD: %f\n", seconds + micros * 1e-6);

//     for (size_t i = 0; i < SIZE * SIZE; i++)
//     {
//         if (c1.data[i] != c2.data[i])
//         {
//             printf("Results do not match!\n");
//             printf("c1[%ld] = %f, c2[%ld] = %f\n", i, c1.data[i], i, c2.data[i]);
//             break;
//         }
//     }

//     free(a.data);
//     free(b.data);
//     free(c1.data);
//     free(c2.data);

//     return 0;
// }