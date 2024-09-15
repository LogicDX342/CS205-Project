#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// #define SIZE (size_t)528
#define BATCH_SIZE 32
#define ITERATIONS (1e9 / (SIZE * SIZE))

void check_result(float *A, float *B, float *C, size_t SIZE, int rank, int size)
{
    float *C_check = (float *)aligned_alloc(32, SIZE * SIZE * sizeof(float));
    for (size_t i = 0; i < SIZE / size; i++)
    {
        for (size_t j = 0; j < SIZE; j++)
        {
            C_check[i * SIZE + j] = 0;
            for (size_t k = 0; k < SIZE; k++)
            {
                C_check[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
            }
        }
    }
    for (size_t i = 0; i < SIZE / size; i++)
    {
        for (size_t j = 0; j < SIZE; j++)
        {
            if (C[i * SIZE + j] != C_check[i * SIZE + j])
            {
                printf("Error: C[%ld][%ld] = %f, C_check[%ld][%ld] = %f\n", i, j, C[i * SIZE + j], i, j, C_check[i * SIZE + j]);
            }
        }
    }
    free(C_check);
}

int matmul(int argc, char *argv[], size_t SIZE)
{
    int rank, size, i, j, k;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    float *A = (float *)aligned_alloc(32, SIZE * SIZE * sizeof(float));
    float *B = (float *)aligned_alloc(32, SIZE * SIZE * sizeof(float));
    float *C = (float *)aligned_alloc(32, SIZE * SIZE * sizeof(float));

    if (rank == 0)
    {
        for (i = 0; i < SIZE; i++)
        {
            for (j = 0; j < SIZE; j++)
            {
                A[i * SIZE + j] = i + j;
                B[i * SIZE + j] = i - j;
                C[i * SIZE + j] = 0;
            }
        }
    }

    double start = MPI_Wtime();

    MPI_Bcast(B, SIZE * SIZE, MPI_FLOAT, 0, comm);

    MPI_Scatter(A, SIZE * SIZE / size, MPI_FLOAT, &A[rank * SIZE * SIZE / size], SIZE * SIZE / size, MPI_FLOAT, 0, comm);
    for (int cnt = 0; cnt < ITERATIONS; cnt++)
    {

        __m256 vecA;
        __m256 vecB;
        __m256 vecC;
        for (size_t ii = 0; ii < SIZE / size; ii += BATCH_SIZE)
        {
            for (size_t jj = 0; jj < SIZE; jj += BATCH_SIZE)
            {
                for (size_t i = ii; i < ii + BATCH_SIZE && i < SIZE / size; i++)
                {
                    vecC = _mm256_setzero_ps();
                    for (size_t j = jj; j < jj + BATCH_SIZE && j < SIZE; j++)
                    {
                        vecA = _mm256_set1_ps(A[(i + rank * SIZE / size) * SIZE + j]);
                        for (size_t k = 0; k < SIZE; k += 8)
                        {
                            vecB = _mm256_load_ps(&B[j * SIZE + k]);
                            vecC = _mm256_load_ps(&C[(i + rank * SIZE / size) * SIZE + k]);
                            vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                            _mm256_store_ps(&C[(i + rank * SIZE / size) * SIZE + k], vecC);
                        }
                    }
                }
            }
        }
    }

    MPI_Gather(&C[rank * SIZE * SIZE / size], SIZE * SIZE / size, MPI_FLOAT, C, SIZE * SIZE / size, MPI_FLOAT, 0, comm);
    MPI_Barrier(comm);
    double finish = MPI_Wtime();

    if (rank == 0)
    {
        printf("\"BM_Matmul<matmul_improved_mpi>/%ld\",%f\n", SIZE, (finish - start) * 1e9 / ITERATIONS);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    for (size_t i = 16; i <= 1024; i += 16)
    {
        matmul(argc, argv, i);
    }
    MPI_Finalize();
}