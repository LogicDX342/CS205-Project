#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

struct timeval start;
struct timeval end;

#ifdef METHOD1
float A_1[n][n];
float B_1[n][n];
float C_1[n][n];
#endif

#ifdef METHOD2
float A_2[n * n];
float B_2[n * n];
float C_2[n * n];
#endif

void method1()
{
#ifdef METHOD1
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A_1[i][j] = (float)rand() / (RAND_MAX);
            B_1[i][j] = (float)rand() / (RAND_MAX);
            C_1[i][j] = 0;
        }
    }
    gettimeofday(&start, NULL);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C_1[i][j] = C_1[i][j] + A_1[i][k] * B_1[k][j];
            }
        }
    }

    gettimeofday(&end, NULL);
    printf("Method1: %ld ms\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000);
#endif
}

void method2()
{
#ifdef METHOD2
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A_2[i * n + j] = (float)rand() / (RAND_MAX);
            B_2[i * n + j] = (float)rand() / (RAND_MAX);
            C_2[i * n + j] = 0;
        }
    }
    gettimeofday(&start, NULL);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C_2[i * n + j] += A_2[i * n + k] * B_2[k * n + j];
            }
        }
    }

    gettimeofday(&end, NULL);
    printf("Method2: %ld ms\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000);
#endif
}

void method3()
{
#ifdef METHOD3
    float *A = (float *)malloc(n * n * sizeof(float));
    float *B = (float *)malloc(n * n * sizeof(float));
    float *C = (float *)malloc(n * n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = (float)rand() / (RAND_MAX);
            B[i * n + j] = (float)rand() / (RAND_MAX);
            C[i * n + j] = 0;
        }
    }
    gettimeofday(&start, NULL);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }

    gettimeofday(&end, NULL);
    printf("Method3: %ld ms\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000);
    free(A);
    free(B);
    free(C);
#endif
}

void method4()
{
#ifdef METHOD4
    float *A = (float *)malloc(n * n * sizeof(float));
    float *B = (float *)malloc(n * n * sizeof(float));
    float *C = (float *)malloc(n * n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = (float)rand() / (RAND_MAX);
            B[i * n + j] = (float)rand() / (RAND_MAX);
            C[i * n + j] = 0;
        }
    }
    gettimeofday(&start, NULL);

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }

    gettimeofday(&end, NULL);
    printf("Method4: %ld ms\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000);
    free(A);
    free(B);
    free(C);
#endif
}

int main()
{
    srand((unsigned)time(NULL));
#ifdef METHOD1
    method1();
#endif
#ifdef METHOD2
    method2();
#endif
#ifdef METHOD3
    method3();
#endif
#ifdef METHOD4
    method4();
#endif
    return 0;
}
