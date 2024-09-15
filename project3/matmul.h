#ifndef MATMUL_H
#define MATMUL_H

#include <stdlib.h>

#define SIZE 2048
#define BATCH_SIZE 32

struct Matrix
{
    size_t rows;
    size_t cols;
    float *data;
};
#ifdef __cplusplus
extern "C"
{
#endif
    void matmul_plain(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_plain(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_reordered(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_simd(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_batched(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_parallel(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb_packedB(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb_packedB_simd(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb_packedB_parallel(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb_packed(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb_packed_simd(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb_packed_parallel(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb_packed_dynamic(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_openblas(struct Matrix *a, struct Matrix *b, struct Matrix *c);
    void matmul_improved_gepb_test(struct Matrix *a, struct Matrix *b, struct Matrix *c, int N0, int K0);

#ifdef __cplusplus
}
#endif

#endif // MATMUL_H