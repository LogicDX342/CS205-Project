#include <cblas.h>
#include <stdio.h>
#include <sys/time.h>

#define TIME_START gettimeofday(&t_start, NULL);
#define TIME_END(time)                                                         \
  gettimeofday(&t_end, NULL);                                                  \
  time = (t_end.tv_sec - t_start.tv_sec) * 1000.0;                             \
  time += (t_end.tv_usec - t_start.tv_usec) / 1000.0;

typedef struct {
  size_t rows;
  size_t cols;
  float *data;        // CPU memory
  float *data_device; // GPU mememory
} Matrix;

Matrix *createMatrix(size_t r, size_t c) {
  size_t len = r * c;
  if (len == 0) {
    fprintf(stderr, "Invalid size. The input should be > 0.\n");
    return NULL;
  }
  Matrix *p = (Matrix *)malloc(sizeof(Matrix));
  if (p == NULL) {
    fprintf(stderr, "Allocate host memory failed.\n");
    goto ERR_TAG;
  }
  p->rows = r;
  p->cols = c;
  p->data = (float *)malloc(sizeof(float) * len);
  if (p->data == NULL) {
    fprintf(stderr, "Allocate host memory failed.\n");
    goto ERR_TAG;
  }
  if (cudaMalloc(&p->data_device, sizeof(float) * len) != cudaSuccess) {
    fprintf(stderr, "Allocate device memory failed.\n");
    goto ERR_TAG;
  }
  return p;
ERR_TAG:
  if (p && p->data)
    free(p->data);
  if (p)
    free(p);
  return NULL;
}

void freeMatrix(Matrix **pp) {
  if (pp == NULL)
    return;
  Matrix *p = *pp;
  if (p != NULL) {
    if (p->data)
      free(p->data);
    if (p->data_device)
      cudaFree(p->data_device);
  }
  *pp = NULL;
}

// a simple function to set all elements to the same value
bool setMatrix(Matrix *pMat, float val) {
  if (pMat == NULL) {
    fprintf(stderr, "NULL pointer.\n");
    return false;
  }
  size_t len = pMat->rows * pMat->cols;
  for (size_t i = 0; i < len; i++)
    pMat->data[i] = val;

  return true;
}

bool mulCPU(const Matrix *pMat1, const Matrix *pMat2, Matrix *pMat3,
            const float alpha, const float beta) {
  if (pMat1 == NULL || pMat2 == NULL || pMat3 == NULL) {
    fprintf(stderr, "Null pointer.\n");
    return false;
  }
  if (pMat1->rows != pMat2->rows || pMat1->cols != pMat2->cols ||
      pMat2->rows != pMat3->rows || pMat2->cols != pMat3->cols) {
    fprintf(stderr, "The 3 matrics are not in the same size.\n");
    return false;
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pMat1->rows,
              pMat1->cols, pMat2->cols, alpha, pMat1->data, pMat1->cols,
              pMat2->data, pMat2->cols, beta, pMat3->data, pMat3->cols);
  return true;
}

#include <cublas_v2.h>

bool mulGPU(const Matrix *pMat1, const Matrix *pMat2, Matrix *pMat3,
            const float alpha, const float beta) {
  if (pMat1 == NULL || pMat2 == NULL || pMat3 == NULL) {
    fprintf(stderr, "Null pointer.\n");
    return false;
  }
  if (pMat1->rows != pMat2->rows || pMat1->cols != pMat2->cols ||
      pMat2->rows != pMat3->rows || pMat2->cols != pMat3->cols) {
    fprintf(stderr, "The 3 matrics are not in the same size.\n");
    return false;
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  size_t len = pMat1->rows * pMat1->cols;

  cudaMemcpy(pMat1->data_device, pMat1->data, sizeof(float) * len,
             cudaMemcpyHostToDevice);
  cudaMemcpy(pMat2->data_device, pMat2->data, sizeof(float) * len,
             cudaMemcpyHostToDevice);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, pMat1->rows, pMat2->cols,
              pMat1->cols, &alpha, pMat1->data_device, pMat1->rows,
              pMat2->data_device, pMat2->rows, &beta, pMat3->data_device,
              pMat3->rows);

  cudaMemcpy(pMat3->data, pMat3->data_device, sizeof(float) * len,
             cudaMemcpyDeviceToHost);

  cublasDestroy(handle);

  return true;
}

int main() {

  struct timeval t_start, t_end;
  double elapsedTimeCPU, elapsedTimeGPU;

  int dev_count = 0;
  int dev_id = 0;
  cudaGetDeviceCount(&dev_count);
  cudaSetDevice(3);
  cudaGetDevice(&dev_id);
  printf("You have %d cuda devices.\n", dev_count);
  printf("You are using device %d.\n", dev_id);

  printf("Length, CPU Time, GPU Time\n");

  for (int size = 64; size <= 8192; size +=64) {
    Matrix *pMat1 = createMatrix(size, size);
    Matrix *pMat2 = createMatrix(size, size);
    Matrix *pMat3 = createMatrix(size, size);

    setMatrix(pMat1, 1.1f);
    setMatrix(pMat2, 2.2f);
    float alpha = 5.0f;
    float beta = 3.0f;

    TIME_START
    mulCPU(pMat1, pMat2, pMat3, alpha, beta);
    TIME_END(elapsedTimeCPU)

    TIME_START
    mulGPU(pMat1, pMat2, pMat3, alpha, beta);
    TIME_END(elapsedTimeGPU)

    printf("%d, %.6f, %.6f\n", size, elapsedTimeCPU, elapsedTimeGPU);

    freeMatrix(&pMat1);
    freeMatrix(&pMat2);
    freeMatrix(&pMat3);
  }

  return 0;
}