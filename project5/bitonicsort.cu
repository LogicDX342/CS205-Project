#include <algorithm>
#include <cblas.h>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/sort.h>

#define TIME_START gettimeofday(&t_start, NULL);

#define TIME_END(time)                                                         \
  gettimeofday(&t_end, NULL);                                                  \
  time = (t_end.tv_sec - t_start.tv_sec) * 1000.0;                             \
  time += (t_end.tv_usec - t_start.tv_usec) / 1000.0;

#define SORT_ARGS_CHECK(arr, n)                                                \
  if (arr == NULL) {                                                           \
    fprintf(stderr, "Null pointer.\n");                                        \
    return false;                                                              \
  }                                                                            \
  if (n <= 0) {                                                                \
    fprintf(stderr, "Invalid size.\n");                                        \
    return false;                                                              \
  }

#define GPU_ERR_CHK(ans)                                                       \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

size_t log2(size_t n) {
  size_t i = 0;
  while (n >>= 1) {
    i++;
  }
  return i;
}

template <typename T> __device__ void swap(T *arr, size_t i, size_t j) {
  T temp = arr[i];
  arr[i] = arr[j];
  arr[j] = temp;
}

template <typename T>
__global__ void bitonicsort_cell(T *arr, size_t j, size_t k) {
  size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  size_t ij = i ^ j;
  if (ij > i) {
    if ((i & k) == 0) {
      if (arr[i] > arr[ij]) {
        swap(arr, i, ij);
      }
    } else {
      if (arr[i] < arr[ij]) {
        swap(arr, i, ij);
      }
    }
  }
}

template <typename T>
__global__ void initialize_padding(T *arr, size_t size, size_t paddedSize,
                                   T maxVal) {
  size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= size && i < paddedSize) {
    arr[i] = maxVal;
  }
}

template <typename T> bool bitonicsort(T *arr, size_t size) {
  SORT_ARGS_CHECK(arr, size)
  size_t threadsPerBlock = 1024;
  size_t paddedSize = 1;
  while (paddedSize < size) {
    paddedSize <<= 1;
  }
  size_t blocksPerGrid = (paddedSize + threadsPerBlock - 1) / threadsPerBlock;
  T *d_arr;
  cudaMalloc(&d_arr, sizeof(T) * paddedSize);
  cudaMemcpy(d_arr, arr, sizeof(T) * size, cudaMemcpyHostToDevice);
  T maxVal = std::numeric_limits<T>::max();
  initialize_padding<<<blocksPerGrid, threadsPerBlock>>>(d_arr, size,
                                                         paddedSize, maxVal);
  for (size_t k = 2; k <= paddedSize; k <<= 1) {
    for (size_t j = k >> 1; j > 0; j = j >> 1) {
      bitonicsort_cell<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k);
    }
  }
  GPU_ERR_CHK(cudaGetLastError());
  cudaMemcpy(arr, d_arr, sizeof(T) * size, cudaMemcpyDeviceToHost);
  cudaFree(d_arr);
  return true;
}

template <typename T> bool check_sorted(T *arr, size_t n) {
  SORT_ARGS_CHECK(arr, n)
  for (size_t i = 1; i < n; i++) {
    if (arr[i] < arr[i - 1]) {
      printf("Error: Array is not sorted.\n");
      return false;
    }
  }
  return true;
}

template <typename T> T *generate_random_array(size_t size) {
  if (size == 0) {
    fprintf(stderr, "Invalid size.\n");
    return NULL;
  }
  T *arr = new T[size];
  if (arr == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory.\n");
    return NULL;
  }
  for (size_t i = 0; i < size; i++) {
    arr[i] = rand();
  }
  return arr;
}

__global__ void init() {}

using Type = float;

int main() {
  struct timeval t_start, t_end;
  double elapsedTimeGPU, elapsedTimeCPU;

  std::ofstream outfile("output.csv");
  if (!outfile.is_open()) {
    fprintf(stderr, "Error: Failed to open output file.\n");
    return 1;
  }

  outfile << "Length, Bitonic Sort, Thrust Sort\n";
  init<<<1, 1>>>();
  for (int size = 1 << 15; size <= (1 << 25); size += 1 << 15) {

    Type *arr = generate_random_array<Type>(size);

    Type *arr_copy = new Type[size];
    Type *arr_copy2 = new Type[size];

    if (arr_copy == NULL || arr_copy2 == NULL) {
      fprintf(stderr, "Error: Failed to allocate memory.\n");
      return 1;
    }

    memcpy(arr_copy, arr, sizeof(Type) * size);
    TIME_START
    bitonicsort(arr_copy, size);
    TIME_END(elapsedTimeGPU)

    memcpy(arr_copy2, arr, sizeof(Type) * size);
    TIME_START
    thrust::sort(arr_copy2, arr_copy2 + size);
    TIME_END(elapsedTimeCPU)

    // Print CSV row to file
    outfile << size << ", " << elapsedTimeGPU << ", " << elapsedTimeCPU << "\n";

    check_sorted(arr_copy, size);

    delete[] arr;
    delete[] arr_copy;
    delete[] arr_copy2;
  }
  outfile.close();
  return 0;
}