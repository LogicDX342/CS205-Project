#include <algorithm>
#include <cblas.h>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/sort.h>
#define MAX_NESTING_DEPTH(size)                                                \
  min(static_cast<size_t>(2 * log2(size)), static_cast<size_t>(100));

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
__host__ __device__ void selection_sort(T *arr, size_t left, size_t right) {
  for (size_t i = left; i <= right; ++i) {
    T min_val = arr[i];
    size_t min_idx = i;

    for (size_t j = i + 1; j <= right; ++j) {
      T val_j = arr[j];

      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    if (i != min_idx) {
      arr[min_idx] = arr[i];
      arr[i] = min_val;
    }
  }
}

template <typename T>
__host__ __device__ void partition(T *arr, size_t left, size_t right,
                                   size_t *pivot) {
  T pivotValue = arr[(left + right) / 2];
  size_t i = left;
  size_t j = right;
  while (i <= j) {
    while (arr[i] < pivotValue) {
      i++;
    }
    while (arr[j] > pivotValue) {
      j--;
    }
    if (i < j) {
      swap(arr, i, j);
      i++;
      j--;
    } else if (i == j) {
      i++;
      break;
    }
  }
  *pivot = i;
}

template <typename T>
__global__ void partition_proxy(T *arr, size_t left, size_t right,
                                size_t *pivot) {
  partition(arr, left, right, pivot);
}

template <typename T>
void quicksort_cell(T *arr, size_t left, size_t right, size_t depth) {
  if (depth <= 0) {
    selection_sort(arr, left, right);
  } else {
    if (left >= right)
      return;
    size_t pivot;
    partition(arr, left, right, &pivot);
    quicksort_cell(arr, left, pivot - 1, depth - 1);
    quicksort_cell(arr, pivot, right, depth - 1);
  }
}

template <typename T> bool quicksort_plain(T *arr, size_t n) {
  SORT_ARGS_CHECK(arr, n)
  size_t depth = MAX_NESTING_DEPTH(n) quicksort_cell(arr, 0, n - 1, depth);
  return true;
}

template <typename T> bool quicksort_cpp(T *arr, size_t n) {
  SORT_ARGS_CHECK(arr, n)
  std::sort(arr, arr + n);
  return true;
}

template <typename T>
void quicksort_cuda_cell(T *arr, size_t left, size_t right) {
  if (left >= right)
    return;
  size_t size = right - left + 1;
  size_t *d_pivot;
  cudaMalloc(&d_pivot, sizeof(size_t));
  T *d_arr;
  cudaMalloc(&d_arr, sizeof(T) * size);
  cudaMemcpy(d_arr, arr + left, sizeof(T) * size, cudaMemcpyHostToDevice);

  partition_proxy<<<1, 1>>>(d_arr, 0, size - 1, d_pivot);
  GPU_ERR_CHK(cudaGetLastError());

  size_t pivot;
  cudaMemcpy(&pivot, d_pivot, sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(arr + left, d_arr, sizeof(T) * size, cudaMemcpyDeviceToHost);

  quicksort_cuda_cell(arr, left, left + pivot - 1);
  quicksort_cuda_cell(arr, left + pivot, right);

  cudaFree(d_pivot);
  cudaFree(d_arr);
}

template <typename T> bool quicksort_cuda(T *arr, size_t n) {
  SORT_ARGS_CHECK(arr, n)
  quicksort_cuda_cell(arr, 0, n - 1);
  return true;
}

template <typename T>
__global__ void quicksort_cuda_rdc_cell(T *arr, size_t left, size_t right,
                                        size_t depth) {
  if (left >= right)
    return;
  if (depth <= 0) {
    selection_sort(arr, left, right);
    return;
  }
  size_t pivot;
  partition(arr, left, right, &pivot);
  quicksort_cuda_rdc_cell<<<1, 1>>>(arr, left, pivot - 1, depth - 1);
  quicksort_cuda_rdc_cell<<<1, 1>>>(arr, pivot, right, depth - 1);
}

template <typename T> bool quicksort_cuda_rdc(T *arr, int n) {
  SORT_ARGS_CHECK(arr, n)
  T *d_arr;
  size_t depth = MAX_NESTING_DEPTH(n) cudaMalloc(&d_arr, sizeof(T) * n);
  cudaMemcpy(d_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice);
  quicksort_cuda_rdc_cell<<<1, 1>>>(d_arr, 0, n - 1, depth);
  GPU_ERR_CHK(cudaGetLastError());
  cudaMemcpy(arr, d_arr, sizeof(T) * n, cudaMemcpyDeviceToHost);
  cudaFree(d_arr);
  return true;
}

template <typename T>
__global__ void quicksort_cuda_rdc_stream_cell(T *arr, size_t left,
                                               size_t right, size_t depth) {
  if (left >= right)
    return;
  if (depth <= 0 || right - left < 32) {
    selection_sort(arr, left, right);
    return;
  }
  size_t pivot;
  partition(arr, left, right, &pivot);
  cudaStream_t l_stream, r_stream;
  if (left < pivot - 1) {
    cudaStreamCreateWithFlags(&l_stream, cudaStreamNonBlocking);
    quicksort_cuda_rdc_stream_cell<<<1, 1, 0, l_stream>>>(arr, left, pivot - 1,
                                                          depth - 1);
    cudaStreamDestroy(l_stream);
  }
  if (pivot < right) {
    cudaStreamCreateWithFlags(&r_stream, cudaStreamNonBlocking);
    quicksort_cuda_rdc_stream_cell<<<1, 1, 0, r_stream>>>(arr, pivot, right,
                                                          depth - 1);
    cudaStreamDestroy(r_stream);
  }
}

template <typename T> bool quicksort_cuda_rdc_stream(T *arr, size_t n) {
  SORT_ARGS_CHECK(arr, n)
  T *d_arr;
  size_t depth = MAX_NESTING_DEPTH(n) cudaMalloc(&d_arr, sizeof(T) * n);
  cudaMemcpy(d_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice);
  quicksort_cuda_rdc_stream_cell<<<1, 1>>>(d_arr, 0, n - 1, depth);
  GPU_ERR_CHK(cudaGetLastError());
  cudaMemcpy(arr, d_arr, sizeof(T) * n, cudaMemcpyDeviceToHost);
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
  double elapsedTimeGPU, elapsedTimeCPU, elapsedTimeTrust;

  std::ofstream outfile("output.csv");
  if (!outfile.is_open()) {
    fprintf(stderr, "Error: Failed to open output file.\n");
    return 1;
  }

  outfile << "Length, CPU Time, GPU Time\n";
  init<<<1, 1>>>();
  for (int length = 1 << 15; length <= (1 << 24); length += 1 << 15) {

    Type *arr = generate_random_array<Type>(length);

    Type *arr_copy = new Type[length];
    Type *arr_copy2 = new Type[length];

    if (arr_copy == NULL || arr_copy2 == NULL) {
      fprintf(stderr, "Error: Failed to allocate memory.\n");
      return 1;
    }

    memcpy(arr_copy, arr, sizeof(Type) * length);
    TIME_START
    quicksort_cpp(arr_copy, length);
    TIME_END(elapsedTimeCPU)

    memcpy(arr_copy2, arr, sizeof(Type) * length);
    TIME_START
    quicksort_cuda_rdc_stream(arr_copy2, length);
    TIME_END(elapsedTimeGPU)

    outfile << length << ", " << elapsedTimeGPU << ", " << elapsedTimeCPU
            << ", " << elapsedTimeTrust << "\n";

    check_sorted(arr_copy2, length);

    delete[] arr;
    delete[] arr_copy;
    delete[] arr_copy2;
  }
  outfile.close();
  return 0;
}
