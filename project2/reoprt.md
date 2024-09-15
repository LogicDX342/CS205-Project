## Performance Evaluation

In this project, I have implemented several ways to calculate the matrix multiplication. The following table shows the performance of each method when n=1024.

| Method | Time (ms) |
| --- | --- |
| Method 1 | 3515 |
| Method 2 | 3540 |
| Method 3 | 3641 |
| Method 4 | 2019 |
| Method 3 + O3 | 723 |
| Method 4 + O3 | 71 |
| Java 1 | 1194 |
| Java 2 | 318 |

The method 1, 2, and 3 are all use simple for loop to calculate the matrix multiplication. The difference between them is the way create and access the matrix. Method 1 use a n*n array to store the matrix, and access the element by `matrix[i][j]`. Method 2 use a 1D array to store the matrix, and access the element by `matrix[i*n+j]`. Method 3 use `malloc` to allocate the memory for the matrix, and access the element by `matrix[i*n+j]`. The performance of these three methods are similar, and the time is around 2600ms.

The method 4 is almost the same as method 3, but change the order of the loop.

```c
// Method 3
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
            C[i*n+j] += A[i*n+k] * B[k*n+j];
        }
    }
}
// Method 4
for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
            C[i*n+j] += A[i*n+k] * B[k*n+j];
        }
    }
}
```

The improvement is because of the cache. The method 3 access the element of B in a row-wise order, and the method 4 access the element of B in a column-wise order. As the CPU fetch the data from memory in a block, access the data in a column-wise order can improve the cache hit rate. As a result, the time of method 4 is significantly smaller than method 3. Form the log of compiler, we can see that in the method 3, only the first loop is vectorized, and in the method 4, both the second and third loop are vectorized. This optimization is also valid for Java, as the time of Java 1 is 1194ms, and the time of Java 2 is 318ms.

> Method 3
>/MatrixMultiplication.c:27:27: optimized: loop vectorized using 16 byte vectors
>/MatrixMultiplication.c:29:31: missed: couldn't vectorize loop
>/MatrixMultiplication.c:29:31: missed: outer-loop already vectorized.
>
> Method 4
> /MatrixMultiplication.c:27:27: missed: couldn't vectorize loop
> /MatrixMultiplication.c:31:30: missed: not vectorized: complicated access pattern.
> /MatrixMultiplication.c:29:31: optimized: loop vectorized using 16 byte vectors
> /MatrixMultiplication.c:29:31: missed: couldn't vectorize loop
> /MatrixMultiplication.c:31:44: missed: not vectorized: complicated access pattern.
> /MatrixMultiplication.c:29:31: optimized: loop vectorized using 16 byte vectors
> /MatrixMultiplication.c:14:23: missed: couldn't vectorize loop

The method 3 + O3 and method 4 + O3 are the same as method 3 and method 4, but add the `-O3` flag when compile the code. The compiler will optimize the code by using the SIMD instruction set, such as AVX. In this case, the compiler use `mulps` and `addps` to calculate the matrix multiplication. Those instructions can calculate 4 pair of float numbers at the same time, and the performance is improved.
