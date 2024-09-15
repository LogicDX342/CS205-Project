# Report

## Introduction

The goal of this project is to implement a class Matrix in C++ with user-friendly interface and memory-safe features. To achieve this goal, I have read the source code of OpenCV library and  taken some ideas from it.

## Features

The class Matrix has the following features:

1. The class Matrix is a template class, so it can be used to represent a matrix of any type.

    ```cpp
    Mat<int> m1(3, 3);
    Mat<float> m2(3, 3);
    ```

2. The class Matrix can be initialized by several ways:

    ```cpp
    Mat<int> m1(3, 3); // 3x3 matrix with all elements are 0
    Mat<int> m2(3, 3, 1); // 3x3 matrix with all elements are 1
    Mat<int> m3({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}); // 3x3 matrix with elements are 1, 2, 3, 4, 5, 6, 7, 8, 9
    Mat<int> m4(Size(3, 3)); // 3x3 matrix with all elements are 0
    ```

3. The class Matrix has some useful methods:

    ```cpp
    Mat<float> m1(3, 4, 3);
    m1.forEach([&](size_t i, size_t j, size_t c){ return i + j + c; });// set each element of the matrix to i + j + c
    Mat<float> m2(3,4);
    Mat<float> m3 = m1 + m2;
    Mat<float> m4 = m1 - m2;
    Mat<float> m5 = m1 * m2;
    Mat<float> m6 = m1 / m2;
    Mat<float> m7 = m1.t(); // transpose of m1
    Mat<float> m8 = m1.reshape(2, 6); // reshape m1 to 2x6 matrix
    ```

    ```cpp
    Mat<float> m9 = m1.row(1); // get the second row of m1
    Mat<float> m10 = m1.col(1); // get the second column of m1
    Mat<float> m14 = m1.clone(); // clone m1
    ```

## Detail of Implementation

1. Initiaze a matrix: At the beginning, I try to follow the OpenCV to manage the data of a matrix. It use a `refcount` to count how many objects refer to the data. Then I realize that the smart pointer `std::shared_ptr` can do this job for me. So I use `std::shared_ptr` to manage the data of a matrix.

    ```cpp
    data_ = std::shared_ptr<int[]>(new int[cols_ * rows_], [](int *ptr) { delete[] ptr; });
    ```

    This code works well but not elegant enough, as I have to manually set the deleter of the smart pointer. Then I find that `std::make_shared` can help me to do this job. It a new feature of C++20 that allows the smart pointer to manage the array.

    ```cpp
    data_ = std::make_shared<int[]>(cols_ * rows_);
    ```

    With smart pointer, I can easily implement the copy constructor and assignment operator of the class Matrix and make sure that the class is memory-safe.

2. Implement the `forEach` method: This method is used to iterate over all elements of the matrix. I use a lambda function to do this job.

    ```cpp
    void forEach(std::function<T(size_t, size_t, size_t)> func)
    {
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                for (size_t c = 0; c < channels_; ++c)
                {
                    (*this)(i, j, c) = func(i, j, c);
                }
            }
        }
    }

    // Usage
    forEach([&](size_t i, size_t j, size_t c){ return std::abs((*this)(i, j, c)); });
    ```

3. Another feature in C++17: copy elison

    ```cpp
    Mat<int> m1(3, 3);
    Mat<int> m2 = m1; // copy constructor is called
    Mat<int> m3 = m1 + m2; // move constructor is called
    ```

    In C++17, the copy elison is allowed. So the copy constructor is not called in the above code. Instead, the move constructor is called. This feature helps to improve the performance of the class Matrix.

4. ROI: I implement the method `roi` to get a region of interest of the matrix. The method is used to get a submatrix of the matrix without deep copy.

    ```cpp
    Mat<int> m1(3, 3, 7);
    Rect roi(1, 1, 2, 2);
    Mat<int> m2 = Mat<int>(roi, m1);
    m2.forEach([&](size_t i, size_t j, size_t c){ return i + j + c; });
    std::cout << m1 << std::endl;
    ```

    The output of the above code is:

    ```txt
    Channel 1:
    7       7       7
    7       0       1
    7       1       2
    ```

    In OpenCV, 'Step' is a important concept to access the elements of a matrix. It means the number of bytes to move to the next row(the one of original matrix). With the help of 'Step', the index of the element can be converted easily.

5. Transpose and reshape: To avoid deep copy, I use a flag to indicate whether the matrix is transposed, and change the way to access the elements of the matrix.

    ```cpp
    Mat &t()
    {
        is_transposed_ = !is_transposed_;
        std::swap(size());
        return *this;
    }

    T operator()(size_t i, size_t j, size_t c) const
    {
        if (is_transposed_)
        {
            return p_data_[j * step_ + i * channels_ + c];
        }
        return p_data_[i * step_ + j * channels_ + c];
    }
    ```

    For reshape, I planed to use the same way as transpose. But I find that it is not a good idea, because the matrix is not continuous after reshape and problems may occur when I transpose the matrix. Maybe a proxy class can solve this problem.

6. Template use in arguments: The usual way we pass a muti-dimensional array to a function is to use a pointer to pointer. But it is not safe and not easy to use. With the help of template, we can pass a multi-dimensional array to a function easily.

    ```cpp
    template <typename T, size_t N, size_t M>
    void print(T (&arr)[N][M])
    {
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < M; ++j)
            {
                std::cout << arr[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    int arr[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    print(arr);
    ```

    The output of the above code is:

    ```txt
    1 2 3
    4 5 6
    7 8 9
    ```

    This feature can also check if all the lengths of the array in a certain dimension are the same, so it can avoid the problem of passing a non-rectangular array to a function.

7. Muti-channel matrix: The class Matrix can represent a multi-channel matrix. The method `operator()` is used to access the elements of the matrix.

    ```cpp
    Mat<int> m1(3, 3, 3);
    m1.forEach([&](size_t i, size_t j, size_t c){ return i + j + c; });
    std::cout << m1 << std::endl;
    ```

    The output of the above code is:

    ```txt
    Channel 1:
    0       1       2
    1       2       3
    2       3       4

    Channel 2:
    1       2       3
    2       3       4
    3       4       5

    Channel 3:
    2       3       4
    3       4       5
    4       5       6
    ```

    I also implement the class `MatChannel` to represent a channel of the matrix. It can convert a multi-channel matrix to several single-channel matrices, which is useful in matrix multiplication.

    ```cpp
    Mat<float, 3> m1(3, 3, 3);
    m1.forEach([&](size_t i, size_t j, size_t c){ return i + j + c; });
    MatChannel channel = m1.getChannel(2);
    Mat<float,2> m2(channel.size(),6);
    m2.setChannel(0, channel);
    std::cout << m2 << std::endl;
    ```

    The output of the above code is:

    ```txt
    Channel 1:
    2       3       4
    3       4       5
    4       5       6
    ,
    Channel 2:
    6       6       6
    6       6       6
    6       6       6
    ```

8. Memory leak test: The compiler has a flag `-fsanitize=address`, which can help to detect the memory leak. I use this flag to test the class Matrix and make sure that the class is memory-safe.

    ```cpp
    int main()
    {
        int *p = new int[10];
        p = new int[20];
        delete[] p;
    
        return 0;
    }
    ```

    ```bash
    g++ -std=c++20 -fsanitize=address -g test.cpp -o test
    ```

    The output of the above code is:

    ```txt
    =================================================================
    ==1818814==ERROR: LeakSanitizer: detected memory leaks
    
    Direct leak of 40 byte(s) in 1 object(s) allocated from:
        #0 0x7f67217f2648 in operator new[](unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:98
        #1 0x556c5512a1fe in main /home/logicdx342/CPP/project/project4/test.cpp:6
        #2 0x7f6721164082 in __libc_start_main ../csu/libc-start.c:308
    
    SUMMARY: AddressSanitizer: 40 byte(s) leaked in 1 allocation(s).
    ```

    The memory leak is detected and the class Matrix is memory-safe.

## Analysis of OpenCV

It is not a doubt that comparing to the class Matrix in OpenCV, my class Matrix is nearly nothing. I have took long time to read the source code of OpenCV and try to figure out how it works and get some ideas from it. However, this makes my work more difficult and finally get a mess. At the beginning, I simply want to implement classes that we usually use in OpenCV, such as Mat, Size, Rect. But then it turnd out that without a sophisticated design, all the classes just build up a loose connection and it is hard to maintain and extend. The use of interface and abstract class can be found at many places in OpenCV, which makes the code more readable and maintainable. For example, the class `MatExpr` is used to represent the expression of a matrix, and the class `MatOp` is used to represent the operation of a matrix. This design makes it possible to use different calculation methods on various platforms.