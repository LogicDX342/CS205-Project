
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include <cblas.h>

#define CHECK_MAT_DIMENSIONS(lhs, rhs)                                                                                             \
    if ((lhs).rows_ != (rhs).rows_ || (lhs).cols_ != (rhs).cols_ || (lhs).channels_ != (rhs).channels_)                            \
    {                                                                                                                              \
        throw std::runtime_error("Mat dimensions must agree. Error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
    }

class Size
{
private:
    size_t rows_;
    size_t cols_;

public:
    Size(size_t rows, size_t cols) : rows_(rows), cols_(cols) {}
    [[nodiscard]] size_t rows() const { return rows_; }
    [[nodiscard]] size_t cols() const { return cols_; }
    bool operator==(const Size &rhs) const
    {
        return rows_ == rhs.rows_ && cols_ == rhs.cols_;
    }
    bool operator!=(const Size &rhs) const
    {
        return !(*this == rhs);
    }
};

class Rect
{
private:
    size_t x_;
    size_t y_;
    size_t width_;
    size_t height_;

public:
    Rect(size_t x, size_t y, size_t width, size_t height) : x_(x), y_(y), width_(width), height_(height) {}
    [[nodiscard]] size_t x() const { return x_; }
    [[nodiscard]] size_t y() const { return y_; }
    [[nodiscard]] size_t width() const { return width_; }
    [[nodiscard]] size_t height() const { return height_; }
};

template <typename T, size_t CHANNELS>
class MatChannel;

template <typename T, size_t CHANNELS = 1>
class Mat
{
private:
    std::shared_ptr<T[]> data_;
    T *p_data_;
    size_t rows_;
    size_t cols_;
    size_t channels_;
    size_t elemSize_;
    size_t step_;
    bool is_transposed_ = false;

public:
    // Constructor
    explicit Mat(size_t rows, size_t cols) : rows_(rows), cols_(cols), channels_(CHANNELS), elemSize_(sizeof(T) * channels_), step_(cols_ * channels_)
    {
        data_ = std::make_shared<T[]>(rows_ * step_);
        p_data_ = data_.get();
    }

    explicit Mat(size_t rows, size_t cols, const T &val) : Mat(rows, cols)
    {
        set(val);
    }

    template <size_t rows_, size_t cols_>
    explicit Mat(const T (&data)[rows_][cols_]) : Mat(size())
    {
        set(data);
    }

    explicit Mat(Size size) : Mat(size.rows(), size.cols()) {}

    explicit Mat(Size size, const T &val) : Mat(size.rows(), size.cols(), val) {}

    explicit Mat(const Rect &roi, Mat &mat)
    {
        if (roi.x() + roi.width() > mat.rows() || roi.y() + roi.height() > mat.cols())
        {
            throw std::runtime_error("Invalid ROI.");
        }
        data_ = mat.data_;
        p_data_ = data_.get() + roi.x() * mat.step_ + roi.y() * mat.channels_;
        rows_ = roi.width();
        cols_ = roi.height();
        channels_ = mat.channels_;
        elemSize_ = mat.elemSize_;
        step_ = mat.step_;
    }

    explicit Mat(const MatChannel<T, 1> &rhs)
    {
        if (rhs.rows() == 0 || rhs.cols() == 0)
        {
            throw std::runtime_error("Invalid MatChannel.");
        }
        rows_ = rhs.rows();
        cols_ = rhs.cols();
        channels_ = 1;
        elemSize_ = sizeof(T);
        step_ = cols_;
        data_ = std::make_shared<T[]>(rows_ * step_);
        p_data_ = data_.get();
        forEach([&](size_t i, size_t j, size_t c)
                { return rhs(i, j); });
    }

    Mat(const Mat &rhs) = default;

    // Destructor
    ~Mat()
    {
        data_.reset();
    }

    // Member Variables
    [[nodiscard]] size_t rows() const { return rows_; }
    [[nodiscard]] size_t cols() const { return cols_; }
    [[nodiscard]] size_t channels() const { return channels_; }
    [[nodiscard]] size_t elemSize() const { return elemSize_; }
    [[nodiscard]] size_t elemSize1() const { return sizeof(T); }
    [[nodiscard]] size_t step() const { return step_; }
    [[nodiscard]] Size size() const { return {rows_, cols_}; }
    [[nodiscard]] std::shared_ptr<T[]> data() const { return data_; }

    // Operator Overloading
    Mat &operator=(const Mat &rhs) = default;

    T operator()(size_t i, size_t j, size_t c = 0) const
    {
        if (is_transposed_)
        {
            return p_data_[j * step_ + i * channels_ + c];
        }
        return p_data_[i * step_ + j * channels_ + c];
    }
    T &operator()(size_t i, size_t j, size_t c = 0)
    {
        if (is_transposed_)
        {
            return p_data_[j * step_ + i * channels_ + c];
        }
        return p_data_[i * step_ + j * channels_ + c];
    }
    Mat &operator+=(const Mat &rhs)
    {
        CHECK_MAT_DIMENSIONS(*this, rhs)
        forEach([&](size_t i, size_t j, size_t c)
                { return (*this)(i, j, c) += rhs(i, j, c); });
        return *this;
    }
    Mat &operator-=(const Mat &rhs)
    {
        CHECK_MAT_DIMENSIONS(*this, rhs)
        forEach([&](size_t i, size_t j, size_t c)
                { return (*this)(i, j, c) -= rhs(i, j, c); });
        return *this;
    }
    Mat &operator*=(const Mat &rhs)
    {
        if (cols_ != rhs.rows_)
        {
            throw std::runtime_error("Mat dimensions must agree.");
        }
        Mat result = Mat(size());
        // data type
        if (std::is_same_v<T, float> ||
            std::is_same_v<T, double>)
        {
            for (size_t channel = 0; channel < channels_; ++channel)
            {
                Mat a(size());
                Mat b(rhs.size());
                Mat c(result.size());
                a.setChannel(0, getChannel(channel));
                b.setChannel(0, rhs.getChannel(channel));

                if (std::is_same_v<T, float>)
                {
                    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_, rhs.cols_, cols_, 1.0, a.data().get(), cols_, b.data().get(), rhs.cols_, 0.0, c.data().get(), rhs.cols_);
                }
                else if (std::is_same_v<T, double>)
                {
                    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_, rhs.cols_, cols_, 1.0, a.data().get(), cols_, b.data().get(), rhs.cols_, 0.0, c.data().get(), rhs.cols_);
                }
                result.setChannel(channel, c.getChannel(0));
            }
        }
        else
        {
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < rhs.cols_; ++j)
                {
                    for (size_t k = 0; k < cols_; ++k)
                    {
                        for (size_t c = 0; c < channels_; ++c)
                        {
                            result(i, j, c) += (*this)(i, k, c) * rhs(k, j, c);
                        }
                    }
                }
            }
        }
        return *this;
    }

    Mat &operator/=(const Mat &rhs)
    {
        CHECK_MAT_DIMENSIONS(*this, rhs)
        forEach([&](size_t i, size_t j, size_t c)
                {return (*this)(i, j, c) /= rhs(i, j, c); });
        return *this;
    }
    Mat &operator%=(const Mat &rhs)
    {
        CHECK_MAT_DIMENSIONS(*this, rhs)
        forEach([&](size_t i, size_t j, size_t c)
                { (*this)(i, j, c) %= rhs(i, j, c); });
        return *this;
    }
    Mat &operator+=(const T &val)
    {
        forEach([&](size_t i, size_t j, size_t c)
                { (*this)(i, j, c) += val; });
        return *this;
    }
    Mat &operator-=(const T &val)
    {
        forEach([&](size_t i, size_t j, size_t c)
                { (*this)(i, j, c) -= val; });
        return *this;
    }
    Mat &operator*=(const T &val)
    {
        forEach([&](size_t i, size_t j, size_t c)
                { (*this)(i, j, c) *= val; });
        return *this;
    }
    Mat &operator/=(const T &val)
    {
        forEach([&](size_t i, size_t j, size_t c)
                { (*this)(i, j, c) /= val; });
        return *this;
    }
    Mat &operator%=(const T &val)
    {
        forEach([&](size_t i, size_t j, size_t c)
                { (*this)(i, j, c) %= val; });
        return *this;
    }
    Mat operator+(const Mat &rhs) const
    {
        Mat result(*this);
        return result += rhs;
    }
    Mat operator-(const Mat &rhs) const
    {
        Mat result(*this);
        return result -= rhs;
    }
    Mat operator*(const Mat &rhs) const
    {
        Mat result(*this);
        return result *= rhs;
    }
    Mat operator/(const Mat &rhs) const
    {
        Mat result(*this);
        return result /= rhs;
    }
    Mat operator%(const Mat &rhs) const
    {
        Mat result(*this);
        return result %= rhs;
    }
    Mat operator+(const T &val) const
    {
        Mat result(*this);
        return result += val;
    }
    Mat operator-(const T &val) const
    {
        Mat result(*this);
        return result -= val;
    }
    Mat operator*(const T &val) const
    {
        Mat result(*this);
        return result *= val;
    }
    Mat operator/(const T &val) const
    {
        Mat result(*this);
        return result /= val;
    }
    Mat operator%(const T &val) const
    {
        Mat result(*this);
        return result %= val;
    }

    // Mat Operations
    T &at(size_t i, size_t j, size_t c = 0)
    {
        assert(i < rows_ && j < cols_ && c < channels_);
        return (*this)(i, j, c);
    }

    [[nodiscard]] const T &at(size_t i, size_t j, size_t c = 0) const
    {
        assert(i < rows_ && j < cols_ && c < channels_);
        return (*this)(i, j, c);
    }

    [[nodiscard]] Mat abs() const
    {
        Mat result(*this);
        forEach([&](size_t i, size_t j, size_t c)
                { return std::abs((*this)(i, j, c)); });
        return result;
    }

    [[nodiscard]] Mat max(const Mat &rhs) const
    {
        CHECK_MAT_DIMENSIONS(*this, rhs)
        Mat result(*this);
        forEach([&](size_t i, size_t j, size_t c)
                { return std::max((*this)(i, j, c), rhs(i, j, c)); });
        return result;
    }

    [[nodiscard]] Mat min(const Mat &rhs) const
    {
        CHECK_MAT_DIMENSIONS(*this, rhs)
        Mat result(*this);
        forEach([&](size_t i, size_t j, size_t c)
                { return std::min((*this)(i, j, c), rhs(i, j, c)); });
        return result;
    }

    void set(const T &val)
    {
        forEach([&](size_t /*i*/, size_t /*j*/, size_t /*c*/)
                { return val; });
    }

    template <size_t rows_, size_t cols_>
    void set(const T (&data)[rows_][cols_])
    {
        assert(rows_ == rows_ && cols_ == cols_);
        forEach([&](size_t i, size_t j, size_t c)
                { (*this)(i, j, c) = data[i][j]; });
    }

    Mat &t()
    {
        is_transposed_ = !is_transposed_;
        std::swap(rows_, cols_);
        return *this;
    }

    Mat &reshape(size_t rows, size_t cols)
    {
        if (step_ != cols_ * channels_)
        {
            throw std::runtime_error("The matrix is not continuous, cannot reshape.");
        }
        if (rows * cols != rows_ * cols_)
        {
            throw std::runtime_error("Invalid reshape dimensions.");
        }
        rows_ = rows;
        cols_ = cols;
        step_ = cols_ * channels_;
        return *this;
    }

    [[nodiscard]] Mat &col(size_t j) 
    {
        if (j >= cols_)
        {
            throw std::runtime_error("Index out of bounds.");
        }
        Mat<T, CHANNELS> result(rows_, 1);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t c = 0; c < channels_; ++c)
            {
                result(i, 0, c) = (*this)(i, j, c);
            }
        }
        return result;
    }

    [[nodiscard]] Mat &row(size_t i) 
    {
        if (i >= rows_)
        {
            throw std::runtime_error("Index out of bounds.");
        }
        Mat<T, CHANNELS> result(1, cols_);
        for (size_t j = 0; j < cols_; ++j)
        {
            for (size_t c = 0; c < channels_; ++c)
            {
                result(0, j, c) = (*this)(i, j, c);
            }
        }
        return result;
    }

    [[nodiscard]] Mat clone() const
    {
        Mat result(size());
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                for (size_t c = 0; c < channels_; ++c)
                {
                    result(i, j, c) = (*this)(i, j, c);
                }
            }
        }
        return result;
    }

    [[nodiscard]] MatChannel<T, CHANNELS> getChannel(size_t channel) const
    {
        if (channel >= channels_)
        {
            throw std::runtime_error("Index out of bounds.");
        }
        return MatChannel(*this, channel);
    }

    template <size_t CHANNELS_>
    void setChannel(size_t channel, const MatChannel<T, CHANNELS_> &mat_channel)
    {
        if (channel >= channels_)
        {
            throw std::runtime_error("Index out of bounds.");
        }
        if (mat_channel.rows() != rows_ || mat_channel.cols() != cols_)
        {
            throw std::runtime_error("Mat dimensions must agree.");
        }
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                (*this)(i, j, channel) = mat_channel(i, j);
            }
        }
    }

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
};

template <typename T, size_t channels>
std::ostream &operator<<(std::ostream &os, const Mat<T, channels> &mat)
{
    for (size_t c = 0; c < channels; ++c)
    {
        os << "Channel " << c + 1 << ":\n";
        for (size_t i = 0; i < mat.rows(); ++i)
        {
            for (size_t j = 0; j < mat.cols(); ++j)
            {
                os << mat(i, j, c);
                if (j < mat.cols() - 1)
                {
                    os << "\t";
                }
            }
            os << "\n";
        }
        if (c < channels - 1)
        {
            os << ",\n";
        }
    }
    return os;
}

template <typename T, size_t CHANNELS>
class MatChannel
{
private:
    Mat<T, CHANNELS> mat_;
    size_t channel_;

public:
    MatChannel(const Mat<T, CHANNELS> &rhs, size_t channel) : mat_(rhs), channel_(channel) {}
    T operator()(size_t i, size_t j, size_t c = 0) const
    {
        return mat_(i, j, channel_);
    }
    T &operator()(size_t i, size_t j, size_t c = 0)
    {
        return mat_(i, j, channel_);
    }
    [[nodiscard]] size_t rows() const { return mat_.rows(); }
    [[nodiscard]] size_t cols() const { return mat_.cols(); }
    [[nodiscard]] Size size() const { return mat_.size(); }
};