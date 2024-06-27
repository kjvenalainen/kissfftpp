// Copyright (c) 2024 Kevin Venalainen
//
// This file is part of KissFFT++.
//

#ifndef KISS_FFT_PP_H
#define KISS_FFT_PP_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

// If not explicitly set, then enable based on debug status.
#ifndef KFFTPP_NO_CONTRACT_CHECKING
#if defined(NDEBUG)
#define KFFTPP_NO_CONTRACT_CHECKING 1
#else
#define KFFTPP_NO_CONTRACT_CHECKING 0
#endif
#endif

// If not explicitly set, then disable exceptions based on common compiler
// flags.
#ifndef KFFTPP_NO_EXCEPTIONS
#if !(defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND))
#define KFFTPP_NO_EXCEPTIONS 1
#else
#define KFFTPP_NO_EXCEPTIONS 0
#endif
#endif

#if !KFFTPP_NO_EXCEPTIONS && !KFFTPP_NO_CONTRACT_CHECKING
#include <stdexcept>
#endif

// There are 3 possible outcomes for error conditions:
// 1. KFFTPP_NO_CONTRACT_CHECKING is true - no runtime checks will be performed
// whatsoever.
// 2. KFFTPP_NO_CONTRACT_CHECKING is false, and KFFTPP_NO_EXCEPTIONS is false -
// runtime checks will be performed, and std::logic_error will be thrown.
// 3. KFFTPP_NO_CONTRACT_CHECKING is false, and KFFTPP_NO_EXCEPTIONS is true -
// runtime checks will be performed, and std::terminate will be called.
#if defined(KFFTPP_NO_CONTRACT_CHECKING)
#define KFFTPP_ASSERT(cond, msg) ((void)0)
#elif defined(KFFTPP_NO_EXCEPTIONS)
#define KFFTPP_ASSERT(cond, msg) \
  if (!(cond)) {                 \
    std::terminate();            \
  }
#else
#define KFFTPP_ASSERT(cond, msg) \
  if (!(cond)) {                 \
    throw std::logic_error(msg); \
  }
#endif

#ifndef KFFTPP_HALF_PI
#define KFFTPP_HALF_PI 1.5707963267948966192313216916397514420986L
#endif  // KFFTPP_HALF_PI

#ifndef KFFTPP_PI
#define KFFTPP_PI 3.1415926535897932384626433832795028841972L
#endif  // KFFTPP_PI

namespace kfft {

// Same as std::dynamic_extent from C++20.
constexpr size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

// Lightweight span implementation.
template <typename T, size_t Extent = dynamic_extent>
class span {
  template <size_t E>
  struct extent_type {
    constexpr explicit extent_type(size_t) noexcept = default;
    static constexpr size_t value_ = E;
  };

  template <>
  struct extent_type<dynamic_extent> {
    constexpr explicit extent_type(size_t value) noexcept : value_(value) {};
    size_t value_;
  };

 public:
  // Constructors.
  template <
      size_t _Extent = Extent,
      typename std::enable_if<_Extent != dynamic_extent, bool>::type = true>
  constexpr explicit span(T* ptr) noexcept : ptr_(ptr), size_(_Extent) {}
  template <
      size_t _Extent = Extent,
      typename std::enable_if<_Extent == dynamic_extent, bool>::type = true>
  constexpr span(T* ptr, size_t size) noexcept : ptr_(ptr), size_(size) {}
  template <
      size_t _Extent = Extent,
      typename std::enable_if<_Extent != dynamic_extent, bool>::type = true>
  constexpr span(T (&arr)[_Extent]) noexcept : ptr_(arr), size_(_Extent) {}
  constexpr span(std::vector<T> vec) noexcept
      : ptr_(vec.data()), size_(vec.size()) {}
  constexpr span(const span&) noexcept = default;
  constexpr span& operator=(const span&) noexcept = default;
  constexpr span(span&&) = default;
  constexpr span& operator=(span&&) = default;

  // Accessors.
  constexpr T* data() const noexcept { return ptr_; }
  constexpr size_t size() const noexcept { return size_.value_; }
  constexpr T& operator[](size_t i) const {
    KFFTPP_ASSERT(i < size_.value_, "Index out of bounds");
    return ptr_[i];
  }

  // Create a subspan of this span.
  template <size_t _Offset, size_t _Count = dynamic_extent>
  constexpr auto subspan() const noexcept
      -> span<T, _Count != dynamic_extent ? _Count : Extent - _Offset> {
    static_assert(_Offset <= Extent,
                  "span<T, N>::subspan<Offset, Count>(): Offset out of range");
    static_assert(
        _Count == dynamic_extent || _Count <= Extent - _Offset,
        "span<T, N>::subspan<Offset, Count>(): Offset + Count out of range");

    using _ReturnType =
        span<T, _Count != dynamic_extent ? _Count : Extent - _Offset>;
    return _ReturnType{data() + _Offset,
                       _Count == dynamic_extent ? size() - _Offset : _Count};
  }
  constexpr span subspan(size_t offset, size_t count = dynamic_extent) const {
    KFFTPP_ASSERT(offset <= size_.value_, "Offset out of bounds");
    KFFTPP_ASSERT(count == dynamic_extent || offset + count <= size_.value_,
                  "Count out of bounds");
    return span(ptr_ + offset,
                count == dynamic_extent ? size_.value_ - offset : count);
  }

 private:
  T* ptr_;
  extent_type<Extent> size_;
};

// Performs no scaling for forward or inverse FFTs.
struct NoScaling {
  template <typename T, bool Inverse>
  static constexpr T Scale(const T& x, const float& /* N */) {
    return x;
  }
};

// Scales forward FFT by 1, and inverse by 1/N. This matches with MATLAB's FFT.
struct InverseOneByNScaling {
  // Forward FFT scales by 1.
  template <typename T, bool Inverse,
            typename std::enable_if_t<!Inverse, bool> = true>
  static constexpr T Scale(const T& x, const float& /* N */) {
    return x;
  }

  // Inverse FFT scales by 1/N.
  template <typename T, bool Inverse,
            typename std::enable_if_t<Inverse, bool> = true>
  static constexpr T Scale(const T& x, const float& N) {
    return x / N;
  }
};

namespace internal {

// Use custom complex number implementation to avoid slow std::complex
// operations due to NaN propagation rules.
template <typename T>
class complex {
 public:
  constexpr complex(const T& real = T(), const T& imag = T())
      : real_(real), imag_(imag) {}
  constexpr complex(const complex&) = default;
  constexpr complex(complex&&) = default;

  constexpr T real() const { return real_; }
  constexpr void real(T value) { real_ = value; }
  constexpr T imag() const { return imag_; }
  constexpr void imag(T value) { imag_ = value; }

  constexpr complex& operator=(const complex&) = default;
  constexpr complex& operator+=(const complex& rhs) {
    real_ += rhs.real_;
    imag_ += rhs.imag_;
    return *this;
  }
  constexpr complex& operator-=(const complex& rhs) {
    real_ -= rhs.real_;
    imag_ -= rhs.imag_;
    return *this;
  }
  constexpr complex& operator*=(const complex& rhs) {
    const T real = real_ * rhs.real_ - imag_ * rhs.imag_;
    const T imag = real_ * rhs.imag_ + imag_ * rhs.real_;
    real_ = real;
    imag_ = imag;
    return *this;
  }
  constexpr complex& operator/=(const complex& rhs) {
    const T div = rhs.real_ * rhs.real_ + rhs.imag_ * rhs.imag_;
    const T real = (real_ * rhs.real_ + imag_ * rhs.imag_) / div;
    const T imag = (imag_ * rhs.real_ - real_ * rhs.imag_) / div;
    real_ = real;
    imag_ = imag;
    return *this;
  }

  constexpr complex& operator+=(const T& rhs) noexcept {
    real_ += rhs;
    return *this;
  }
  constexpr complex& operator-=(const T& rhs) noexcept {
    real_ -= rhs;
    return *this;
  }
  constexpr complex& operator*=(const T& rhs) noexcept {
    real_ *= rhs;
    imag_ *= rhs;
    return *this;
  }
  constexpr complex& operator/=(const T& rhs) noexcept {
    real_ /= rhs;
    imag_ /= rhs;
    return *this;
  }

 private:
  T real_;
  T imag_;
};

template <typename T>
constexpr complex<T> operator+(const complex<T>& val) {
  return {val.real(), val.imag()};
}
template <typename T>
constexpr complex<T> operator-(const complex<T>& val) {
  return {-val.real(), -val.imag()};
}
template <typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) += rhs;
}
template <typename T>
constexpr complex<T> operator+(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) += rhs;
}
template <typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const T& rhs) {
  return complex<T>(lhs) += rhs;
}
template <typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) -= rhs;
}
template <typename T>
constexpr complex<T> operator-(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) -= rhs;
}
template <typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const T& rhs) {
  return complex<T>(lhs) -= rhs;
}
template <typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) *= rhs;
}
template <typename T>
constexpr complex<T> operator*(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) *= rhs;
}
template <typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const T& rhs) {
  return complex<T>(lhs) *= rhs;
}
template <typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) /= rhs;
}
template <typename T>
constexpr complex<T> operator/(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) /= rhs;
}
template <typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const T& rhs) {
  return complex<T>(lhs) /= rhs;
}

// Compute twiddle factors for FFT of length N.
template <typename T>
static constexpr std::vector<internal::complex<T>> ComputeTwiddles(
    size_t N, bool inverse) {
  auto twiddles = std::vector<internal::complex<T>>(N);
  const double phase = inverse ? 2 * KFFTPP_PI / N : -2 * KFFTPP_PI / N;
  for (size_t i = 0; i < N; ++i) {
    const double phaseArg = phase * i;
    twiddles[i] = {static_cast<T>(std::cos(phaseArg)),
                   static_cast<T>(std::sin(phaseArg))};
  }
  return twiddles;
}

// Given a FFT length N, factorize it into a sequence of `p, m` pairs where `p`
// is the FFT radix and `m` is the length of the FFT at that stage.
static std::vector<size_t> Factorize(size_t N) {
  auto factors = std::vector<size_t>();
  size_t p = 4;
  const auto floorSqrt =
      static_cast<size_t>(std::floor(std::sqrt(static_cast<double>(N))));
  do {
    while (N % p) {
      switch (p) {
        case 4:
          p = 2;
          break;
        case 2:
          p = 3;
          break;
        default:
          p += 2;
          break;
      }
      if (p > floorSqrt) p = N;  // no more factors, skip to end
    }
    N /= p;
    factors.push_back(p);
    factors.push_back(N);
  } while (N > 1);
  return factors;
}

// Given a factorized FFT, compute the maximum required scratch array length for
// the generic butterfly operations. If there are no generic butterflies, then
// the scratch space required is 0.
static constexpr size_t RequiredScratchLength(
    const std::vector<size_t>& factors) {
  constexpr std::array<size_t, 4> NON_GENERIC_BUTTERFLY_RADICES = {2, 3, 4, 5};

  size_t scratchLength = 0;
  for (size_t i = 0; i < factors.size(); i += 2) {
    bool generic = true;
    for (size_t pi = 0; pi < NON_GENERIC_BUTTERFLY_RADICES.size(); ++pi) {
      if (factors[i] == NON_GENERIC_BUTTERFLY_RADICES[pi]) {
        // Not a generic butterfly, skip.
        generic = false;
        break;
      }
    }
    if (!generic) {
      continue;
    }

    // Generic butterfly, compute scratch size.
    scratchLength = std::max(scratchLength, static_cast<size_t>(factors[i]));
  }

  return scratchLength;
}

template <typename T, bool Inverse>
static constexpr void Butterfly2(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  for (size_t i = 0; i < m; ++i) {
    const auto xi = x[m + i] * twiddles[i * stride];
    x[m + i] = x[i] - xi;
    x[i] += xi;
  }
}

template <typename T, bool Inverse>
static constexpr void Butterfly3(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  const size_t m2 = 2 * m;
  for (size_t i = 0; i < m; ++i) {
    std::array<T, 5> xi;
    xi[1] = x[m + i] * twiddles[i * stride];
    xi[2] = x[m2 + i] * twiddles[i * 2 * stride];
    xi[3] = xi[1] + xi[2];
    xi[0] = xi[1] - xi[2];
    x[m + i] = {
        x[i].real() - (xi[3].real() * 0.5f),
        x[i].imag() - (xi[3].imag() * 0.5f),
    };
    xi[0] *= twiddles[m * stride].imag();
    x[i] += xi[3];
    x[m2 + i] = {x[m + i].real() + xi[0].imag(),
                 x[m + i].imag() - xi[0].real()};
    x[m + i] = {x[m + i].real() - xi[0].imag(), x[m + i].imag() + xi[0].real()};
  }
}

template <typename T, bool Inverse,
          typename std::enable_if_t<!Inverse, bool> = true>
static constexpr void Butterfly4(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  const size_t m2 = 2 * m;
  const size_t m3 = 3 * m;
  for (size_t i = 0; i < m; ++i) {
    std::array<T, 6> xi;
    xi[0] = x[m + i] * twiddles[i * stride];
    xi[1] = x[m2 + i] * twiddles[i * 2 * stride];
    xi[2] = x[m3 + i] * twiddles[i * 3 * stride];
    xi[3] = xi[0] + xi[2];
    xi[4] = xi[0] - xi[2];
    xi[5] = x[i] - xi[1];
    x[i] += xi[1];
    x[m2 + i] = x[i] - xi[3];
    x[i] += xi[3];

    x[m + i] = {xi[5].real() + xi[4].imag(), xi[5].imag() - xi[4].real()};
    x[m3 + i] = {xi[5].real() - xi[4].imag(), xi[5].imag() + xi[4].real()};
  }
}

template <typename T, bool Inverse,
          typename std::enable_if_t<Inverse, bool> = true>
static constexpr void Butterfly4(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  const size_t m2 = 2 * m;
  const size_t m3 = 3 * m;
  for (size_t i = 0; i < m; ++i) {
    std::array<T, 6> xi;
    xi[0] = x[m + i] * twiddles[i * stride];
    xi[1] = x[m2 + i] * twiddles[i * 2 * stride];
    xi[2] = x[m3 + i] * twiddles[i * 3 * stride];
    xi[3] = xi[0] + xi[2];
    xi[4] = xi[0] - xi[2];
    xi[5] = x[i] - xi[1];
    x[i] += xi[1];
    x[m2 + i] = x[i] - xi[3];
    x[i] += xi[3];

    x[m + i] = {xi[5].real() - xi[4].imag(), xi[5].imag() + xi[4].real()};
    x[m3 + i] = {xi[5].real() + xi[4].imag(), xi[5].imag() - xi[4].real()};
  }
}

template <typename T, bool Inverse>
static constexpr void Butterfly5(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  const size_t m2 = 2 * m;
  const size_t m3 = 3 * m;
  const size_t m4 = 4 * m;
  for (size_t i = 0; i < m; ++i) {
    std::array<T, 13> xi;
    xi[0] = x[i];
    xi[1] = x[m + i] * twiddles[i * stride];
    xi[2] = x[m2 + i] * twiddles[i * 2 * stride];
    xi[3] = x[m3 + i] * twiddles[i * 3 * stride];
    xi[4] = x[m4 + i] * twiddles[i * 4 * stride];
    xi[7] = xi[1] + xi[4];
    xi[10] = xi[1] - xi[4];
    xi[8] = xi[2] + xi[3];
    xi[9] = xi[2] - xi[3];
    x[i] = {
        xi[0].real() + xi[7].real() + xi[8].real(),
        xi[0].imag() + xi[7].imag() + xi[8].imag(),
    };
    xi[5] = {
        xi[0].real() + xi[7].real() * twiddles[m * stride].real() +
            xi[8].real() * twiddles[m * 2 * stride].real(),
        xi[0].imag() + xi[7].imag() * twiddles[m * stride].real() +
            xi[8].imag() * twiddles[m * 2 * stride].real(),
    };
    xi[6] = {
        xi[10].imag() * twiddles[m * stride].imag() +
            xi[9].imag() * twiddles[m * 2 * stride].imag(),
        -xi[10].real() * twiddles[m * stride].imag() -
            xi[9].real() * twiddles[m * 2 * stride].imag(),
    };
    x[m + i] = xi[5] - xi[6];
    x[m4 + i] = xi[5] + xi[6];
    xi[11] = {
        xi[0].real() + xi[7].real() * twiddles[m * 2 * stride].real() +
            xi[8].real() * twiddles[m * stride].real(),
        xi[0].imag() + xi[7].imag() * twiddles[m * 2 * stride].real() +
            xi[8].imag() * twiddles[m * stride].real(),
    };
    xi[12] = {
        xi[9].imag() * twiddles[m * stride].imag() -
            xi[10].imag() * twiddles[m * 2 * stride].imag(),
        xi[10].real() * twiddles[m * 2 * stride].imag() -
            xi[9].real() * twiddles[m * stride].imag(),
    };
    x[m2 + i] = xi[11] + xi[12];
    x[m3 + i] = xi[11] - xi[12];
  }
}

template <typename T, bool Inverse>
static constexpr void ButterflyGeneric(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m,
    const size_t p, const size_t N,
    std::vector<internal::complex<float>>& scratch) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      scratch[j] = x[i + j * m];
    }

    for (size_t j = 0; j < p; ++j) {
      x[i + j * m] = scratch[0];
      size_t twIdx = 0;
      for (size_t k = 1; k < p; ++k) {
        twIdx += stride * (i + j * m);
        if (twIdx >= N) {
          twIdx -= N;
        }
        x[i + j * m] += scratch[k] * twiddles[twIdx];
      }
    }
  }
}

template <typename T, bool Inverse, typename Scaling>
static constexpr void FftRecursive(
    const span<T> x, span<T> y, const size_t inputStride,
    const size_t factorStride, const size_t recursionIndex,
    const span<size_t> factors,
    const std::vector<internal::complex<float>>& twiddles, const size_t N,
    std::vector<internal::complex<float>>& scratch) {
  const auto p = factors[2 * recursionIndex];  // FFT radix for this stage.
  const auto m =
      factors[2 * recursionIndex + 1];  // Length of this FFT stage / radix.

  if (m == 1) {
    // Copy strided input to output, scaling as needed.
    for (size_t i = 0; i < p * m; ++i) {
      y[i] = Scaling::template Scale<T, Inverse>(
          x[i * inputStride * factorStride], N);
    }
  } else {
    for (size_t i = 0; i < p; ++i) {
      // Decimation in time algorithm:
      // Perform p instances of smaller DFTs of size m,
      // each one with a decimated (srided) input.
      FftRecursive<T, Inverse, Scaling>(
          x.subspan(i * factorStride * inputStride), y.subspan(i * m),
          inputStride, factorStride * p, recursionIndex + 1, factors, twiddles,
          N, scratch);
    }
  }

  // Recombine the p smaller DFTs.
  switch (p) {
    case 2:
      Butterfly2<T, Inverse>(y, factorStride, twiddles, m);
      break;
    case 3:
      Butterfly3<T, Inverse>(y, factorStride, twiddles, m);
      break;
    case 4:
      Butterfly4<T, Inverse>(y, factorStride, twiddles, m);
      break;
    case 5:
      Butterfly5<T, Inverse>(y, factorStride, twiddles, m);
      break;
    default:
      ButterflyGeneric<T, Inverse>(y, factorStride, twiddles, m, p, N, scratch);
      break;
  }
}

}  // namespace internal

// Main FFT class.
class FFT {
 public:
  FFT(size_t N) noexcept
      : N_(N),
        factors_(internal::Factorize(N_)),
        twiddlesForward_(internal::ComputeTwiddles<float>(N_, false)),
        twiddlesInverse_(internal::ComputeTwiddles<float>(N_, true)),
        scratch_(internal::RequiredScratchLength(factors_)) {}
  FFT(const FFT&) = default;
  FFT& operator=(const FFT&) = default;
  FFT(FFT&&) = default;
  FFT& operator=(FFT&&) = default;

  // Forward complex-to-complex FFT.
  template <typename Scaling = InverseOneByNScaling>
  void fft(const std::vector<std::complex<float>>& x,
           std::vector<std::complex<float>>& y) noexcept {
    // Convert to internal complex type.
    auto& x_ = reinterpret_cast<const span<kfft::internal::complex<float>>&>(x);
    auto& y_ = reinterpret_cast<span<kfft::internal::complex<float>>&>(y);
    internal::FftRecursive<internal::complex<float>, false, Scaling>(
        x_, y_, 1, 1, 0, factors_, twiddlesForward_, N_, scratch_);
  }

  // Inverse complex-to-complex FFT.
  template <typename Scaling = InverseOneByNScaling>
  void ifft(const std::vector<std::complex<float>>& x,
            std::vector<std::complex<float>>& y) noexcept {
    // Convert to internal complex type.
    auto& x_ = reinterpret_cast<const span<kfft::internal::complex<float>>&>(x);
    auto& y_ = reinterpret_cast<span<kfft::internal::complex<float>>&>(y);
    internal::FftRecursive<internal::complex<float>, true, Scaling>(
        x_, y_, 1, 1, 0, factors_, twiddlesInverse_, N_, scratch_);
  }

 private:
  size_t N_;
  std::vector<size_t> factors_;
  std::vector<internal::complex<float>> twiddlesForward_;
  std::vector<internal::complex<float>> twiddlesInverse_;
  std::vector<internal::complex<float>> scratch_;
};

}  // namespace kfft

#endif  // KISS_FFT_PP_H