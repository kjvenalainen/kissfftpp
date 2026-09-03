// Copyright (c) 2024 Kevin Venalainen
//
// This file is part of KissFFT++.
//

#ifndef KISS_FFT_PP_H
#define KISS_FFT_PP_H

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
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
#if KFFTPP_NO_CONTRACT_CHECKING
#define KFFTPP_ASSERT(cond, msg) ((void)0)
#elif KFFTPP_NO_EXCEPTIONS
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
  // Extent type to handle static extent, where the size is part of the type
  // itself.
  template <size_t E>
  struct extent_type {
    constexpr explicit extent_type(size_t) noexcept {};
    static constexpr size_t value_ = E;
  };

  // Specialization for dynamic_extent.
  template <>
  struct extent_type<dynamic_extent> {
    constexpr explicit extent_type(size_t value) noexcept : value_(value) {};
    size_t value_;
  };

 public:
  static constexpr std::size_t extent = Extent;

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
  template <
      size_t _Extent = Extent,
      typename std::enable_if<_Extent == dynamic_extent, bool>::type = true>
  constexpr span(std::vector<T>& vec) noexcept
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
  constexpr span subspan(size_t offset, size_t count = dynamic_extent) const {
    KFFTPP_ASSERT(offset <= size_.value_, "Offset out of bounds");
    KFFTPP_ASSERT(count == dynamic_extent || offset + count <= size_.value_,
                  "Count out of bounds");
    return span(data() + offset,
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
  T* x0 = x.data();
  T* x1 = x0 + m;
  const auto* twiddle = twiddles.data();
  for (size_t i = 0; i < m; ++i) {
    const auto xi = *x1 * *twiddle;
    *x1 = *x0 - xi;
    *x0 += xi;
    ++x0;
    ++x1;
    twiddle += stride;
  }
}

template <typename T, bool Inverse>
static constexpr void Butterfly3(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  T* x0 = x.data();
  T* x1 = x0 + m;
  T* x2 = x1 + m;
  const auto* twiddle1 = twiddles.data();
  const auto* twiddle2 = twiddle1;
  const auto epi3 = twiddles[m * stride].imag();
  for (size_t i = 0; i < m; ++i) {
    std::array<T, 5> xi;
    xi[1] = *x1 * *twiddle1;
    xi[2] = *x2 * *twiddle2;
    xi[3] = xi[1] + xi[2];
    xi[0] = xi[1] - xi[2];
    *x1 = {
        x0->real() - (xi[3].real() * 0.5f),
        x0->imag() - (xi[3].imag() * 0.5f),
    };
    xi[0] *= epi3;
    *x0 += xi[3];
    *x2 = {x1->real() + xi[0].imag(), x1->imag() - xi[0].real()};
    *x1 = {x1->real() - xi[0].imag(), x1->imag() + xi[0].real()};
    ++x0;
    ++x1;
    ++x2;
    twiddle1 += stride;
    twiddle2 += 2 * stride;
  }
}

template <typename T, bool Inverse,
          typename std::enable_if_t<!Inverse, bool> = true>
static constexpr void Butterfly4(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  T* x0 = x.data();
  T* x1 = x0 + m;
  T* x2 = x1 + m;
  T* x3 = x2 + m;
  const auto* twiddle1 = twiddles.data();
  const auto* twiddle2 = twiddle1;
  const auto* twiddle3 = twiddle1;
  for (size_t i = 0; i < m; ++i) {
    std::array<T, 6> xi;
    xi[0] = *x1 * *twiddle1;
    xi[1] = *x2 * *twiddle2;
    xi[2] = *x3 * *twiddle3;
    xi[3] = xi[0] + xi[2];
    xi[4] = xi[0] - xi[2];
    xi[5] = *x0 - xi[1];
    *x0 += xi[1];
    *x2 = *x0 - xi[3];
    *x0 += xi[3];

    *x1 = {xi[5].real() + xi[4].imag(), xi[5].imag() - xi[4].real()};
    *x3 = {xi[5].real() - xi[4].imag(), xi[5].imag() + xi[4].real()};
    ++x0;
    ++x1;
    ++x2;
    ++x3;
    twiddle1 += stride;
    twiddle2 += 2 * stride;
    twiddle3 += 3 * stride;
  }
}

template <typename T, bool Inverse,
          typename std::enable_if_t<Inverse, bool> = true>
static constexpr void Butterfly4(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  T* x0 = x.data();
  T* x1 = x0 + m;
  T* x2 = x1 + m;
  T* x3 = x2 + m;
  const auto* twiddle1 = twiddles.data();
  const auto* twiddle2 = twiddle1;
  const auto* twiddle3 = twiddle1;
  for (size_t i = 0; i < m; ++i) {
    std::array<T, 6> xi;
    xi[0] = *x1 * *twiddle1;
    xi[1] = *x2 * *twiddle2;
    xi[2] = *x3 * *twiddle3;
    xi[3] = xi[0] + xi[2];
    xi[4] = xi[0] - xi[2];
    xi[5] = *x0 - xi[1];
    *x0 += xi[1];
    *x2 = *x0 - xi[3];
    *x0 += xi[3];

    *x1 = {xi[5].real() - xi[4].imag(), xi[5].imag() + xi[4].real()};
    *x3 = {xi[5].real() + xi[4].imag(), xi[5].imag() - xi[4].real()};
    ++x0;
    ++x1;
    ++x2;
    ++x3;
    twiddle1 += stride;
    twiddle2 += 2 * stride;
    twiddle3 += 3 * stride;
  }
}

template <typename T, bool Inverse>
static constexpr void Butterfly5(
    span<T> x, const size_t stride,
    const std::vector<internal::complex<float>>& twiddles, const size_t m) {
  T* x0 = x.data();
  T* x1 = x0 + m;
  T* x2 = x1 + m;
  T* x3 = x2 + m;
  T* x4 = x3 + m;
  const auto* twiddle1 = twiddles.data();
  const auto* twiddle2 = twiddle1;
  const auto* twiddle3 = twiddle1;
  const auto* twiddle4 = twiddle1;
  const auto ya = twiddles[m * stride];
  const auto yb = twiddles[m * 2 * stride];
  for (size_t i = 0; i < m; ++i) {
    std::array<T, 13> xi;
    xi[0] = *x0;
    xi[1] = *x1 * *twiddle1;
    xi[2] = *x2 * *twiddle2;
    xi[3] = *x3 * *twiddle3;
    xi[4] = *x4 * *twiddle4;
    xi[7] = xi[1] + xi[4];
    xi[10] = xi[1] - xi[4];
    xi[8] = xi[2] + xi[3];
    xi[9] = xi[2] - xi[3];
    *x0 = {
        xi[0].real() + xi[7].real() + xi[8].real(),
        xi[0].imag() + xi[7].imag() + xi[8].imag(),
    };
    xi[5] = {
        xi[0].real() + xi[7].real() * ya.real() +
            xi[8].real() * yb.real(),
        xi[0].imag() + xi[7].imag() * ya.real() +
            xi[8].imag() * yb.real(),
    };
    xi[6] = {
        xi[10].imag() * ya.imag() + xi[9].imag() * yb.imag(),
        -xi[10].real() * ya.imag() - xi[9].real() * yb.imag(),
    };
    *x1 = xi[5] - xi[6];
    *x4 = xi[5] + xi[6];
    xi[11] = {
        xi[0].real() + xi[7].real() * yb.real() +
            xi[8].real() * ya.real(),
        xi[0].imag() + xi[7].imag() * yb.real() +
            xi[8].imag() * ya.real(),
    };
    xi[12] = {
        xi[9].imag() * ya.imag() - xi[10].imag() * yb.imag(),
        xi[10].real() * yb.imag() - xi[9].real() * ya.imag(),
    };
    *x2 = xi[11] + xi[12];
    *x3 = xi[11] - xi[12];
    ++x0;
    ++x1;
    ++x2;
    ++x3;
    ++x4;
    twiddle1 += stride;
    twiddle2 += 2 * stride;
    twiddle3 += 3 * stride;
    twiddle4 += 4 * stride;
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
    // The final radix-2 and radix-4 stages always use twiddle factor one.
    // Evaluate them directly so the common power-of-two transforms do not
    // spend complex multiplies loading and multiplying by that identity.
    const size_t stride = inputStride * factorStride;
    if (p == 2) {
      const auto x0 = Scaling::template Scale<T, Inverse>(x[0], N);
      const auto x1 = Scaling::template Scale<T, Inverse>(x[stride], N);
      y[0] = x0 + x1;
      y[1] = x0 - x1;
      return;
    }
    if (p == 4) {
      const auto x0 = Scaling::template Scale<T, Inverse>(x[0], N);
      const auto x1 = Scaling::template Scale<T, Inverse>(x[stride], N);
      const auto x2 = Scaling::template Scale<T, Inverse>(x[2 * stride], N);
      const auto x3 = Scaling::template Scale<T, Inverse>(x[3 * stride], N);
      const auto x0PlusX2 = x0 + x2;
      const auto x0MinusX2 = x0 - x2;
      const auto x1PlusX3 = x1 + x3;
      const auto x1MinusX3 = x1 - x3;
      y[0] = x0PlusX2 + x1PlusX3;
      y[2] = x0PlusX2 - x1PlusX3;
      if (Inverse) {
        y[1] = {x0MinusX2.real() - x1MinusX3.imag(),
                x0MinusX2.imag() + x1MinusX3.real()};
        y[3] = {x0MinusX2.real() + x1MinusX3.imag(),
                x0MinusX2.imag() - x1MinusX3.real()};
      } else {
        y[1] = {x0MinusX2.real() + x1MinusX3.imag(),
                x0MinusX2.imag() - x1MinusX3.real()};
        y[3] = {x0MinusX2.real() - x1MinusX3.imag(),
                x0MinusX2.imag() + x1MinusX3.real()};
      }
      return;
    }

    // Copy strided input to output, scaling as needed.
    for (size_t i = 0; i < p; ++i) {
      y[i] = Scaling::template Scale<T, Inverse>(x[i * stride], N);
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

// Real-to-complex FFT using the same packed, half-length complex transform as
// kiss_fftr. The forward transform produces N / 2 + 1 bins: DC through the
// Nyquist frequency. The inverse accepts that same non-redundant spectrum.
class RealFFT {
 public:
  explicit RealFFT(size_t N)
      : N_(N),
        ncfft_(ValidateLength(N)),
        factors_(internal::Factorize(ncfft_)),
        twiddlesForward_(internal::ComputeTwiddles<float>(ncfft_, false)),
        twiddlesInverse_(internal::ComputeTwiddles<float>(ncfft_, true)),
        superTwiddlesForward_(ComputeSuperTwiddles(false)),
        superTwiddlesInverse_(ComputeSuperTwiddles(true)),
        packedInput_(ncfft_),
        packedOutput_(ncfft_),
        scratch_(internal::RequiredScratchLength(factors_)) {}

  RealFFT(const RealFFT&) = default;
  RealFFT& operator=(const RealFFT&) = default;
  RealFFT(RealFFT&&) = default;
  RealFFT& operator=(RealFFT&&) = default;

  // Forward real-to-complex FFT. `x` must contain N samples and `y` must
  // contain N / 2 + 1 frequency bins.
  template <typename Scaling = InverseOneByNScaling>
  void fft(const std::vector<float>& x, std::vector<std::complex<float>>& y) {
    KFFTPP_ASSERT(x.size() == N_, "Input size must equal the FFT length");
    KFFTPP_ASSERT(y.size() == ncfft_ + 1, "Output size must equal N / 2 + 1");

    // Pack even samples in the real component and odd samples in the
    // imaginary component, then transform the packed signal once.
    for (size_t i = 0; i < ncfft_; ++i) {
      packedInput_[i] = {x[2 * i], x[2 * i + 1]};
    }
    internal::FftRecursive<internal::complex<float>, false, NoScaling>(
        span<internal::complex<float>>(packedInput_),
        span<internal::complex<float>>(packedOutput_), 1, 1, 0, factors_,
        twiddlesForward_, ncfft_, scratch_);

    const auto dc = packedOutput_[0];
    StoreScaled<Scaling, false>(y[0], {dc.real() + dc.imag(), 0.0f}, N_);
    StoreScaled<Scaling, false>(y[ncfft_], {dc.real() - dc.imag(), 0.0f}, N_);

    // This is the recombination step from kiss_fftr. At the midpoint both
    // assignments address the same bin; retain the C implementation's order.
    for (size_t k = 1; k <= ncfft_ / 2; ++k) {
      const auto fpk = packedOutput_[k];
      const auto fpnk = internal::complex<float>(
          packedOutput_[ncfft_ - k].real(), -packedOutput_[ncfft_ - k].imag());
      const auto f1k = fpk + fpnk;
      const auto f2k = fpk - fpnk;
      const auto tw = f2k * superTwiddlesForward_[k - 1];

      StoreScaled<Scaling, false>(y[k], (f1k + tw) * 0.5f, N_);
      StoreScaled<Scaling, false>(
          y[ncfft_ - k],
          {0.5f * (f1k.real() - tw.real()), 0.5f * (tw.imag() - f1k.imag())},
          N_);
    }
  }

  // Inverse complex-to-real FFT. With the default scaling, ifft(fft(x)) == x.
  template <typename Scaling = InverseOneByNScaling>
  void ifft(const std::vector<std::complex<float>>& x, std::vector<float>& y) {
    KFFTPP_ASSERT(x.size() == ncfft_ + 1, "Input size must equal N / 2 + 1");
    KFFTPP_ASSERT(y.size() == N_, "Output size must equal the FFT length");

    packedInput_[0] = {x[0].real() + x[ncfft_].real(),
                       x[0].real() - x[ncfft_].real()};

    // This is the inverse recombination step from kiss_fftri. The DC and
    // Nyquist imaginary components are intentionally ignored, as in KissFFT.
    for (size_t k = 1; k <= ncfft_ / 2; ++k) {
      const internal::complex<float> fk(x[k].real(), x[k].imag());
      const internal::complex<float> fnkc(x[ncfft_ - k].real(),
                                          -x[ncfft_ - k].imag());
      const auto fek = fk + fnkc;
      const auto fok = (fk - fnkc) * superTwiddlesInverse_[k - 1];
      packedInput_[k] = fek + fok;
      packedInput_[ncfft_ - k] = fek - fok;
      packedInput_[ncfft_ - k].imag(-packedInput_[ncfft_ - k].imag());
    }

    internal::FftRecursive<internal::complex<float>, true, NoScaling>(
        span<internal::complex<float>>(packedInput_),
        span<internal::complex<float>>(packedOutput_), 1, 1, 0, factors_,
        twiddlesInverse_, ncfft_, scratch_);

    for (size_t i = 0; i < ncfft_; ++i) {
      y[2 * i] =
          Scaling::template Scale<float, true>(packedOutput_[i].real(), N_);
      y[2 * i + 1] =
          Scaling::template Scale<float, true>(packedOutput_[i].imag(), N_);
    }
  }

 private:
  static size_t ValidateLength(size_t N) {
    KFFTPP_ASSERT(N >= 2 && N % 2 == 0,
                  "Real FFT length must be a non-zero even number");
    return N / 2;
  }

  std::vector<internal::complex<float>> ComputeSuperTwiddles(
      bool inverse) const {
    auto twiddles = std::vector<internal::complex<float>>(ncfft_ / 2);
    const float direction = inverse ? 1.0f : -1.0f;
    for (size_t i = 0; i < twiddles.size(); ++i) {
      const double phase =
          direction * KFFTPP_PI * (static_cast<double>(i + 1) / ncfft_ + 0.5);
      twiddles[i] = {static_cast<float>(std::cos(phase)),
                     static_cast<float>(std::sin(phase))};
    }
    return twiddles;
  }

  template <typename Scaling, bool Inverse>
  static void StoreScaled(std::complex<float>& destination,
                          const internal::complex<float>& value,
                          const float N) {
    const auto scaled =
        Scaling::template Scale<internal::complex<float>, Inverse>(value, N);
    destination = {scaled.real(), scaled.imag()};
  }

  size_t N_;
  size_t ncfft_;
  std::vector<size_t> factors_;
  std::vector<internal::complex<float>> twiddlesForward_;
  std::vector<internal::complex<float>> twiddlesInverse_;
  std::vector<internal::complex<float>> superTwiddlesForward_;
  std::vector<internal::complex<float>> superTwiddlesInverse_;
  std::vector<internal::complex<float>> packedInput_;
  std::vector<internal::complex<float>> packedOutput_;
  std::vector<internal::complex<float>> scratch_;
};

}  // namespace kfft

#endif  // KISS_FFT_PP_H
