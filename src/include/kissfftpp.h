// Copyright (c) 2024 Kevin Venalainen
//
// This file is part of KissFFT++.
//

#ifndef KISS_FFT_PP_H
#define KISS_FFT_PP_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// If not explicitly set, then enable based on debug status.
#ifndef KFFTPP_NO_CONTRACT_CHECKING
#if defined(NDEBUG)
#define KFFTPP_NO_CONTRACT_CHECKING
#endif
#endif

// If not explicitly set, then disable exceptions based on common compiler
// flags.
#ifndef KFFTPP_NO_EXCEPTIONS
#if !(defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND))
#define KFFTPP_NO_EXCEPTIONS
#endif
#endif

#if !defined(KFFTPP_NO_EXCEPTIONS) && !defined(KFFTPP_NO_CONTRACT_CHECKING)
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
#define KFFTPP_ASSERT(cond, msg)                                               \
  if (!(cond)) {                                                               \
    std::terminate();                                                          \
  }
#else
#define KFFTPP_ASSERT(cond, msg)                                               \
  if (!(cond)) {                                                               \
    throw std::logic_error(msg);                                               \
  }
#endif

#ifndef KFFTPP_HALF_PI
#define KFFTPP_HALF_PI 1.5707963267948966192313216916397514420986L
#endif // KFFTPP_HALF_PI

#ifndef KFFTPP_PI
#define KFFTPP_PI 3.1415926535897932384626433832795028841972L
#endif // KFFTPP_PI

namespace kfft {

namespace internal {

// Lightweight span implementation.
template <typename T> class span {
public:
  // Placeholder for getting all elements during subspan.
  static constexpr size_t ALL = SIZE_MAX;

  // Constructors.
  span(T *ptr, size_t size) noexcept : ptr_(ptr), size_(size) {}
  span(const span &) = default;
  span(span &&) = default;

  // Accessors.
  T *data() const noexcept { return ptr_; }
  size_t size() const noexcept { return size_; }
  T &operator[](size_t i) const {
    KFFTPP_ASSERT(i < size_, "Index out of bounds");
    return ptr_[i];
  }

  // Create a subspan of this span.
  span subspan(size_t offset, size_t count = ALL) const {
    KFFTPP_ASSERT(offset <= size_, "Offset out of bounds");
    KFFTPP_ASSERT(count == ALL || offset + count <= size_,
                  "Count out of bounds");
    return span(ptr_ + offset, count == ALL ? size_ - offset : count);
  }

private:
  T *ptr_;
  size_t size_;
};

// Compute twiddle factors for FFT of length N.
template <typename T>
static constexpr std::vector<std::complex<T>> ComputeTwiddles(size_t N,
                                                              bool inverse) {
  auto twiddles = std::vector<std::complex<T>>(N);
  const double phase = inverse ? 2 * KFFTPP_PI / N : -2 * KFFTPP_PI / N;
  for (size_t i = 0; i < N; ++i) {
    const double phaseArg = phase * i;
    twiddles[i] = {static_cast<T>(std::cos(phaseArg)),
                   static_cast<T>(std::sin(phaseArg))};
  }
  return twiddles;
}

// Factorize N, first as powers of 4, then powers of 2, then remaining
// primes.
static std::vector<int> Factorize(size_t N) {
  auto factors = std::vector<int>();
  int p = 4;
  double floorSqrt = std::floor(std::sqrt(static_cast<double>(N)));
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
      if (p > floorSqrt)
        p = N; // no more factors, skip to end
    }
    N /= p;
    factors.push_back(p);
    factors.push_back(N);
  } while (N > 1);
  return factors;
}

template <typename T>
static constexpr void
Butterfly2(internal::span<T> x, const size_t stride,
           const std::vector<std::complex<float>> &twiddles, const size_t m) {
  for (size_t i = 0; i < m; ++i) {
    const auto xi = x[m + i] * twiddles[i * stride];
    x[m + i] = x[i] - xi;
    x[i] += xi;
  }
}

template <typename T>
static constexpr void
Butterfly4(internal::span<T> x, const size_t stride,
           const std::vector<std::complex<float>> &twiddles, const size_t m) {
  const int m2 = 2 * m;
  const int m3 = 3 * m;
  const size_t stride2 = 2 * stride;
  const size_t stride3 = 3 * stride;
  std::array<T, 6> scratch;
  for (size_t i = 0; i < m; ++i) {
    scratch[0] = x[m + i] * twiddles[i * stride];
    scratch[1] = x[m2 + i] * twiddles[i * stride2];
    scratch[2] = x[m3 + i] * twiddles[i * stride3];
    scratch[5] = x[i] - scratch[1];
    x[i] += scratch[1];
    scratch[3] = scratch[0] + scratch[2];
    scratch[4] = scratch[0] - scratch[2];
    x[m2 + i] = x[i] - scratch[3];
    x[i] += scratch[3];

    // TODO: Different for inverse.
    x[m + i] = {scratch[5].real() + scratch[4].imag(),
                scratch[5].imag() - scratch[4].real()};
    x[m3 + i] = {scratch[5].real() - scratch[4].imag(),
                 scratch[5].imag() + scratch[4].real()};
  }
}

template <typename T>
static constexpr void
FftRecursive(const internal::span<const T> x, internal::span<T> y,
             const size_t inputStride, const size_t factorStride,
             const size_t recursionIndex, const std::vector<int> &factors,
             const std::vector<std::complex<float>> &twiddles) {
  // // recombine the p smaller DFTs
  // switch (p) {
  //     case 2: kf_bfly2(Fout,fstride,st,m); break;
  //     case 3: kf_bfly3(Fout,fstride,st,m); break;
  //     case 4: kf_bfly4(Fout,fstride,st,m); break;
  //     case 5: kf_bfly5(Fout,fstride,st,m); break;
  //     default: kf_bfly_generic(Fout,fstride,st,m,p); break;
  // }

  const int p = factors[2 * recursionIndex]; // FFT radix for this stage.
  const int m =
      factors[2 * recursionIndex + 1]; // Length of this FFT stage / radix.

  if (m == 1) {
    // Copy strided input to output.
    for (size_t i = 0; i < p * m; ++i) {
      y[i] = x[i * inputStride * factorStride];
    }
  } else {
    for (size_t i = 0; i < p; ++i) {
      // Decimation in time algorithm:
      // Perform p instances of smaller DFTs of size m,
      // each one with a decimated (srided) input.
      FftRecursive(x.subspan(i * factorStride * inputStride), y.subspan(i * m),
                   inputStride, factorStride * p, recursionIndex + 1, factors,
                   twiddles);
    }
  }

  // Recombine the p smaller DFTs.
  switch (p) {
  case 2:
    Butterfly2(y, factorStride, twiddles, m);
    break;
  case 3:
    // Butterfly for radix 3.
    break;
  case 4:
    Butterfly4(y, factorStride, twiddles, m);
    break;
  case 5:
    // Butterfly for radix 5.
    break;
  default:
    // Butterfly for generic radix.
    break;
  }
}

} // namespace internal

// Main FFT class.
class FFT {
public:
  FFT(size_t N, bool inverse) noexcept
      : N_(N), inverse_(inverse), factors_(internal::Factorize(N_)),
        twiddles_(internal::ComputeTwiddles<float>(N_, inverse_)) {}
  FFT(const FFT &) = default;
  FFT(FFT &&) = default;

  void fft(const std::vector<std::complex<float>> &x,
           std::vector<std::complex<float>> &y) const noexcept {
    internal::FftRecursive<std::complex<float>>(
        internal::span<const std::complex<float>>(x.data(), N_),
        internal::span<std::complex<float>>(y.data(), N_), 1, 1, 0, factors_,
        twiddles_);
  }

private:
  size_t N_;
  bool inverse_;
  std::vector<int> factors_;
  std::vector<std::complex<float>> twiddles_;
};

} // namespace kfft

#endif // KISS_FFT_PP_H