// MIT License
//
// Copyright (c) 2024 Kevin Venalainen
//
// This is the main header file for the kissfftpp library.

#ifndef KISS_FFT_PP_H
#define KISS_FFT_PP_H

#include <cstddef>
#include <cstdint>

#ifndef KFFTPP_HALF_PI
#define KFFTPP_HALF_PI 1.5707963267948966192313216916397514420986L
#endif // KFFTPP_HALF_PI

#ifndef KFFTPP_PI
#define KFFTPP_PI 3.1415926535897932384626433832795028841972L
#endif // KFFTPP_PI

namespace kissfftpp {

// Main FFT class.
class FFT {
public:
  FFT() = default;
  FFT(const FFT &) = default;
  FFT(FFT &&) = default;

  void fft(const float *x, float *y, const size_t len) const noexcept {}
};

} // namespace kissfftpp

#endif // KISS_FFT_PP_H