// Copyright (c) 2024 Kevin Venalainen
//
// This file is part of KissFFT++.
//

#include <cstddef>
#include <complex>
#include <iostream>
#include <vector>

#include "include/kissfftpp.h"

int main() {
  constexpr std::size_t kLength = 8;
  std::vector<std::complex<float>> input(kLength, {0.0f, 0.0f});
  std::vector<std::complex<float>> spectrum(kLength);
  input[0] = {1.0f, 0.0f};

  kfft::FFT fft(kLength);
  fft.fft(input, spectrum);

  for (const auto& bin : spectrum) {
    std::cout << bin << '\n';
  }

  return 0;
}
