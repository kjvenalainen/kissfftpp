// Copyright (c) 2024 Kevin Venalainen
//
// This file is part of KissFFT++.
//

#include <array>
#include <cmath>

#include "kissfft/kiss_fft.h"
#include "src/include/kissfftpp.h"
#include "testDataKissfft.h"
#include "testUtils.h"
#include "gtest/gtest.h"

using namespace kfftTestData;

TEST(KissFft, KissFft) {
  auto kfft = kiss_fft_alloc(512, 0, NULL, 0);
  EXPECT_NE(kfft, nullptr);

  std::vector<kiss_fft_cpx> x(KISS_FFT_X_512.size());
  std::vector<kiss_fft_cpx> y(KISS_FFT_X_512.size(), {0, 0});

  for (size_t i = 0; i < KISS_FFT_X_512.size(); i++) {
    x[i].r = KISS_FFT_X_512[i];
    x[i].i = 0;
  }

  kiss_fft(kfft, x.data(), y.data());

  // Print y with 7 decimals precision
  // for (size_t i = 0; i < KISS_FFT_X_512.size(); i++) {
  //   std::cout << "{ " << std::fixed << std::setprecision(7) << y[i].r << ", "
  //             << y[i].i << "}, ";
  // }

  for (size_t i = 0; i < KISS_FFT_X_512.size(); i++) {
    EXPECT_NEAR(y[i].r, KISS_FFT_Y_512[i].r, 1e-3);
    EXPECT_NEAR(y[i].i, KISS_FFT_Y_512[i].i, 1e-3);
  }

  kiss_fft_free(kfft);
}

TEST(KissFftpp, KissFftpp) {
  kfft::FFT kfft(512, false);
  auto x = std::vector<std::complex<float>>(KISS_FFT_X_512.size());
  auto y = std::vector<std::complex<float>>(KISS_FFT_X_512.size(), {0, 0});

  for (size_t i = 0; i < KISS_FFT_X_512.size(); i++) {
    x[i] = {KISS_FFT_X_512[i], 0};
  }

  kfft.fft(x, y);

  // Print y with 7 decimals precision
  // for (size_t i = 0; i < KISS_FFT_X_512.size(); i++) {
  //   std::cout << "{ " << std::fixed << std::setprecision(7) << y[i].real() <<
  //   ", "
  //             << y[i].imag() << "}, ";
  // }

  for (size_t i = 0; i < KISS_FFT_X_512.size(); i++) {
    EXPECT_NEAR(y[i].real(), KISS_FFT_Y_512[i].r, 1e-3);
    EXPECT_NEAR(y[i].imag(), KISS_FFT_Y_512[i].i, 1e-3);
  }
}

TEST(KissFftpp, KissFftCompare) {
  constexpr static size_t N = 16;
  std::vector<kiss_fft_cpx> x0(N);
  std::vector<kiss_fft_cpx> y0(N, {0, 0});

  auto kfft = kiss_fft_alloc(N, 0, NULL, 0);
  EXPECT_NE(kfft, nullptr);

  for (size_t i = 0; i < N; i++) {
    x0[i] = {KISS_FFT_X_512[i], 0};
  }

  kiss_fft(kfft, x0.data(), y0.data());

  std::vector<std::complex<float>> x1(N);
  std::vector<std::complex<float>> y1(N, {0, 0});

  kfft::FFT kfftpp(N, false);

  for (size_t i = 0; i < N; i++) {
    x1[i] = {KISS_FFT_X_512[i], 0};
  }

  kfftpp.fft(x1, y1);

  // Print y with 7 decimals precision
  // for (size_t i = 0; i < KISS_FFT_X_512.size(); i++) {
  //   std::cout << "{ " << std::fixed << std::setprecision(7) << y[i].real() <<
  //   ", "
  //             << y[i].imag() << "}, ";
  // }

  for (size_t i = 0; i < N; i++) {
    std::cout << "{ " << std::fixed << std::setprecision(7) << y0[i].r << ", "
              << y0[i].i << "}, ";
  }

  std::cout << std::endl << std::endl;

  for (size_t i = 0; i < N; i++) {
    std::cout << "{ " << std::fixed << std::setprecision(7) << y1[i].real()
              << ", " << y1[i].imag() << "}, ";
  }

  for (size_t i = 0; i < N; i++) {
    EXPECT_NEAR(y0[i].r, y1[i].real(), 1e-3);
    EXPECT_NEAR(y0[i].i, y1[i].imag(), 1e-3);
  }
}

TEST(KissFftpp, KissFftPerformance) {
  // Run a performance test using many random length 512 ffts.

  constexpr static size_t N = 512;
  constexpr static size_t NUM_TESTS = 100000;
  std::vector<kiss_fft_cpx> x0(N);
  std::vector<kiss_fft_cpx> y0(N, {0, 0});

  auto kfft = kiss_fft_alloc(N, 0, NULL, 0);
  EXPECT_NE(kfft, nullptr);

  std::vector<std::complex<float>> x1(N);
  std::vector<std::complex<float>> y1(N, {0, 0});

  kfft::FFT kfftpp(N, false);

  kiss_fft(kfft, x0.data(), y0.data());
  kfftpp.fft(x1, y1);

  std::chrono::duration<double> durationKissFft =
      std::chrono::duration<double>::zero();
  std::chrono::duration<double> durationKissFftpp =
      std::chrono::duration<double>::zero();

  for (size_t i = 0; i < NUM_TESTS; i++) {
    // Generate random input
    for (size_t j = 0; j < N; j++) {
      x0[j] = {static_cast<float>(rand()) / RAND_MAX,
               static_cast<float>(rand()) / RAND_MAX};
      x1[j] = {x0[j].r, x0[j].i};
    }

    if (rand() % 2) {
      // Run kiss_fft first.

      // Measure KissFFT exeuction time.
      std::chrono::high_resolution_clock::time_point start =
          std::chrono::high_resolution_clock::now();

      kiss_fft(kfft, x0.data(), y0.data());

      std::chrono::high_resolution_clock::time_point end =
          std::chrono::high_resolution_clock::now();
      durationKissFft +=
          std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                    start);

      // Measure KissFFTPP exeuction time.
      start = std::chrono::high_resolution_clock::now();

      kfftpp.fft(x1, y1);

      end = std::chrono::high_resolution_clock::now();
      durationKissFftpp +=
          std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                    start);
    } else {
      // Run kiss_fft++ first.

      // Measure KissFFTPP exeuction time.
      std::chrono::high_resolution_clock::time_point start =
          std::chrono::high_resolution_clock::now();

      kfftpp.fft(x1, y1);

      std::chrono::high_resolution_clock::time_point end =
          std::chrono::high_resolution_clock::now();
      durationKissFftpp +=
          std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                    start);

      // Measure KissFFT exeuction time.
      start = std::chrono::high_resolution_clock::now();

      kiss_fft(kfft, x0.data(), y0.data());

      end = std::chrono::high_resolution_clock::now();
      durationKissFft +=
          std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                    start);
    }
  }

  // Print results.
  std::cout << "KissFFT: " << durationKissFft.count() << " seconds"
            << std::endl;
  std::cout << "KissFFTPP: " << durationKissFftpp.count() << " seconds"
            << std::endl;
}