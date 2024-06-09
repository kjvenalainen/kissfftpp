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

// Tests for kfft::internal::complex
TEST(KissFftppInternal, Complex) {
  kfft::internal::complex<float> c0(1, 2);
  kfft::internal::complex<float> c1(3, 4);

  // Accessors.
  EXPECT_EQ(c0.real(), 1);
  EXPECT_EQ(c0.imag(), 2);

  EXPECT_EQ(c1.real(), 3);
  EXPECT_EQ(c1.imag(), 4);

  // Modifiers.
  c0.real(5);
  c0.imag(6);
  EXPECT_EQ(c0.real(), 5);
  EXPECT_EQ(c0.imag(), 6);
  c0 = {1, 2};

  // Member arithmetic operators.
  auto c2 = c0 + c1;
  EXPECT_EQ(c2.real(), 4);
  EXPECT_EQ(c2.imag(), 6);

  auto c3 = c0 - c1;
  EXPECT_EQ(c3.real(), -2);
  EXPECT_EQ(c3.imag(), -2);

  auto c4 = c0 * c1;
  EXPECT_EQ(c4.real(), -5);
  EXPECT_EQ(c4.imag(), 10);

  auto c5 = c0 / c1;
  EXPECT_NEAR(c5.real(), 0.44, 1e-3);
  EXPECT_NEAR(c5.imag(), 0.08, 1e-3);

  // Non-member arithmetic operators.
  auto c6 = c0 + 2.0f;
  EXPECT_EQ(c6.real(), 3);
  EXPECT_EQ(c6.imag(), 2);

  auto c7 = c0 - 2.0f;
  EXPECT_EQ(c7.real(), -1);
  EXPECT_EQ(c7.imag(), 2);

  auto c8 = c0 * 2.0f;
  EXPECT_EQ(c8.real(), 2);
  EXPECT_EQ(c8.imag(), 4);

  auto c9 = c0 / 2.0f;
  EXPECT_EQ(c9.real(), 0.5);
  EXPECT_EQ(c9.imag(), 1);

  auto c10 = 2.0f + c0;
  EXPECT_EQ(c10.real(), 3);
  EXPECT_EQ(c10.imag(), 2);

  auto c11 = 2.0f - c0;
  EXPECT_EQ(c11.real(), 1);
  EXPECT_EQ(c11.imag(), -2);

  auto c12 = 2.0f * c0;
  EXPECT_EQ(c12.real(), 2);
  EXPECT_EQ(c12.imag(), 4);

  auto c13 = 2.0f / c0;
  EXPECT_NEAR(c13.real(), 0.4, 1e-7);
  EXPECT_NEAR(c13.imag(), -0.8, 1e-7);

  auto c14 = -c0;
  EXPECT_EQ(c14.real(), -1);
}

// Tests for kfft::internal::span
TEST(KissFftppInternal, Span) {
  std::vector<float> v = {1, 2, 3, 4, 5};
  kfft::internal::span<float> s(v.data(), v.size());

  // Accessors.
  EXPECT_EQ(s.size(), v.size());
  EXPECT_EQ(s.data(), v.data());
  EXPECT_EQ(s[0], 1);
  EXPECT_EQ(s[1], 2);
  EXPECT_EQ(s[2], 3);
  EXPECT_EQ(s[3], 4);
  EXPECT_EQ(s[4], 5);

  // Modifiers.
  s[0] = 6;
  EXPECT_EQ(s[0], 6);

  // Subspan.
  auto s2 = s.subspan(1, 3);
  EXPECT_EQ(s2.size(), 3);
  EXPECT_EQ(s2[0], 2);
  EXPECT_EQ(s2[1], 3);
  EXPECT_EQ(s2[2], 4);

// Bounds checking.
#if !defined(KFFTPP_NO_CONTRACT_CHECKING) && !defined(KFFTPP_NO_EXCEPTIONS)
  EXPECT_THROW(s[5], std::logic_error);
  EXPECT_THROW(s.subspan(1, 5), std::logic_error);
  EXPECT_THROW(s.subspan(5, 1), std::logic_error);
#endif
}

TEST(KissFftppInternal, RequiredScratchLength) {
  EXPECT_EQ(kfft::internal::RequiredScratchLength({2, 1}), 0);
  EXPECT_EQ(kfft::internal::RequiredScratchLength({3, 1}), 0);
  EXPECT_EQ(kfft::internal::RequiredScratchLength({4, 1}), 0);
  EXPECT_EQ(kfft::internal::RequiredScratchLength({5, 1}), 0);
  EXPECT_EQ(kfft::internal::RequiredScratchLength({6, 1}), 6);
  EXPECT_EQ(kfft::internal::RequiredScratchLength({6, 1, 13, 2}), 13);
}

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
  constexpr static size_t N = 1560;
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

  // for (size_t i = 0; i < N; i++) {
  //   std::cout << "{ " << std::fixed << std::setprecision(7) << y0[i].r << ",
  //   "
  //             << y0[i].i << "}, ";
  // }

  // std::cout << std::endl << std::endl;

  // for (size_t i = 0; i < N; i++) {
  //   std::cout << "{ " << std::fixed << std::setprecision(7) << y1[i].real()
  //             << ", " << y1[i].imag() << "}, ";
  // }

  for (size_t i = 0; i < N; i++) {
    const auto norm = std::sqrt(y0[i].r * y0[i].r + y0[i].i * y0[i].i);
    EXPECT_NEAR(y0[i].r, y1[i].real(), 1e-3 * norm);
    EXPECT_NEAR(y0[i].i, y1[i].imag(), 1e-3 * norm);
  }
}

TEST(KissFftpp, KissFftPerformance) {
  // Run a performance test using many random length 512 ffts.

  constexpr static size_t N = 1560;
  constexpr static size_t NUM_TESTS = 10000;
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