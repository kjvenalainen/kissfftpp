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
  constexpr kfft::internal::complex<float> c1(3, 4);

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

// Tests for kfft::span
TEST(KissFftppInternal, Span) {
  std::vector<float> v = {1, 2, 3, 4, 5};
  kfft::span<float> s(v.data(), v.size());

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

// Assert correct outputs for complex<float> input and output.
TEST(KissFftpp, CorrectnessComplexFloat) {
  for (size_t index = 0; index < TEST_FFT_SIZES.size(); index++) {
    size_t length = TEST_FFT_SIZES[index];

    std::vector<std::complex<float>> x(length);
    std::vector<std::complex<float>> y(length, {0, 0});

    auto kfftpp = kfft::FFT(length);

    for (size_t i = 0; i < length; i++) {
      x[i] = {TEST_DATA_INPUT[2 * i], TEST_DATA_INPUT[2 * i + 1]};
    }

    kfftpp.fft<kfft::InverseOneByNScaling>(x, y);

    for (size_t i = 0; i < length; i++) {
      EXPECT_NEAR_RELATIVE(y[i].real(),
                           TEST_DATA_OUTPUT_COMPLEX_FLOAT[index][i].real(), 0.1)
          << "index: " << index << " i: " << i;
      EXPECT_NEAR_RELATIVE(y[i].imag(),
                           TEST_DATA_OUTPUT_COMPLEX_FLOAT[index][i].imag(), 0.1)
          << "index: " << index << " i: " << i;
    }

    // Test inverse, asserting that iffft(fft(x)) = x (with default 1/N scaling
    // on the inverse).
    kfftpp.ifft<kfft::InverseOneByNScaling>(y, x);

    for (size_t i = 0; i < length; i++) {
      EXPECT_NEAR_RELATIVE(x[i].real(), TEST_DATA_INPUT[2 * i], 0.1)
          << "index: " << index << " i: " << i;
      EXPECT_NEAR_RELATIVE(x[i].imag(), TEST_DATA_INPUT[2 * i + 1], 0.1)
          << "index: " << index << " i: " << i;
    }

    // Test inverse with no scaling, resulting in iffft(fft(x)) = x * N.
    kfftpp.ifft<kfft::NoScaling>(y, x);

    for (size_t i = 0; i < length; i++) {
      EXPECT_NEAR_RELATIVE(x[i].real(), TEST_DATA_INPUT[2 * i] * length, 0.1)
          << "index: " << index << " i: " << i;
      EXPECT_NEAR_RELATIVE(x[i].imag(), TEST_DATA_INPUT[2 * i + 1] * length,
                           0.1)
          << "index: " << index << " i: " << i;
    }
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

  kfft::FFT kfftpp(N);

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

      kfftpp.fft<kfft::NoScaling>(x1, y1);

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

TEST(KissFft, DISABLED_PrintTestData) {
  for (size_t index = 0; index < TEST_FFT_SIZES.size(); index++) {
    size_t length = TEST_FFT_SIZES[index];
    std::vector<kiss_fft_cpx> x(length);
    std::vector<kiss_fft_cpx> y(length, {0, 0});

    auto kfft = kiss_fft_alloc(length, 0, NULL, 0);
    EXPECT_NE(kfft, nullptr);

    for (size_t i = 0; i < length; i++) {
      x[i] = {TEST_DATA_INPUT[2 * i], TEST_DATA_INPUT[2 * i + 1]};
    }

    kiss_fft(kfft, x.data(), y.data());

    // Print with 7 decimal precision.
    std::cout << std::fixed << std::setprecision(7);
    std::cout << "constexpr static std::array<std::complex<float>, " << length
              << "> TEST_DATA_OUTPUT_COMPLEX_FLOAT_" << length << " = { ";
    for (size_t i = 0; i < length; i++) {
      std::cout << "std::complex<float>(" << y[i].r << "," << y[i].i << "), ";
    }
    std::cout << "};" << std::endl;

    kiss_fft_free(kfft);

    for (size_t i = 0; i < length; i++) {
      EXPECT_NEAR(y[i].r, TEST_DATA_OUTPUT_COMPLEX_FLOAT[index][i].real(), 1e-3)
          << "index: " << index << " i: " << i;
      EXPECT_NEAR(y[i].i, TEST_DATA_OUTPUT_COMPLEX_FLOAT[index][i].imag(), 1e-3)
          << "index: " << index << " i: " << i;
    }
  }

  std::cout << " constexpr static std::array<kfft::span<const "
               "std::complex<float>>, "
            << TEST_FFT_SIZES.size() << "> TEST_DATA_OUTPUT_COMPLEX_FLOAT = {";
  for (size_t index = 0; index < TEST_FFT_SIZES.size(); index++) {
    std::cout << "kfft::span<const "
                 "std::complex<float>>(&TEST_DATA_OUTPUT_COMPLEX_FLOAT_"
              << TEST_FFT_SIZES[index] << "[0], " << TEST_FFT_SIZES[index]
              << "), ";
  }
  std::cout << "};" << std::endl;
}