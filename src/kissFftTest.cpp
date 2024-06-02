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
