// Copyright (c) 2024 Kevin Venalainen
//
// This file is part of KissFFT++.
//

#include <chrono>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "include/kissfftpp.h"
#include "kissfft/kiss_fft.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kWarmupIterations = 10;
constexpr int kMeasuredIterations = 1000;
constexpr int kSizeColumnWidth = 8;
constexpr int kTimeColumnWidth = 18;
constexpr int kSpeedupColumnWidth = 10;

struct Result {
  double milliseconds;
  float checksum;
};

Result BenchmarkKissFFT(const std::vector<kiss_fft_cpx>& input,
                        std::vector<kiss_fft_cpx>& output,
                        kiss_fft_cfg plan, int iterations) {
  const auto start = Clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    kiss_fft(plan, input.data(), output.data());
  }
  const auto end = Clock::now();

  float checksum = 0.0f;
  for (const kiss_fft_cpx& value : output) {
    checksum += value.r + value.i;
  }

  return {std::chrono::duration<double, std::milli>(end - start).count(),
          checksum};
}

Result BenchmarkKissFFTpp(const std::vector<std::complex<float>>& input,
                          std::vector<std::complex<float>>& output,
                          kfft::FFT& plan, int iterations) {
  const auto start = Clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    plan.fft<kfft::NoScaling>(input, output);
  }
  const auto end = Clock::now();

  float checksum = 0.0f;
  for (const std::complex<float>& value : output) {
    checksum += value.real() + value.imag();
  }

  return {std::chrono::duration<double, std::milli>(end - start).count(),
          checksum};
}

int ParseIterations(int argc, char** argv) {
  if (argc == 1) {
    return kMeasuredIterations;
  }
  if (argc == 3 && std::string(argv[1]) == "--iterations") {
    const int iterations = std::atoi(argv[2]);
    if (iterations > 0) {
      return iterations;
    }
  }

  std::cerr << "Usage: " << argv[0]
            << " [--iterations positive_integer]\n";
  return 0;
}

std::string FormatValue(double value, int precision, const char* suffix = "") {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value << suffix;
  return stream.str();
}

void PrintTableDivider() {
  std::cout << '+' << std::string(kSizeColumnWidth + 2, '-') << '+'
            << std::string(kTimeColumnWidth + 2, '-') << '+'
            << std::string(kTimeColumnWidth + 2, '-') << '+'
            << std::string(kSpeedupColumnWidth + 2, '-') << "+\n";
}

void PrintTableHeader() {
  std::cout << "| " << std::left << std::setw(kSizeColumnWidth) << "size"
            << "| " << std::setw(kTimeColumnWidth) << "kissfft (us/fft)"
            << "| " << std::setw(kTimeColumnWidth) << "kissfftpp (us/fft)"
            << "| " << std::setw(kSpeedupColumnWidth) << "speedup"
            << "|\n";
}

void PrintTableRow(size_t size, double kissMicroseconds,
                   double kissppMicroseconds) {
  std::cout << "| " << std::right << std::setw(kSizeColumnWidth) << size
            << "| " << std::setw(kTimeColumnWidth)
            << FormatValue(kissMicroseconds, 3) << "| "
            << std::setw(kTimeColumnWidth) << FormatValue(kissppMicroseconds, 3)
            << "| " << std::setw(kSpeedupColumnWidth)
            << FormatValue(kissMicroseconds / kissppMicroseconds, 2, "x")
            << "|\n";
}

}  // namespace

int main(int argc, char** argv) {
  const int iterations = ParseIterations(argc, argv);
  if (iterations == 0) {
    return EXIT_FAILURE;
  }

  // These cover small transforms, common audio block sizes, mixed-radix
  // transforms, and larger FFTs. 441 also exercises the generic radix-7
  // butterfly.
  const std::vector<size_t> sizes = {16,  32,   64,   128,  256,
                                     441, 480,  512,  1000, 1024,
                                     1536, 2048, 4096, 8192};
  volatile float checksumSink = 0.0f;
  std::vector<double> kissMicroseconds;
  std::vector<double> kissppMicroseconds;
  kissMicroseconds.reserve(sizes.size());
  kissppMicroseconds.reserve(sizes.size());

  std::cout << "FFT benchmark (" << iterations
            << " measured iterations per implementation)\n";

  for (size_t sizeIndex = 0; sizeIndex < sizes.size(); ++sizeIndex) {
    const size_t size = sizes[sizeIndex];
    std::vector<kiss_fft_cpx> kissInput(size);
    std::vector<kiss_fft_cpx> kissOutput(size);
    std::vector<std::complex<float>> kissppInput(size);
    std::vector<std::complex<float>> kissppOutput(size);

    for (size_t index = 0; index < size; ++index) {
      const float phase = static_cast<float>(index + 1);
      const float real = (phase * 0.25f) - 1.0f;
      const float imag = (phase * phase * 0.001f) - 0.5f;
      kissInput[index] = {real, imag};
      kissppInput[index] = {real, imag};
    }

    kiss_fft_cfg kissPlan = kiss_fft_alloc(static_cast<int>(size), 0,
                                            nullptr, nullptr);
    if (kissPlan == nullptr) {
      std::cerr << "Unable to allocate kissfft plan for size " << size << '\n';
      return EXIT_FAILURE;
    }
    kfft::FFT kissppPlan(size);

    // Warm both implementations before collecting timings.
    BenchmarkKissFFT(kissInput, kissOutput, kissPlan, kWarmupIterations);
    BenchmarkKissFFTpp(kissppInput, kissppOutput, kissppPlan,
                       kWarmupIterations);

    Result kissResult;
    Result kissppResult;
    if (sizeIndex % 2 == 0) {
      kissResult = BenchmarkKissFFT(kissInput, kissOutput, kissPlan,
                                    iterations);
      kissppResult = BenchmarkKissFFTpp(kissppInput, kissppOutput,
                                        kissppPlan, iterations);
    } else {
      kissppResult = BenchmarkKissFFTpp(kissppInput, kissppOutput,
                                        kissppPlan, iterations);
      kissResult = BenchmarkKissFFT(kissInput, kissOutput, kissPlan,
                                    iterations);
    }

    checksumSink += kissResult.checksum + kissppResult.checksum;
    const double kissTimeMicroseconds =
        kissResult.milliseconds * 1000.0 / iterations;
    const double kissppTimeMicroseconds =
        kissppResult.milliseconds * 1000.0 / iterations;
    kissMicroseconds.push_back(kissTimeMicroseconds);
    kissppMicroseconds.push_back(kissppTimeMicroseconds);

    kiss_fft_free(kissPlan);
  }

  PrintTableDivider();
  PrintTableHeader();
  PrintTableDivider();
  for (size_t i = 0; i < sizes.size(); ++i) {
    PrintTableRow(sizes[i], kissMicroseconds[i], kissppMicroseconds[i]);
  }
  PrintTableDivider();

  // Keep the transform results observable without polluting the timed region.
  if (checksumSink == 0.123456f) {
    std::cerr << "checksum: " << checksumSink << '\n';
  }
  return EXIT_SUCCESS;
}
