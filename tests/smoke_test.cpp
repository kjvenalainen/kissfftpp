#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "kissfftpp.h"

namespace {

constexpr float kTolerance = 1e-4f;
constexpr float kPi = 3.14159265358979323846f;

bool NearlyEqual(float actual, float expected) {
  return std::fabs(actual - expected) <= kTolerance;
}

bool NearlyEqual(std::complex<float> actual, std::complex<float> expected) {
  return NearlyEqual(actual.real(), expected.real()) &&
         NearlyEqual(actual.imag(), expected.imag());
}

std::vector<std::complex<float>> NaiveDft(
    const std::vector<std::complex<float>>& input) {
  const size_t length = input.size();
  std::vector<std::complex<float>> output(length);
  for (size_t bin = 0; bin < length; ++bin) {
    for (size_t sample = 0; sample < length; ++sample) {
      const float phase = -2.0f * kPi * static_cast<float>(bin * sample) /
                          static_cast<float>(length);
      output[bin] +=
          input[sample] * std::complex<float>(std::cos(phase), std::sin(phase));
    }
  }
  return output;
}

bool TestComplexRoundTrip() {
  const std::vector<std::complex<float>> input = {
      {1.0f, 0.0f},  {2.0f, -1.0f}, {0.0f, 2.0f},  {3.0f, 0.5f},
      {-2.0f, 1.0f}, {1.0f, 3.0f},  {0.0f, -2.0f}, {4.0f, 1.0f},
  };
  std::vector<std::complex<float>> values = input;
  const auto expected = NaiveDft(input);

  kfft::FFT plan(values.size());
  plan.fft<kfft::NoScaling>(values, values);
  for (size_t i = 0; i < values.size(); ++i) {
    if (!NearlyEqual(values[i], expected[i])) {
      return false;
    }
  }

  plan.ifft(values, values);
  for (size_t i = 0; i < values.size(); ++i) {
    if (!NearlyEqual(values[i], input[i])) {
      return false;
    }
  }
  return true;
}

bool TestRealRoundTrip() {
  const std::vector<float> input = {1.0f, -2.0f, 3.0f, 0.5f,
                                    4.0f, -1.0f, 2.0f, 0.0f};
  std::vector<std::complex<float>> spectrum(input.size() / 2 + 1);
  std::vector<float> output(input.size());

  kfft::RealFFT plan(input.size());
  plan.fft(input, spectrum);
  plan.ifft(spectrum, output);
  for (size_t i = 0; i < input.size(); ++i) {
    if (!NearlyEqual(output[i], input[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!TestComplexRoundTrip() || !TestRealRoundTrip()) {
    std::cerr << "kissfftpp smoke test failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
