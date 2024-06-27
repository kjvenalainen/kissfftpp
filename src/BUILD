load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")

package(default_visibility = ["//visibility:public"])

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = [
        "*",
    ],
)

# Compile the original kissfft library.
cc_library(
  name = "kissfft",
  hdrs = glob(["kissfft/*.h"]),
  srcs = glob(["kissfft/*.c"]),
  visibility = ["//visibility:public"],
  copts = [
    "-O3",
    "-ffast-math",
  ],
)

# Compile the KissFFT++ header.
cc_library(
  name = "kissfftpp",
  hdrs = ["include/kissfftpp.h"],
  visibility = ["//visibility:public"],
  includes = ["include"],
)

cc_binary(
  name = "main",
  deps = [
    ":kissfftpp",
    ":kissfft",
  ],
  srcs = [
    "main.cpp",
  ],
)

# The main unit test binary.
cc_test(
  name = "gtest",
  features = ["parse_headers"],
  srcs = [
    "testUtils.h",
    "testDataKissfft.h",
    "kissFftTest.cpp",
  ],
  defines = [
    "KFFTPP_NO_CONTRACT_CHECKING=0",
    "KFFTPP_NO_EXCEPTIONS=0",
  ],
  deps = [
    "@gtest",
    "@gtest//:gtest_main",
    ":kissfftpp",
    ":kissfft",
  ],
  copts = [
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Wsign-conversion",
    "-Wsign-compare",
  ]
)
