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
)

cc_binary(
  name = "main",
  deps = [
    "//src/include:kissfftpp",
    ":kissfft",
  ],
  srcs = [
    "main.cpp",
  ],
)

# The main unit test binary.
cc_test(
  name = "gtest",
  srcs = [
    "testUtils.h",
    "testDataKissfft.h",
    "kissFftTest.cpp",
  ],
  deps = [
    "@gtest",
    "@gtest//:gtest_main",
    "//src/include:kissfftpp",
    ":kissfft",
  ],
)
