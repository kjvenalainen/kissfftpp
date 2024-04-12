load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "main",
  deps = [
    "//src/include:kissfftpp",
  ],
  srcs = [
    "main.cpp",
  ],
)

# The main unit test binary.
cc_test(
  name = "gtest",
  srcs = [
    "kissFftTest.cpp",
  ],
  deps = [
    "@gtest",
    "@gtest//:gtest_main",
    "//src/include:kissfftpp",
  ],
)
