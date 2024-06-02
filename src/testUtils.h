// Copyright (c) 2024 Kevin Venalainen
//
// This file is part of KissFFT++.
//

#pragma once

#include <cstddef>
#include <initializer_list>
#include <utility>

#include "gtest/gtest.h"

namespace helper {

// Calls the given function with the tuple elements as arguments.
template <typename Function, typename Tuple, size_t... I>
auto Call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

} // namespace helper

// Invokes test and expected functions with each tuple in `values` and calls
// EXPECT_EQ on each result.
template <typename FuncType0, typename FuncType1, typename... Args>
constexpr void ExpectEq(FuncType0 testFunc, FuncType1 expectedFunc,
                        std::initializer_list<std::tuple<Args...>> values) {
  constexpr auto size = std::tuple_size<std::tuple<Args...>>::value;
  for (const std::tuple<Args...> &args : values) {
    EXPECT_EQ(
        helper::Call(testFunc, args, std::make_index_sequence<size>()),
        helper::Call(expectedFunc, args, std::make_index_sequence<size>()))
        << "args[0]: " << std::get<0>(args);
  }
}
