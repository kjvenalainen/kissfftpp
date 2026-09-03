cmake_minimum_required(VERSION 3.16)

project(kissfftpp VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(KISSFFTPP_BUILD_TESTING "Build the self-contained smoke test" ON)

add_library(kissfftpp INTERFACE)
add_library(kissfftpp::kissfftpp ALIAS kissfftpp)

target_compile_features(kissfftpp INTERFACE cxx_std_14)
target_include_directories(kissfftpp
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(KISSFFTPP_BUILD_TESTING)
  enable_testing()
  add_executable(kissfftpp_smoke tests/smoke_test.cpp)
  target_link_libraries(kissfftpp_smoke PRIVATE kissfftpp::kissfftpp)
  add_test(NAME kissfftpp.smoke COMMAND kissfftpp_smoke)
endif()

install(TARGETS kissfftpp EXPORT kissfftppTargets)
install(DIRECTORY src/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES README.md LICENSE CONTRIBUTING.md SECURITY.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(EXPORT kissfftppTargets
  FILE kissfftppTargets.cmake
  NAMESPACE kissfftpp::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kissfftpp)

configure_package_config_file(
  cmake/kissfftppConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/kissfftppConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kissfftpp)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/kissfftppConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/kissfftppConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/kissfftppConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kissfftpp)
