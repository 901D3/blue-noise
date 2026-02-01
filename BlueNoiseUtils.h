#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <random>

#include <algorithm>
#include <limits>
#include <utility>
#include <queue>
#include <type_traits>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstddef>

#include <fstream>
#include <iomanip>

#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int);
extern "C" __declspec(dllimport) int __stdcall SetConsoleCP(unsigned int);

constexpr unsigned int CP_UTF8 = 65001;
#endif

#if defined(DEBUG) || defined(_DEBUG) || defined(PerfDebugMain) || defined(PerfDebugUtils) || defined(PerfDebugUtils2)
#include <ctime>
#endif

extern std::mt19937 rng;
extern std::uniform_real_distribution<double> randNormalized;

extern double PI;
extern double twoPI;
extern double halfPI;

namespace BlueNoiseUtils
{
  template <typename T>
  void saveArrayAsJSON(
      const std::string &,
      const std::vector<T> &);

  void saveAsBMP24BPPGrayscale(
      const std::string &,
      const std::vector<uint8_t> &,
      uint32_t,
      uint32_t);

  void saveAsBMP24BPP(
      const std::string &,
      const std::vector<uint8_t> &,
      uint32_t,
      uint32_t);

  void saveAsPBMASCIIGrayscale(
      const std::string &,
      const std::vector<uint8_t> &,
      uint32_t,
      uint32_t);

  void saveAsPGMASCIIGrayscale(
      const std::string &,
      const std::vector<uint16_t> &,
      uint32_t,
      uint32_t,
      uint16_t);

  void saveAsPFMGrayscaleFloat32(
      const std::string &filename,
      const std::vector<float> &array,
      uint32_t width,
      uint32_t height,
      float scaleFactor);

  template <typename T>
  void shuffleInPlace(std::vector<T> &);

  template <typename T>
  void randomSamplesInPlace(
      std::vector<T> &,
      uint32_t,
      uint32_t);

  template <
      typename T1,
      typename T2,
      typename T3>
  void convolveWrapAroundInPlace(
      std::vector<T1> &,
      std::vector<T2> &,
      uint32_t,
      uint32_t,
      std::vector<T3> &,
      uint32_t,
      uint32_t);

  template <
      typename T1,
      typename T2>
  void convolveAddWrapAroundInPlace(
      std::vector<T1> &,
      uint32_t,
      uint32_t,
      uint32_t,
      double,
      std::vector<T2> &,
      uint32_t,
      uint32_t);

  template <
      typename T1,
      typename T2,
      typename T3>
  void convolveInPlace(
      std::vector<T1> &inArray,
      std::vector<T2> &blurredArray,
      uint32_t width,
      uint32_t height,
      std::vector<T3> &kernelArray,
      uint32_t kernelWidth,
      uint32_t kernelHeight);

  template <typename T>
  void bilinearAddWrapAroundInPlace(
      std::vector<T> &inArray,
      uint32_t width,
      uint32_t height,
      double idxX,
      double idxY,
      double amount);

  template <typename T>
  T bilinearLookupWrapAroundInPlace(
      std::vector<T> &inArray,
      uint32_t width,
      uint32_t height,
      double idxX,
      double idxY);

  template <
      typename T1,
      typename T2>
  void centeredCosineFourierTransform2DInPlace(
      std::vector<T1> &,
      std::vector<T2> &,
      uint32_t,
      uint32_t);

  template <typename T>
  void radialDomainBuildWrapAroundDiscreteInPlace(
      std::vector<T> &,
      std::vector<T> &,
      uint32_t,
      uint32_t,
      std::vector<T> &,
      uint32_t);

  template <
      typename T1,
      typename T2>
  void differentialDomainBuildWrapAroundScalableInPlace(
      std::vector<T1> &,
      std::vector<T1> &,
      uint32_t,
      uint32_t,
      std::vector<T2> &,
      uint32_t,
      uint32_t,
      uint32_t);

  template <
      typename T1,
      typename T2>
  void differentialDomainAddWrapAroundScalableInPlace(
      std::vector<T1> &,
      std::vector<T1> &,
      uint32_t,
      uint32_t,
      T1,
      T1,
      std::vector<T2> &,
      uint32_t,
      uint32_t,
      uint32_t);

  template <
      typename T1,
      typename T2>
  void differentialDomainBuildBilinearWrapAroundScalableInPlace(
      std::vector<T1> &,
      std::vector<T1> &,
      uint32_t,
      uint32_t,
      std::vector<T2> &,
      uint32_t,
      uint32_t,
      uint32_t);

  template <
      typename T1,
      typename T2>
  void differentialDomainAddBilinearWrapAroundScalableInPlace(
      std::vector<T1> &,
      std::vector<T1> &,
      uint32_t,
      uint32_t,
      T1,
      T1,
      std::vector<T2> &,
      uint32_t,
      uint32_t,
      uint32_t);

  template <
      typename T1,
      typename T2>
  void differentialDomainUpdateBilinearWrapAroundScalableInPlace(
      std::vector<T1> &,
      std::vector<T1> &,
      uint32_t,
      uint32_t,
      T1,
      T1,
      std::vector<T2> &,
      uint32_t,
      uint32_t,
      uint32_t,
      uint32_t);
}

#include "BlueNoiseUtilsSave.inl"
#include "BlueNoiseUtils.inl"