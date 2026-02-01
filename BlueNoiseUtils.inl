#pragma once

#define NOMINMAX
namespace BlueNoiseUtils
{
  template <typename T>
  void shuffleInPlace(std::vector<T> &array)
  {
    std::shuffle(array.begin(), array.end(), rng);
  }

  template <typename T>
  void randomSamplesInPlace(
      std::vector<T> &array,
      uint32_t range,
      uint32_t length)
  {
    for (uint32_t i = 0; i < length; i++)
    {
      array[i] = randNormalized(rng) * range;
    }
  }

  template <
      typename T1,
      typename T2,
      typename T3>
  void convolveWrapAroundInPlace(
      std::vector<T1> &inArray,
      std::vector<T2> &blurredArray,
      uint32_t width,
      uint32_t height,
      std::vector<T3> &kernelArray,
      uint32_t kernelWidth,
      uint32_t kernelHeight)
  {

#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    int32_t halfKernelWidth = kernelWidth / 2;
    int32_t halfKernelHeight = kernelHeight / 2;

    for (uint32_t idxY = 0; idxY < height; idxY++)
    {
      uint32_t idxYOffs = idxY * width;

      int32_t baseConvolveIdxY = idxY - halfKernelHeight;
      if (baseConvolveIdxY < 0)
      {
        baseConvolveIdxY = (baseConvolveIdxY + height) % height;
      }

      for (uint32_t idxX = 0; idxX < width; idxX++)
      {
        int32_t baseConvolveIdxX = idxX - halfKernelWidth;
        if (baseConvolveIdxX < 0)
        {
          baseConvolveIdxX = (baseConvolveIdxX + width) % width;
        }

        uint32_t convolveIdxY = baseConvolveIdxY;
        double total = 0;

        for (uint32_t kernelIdxY = 0; kernelIdxY < kernelHeight; kernelIdxY++)
        {
          uint32_t convolveIdxYOffs = convolveIdxY * width;
          uint32_t kernelIdxYOffs = kernelIdxY * kernelWidth;

          uint32_t convolveIdxX = baseConvolveIdxX;

          for (uint32_t kernelIdxX = 0; kernelIdxX < kernelWidth; kernelIdxX++)
          {
            total +=
                inArray[convolveIdxYOffs + convolveIdxX] *
                kernelArray[kernelIdxYOffs + kernelIdxX];

            if (++convolveIdxX == width)
            {
              convolveIdxX = 0;
            }
          }

          if (++convolveIdxY == height)
          {
            convolveIdxY = 0;
          }
        }

        blurredArray[idxYOffs + idxX] = total;
      }
    }

#ifdef PerfDebugUtils
    std::cout
        << "convolveWrapAroundInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <
      typename T1,
      typename T2>
  void convolveAddWrapAroundInPlace(
      std::vector<T1> &blurredArray,
      uint32_t width,
      uint32_t height,
      uint32_t idx,
      double amount,
      std::vector<T2> &kernelArray,
      uint32_t kernelWidth,
      uint32_t kernelHeight)
  {
#ifdef PerfDebugUtils2
    clock_t t0 = clock();
#endif

    int32_t convolveIdxY =
        static_cast<uint32_t>(std::floor(idx / width)) -
        static_cast<uint32_t>(std::floor(kernelHeight / 2));

    int32_t baseConvolveIdxX =
        std::fmod(idx, width) -
        static_cast<uint32_t>(std::floor(kernelWidth / 2));

    if (convolveIdxY < 0)
    {
      convolveIdxY = std::fmod(convolveIdxY, height);
      convolveIdxY += height;
    }

    if (baseConvolveIdxX < 0)
    {
      baseConvolveIdxX = std::fmod(baseConvolveIdxX, width);
      baseConvolveIdxX += width;
    }

    for (uint32_t kernelIdxY = 0; kernelIdxY < kernelHeight; kernelIdxY++)
    {
      uint32_t convolveIdxYOffs = convolveIdxY * width;
      uint32_t kernelIdxYOffs = kernelIdxY * kernelWidth;

      uint32_t convolveIdxX = baseConvolveIdxX;

      for (uint32_t kernelIdxX = 0; kernelIdxX < kernelWidth; kernelIdxX++)
      {
        blurredArray[convolveIdxYOffs + convolveIdxX] +=
            kernelArray[kernelIdxYOffs + kernelIdxX] * amount;

        if (++convolveIdxX == width)
        {
          convolveIdxX = 0;
        }
      }

      if (++convolveIdxY == height)
      {
        convolveIdxY = 0;
      }
    }

#ifdef PerfDebugUtils2
    std::cout
        << "convolveAddWrapAroundInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

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
      uint32_t kernelHeight)
  {

#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    int32_t halfKernelWidth = kernelWidth / 2;
    int32_t halfKernelHeight = kernelHeight / 2;

    uint32_t moreHeight = height + halfKernelHeight;
    uint32_t moreWidth = width + halfKernelWidth;

    for (uint32_t idxY = 0; idxY < height; idxY++)
    {
      uint32_t idxYOffs = idxY * width;
      int32_t kernelCenterConvolveIdxY = idxY - halfKernelHeight;

      uint32_t kernelStartIdxY =
          std::max(0, halfKernelHeight - static_cast<int32_t>(idxY));
      uint32_t kernelEndIdxY =
          std::min(kernelHeight, moreHeight - static_cast<int32_t>(idxY));

      for (uint32_t idxX = 0; idxX < width; idxX++)
      {
        int32_t kernelCenterConvolveIdxX = idxX - halfKernelWidth;
        double total = 0;

        uint32_t kernelStartIdxX =
            std::max(0, halfKernelWidth - static_cast<int32_t>(idxX));
        uint32_t kernelEndIdxX =
            std::min(kernelWidth, moreWidth - static_cast<int32_t>(idxX));

        for (uint32_t kernelIdxY = kernelStartIdxY; kernelIdxY < kernelEndIdxY; kernelIdxY++)
        {
          uint32_t convolveIdxYOffs = (kernelCenterConvolveIdxY + kernelIdxY) * width;
          uint32_t kernelIdxYOffs = kernelIdxY * kernelWidth;

          for (uint32_t kernelIdxX = kernelStartIdxX; kernelIdxX < kernelEndIdxX; kernelIdxX++)
          {
            total +=
                inArray[convolveIdxYOffs +
                        kernelCenterConvolveIdxX + kernelIdxX] *
                kernelArray[kernelIdxYOffs + kernelIdxX];
          }
        }

        blurredArray[idxYOffs + idxX] = total;
      }
    }

#ifdef PerfDebugUtils
    std::cout
        << "convolveInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <typename T>
  void bilinearAddWrapAroundInPlace(
      std::vector<T> &inArray,
      uint32_t width,
      uint32_t height,
      double idxX,
      double idxY,
      double amount)
  {
#ifdef PerfDebugUtils2
    clock_t t0 = clock();
#endif

    uint32_t floorIdxX = static_cast<uint32_t>(idxX);
    uint32_t floorIdxY = static_cast<uint32_t>(idxY);

    uint32_t nextFloorIdxX = floorIdxX + 1;
    uint32_t nextFloorIdxY = floorIdxY + 1;

    if (nextFloorIdxX == width)
    {
      nextFloorIdxX = 0;
    }
    if (nextFloorIdxY == height)
    {
      nextFloorIdxY = 0;
    }

    uint32_t yOffs0 = floorIdxY * width;
    uint32_t yOffs1 = nextFloorIdxY * width;

    double fractionalX = idxX - floorIdxX;
    double fractionalY = idxY - floorIdxY;

    double invFractionalY = 1 - fractionalY;

    double leftAmount = amount * (1 - fractionalX);
    double rightAmount = amount * fractionalX;

    T *data = inArray.data();

    data[yOffs0 + floorIdxX] += leftAmount * invFractionalY;
    data[yOffs0 + nextFloorIdxX] += rightAmount * invFractionalY;
    data[yOffs1 + floorIdxX] += leftAmount * fractionalY;
    data[yOffs1 + nextFloorIdxX] += rightAmount * fractionalY;

#ifdef PerfDebugUtils2
    std::cout
        << "bilinearAddWrapAroundInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <typename T>
  T bilinearLookupWrapAroundInPlace(
      std::vector<T> &inArray,
      uint32_t width,
      uint32_t height,
      double idxX,
      double idxY)
  {
#ifdef PerfDebugUtils2
    clock_t t0 = clock();
#endif

    uint32_t floorIdxX = static_cast<uint32_t>(idxX);
    uint32_t floorIdxY = static_cast<uint32_t>(idxY);

    uint32_t nextFloorIdxX = floorIdxX + 1;
    uint32_t nextFloorIdxY = floorIdxY + 1;

    if (nextFloorIdxX == width)
    {
      nextFloorIdxX = 0;
    }
    if (nextFloorIdxY == height)
    {
      nextFloorIdxY = 0;
    }

    uint32_t yOffs0 = floorIdxY * width;
    uint32_t yOffs1 = nextFloorIdxY * width;

    double fractionalX = idxX - floorIdxX;
    double fractionalY = idxY - floorIdxY;

    double invFractionalX = 1 - fractionalX;

    T *data = inArray.data();

    T top =
        data[yOffs0 + floorIdxX] * invFractionalX +
        data[yOffs0 + nextFloorIdxX] * fractionalX;

    T bottom =
        data[yOffs1 + floorIdxX] * invFractionalX +
        data[yOffs1 + nextFloorIdxX] * fractionalX;

#ifdef PerfDebugUtils2
    std::cout
        << "bilinearLookupWrapAroundInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif

    return (bottom - top) * fractionalY + top;
  }

  template <
      typename T1,
      typename T2>
  void centeredCosineFourierTransform2DInPlace(
      std::vector<T1> &powerDomain,
      std::vector<T2> &correlationDomain,
      uint32_t width,
      uint32_t height)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;

    for (uint32_t idxY0 = 0; idxY0 < height; idxY0++)
    {
      uint32_t y0Offs = idxY0 * width;
      double y0Shifted = (idxY0 - halfHeight) / height;

      for (uint32_t idxX0 = 0; idxX0 < width; idxX0++)
      {
        double sum = 0;
        double x0Shifted = (idxX0 - halfWidth) / width;

        for (uint32_t idxY1 = 0; idxY1 < height; idxY1++)
        {
          uint32_t y1Offs = idxY1 * width;
          double baseAngleY = static_cast<double>(idxY1 * idxY0) / height;

          double y1Shifted = idxY1 - halfHeight;

          for (uint32_t idxX1 = 0; idxX1 < width; idxX1++)
          {
            sum += powerDomain[y1Offs + idxX1] *
                   std::cos(
                       twoPI *
                       (x0Shifted * (idxX1 - halfWidth) + y0Shifted * y1Shifted));
          }
        }

        correlationDomain[y0Offs + idxX0] = sum;
      }
    }

#ifdef PerfDebugUtils
    std::cout
        << "centeredCosineFourierTransform2DInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <typename T>
  void radialDomainBuildWrapAroundDiscreteInPlace(
      std::vector<T> &sampleIdxXArray,
      std::vector<T> &sampleIdxYArray,
      uint32_t width,
      uint32_t height,
      std::vector<T> &radialDomain,
      uint32_t samples)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;

    double negHalfWidth = -halfWidth;
    double negHalfHeight = -halfHeight;

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      double sampleIdxX = sampleIdxXArray[sample];
      double sampleIdxY = sampleIdxYArray[sample];

      for (uint32_t candidate = 0; candidate < samples; candidate++)
      {
        if (sample == candidate)
        {
          continue;
        }

        double distanceX = sampleIdxX - sampleIdxXArray[candidate];
        double distanceY = sampleIdxY - sampleIdxYArray[candidate];

        if (distanceX > halfWidth)
        {
          distanceX -= width;
        }
        else if (distanceX < negHalfWidth)
        {
          distanceX += width;
        }

        if (distanceY > halfHeight)
        {
          distanceY -= height;
        }
        else if (distanceY < negHalfHeight)
        {
          distanceY += height;
        }

        radialDomain[std::abs(distanceX) + std::abs(distanceY)]++;
      }
    }

#ifdef PerfDebugUtils
    std::cout
        << "radialDomainBuildWrapAroundDiscreteInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <
      typename T1,
      typename T2>
  void differentialDomainBuildWrapAroundScalableInPlace(
      std::vector<T1> &sampleIdxXArray,
      std::vector<T1> &sampleIdxYArray,
      uint32_t width,
      uint32_t height,
      std::vector<T2> &differentialDomain,
      uint32_t differentialDomainWidth,
      uint32_t differentialDomainHeight,
      uint32_t samples)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;

    double negHalfWidth = -halfWidth;
    double negHalfHeight = -halfHeight;

    double rescaledWidth = static_cast<double>(differentialDomainWidth) / width;
    double rescaledHeight = static_cast<double>(differentialDomainHeight) / height;

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      double sampleIdxX = sampleIdxXArray[sample];
      double sampleIdxY = sampleIdxYArray[sample];

      for (uint32_t candidate = sample + 1; candidate < samples; candidate++)
      {
        double distanceX = sampleIdxX - sampleIdxXArray[candidate];
        double distanceY = sampleIdxY - sampleIdxYArray[candidate];

        if (distanceX > halfWidth)
        {
          distanceX -= width;
        }
        else if (distanceX < negHalfWidth)
        {
          distanceX += width;
        }

        if (distanceY > halfHeight)
        {
          distanceY -= height;
        }
        else if (distanceY < negHalfHeight)
        {
          distanceY += height;
        }

        differentialDomain[static_cast<uint32_t>(std::floor((distanceY + halfHeight) * rescaledHeight)) *
                               differentialDomainWidth +
                           static_cast<uint32_t>(std::floor((distanceX + halfWidth) * rescaledWidth))]++;

        differentialDomain[static_cast<uint32_t>(std::floor((-distanceY + halfHeight) * rescaledHeight)) *
                               differentialDomainWidth +
                           static_cast<uint32_t>(std::floor((-distanceX + halfWidth) * rescaledWidth))]++;
      }
    }

#ifdef PerfDebugUtils
    std::cout
        << "differentialDomainBuildWrapAroundScalableInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <
      typename T1,
      typename T2>
  void differentialDomainAddWrapAroundScalableInPlace(
      std::vector<T1> &sampleIdxXArray,
      std::vector<T1> &sampleIdxYArray,
      uint32_t width,
      uint32_t height,
      T1 idxX,
      T1 idxY,
      std::vector<T2> &differentialDomain,
      uint32_t differentialDomainWidth,
      uint32_t differentialDomainHeight,
      uint32_t samples)
  {
#ifdef PerfDebugUtils2
    clock_t t0 = clock();
#endif

    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;

    double negHalfWidth = -halfWidth;
    double negHalfHeight = -halfHeight;

    double rescaledWidth = static_cast<double>(differentialDomainWidth) / width;
    double rescaledHeight = static_cast<double>(differentialDomainHeight) / height;

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      double distanceX = sampleIdxXArray[sample] - idxX;
      double distanceY = sampleIdxYArray[sample] - idxY;

      if (distanceX > halfWidth)
      {
        distanceX -= width;
      }
      else if (distanceX < negHalfWidth)
      {
        distanceX += width;
      }

      if (distanceY > halfHeight)
      {
        distanceY -= height;
      }
      else if (distanceY < negHalfHeight)
      {
        distanceY += height;
      }

      differentialDomain[static_cast<uint32_t>(std::floor((distanceY + halfHeight) * rescaledHeight)) *
                             differentialDomainWidth +
                         static_cast<uint32_t>(std::floor((distanceX + halfWidth) * rescaledWidth))]++;

      differentialDomain[static_cast<uint32_t>(std::floor((-distanceY + halfHeight) * rescaledHeight)) *
                             differentialDomainWidth +
                         static_cast<uint32_t>(std::floor((-distanceX + halfWidth) * rescaledWidth))]++;
    }

#ifdef PerfDebugUtils2
    std::cout
        << "differentialDomainAddWrapAroundScalableInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <
      typename T1,
      typename T2>
  void differentialDomainUpdateWrapAroundScalableInPlace(
      std::vector<T1> &sampleIdxXArray,
      std::vector<T1> &sampleIdxYArray,
      uint32_t width,
      uint32_t height,
      T1 newIdxX,
      T1 newIdxY,
      std::vector<T2> &differentialDomain,
      uint32_t differentialDomainWidth,
      uint32_t differentialDomainHeight,
      uint32_t idxOfOldSample,
      uint32_t samples)
  {
#ifdef PerfDebugUtils2
    clock_t t0 = clock();
#endif

    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;

    double negHalfWidth = -halfWidth;
    double negHalfHeight = -halfHeight;

    double rescaledWidth = static_cast<double>(differentialDomainWidth) / width;
    double rescaledHeight = static_cast<double>(differentialDomainHeight) / height;

    T1 oldSampleX = sampleIdxXArray[idxOfOldSample];
    T1 oldSampleY = sampleIdxYArray[idxOfOldSample];

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      if (sample == idxOfOldSample)
      {
        continue;
      }

      double sampleIdxX = sampleIdxXArray[sample];
      double sampleIdxY = sampleIdxYArray[sample];

      double distanceOldX = sampleIdxX - oldSampleX;
      double distanceOldY = sampleIdxY - oldSampleY;

      if (distanceOldX > halfWidth)
      {
        distanceOldX -= width;
      }
      else if (distanceOldX < negHalfWidth)
      {
        distanceOldX += width;
      }

      if (distanceOldY > halfHeight)
      {
        distanceOldY -= height;
      }
      else if (distanceOldY < negHalfHeight)
      {
        distanceOldY += height;
      }

      differentialDomain[static_cast<uint32_t>(std::floor((distanceOldY + halfHeight) * rescaledHeight)) *
                             differentialDomainWidth +
                         static_cast<uint32_t>(std::floor((distanceOldX + halfWidth) * rescaledWidth))]--;

      differentialDomain[static_cast<uint32_t>(std::floor((-distanceOldY + halfHeight) * rescaledHeight)) *
                             differentialDomainWidth +
                         static_cast<uint32_t>(std::floor((-distanceOldX + halfWidth) * rescaledWidth))]--;

      double distanceNewX = sampleIdxX - newIdxX;
      double distanceNewY = sampleIdxY - newIdxY;

      if (distanceNewX > halfWidth)
      {
        distanceNewX -= width;
      }
      else if (distanceNewX < negHalfWidth)
      {
        distanceNewX += width;
      }

      if (distanceNewY > halfHeight)
      {
        distanceNewY -= height;
      }
      else if (distanceNewY < negHalfHeight)
      {
        distanceNewY += height;
      }

      differentialDomain[static_cast<uint32_t>(std::floor((distanceNewY + halfHeight) * rescaledHeight)) *
                             differentialDomainWidth +
                         static_cast<uint32_t>(std::floor((distanceNewX + halfWidth) * rescaledWidth))]++;

      differentialDomain[static_cast<uint32_t>(std::floor((-distanceNewY + halfHeight) * rescaledHeight)) *
                             differentialDomainWidth +
                         static_cast<uint32_t>(std::floor((-distanceNewX + halfWidth) * rescaledWidth))]++;
    }

#ifdef PerfDebugUtils2
    std::cout
        << "differentialDomainUpdateWrapAroundScalableInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <
      typename T1,
      typename T2>
  void differentialDomainBuildBilinearWrapAroundScalableInPlace(
      std::vector<T1> &sampleIdxXArray,
      std::vector<T1> &sampleIdxYArray,
      uint32_t width,
      uint32_t height,
      std::vector<T2> &differentialDomain,
      uint32_t differentialDomainWidth,
      uint32_t differentialDomainHeight,
      uint32_t samples)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;

    double negHalfWidth = -halfWidth;
    double negHalfHeight = -halfHeight;

    double rescaledWidth = static_cast<double>(differentialDomainWidth) / width;
    double rescaledHeight = static_cast<double>(differentialDomainHeight) / height;

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      double sampleIdxX = sampleIdxXArray[sample];
      double sampleIdxY = sampleIdxYArray[sample];

      for (uint32_t candidate = sample + 1; candidate < samples; candidate++)
      {
        double distanceX = sampleIdxX - sampleIdxXArray[candidate];
        double distanceY = sampleIdxY - sampleIdxYArray[candidate];

        if (distanceX > halfWidth)
        {
          distanceX -= width;
        }
        else if (distanceX < negHalfWidth)
        {
          distanceX += width;
        }

        if (distanceY > halfHeight)
        {
          distanceY -= height;
        }
        else if (distanceY < negHalfHeight)
        {
          distanceY += height;
        }

        bilinearAddWrapAroundInPlace(
            differentialDomain,
            differentialDomainWidth,
            differentialDomainHeight,
            (distanceX + halfWidth) * rescaledWidth,
            (distanceY + halfHeight) * rescaledHeight,
            1);

        bilinearAddWrapAroundInPlace(
            differentialDomain,
            differentialDomainWidth,
            differentialDomainHeight,
            (-distanceX + halfWidth) * rescaledWidth,
            (-distanceY + halfHeight) * rescaledHeight,
            1);
      }
    }

#ifdef PerfDebugUtils
    std::cout
        << "differentialDomainBuildWrapAroundScalableInPlace: "
        << (clock() - t0)
        << "ms"
        << std::endl;
#endif
  }

  template <
      typename T1,
      typename T2>
  void differentialDomainAddBilinearWrapAroundScalableInPlace(
      std::vector<T1> &sampleIdxXArray,
      std::vector<T1> &sampleIdxYArray,
      uint32_t width,
      uint32_t height,
      T1 idxX,
      T1 idxY,
      std::vector<T2> &differentialDomain,
      uint32_t differentialDomainWidth,
      uint32_t differentialDomainHeight,
      uint32_t samples)
  {
#ifdef PerfDebugUtils2
    clock_t t0 = clock();
#endif

    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;

    double negHalfWidth = -halfWidth;
    double negHalfHeight = -halfHeight;

    double rescaledWidth = static_cast<double>(differentialDomainWidth) / width;
    double rescaledHeight = static_cast<double>(differentialDomainHeight) / height;

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      double distanceX = sampleIdxXArray[sample] - idxX;
      double distanceY = sampleIdxYArray[sample] - idxY;

      if (distanceX > halfWidth)
      {
        distanceX -= width;
      }
      else if (distanceX < negHalfWidth)
      {
        distanceX += width;
      }

      if (distanceY > halfHeight)
      {
        distanceY -= height;
      }
      else if (distanceY < negHalfHeight)
      {
        distanceY += height;
      }

      bilinearAddWrapAroundInPlace(
          differentialDomain,
          differentialDomainWidth,
          differentialDomainHeight,
          (distanceX + halfWidth) * rescaledWidth,
          (distanceY + halfHeight) * rescaledHeight,
          1);

      bilinearAddWrapAroundInPlace(
          differentialDomain,
          differentialDomainWidth,
          differentialDomainHeight,
          (-distanceX + halfWidth) * rescaledWidth,
          (-distanceY + halfHeight) * rescaledHeight,
          1);
    }

#ifdef PerfDebugUtils2
    std::cout
        << "differentialDomainAddConvolveWrapAroundScalableInPlace: "
        << (clock() - t0)
        << "ms" << std::endl;
#endif
  }

  template <
      typename T1,
      typename T2>
  void differentialDomainUpdateBilinearWrapAroundScalableInPlace(
      std::vector<T1> &sampleIdxXArray,
      std::vector<T1> &sampleIdxYArray,
      uint32_t width,
      uint32_t height,
      T1 newIdxX,
      T1 newIdxY,
      std::vector<T2> &differentialDomain,
      uint32_t differentialDomainWidth,
      uint32_t differentialDomainHeight,
      uint32_t idxOfOldSample,
      uint32_t samples)
  {
#ifdef PerfDebugUtils2
    clock_t t0 = clock();
#endif

    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;

    double negHalfWidth = -halfWidth;
    double negHalfHeight = -halfHeight;

    double rescaledWidth = static_cast<double>(differentialDomainWidth) / width;
    double rescaledHeight = static_cast<double>(differentialDomainHeight) / height;

    T1 oldSampleX = sampleIdxXArray[idxOfOldSample];
    T1 oldSampleY = sampleIdxYArray[idxOfOldSample];

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      if (sample == idxOfOldSample)
      {
        continue;
      }

      double sampleIdxX = sampleIdxXArray[sample];
      double sampleIdxY = sampleIdxYArray[sample];

      double distanceOldX = sampleIdxX - oldSampleX;
      double distanceOldY = sampleIdxY - oldSampleY;

      if (distanceOldX > halfWidth)
      {
        distanceOldX -= width;
      }
      else if (distanceOldX < negHalfWidth)
      {
        distanceOldX += width;
      }

      if (distanceOldY > halfHeight)
      {
        distanceOldY -= height;
      }
      else if (distanceOldY < negHalfHeight)
      {
        distanceOldY += height;
      }

      bilinearAddWrapAroundInPlace(
          differentialDomain,
          differentialDomainWidth,
          differentialDomainHeight,
          (distanceOldX + halfWidth) * rescaledWidth,
          (distanceOldY + halfHeight) * rescaledHeight,
          -1);

      bilinearAddWrapAroundInPlace(
          differentialDomain,
          differentialDomainWidth,
          differentialDomainHeight,
          (-distanceOldX + halfWidth) * rescaledWidth,
          (-distanceOldY + halfHeight) * rescaledHeight,
          -1);

      double distanceNewX = sampleIdxX - newIdxX;
      double distanceNewY = sampleIdxY - newIdxY;

      if (distanceNewX > halfWidth)
      {
        distanceNewX -= width;
      }
      else if (distanceNewX < negHalfWidth)
      {
        distanceNewX += width;
      }

      if (distanceNewY > halfHeight)
      {
        distanceNewY -= height;
      }
      else if (distanceNewY < negHalfHeight)
      {
        distanceNewY += height;
      }

      bilinearAddWrapAroundInPlace(
          differentialDomain,
          differentialDomainWidth,
          differentialDomainHeight,
          (distanceNewX + halfWidth) * rescaledWidth,
          (distanceNewY + halfHeight) * rescaledHeight,
          1);

      bilinearAddWrapAroundInPlace(
          differentialDomain,
          differentialDomainWidth,
          differentialDomainHeight,
          (-distanceNewX + halfWidth) * rescaledWidth,
          (-distanceNewY + halfHeight) * rescaledHeight,
          1);
    }

#ifdef PerfDebugUtils2
    std::cout << "differentialDomainUpdateConvolveWrapAroundScalableInPlace: "
              << (clock() - t0)
              << "ms"
              << std::endl;
#endif
  }
}
