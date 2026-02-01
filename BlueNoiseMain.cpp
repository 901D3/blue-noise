// GPLv3

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

#include "BlueNoiseUtils.h"

#if defined(DEBUG) || defined(_DEBUG) || defined(PerfDebugMain) || defined(PerfDebugUtils) || defined(PerfDebugUtils2)
#include <ctime>
#endif

double PI = std::acos(0.) * 2.;
double twoPI = PI * 2.;
double halfPI = PI / 2.;

std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<double> randNormalized(0., 1.);

uint32_t width = 64;
uint32_t height = 64;
uint32_t samples = 100;
// iteration lớn hơn sẽ lãng phí
uint32_t iterations = 15;

uint32_t algo = 0;

double sigma = 2;
double kernelSizeMultiplier = 3;
bool useAdaptiveSigma = false;
double adaptiveSigmaScale = 0.5;
bool useCosineGaussian = false;
double stepScale = 0.01;

uint32_t DDAWidth = 64;
uint32_t DDAHeight = 64;

std::string DDATarget = "";

void loadPFMGrayscaleInPlace(
    const std::string &filename,
    std::vector<float> &array)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    std::cerr << "Không thể mở tệp để tải: " << filename << "\n";
    std::exit(0);
  }

  std::string token;

  std::getline(file, token);
  if (token != "Pf")
  {
    std::cerr << "Không phải PFM grayscale (Pf)\n";
    std::exit(0);
  }

  std::getline(file, token, ' ');
  uint32_t width = static_cast<uint32_t>(std::stoul(token));

  std::getline(file, token);
  uint32_t height = static_cast<uint32_t>(std::stoul(token));

  std::getline(file, token);
  float scale = static_cast<float>(std::stof(token));

  file.read(reinterpret_cast<char *>(array.data()),
            width * height * sizeof(float));
}

template <typename T>
void saveAsBMP24BPPWrapper(
    const std::string &filename,
    const std::vector<T> &array,
    uint32_t width,
    uint32_t height)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
#endif

  uint32_t squaredSize = width * height;

  std::vector<uint8_t> image(squaredSize);

  double highestValue = -std::numeric_limits<T>::max();
  double lowestValue = std::numeric_limits<T>::max();

  for (uint32_t i = 0; i < squaredSize; i++)
  {
    T value = array[i];

    if (value > highestValue)
    {
      highestValue = value;
    }

    if (value < lowestValue)
    {
      lowestValue = value;
    }
  }

  double denom = 255. / (highestValue - lowestValue);

  for (uint32_t i = 0; i < squaredSize; i++)
  {
    image[i] = static_cast<uint8_t>((array[i] - lowestValue) * denom);
  }

  BlueNoiseUtils::saveAsBMP24BPPGrayscale(filename, image, width, height);

#ifdef PerfDebugMain
  std::cout
      << "saveAsBMP24BPPWrapper: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

void samplesSaveAsBMP24BPPWrapper(
    const std::string &filename,
    std::vector<double> &sampleIdxXArray,
    std::vector<double> &sampleIdxYArray,
    uint32_t width,
    uint32_t height)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
#endif

  uint32_t squaredSize = width * height;

  std::vector<uint8_t> image(squaredSize);

  for (uint32_t i = 0; i < squaredSize; i++)
  {
    image[static_cast<uint32_t>(std::floor(sampleIdxYArray[i])) * width +
          static_cast<uint32_t>(std::floor(sampleIdxXArray[i]))] = 255;
  }

  BlueNoiseUtils::saveAsBMP24BPPGrayscale(filename, image, width, height);

#ifdef PerfDebugMain
  std::cout
      << "saveAsBMP24BPPWrapper: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

void saveAsPGMASCIIGrayscaleWrapper(
    const std::string &filename,
    const std::vector<uint16_t> &array,
    uint32_t width,
    uint32_t height)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
#endif

  uint32_t squaredSize = width * height;

  std::vector<uint16_t> image(squaredSize);

  double highestValue = -std::numeric_limits<uint16_t>::max();

  for (uint32_t i = 0; i < squaredSize; i++)
  {
    uint16_t value = array[i];

    if (value > highestValue)
    {
      highestValue = value;
    }
  }

  BlueNoiseUtils::saveAsPGMASCIIGrayscale(filename, array, width, height, highestValue);

#ifdef PerfDebugMain
  std::cout
      << "saveAsPBMASCIIGrayscaleWrapper: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

// tạo Gaussian kernel, mutate array tại chỗ
template <typename T>
void gaussianKernelGenerateInPlace(
    std::vector<T> &kernelArray,
    double sigma,
    uint32_t radius)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
#endif

  int32_t signedRadius = static_cast<int32_t>(radius);
  uint32_t kernelSize = 2 * radius + 1;
  uint32_t squaredSize = kernelSize * kernelSize;

  double invTwoSigma = 1 / (2 * sigma * sigma);
  double sum = 0;

  for (int32_t idxY = -signedRadius; idxY <= signedRadius; idxY++)
  {
    // cache 1 số thứ
    int32_t twoY = idxY * idxY;
    int32_t idxYOffs = (idxY + signedRadius) * kernelSize;

    for (int32_t idxX = -signedRadius; idxX <= signedRadius; idxX++)
    {
      T value = std::exp(-(idxX * idxX + twoY) * invTwoSigma);

      kernelArray[idxYOffs + (idxX + signedRadius)] = value;
      sum += value;
    }
  }

  for (int32_t i = 0; i < squaredSize; i++)
  {
    kernelArray[i] /= sum;
  }

#ifdef PerfDebugMain
  std::cout
      << "gaussianKernelGenerateInPlace: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

template <typename T>
void cosineGaussianKernelGenerateInPlace(
    std::vector<T> &kernelArray,
    double sigma,
    uint32_t radius)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
#endif

  int32_t signedRadius = static_cast<int32_t>(radius);
  uint32_t kernelSize = 2 * radius + 1;
  uint32_t squaredSize = kernelSize * kernelSize;

  double invTwoSigma = 1 / (2 * sigma * sigma);
  double sum = 0;

  kernelArray.resize(squaredSize);

  for (int32_t idxY = -signedRadius; idxY <= signedRadius; idxY++)
  {
    // cache 1 số thứ
    int32_t twoY = idxY * idxY;
    int32_t idxYOffs = (idxY + signedRadius) * kernelSize;

    for (int32_t idxX = -signedRadius; idxX <= signedRadius; idxX++)
    {
      T value = std::sin((PI / 2) * std::exp(-(idxX * idxX + twoY) * invTwoSigma));

      kernelArray[idxYOffs + (idxX + signedRadius)] = value;
      sum += value;
    }
  }

  for (int32_t i = 0; i < squaredSize; i++)
  {
    kernelArray[i] /= sum;
  }

#ifdef PerfDebugMain
  std::cout
      << "cosineGaussianKernelGenerateInPlace: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

template <typename T>
void gaussianKernelDerivativeGenerateInPlace(
    std::vector<T> &kernelDerivativeXArray,
    std::vector<T> &kernelDerivativeYArray,
    double sigma,
    uint32_t radius)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
#endif

  int32_t signedRadius = static_cast<int32_t>(radius);
  uint32_t kernelSize = 2 * radius + 1;

  double twoSigma = sigma * sigma;
  double inv2TwoSigma = 1 / (2 * twoSigma);

  for (int32_t idxY = -signedRadius; idxY <= signedRadius; idxY++)
  {
    int32_t twoY = idxY * idxY;
    int32_t idxYOffs = (idxY + signedRadius) * kernelSize;

    for (int32_t idxX = -signedRadius; idxX <= signedRadius; idxX++)
    {
      uint32_t idx = idxYOffs + (idxX + signedRadius);

      T value = std::exp(-(idxX * idxX + twoY) * inv2TwoSigma);
      T scale = value * value / twoSigma;

      kernelDerivativeXArray[idx] = -idxX * scale;
      kernelDerivativeYArray[idx] = -idxY * scale;
    }
  }

#ifdef PerfDebugMain
  std::cout
      << "gaussianKernelDerivativeGenerateInPlace: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

template <typename T1,
          typename T2>
void voidAndClusterCandidateWrapAroundInPlace(
    std::vector<T1> &inArray,
    uint32_t width,
    uint32_t height,
    std::vector<T2> &kernelArray,
    uint32_t kernelWidth,
    uint32_t kernelHeight)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
#endif

  uint32_t squaredSize = width * height;

  std::vector<double> blurredArray(squaredSize);

  // làm mờ, kết quả sẽ đc lưu vào blurredArray
  BlueNoiseUtils::convolveWrapAroundInPlace(
      inArray,
      blurredArray,
      width,
      height,
      kernelArray,
      kernelWidth,
      kernelHeight);

  std::priority_queue<
      std::pair<double, uint32_t>>
      clusterPriorityQueue;

  std::priority_queue<
      std::pair<double, uint32_t>,
      std::vector<std::pair<double, uint32_t>>,
      std::greater<>>
      voidPriorityQueue;

  for (uint32_t i = 0; i < squaredSize; i++)
  {
    if (inArray[i] == 1)
    {
      clusterPriorityQueue.emplace(blurredArray[i], i);
    }
    else
    {
      voidPriorityQueue.emplace(blurredArray[i], i);
    }
  }

  uint32_t clusterIdx = 0;
  uint32_t voidIdx = 0;

  uint32_t idxY = 0;
  int32_t convolveIdxY = 0;
  int32_t baseConvolveIdxX = 0;

  while (true)
  {
    // tìm chấm có năng lượng lớn nhất và lưu địa chỉ của chấm đó
    while (true)
    {
      // lấy phần tử đầu tiên(giá trị của first cao nhất)
      auto topElement = clusterPriorityQueue.top();

      // save
      auto value = topElement.first;
      auto index = topElement.second;

      // bỏ phần tử đầu tiên đó khỏi danh sách
      clusterPriorityQueue.pop();

      // vì danh sách luôn đc sắp xếp với first cao nhất, có thể break sớm đc
      if (inArray[index] == 1 && blurredArray[index] == value)
      {
        clusterIdx = index;
        break;
      }
    }

    // flip chấm đó thành 0
    inArray[clusterIdx] = 0;

    // update blurredArray
    BlueNoiseUtils::convolveAddWrapAroundInPlace(
        blurredArray,
        width,
        height,
        clusterIdx,
        -1,
        kernelArray,
        kernelWidth,
        kernelHeight);

    {
      idxY = clusterIdx / width;

      convolveIdxY = idxY - (kernelHeight / 2);
      if (convolveIdxY < 0)
      {
        convolveIdxY = (convolveIdxY + height) % height;
      }

      baseConvolveIdxX = clusterIdx - idxY * width - (kernelWidth / 2);
      if (baseConvolveIdxX < 0)
      {
        baseConvolveIdxX = (baseConvolveIdxX + width) % width;
      }

      for (uint32_t kernelIdxY = 0; kernelIdxY < kernelHeight; kernelIdxY++)
      {
        uint32_t convolveIdxYOffs = convolveIdxY * width;
        uint32_t convolveIdxX = baseConvolveIdxX;

        for (uint32_t kernelIdxX = 0; kernelIdxX < kernelWidth; kernelIdxX++)
        {
          uint32_t convolveIdx = convolveIdxYOffs + convolveIdxX;

          if (inArray[convolveIdx] == 1)
          {
            clusterPriorityQueue.emplace(
                blurredArray[convolveIdx],
                convolveIdx);
          }
          else
          {
            voidPriorityQueue.emplace(
                blurredArray[convolveIdx],
                convolveIdx);
          }

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
    }

    // tìm chấm đen có năng lượng thấp nhất và lưu địa chỉ của chấm đó
    while (true)
    {
      auto topElement = voidPriorityQueue.top();
      auto value = topElement.first;
      auto index = topElement.second;

      voidPriorityQueue.pop();

      if (inArray[index] == 0 && blurredArray[index] == value)
      {
        voidIdx = index;
        break;
      }
    }

    // nếu chấm trắng với năng lượng cao nhất và chấm đen với năng lượng thấp nhất
    // là cùng 1 vị trí, khôi phục lại chấm trắng đã flip và thoát vòng lặp
    if (clusterIdx == voidIdx)
    {
      inArray[clusterIdx] = 1;
      break;
    }

    // flip chấm đen đó thành 1
    inArray[voidIdx] = 1;

    // update blurredArray
    BlueNoiseUtils::convolveAddWrapAroundInPlace(
        blurredArray,
        width,
        height,
        voidIdx,
        1,
        kernelArray,
        kernelWidth,
        kernelHeight);

    {
      idxY = voidIdx / width;

      convolveIdxY = idxY - (kernelHeight / 2);
      if (convolveIdxY < 0)
      {
        convolveIdxY = (convolveIdxY + height) % height;
      }

      baseConvolveIdxX = (voidIdx - idxY * width) - (kernelWidth / 2);
      if (baseConvolveIdxX < 0)
      {
        baseConvolveIdxX = (baseConvolveIdxX + width) % width;
      }

      for (uint32_t kernelIdxY = 0; kernelIdxY < kernelHeight; kernelIdxY++)
      {
        uint32_t convolveIdxYOffs = convolveIdxY * width;
        uint32_t kernelIdxYOffs = kernelIdxY * kernelWidth;

        uint32_t convolveIdxX = baseConvolveIdxX;

        for (uint32_t kernelIdxX = 0; kernelIdxX < kernelWidth; kernelIdxX++)
        {
          uint32_t convolveIdx = convolveIdxYOffs + convolveIdxX;

          if (inArray[convolveIdx] == 1)
          {
            clusterPriorityQueue.emplace(
                blurredArray[convolveIdx],
                convolveIdx);
          }
          else
          {
            voidPriorityQueue.emplace(
                blurredArray[convolveIdx],
                convolveIdx);
          }

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
    }

    clusterIdx = 0;
    voidIdx = 0;
  }

#ifdef PerfDebugMain
  std::cout
      << "voidAndClusterCandidateWrapAroundInPlace: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

// Gaussian Blue Noise với toroidal, mutate array tại chỗ
template <typename T>
void gaussianBlueNoiseWrapAroundInPlace(
    std::vector<T> &sampleIdxXArray,
    std::vector<T> &sampleIdxYArray,
    uint32_t width,
    uint32_t height,
    uint32_t samples,
    double sigma,
    uint32_t iterations)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
#endif

  double halfWidth = static_cast<double>(width) * 0.5;
  double halfHeight = static_cast<double>(height) * 0.5;

  // cache
  double invTwoSigma = 1. / (2. * sigma * sigma);

  for (uint32_t iteration = 0; iteration < iterations; iteration++)
  {
    for (uint32_t sample = 0; sample < samples; sample++)
    {
      double pushX = 0;
      double pushY = 0;

      double sampleIdxX = sampleIdxXArray[sample];
      double sampleIdxY = sampleIdxYArray[sample];

      // các sample còn lại là candidate
      for (uint32_t candidate = 0; candidate < samples; candidate++)
      {
        // bỏ qua những candidate có cùng index với sample
        if (sample == candidate)
        {
          continue;
        }

        // khoảng cách từ sample đến candidate
        double distanceX = sampleIdxX - sampleIdxXArray[candidate];
        double distanceY = sampleIdxY - sampleIdxYArray[candidate];

        if (distanceX > halfWidth)
        {
          distanceX -= width;
        }
        else if (distanceX < -halfWidth)
        {
          distanceX += width;
        }

        if (distanceY > halfHeight)
        {
          distanceY -= height;
        }
        else if (distanceY < -halfHeight)
        {
          distanceY += height;
        }

        // thay đổi ptrình tùy thích
        double gaussian = std::exp(-(distanceX * distanceX + distanceY * distanceY) * invTwoSigma);

        pushX += distanceX * gaussian;
        pushY += distanceY * gaussian;
      }

      double movedSampleIdxX = sampleIdxX + pushX;
      double movedSampleIdxY = sampleIdxY + pushY;

      if (movedSampleIdxX >= width)
      {
        movedSampleIdxX = std::fmod(movedSampleIdxX, width);
      }
      else if (movedSampleIdxX < 0)
      {
        movedSampleIdxX = std::fmod(movedSampleIdxX, width);
        movedSampleIdxX += width;
      }

      if (movedSampleIdxY >= height)
      {
        movedSampleIdxY = std::fmod(movedSampleIdxY, height);
      }
      else if (movedSampleIdxY < 0)
      {
        movedSampleIdxY = std::fmod(movedSampleIdxY, height);
        movedSampleIdxY += height;
      }

      sampleIdxXArray[sample] = movedSampleIdxX;
      sampleIdxYArray[sample] = movedSampleIdxY;
    }
  }

#ifdef PerfDebugMain
  std::cout
      << "gaussianBlueNoiseWrapAroundInPlace: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

template <typename T1,
          typename T2>
void generalSpectrumNoiseWrapAroundScalableInPlace(
    std::vector<T1> &sampleIdxXArray,
    std::vector<T1> &sampleIdxYArray,
    uint32_t width,
    uint32_t height,
    std::vector<T2> &targetDDA,
    uint32_t DDAWidth,
    uint32_t DDAHeight,
    double stepScale,
    uint32_t samples,
    uint32_t iterations,
    std::vector<T2> &kernelArray,
    std::vector<T2> &kernelDerivativeXArray,
    std::vector<T2> &kernelDerivativeYArray,
    uint32_t kernelWidth,
    uint32_t kernelHeight)
{
#ifdef PerfDebugMain
  clock_t t0 = clock();
  uint32_t iter = 0;
  uint32_t interval = 50;
#endif

  uint32_t squaredSize = width * height;
  uint32_t DDASquaredSize = DDAWidth * DDAHeight;

  double halfWidth = width * 0.5;
  double halfHeight = height * 0.5;

  double negHalfWidth = -halfWidth;
  double negHalfHeight = -halfHeight;

  double rescaledWidth = static_cast<double>(DDAWidth) / width;
  double rescaledHeight = static_cast<double>(DDAHeight) / height;

  double scaledSquaredSize = stepScale * std::sqrt(DDASquaredSize);

  std::vector<double> currentDDA(DDASquaredSize);
  std::vector<double> errorDDA(DDASquaredSize);

  std::vector<double> errorDerivativeX(DDASquaredSize);
  std::vector<double> errorDerivativeY(DDASquaredSize);

  std::vector<double> forceX(DDASquaredSize);
  std::vector<double> forceY(DDASquaredSize);

  BlueNoiseUtils::differentialDomainBuildBilinearWrapAroundScalableInPlace(
      sampleIdxXArray,
      sampleIdxYArray,
      width,
      height,
      currentDDA,
      DDAWidth,
      DDAHeight,
      samples);

  for (uint32_t iteration = 0; iteration < iterations; iteration++)
  {
    std::fill(forceX.begin(), forceX.end(), 0);
    std::fill(forceY.begin(), forceY.end(), 0);

    for (uint32_t i = 0; i < DDASquaredSize; i++)
    {
      errorDDA[i] = currentDDA[i] - targetDDA[i];
    }

    BlueNoiseUtils::convolveWrapAroundInPlace(
        errorDDA,
        errorDerivativeX,
        DDAWidth,
        DDAHeight,
        kernelDerivativeXArray,
        kernelWidth,
        kernelHeight);

    BlueNoiseUtils::convolveWrapAroundInPlace(
        errorDDA,
        errorDerivativeY,
        DDAWidth,
        DDAHeight,
        kernelDerivativeYArray,
        kernelWidth,
        kernelHeight);

#ifdef PerfDebugMain
    if (iter++ % interval == 0)
    {
      saveAsBMP24BPPWrapper("errorDDA" + std::to_string(iter) + ".bmp", errorDDA, DDAWidth, DDAHeight);
      saveAsBMP24BPPWrapper("errorDerivativeX" + std::to_string(iter) + ".bmp", errorDerivativeX, DDAWidth, DDAHeight);
      saveAsBMP24BPPWrapper("errorDerivativeY" + std::to_string(iter) + ".bmp", errorDerivativeY, DDAWidth, DDAHeight);
    }
#endif

    double maxPower = -std::numeric_limits<double>::infinity();

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      double accumulateX = 0;
      double accumulateY = 0;

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

        double lookupIdxX = (distanceX + halfWidth) * rescaledWidth;
        double lookupIdxY = (distanceY + halfHeight) * rescaledHeight;

        if (lookupIdxX >= DDAWidth)
        {
          lookupIdxX -= DDAWidth;
        }
        if (lookupIdxY >= DDAHeight)
        {
          lookupIdxY -= DDAHeight;
        }

        accumulateX += BlueNoiseUtils::bilinearLookupWrapAroundInPlace(
            errorDerivativeX,
            DDAWidth,
            DDAHeight,
            lookupIdxX,
            lookupIdxY);

        accumulateY += BlueNoiseUtils::bilinearLookupWrapAroundInPlace(
            errorDerivativeY,
            DDAWidth,
            DDAHeight,
            lookupIdxX,
            lookupIdxY);
      }

      forceX[sample] = accumulateX;
      forceY[sample] = accumulateY;

      double powerValue =
          accumulateX * accumulateX +
          accumulateY * accumulateY;

      if (powerValue > maxPower)
      {
        maxPower = powerValue;
      }
    }

    double finalStep = scaledSquaredSize / std::sqrt(maxPower);

    for (uint32_t sample = 0; sample < samples; sample++)
    {
      double *sampleIdxX = &sampleIdxXArray[sample];
      double *sampleIdxY = &sampleIdxYArray[sample];

      double newIdxX = *sampleIdxX + forceX[sample] * finalStep;
      double newIdxY = *sampleIdxY + forceY[sample] * finalStep;

      if (newIdxX >= width)
      {
        newIdxX -= width;
      }
      else if (newIdxX < 0)
      {
        newIdxX += width;
      }

      if (newIdxY >= height)
      {
        newIdxY -= height;
      }
      else if (newIdxY < 0)
      {
        newIdxY += height;
      }

      BlueNoiseUtils::differentialDomainUpdateBilinearWrapAroundScalableInPlace(
          sampleIdxXArray,
          sampleIdxYArray,
          width,
          height,
          newIdxX,
          newIdxY,
          currentDDA,
          DDAWidth,
          DDAHeight,
          sample,
          samples);

      *sampleIdxX = newIdxX;
      *sampleIdxY = newIdxY;
    }
  }

#ifdef PerfDebugMain
  std::cout
      << "generalSpectrumNoiseWrapAroundScalable: "
      << (clock() - t0)
      << "ms" << "\n";
#endif
}

int main(int argc, char *argv[])
{
  // Fix UTF-8
  SetConsoleOutputCP(CP_UTF8);
  setvbuf(stdout, nullptr, _IOFBF, 1000);
  std::cout << "# BlueNoise #" << "\n";
  std::cout << "# Bởi 901D3 #" << "\n";
  std::cout << "Được port sang C++ từ bản JS gốc: https://github.com/901D3/blue-noise.js" << "\n";

  if (argc == 1)
  {
    std::cout << "Không có argument nào được parse, mặc định thành" << "\n";
    std::cout << "<uint32_t> width = " << width << "\n";
    std::cout << "<uint32_t> height = " << height << "\n";
    std::cout << "<double> sigma = " << sigma << "\n";
    std::cout << "<double> kernelSizeMultiplier = " << kernelSizeMultiplier << "\n";
    std::cout << "<uint32_t> samples = " << samples << "\n";
    std::cout << "<uint32_t> iterations = " << iterations << "\n";
  }
  else if (argc > 1)
  {
    for (uint32_t argIdx = 0; argIdx + 1 < argc; argIdx++)
    {
      if (std::string(argv[argIdx]) == "-algo")
      {
        algo = static_cast<uint32_t>(std::stoi(argv[argIdx + 1]));
      }
      else if (std::string(argv[argIdx]) == "-width")
      {
        width = static_cast<uint32_t>(std::stoi(argv[argIdx + 1]));
      }
      else if (std::string(argv[argIdx]) == "-height")
      {
        height = static_cast<uint32_t>(std::stoi(argv[argIdx + 1]));
      }
      else if (std::string(argv[argIdx]) == "-sigma")
      {
        sigma = std::stod(argv[argIdx + 1]);
      }
      else if (std::string(argv[argIdx]) == "-kernel_size_multiplier")
      {
        kernelSizeMultiplier = std::stod(argv[argIdx + 1]);
      }
      else if (std::string(argv[argIdx]) == "-use_cosine_gaussian")
      {
        useCosineGaussian = true;
      }
      else if (std::string(argv[argIdx]) == "-use_adaptive_sigma")
      {
        useAdaptiveSigma = true;
      }
      else if (std::string(argv[argIdx]) == "-adaptive_sigma_scale")
      {
        adaptiveSigmaScale = std::stod(argv[argIdx + 1]);
      }
      else if (std::string(argv[argIdx]) == "-samples")
      {
        samples = static_cast<uint32_t>(std::stoi(argv[argIdx + 1]));
      }
      else if (std::string(argv[argIdx]) == "-iterations")
      {
        iterations = static_cast<uint32_t>(std::stoi(argv[argIdx + 1]));
      }
      else if (std::string(argv[argIdx]) == "-step_scale")
      {
        stepScale = std::stod(argv[argIdx + 1]);
      }
      else if (std::string(argv[argIdx]) == "-DDA_width")
      {
        DDAWidth = static_cast<uint32_t>(std::stoi(argv[argIdx + 1]));
      }
      else if (std::string(argv[argIdx]) == "-DDA_height")
      {
        DDAHeight = static_cast<uint32_t>(std::stoi(argv[argIdx + 1]));
      }
      else if (std::string(argv[argIdx]) == "-DDA_target")
      {
        DDATarget = std::string(argv[argIdx + 1]);
      }
    }
  }

  uint32_t squaredSize = width * height;

  if (useAdaptiveSigma)
  {
    sigma = std::sqrtf(static_cast<double>(squaredSize) / samples) * adaptiveSigmaScale;

    std::cout << "Sử dụng giá trị sigma tự động" << "\n";
    std::cout << "sigma = " << sigma << "\n";
  }

  uint32_t radius = static_cast<uint32_t>(std::floor(kernelSizeMultiplier * sigma));
  uint32_t kernelWidth = 2 * radius + 1;
  uint32_t kernelHeight = kernelWidth;
  uint32_t squaredKernelSize = kernelWidth * kernelHeight;

  std::vector<float> kernelArray(squaredKernelSize);

  if (useCosineGaussian)
  {
    std::cout << "Sử dụng cosine-Gaussian" << "\n";
    cosineGaussianKernelGenerateInPlace(
        kernelArray,
        sigma,
        radius);
  }
  else
  {
    gaussianKernelGenerateInPlace(
        kernelArray,
        sigma,
        radius);
  }

  // 2 array với sampleIdxXArray[i] là hoành độ, sampleIdxYArray[i] là tung độ của 1 chấm
  // với điều kiện cả 2 array phải = nhau về length và ko có element nào là số ko xác định
  std::vector<double> sampleIdxXArray(samples);
  std::vector<double> sampleIdxYArray(samples);

  BlueNoiseUtils::randomSamplesInPlace(sampleIdxXArray, width, samples);
  BlueNoiseUtils::randomSamplesInPlace(sampleIdxYArray, height, samples);

  if (algo == 0)
  {
    std::vector<double> result(squaredSize);
    std::vector<double> resultBlurred(squaredSize);

    for (uint32_t i = 0; i < samples; i++)
    {
      result[static_cast<uint32_t>(std::floor(sampleIdxYArray[i])) * width +
             static_cast<uint32_t>(std::floor(sampleIdxXArray[i]))] = 1;
    }

    voidAndClusterCandidateWrapAroundInPlace(
        result,
        width,
        height,
        kernelArray,
        kernelWidth,
        kernelHeight);

    BlueNoiseUtils::convolveWrapAroundInPlace(
        result,
        resultBlurred,
        width,
        height,
        kernelArray,
        kernelWidth,
        kernelHeight);

    saveAsBMP24BPPWrapper("result.bmp", result, width, height);
    saveAsBMP24BPPWrapper("resultBlurred.bmp", resultBlurred, width, height);
  }
  else if (algo == 1)
  {
    // khi chấm mỗi sample lên 1 hình ảnh(chuyển sang int trc), kết quả là các chấm đều
    // có những vùng trống, ít chấm hay ko là do giá trị sigma hoặc iterations thấp
    gaussianBlueNoiseWrapAroundInPlace(
        sampleIdxXArray,
        sampleIdxYArray,
        width,
        height,
        samples,
        sigma,
        iterations);

    std::vector<uint8_t> resultImage(squaredSize);

    for (uint32_t i = 0; i < samples; i++)
    {
      resultImage[static_cast<uint32_t>(std::floor(sampleIdxYArray[i])) * width +
                  static_cast<uint32_t>(std::floor(sampleIdxXArray[i]))] = 255;
    }

    BlueNoiseUtils::saveAsBMP24BPPGrayscale("resultImage.bmp", resultImage, width, height);
  }
  else if (algo == 2)
  {
    saveAsBMP24BPPWrapper("kernelArray.bmp", kernelArray, kernelWidth, kernelHeight);
  }
  else if (algo == 3)
  {
    std::vector<float> ccvt;
    loadPFMGrayscaleInPlace("ccvt.pfm", ccvt);

    BlueNoiseUtils::saveArrayAsJSON("ccvt.json", ccvt);
    saveAsBMP24BPPWrapper("ccvt.bmp", ccvt, 128, 1);
  }
  else if (algo == 4)
  {
    std::vector<float> kernelDerivativeXArray(squaredKernelSize);
    std::vector<float> kernelDerivativeYArray(squaredKernelSize);

    gaussianKernelDerivativeGenerateInPlace(
        kernelDerivativeXArray,
        kernelDerivativeYArray,
        sigma,
        radius);

#ifdef PerfDebugMain
    saveAsBMP24BPPWrapper(
        "kernelDerivativeXArray.bmp",
        kernelDerivativeXArray,
        kernelWidth,
        kernelHeight);

    saveAsBMP24BPPWrapper(
        "kernelDerivativeYArray.bmp",
        kernelDerivativeYArray,
        kernelWidth,
        kernelHeight);
#endif

    uint32_t DDASquaredSize = DDAWidth * DDAHeight;

    std::vector<double> targetFT(DDASquaredSize);
    std::vector<float> targetDDA(DDASquaredSize);

    uint32_t halfDDAWidth = DDAWidth * 0.5;
    uint32_t halfDDAHeight = DDAHeight * 0.5;

    if (DDATarget == "")
    {
      double r0 = 1;
      double r1 = 7;
      double peakGain = 1.5;
      double midGain = 1;

      for (uint32_t y = 0; y < DDAHeight; y++)
      {
        uint32_t row = y * DDAWidth;
        int32_t centerY = static_cast<int32_t>(y) - halfDDAHeight;

        for (uint32_t x = 0; x < DDAWidth; x++)
        {
          int32_t centerX = static_cast<int32_t>(x) - halfDDAWidth;

          double r = std::sqrt(centerX * centerX + centerY * centerY);

          double value = 0;

          if (r >= r0 && r < r1 + 1)
          {
            double t = (r - r0) / (r1 - r0);
            value = peakGain * (t * (3 - 2 * t));
          }
          else if (r > r1)
          {
            value = midGain;
          }

          targetFT[row + x] = value;
        }
      }

      BlueNoiseUtils::centeredCosineFourierTransform2DInPlace(
          targetFT,
          targetDDA,
          DDAWidth,
          DDAHeight);
    }
    else
    {
      loadPFMGrayscaleInPlace(DDATarget, targetDDA);

#ifdef PerfDebugMain
      saveAsBMP24BPPWrapper("targetDDA.bmp", targetDDA, DDAWidth, DDAHeight);
#endif
    }

#ifdef PerfDebugMain
    if (DDATarget == "")
    {
      saveAsBMP24BPPWrapper("targetFT.bmp", targetFT, DDAWidth, DDAHeight);

      double temp = targetDDA[halfDDAHeight * DDAWidth + halfDDAWidth];
      targetDDA[halfDDAHeight * DDAWidth + halfDDAWidth] = 0;
      saveAsBMP24BPPWrapper("targetDDA.bmp", targetDDA, DDAWidth, DDAHeight);
      targetDDA[halfDDAHeight * DDAWidth + halfDDAWidth] = temp;
    }

    std::vector<double> beforeDDA(DDASquaredSize);
    BlueNoiseUtils::differentialDomainBuildBilinearWrapAroundScalableInPlace(
        sampleIdxXArray,
        sampleIdxYArray,
        width,
        height,
        beforeDDA,
        DDAWidth,
        DDAHeight,
        samples);

    saveAsBMP24BPPWrapper("beforeDDA.bmp", beforeDDA, DDAWidth, DDAHeight);

    std::vector<uint8_t> resultBeforeImage(squaredSize);
    for (uint32_t i = 0; i < samples; i++)
    {
      resultBeforeImage[static_cast<uint32_t>(std::floor(sampleIdxYArray[i])) * width +
                        static_cast<uint32_t>(std::floor(sampleIdxXArray[i]))] = 255;
    }

    BlueNoiseUtils::saveAsBMP24BPPGrayscale("resultBeforeImage.bmp", resultBeforeImage, width, height);
#endif

    generalSpectrumNoiseWrapAroundScalableInPlace(
        sampleIdxXArray,
        sampleIdxYArray,
        width,
        height,
        targetDDA,
        DDAWidth,
        DDAHeight,
        stepScale,
        samples,
        iterations,
        kernelArray,
        kernelDerivativeXArray,
        kernelDerivativeYArray,
        kernelWidth,
        kernelHeight);

    std::vector<double> afterDDA(DDASquaredSize);
    BlueNoiseUtils::differentialDomainBuildBilinearWrapAroundScalableInPlace(
        sampleIdxXArray,
        sampleIdxYArray,
        width,
        height,
        afterDDA,
        DDAWidth,
        DDAHeight,
        samples);

#ifdef PerfDebugMain
    saveAsBMP24BPPWrapper("afterDDA.bmp", afterDDA, DDAWidth, DDAHeight);
#endif

    std::vector<uint8_t> resultImage(squaredSize);
    for (uint32_t i = 0; i < samples; i++)
    {
      resultImage[static_cast<uint32_t>(std::floor(sampleIdxYArray[i])) * width +
                  static_cast<uint32_t>(std::floor(sampleIdxXArray[i]))] = 1;
    }

    saveAsBMP24BPPWrapper("resultImage.bmp", resultImage, width, height);
    BlueNoiseUtils::saveAsPBMASCIIGrayscale("resultImage.pbm", resultImage, width, height);
  }
  else if (algo == 5)
  {
  }

  return 0;
}