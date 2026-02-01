#include "BlueNoiseUtils.h"

namespace BlueNoiseUtils
{
  void saveAsBMP24BPPGrayscale(
      const std::string &filename,
      const std::vector<uint8_t> &array,
      uint32_t width,
      uint32_t height)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
      std::cerr << "Không thể mở tệp để ghi: " << filename << "\n";
      std::exit(0);
    }

    uint32_t rowStride = width * 3;
    uint32_t rowPadded = (rowStride + 3) & ~3;
    uint32_t imageSize = rowPadded * height;

    uint16_t signature = 0x4D42;
    uint32_t fileHeaderLength = 14;
    uint32_t infoHeaderLength = 40;
    uint32_t headerLength = fileHeaderLength + infoHeaderLength;
    uint32_t fileSize = headerLength + imageSize;

    uint16_t plane = 1;
    uint16_t bitCount = 24;
    uint32_t compression = 0;

    uint8_t padding3[3] = {0, 0, 0};
    uint16_t zero2B = 0;
    uint32_t zero4B = 0;

    file.write(reinterpret_cast<const char *>(&signature), 2);
    file.write(reinterpret_cast<const char *>(&fileSize), 4);
    file.write(reinterpret_cast<const char *>(&zero2B), 2);
    file.write(reinterpret_cast<const char *>(&zero2B), 2);
    file.write(reinterpret_cast<const char *>(&headerLength), 4);

    file.write(reinterpret_cast<const char *>(&infoHeaderLength), 4);
    file.write(reinterpret_cast<const char *>(&width), 4);

    int32_t negHeight = -static_cast<int32_t>(height);
    file.write(reinterpret_cast<const char *>(&negHeight), 4);

    file.write(reinterpret_cast<const char *>(&plane), 2);
    file.write(reinterpret_cast<const char *>(&bitCount), 2);
    file.write(reinterpret_cast<const char *>(&compression), 4);
    file.write(reinterpret_cast<const char *>(&imageSize), 4);

    file.write(reinterpret_cast<const char *>(&zero4B), 4);
    file.write(reinterpret_cast<const char *>(&zero4B), 4);
    file.write(reinterpret_cast<const char *>(&zero4B), 4);
    file.write(reinterpret_cast<const char *>(&zero4B), 4);

    for (uint32_t idxY = 0; idxY < height; idxY++)
    {
      uint32_t yOffs = idxY * width;

      for (uint32_t idxX = 0; idxX < width; idxX++)
      {
        uint8_t g = array[yOffs + idxX];

        uint8_t pixel[3] = {g, g, g};
        file.write(reinterpret_cast<char *>(pixel), 3);
      }

      file.write(reinterpret_cast<char *>(padding3), rowPadded - rowStride);
    }

    std::cout << "Đã lưu tệp: " << filename << "\n";

#ifdef PerfDebugUtils
    std::cout
        << "saveAsBMPGrayscale: "
        << (clock() - t0)
        << "ms"
        << "\n";
#endif
  }

  void saveAsBMP24BPP(
      const std::string &filename,
      const std::vector<uint8_t> &rArray,
      const std::vector<uint8_t> &gArray,
      const std::vector<uint8_t> &bArray,
      uint32_t width,
      uint32_t height)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
      std::cerr << "Không thể mở tệp để ghi: " << filename << "\n";
      std::exit(0);
    }

    uint32_t rowStride = width * 3;
    uint32_t rowPadded = (rowStride + 3) & ~3;
    uint32_t imageSize = rowPadded * height;

    uint16_t signature = 0x4D42;
    uint32_t fileHeaderLength = 14;
    uint32_t infoHeaderLength = 40;
    uint32_t headerLength = fileHeaderLength + infoHeaderLength;
    uint32_t fileSize = headerLength + imageSize;

    uint16_t plane = 1;
    uint16_t bitCount = 24;
    uint32_t compression = 0;

    uint8_t padding3[3] = {0, 0, 0};
    uint16_t zero2B = 0;
    uint32_t zero4B = 0;

    file.write(reinterpret_cast<char *>(&signature), 2);
    file.write(reinterpret_cast<char *>(&fileSize), 4);
    file.write(reinterpret_cast<char *>(&zero2B), 2);
    file.write(reinterpret_cast<char *>(&zero2B), 2);
    file.write(reinterpret_cast<char *>(&headerLength), 4);

    file.write(reinterpret_cast<char *>(&infoHeaderLength), 4);
    file.write(reinterpret_cast<char *>(&width), 4);

    int32_t negHeight = -static_cast<int32_t>(height);
    file.write(reinterpret_cast<char *>(&negHeight), 4);

    file.write(reinterpret_cast<char *>(&plane), 2);
    file.write(reinterpret_cast<char *>(&bitCount), 2);
    file.write(reinterpret_cast<char *>(&compression), 4);
    file.write(reinterpret_cast<char *>(&imageSize), 4);

    file.write(reinterpret_cast<char *>(&zero4B), 4);
    file.write(reinterpret_cast<char *>(&zero4B), 4);
    file.write(reinterpret_cast<char *>(&zero4B), 4);
    file.write(reinterpret_cast<char *>(&zero4B), 4);

    for (uint32_t idxY = 0; idxY < height; idxY++)
    {
      uint32_t yOffs = idxY * width;

      for (uint32_t idxX = 0; idxX < width; idxX++)
      {
        uint32_t idx = yOffs + idxX;

        uint8_t pixel[3] = {rArray[idx], gArray[idx], bArray[idx]};
        file.write(reinterpret_cast<char *>(pixel), 3);
      }

      file.write(reinterpret_cast<char *>(padding3), rowPadded - rowStride);
    }

    std::cout << "Đã lưu tệp: " << filename << "\n";

#ifdef PerfDebugUtils
    std::cout
        << "saveAsBMPGrayscale: "
        << (clock() - t0)
        << "ms"
        << "\n";
#endif
  }

  void saveAsPBMASCIIGrayscale(
      const std::string &filename,
      const std::vector<uint8_t> &array,
      uint32_t width,
      uint32_t height)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    std::ofstream file(filename);
    if (!file.is_open())
    {
      std::cerr << "Không thể mở tệp để ghi: " << filename << "\n";
      std::exit(0);
    }

    file << "P1\n";
    file << width << " " << height << "\n";

    for (uint32_t idxY = 0; idxY < height; idxY++)
    {
      uint32_t yOffs = idxY * width;

      for (uint32_t idxX = 0; idxX < width; idxX++)
      {
        if (array[yOffs + idxX] == 1)
        {
          file << '0';
        }
        else
        {
          file << '1';
        }
      }

      file << '\n';
    }

    std::cout << "Đã lưu tệp: " << filename << "\n";

#ifdef PerfDebugUtils
    std::cout
        << "saveAsPBMASCIIGrayscale: "
        << (clock() - t0)
        << "ms"
        << "\n";
#endif
  }

  void saveAsPGMASCIIGrayscale(
      const std::string &filename,
      const std::vector<uint16_t> &array,
      uint32_t width,
      uint32_t height,
      uint16_t maxVal)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    std::ofstream file(filename);
    if (!file.is_open())
    {
      std::cerr << "Không thể mở tệp để ghi: " << filename << "\n";
      std::exit(0);
    }

    if (array.size() < width * height)
    {
      std::cerr << "Array quá nhỏ\n";
      std::exit(0);
    }

    if (maxVal >= 65536)
    {
      std::cerr << "maxVal >= 65536" << "\n";
      std::exit(0);
    }

    file << "P2\n";
    file << width << " " << height << "\n";
    file << maxVal << "\n";

    for (uint32_t idxY = 0; idxY < height; idxY++)
    {
      uint32_t yOffs = idxY * width;

      for (uint32_t idxX = 0; idxX < width; idxX++)
      {
        file << array[yOffs + idxX] << " ";
      }

      file << "\n";
    }

    std::cout << "Đã lưu tệp: " << filename << "\n";

#ifdef PerfDebugUtils
    std::cout
        << "saveAsPBMASCIIGrayscale: "
        << (clock() - t0)
        << "ms"
        << "\n";
#endif
  }

  void saveAsPFMGrayscaleFloat32(
      const std::string &filename,
      const std::vector<float> &array,
      uint32_t width,
      uint32_t height,
      float scaleFactor)
  {
#ifdef PerfDebugUtils
    clock_t t0 = clock();
#endif

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
      std::cerr << "Không thể mở tệp để ghi: " << filename << "\n";
      std::exit(0);
    }

    if (array.size() < width * height)
    {
      std::cerr << "Array quá nhỏ\n";
      std::exit(0);
    }

    file << "Pf\n";
    file << width << " " << height << "\n";
    file << scaleFactor << "\n";

    for (uint32_t idxY = height; idxY-- > 0;)
    {
      uint32_t yOffs = idxY * width;

      for (uint32_t idxX = 0; idxX < width; idxX++)
      {
        float value = array[yOffs + idxX];

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        if (scaleFactor > 0.f)
        {
          uint32_t u;
          std::memcpy(&u, &value, sizeof(float));
          u = ((u & 0x000000FF) << 24) |
              ((u & 0x0000FF00) << 8) |
              ((u & 0x00FF0000) >> 8) |
              ((u & 0xFF000000) >> 24);
          std::memcpy(&value, &u, sizeof(float));
        }
#endif

        file.write(reinterpret_cast<char *>(&value), sizeof(float));
      }
    }

    std::cout << "Đã lưu tệp: " << filename << "\n";

#ifdef PerfDebugUtils
    std::cout
        << "saveAsPFMGrayscaleFloat32: "
        << (clock() - t0)
        << "ms"
        << "\n";
#endif
  }
}
