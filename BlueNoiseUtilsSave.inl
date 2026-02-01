#pragma once

namespace BlueNoiseUtils
{
  template <typename T>
  void saveArrayAsJSON(
      const std::string &filename,
      const std::vector<T> &array)
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

    uint32_t length = array.size();

    file << std::fixed << std::setprecision(6);
    file << "[\n";

    for (size_t i = 0; i < array.size(); i++)
    {
      file << array[i];

      if (i + 1 < array.size())
      {
        file << ",";
      }
    }

    file << "]";

    file.close();

    std::cout << "Đã lưu tệp: " << filename << "\n";

#ifdef PerfDebugUtils
    std::cout
        << "saveArrayAsJSON: "
        << (clock() - t0)
        << "ms"
        << "\n";
#endif
  }
}
