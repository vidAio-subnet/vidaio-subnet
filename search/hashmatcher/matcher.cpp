#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <bitset>
#include <fftw3.h>
#include <filesystem>
#include <chrono>
#include <regex>
#include <pybind11/functional.h>
#include <fstream>

namespace py = pybind11;
namespace fs = std::filesystem;

using namespace cv;
using namespace std;

__always_inline
double median(double *arr, size_t len) {
    std::sort(arr, arr + len);

    if (len % 2 == 0) {
        return (arr[(len / 2) - 1] + arr[len / 2]) / 2;
    } else {
        return arr[(len + 1 / 2)];
    }
}

__always_inline
double median(uchar *arr, size_t len) {
    std::sort(arr, arr + len);

    if (len % 2 == 0) {
        return (double) (arr[(len / 2) - 1] + arr[len / 2]) / 2;
    } else {
        return arr[(len + 1 / 2)];
    }
}

__always_inline
void set_bit_at(uchar *buf, unsigned int offset, bool val) {
    unsigned int byte_offset = offset / 8;
    unsigned int bit_offset = offset - byte_offset * 8;

    if (val) {
        buf[byte_offset] |= (1 << bit_offset);
    } else {
        buf[byte_offset] &= ~(1 << bit_offset);
    }
}

void hash_to_hex_string_reversed(const uchar *h, char *out, int hash_size) {
    int hash_len = hash_size * hash_size / 4;

    for (unsigned int i = 0; i < hash_len; i += 2) {
        uchar c = (h[i / 2] & 0x80) >> 7 |
                  (h[i / 2] & 0x40) >> 5 |
                  (h[i / 2] & 0x20) >> 3 |
                  (h[i / 2] & 0x10) >> 1 |
                  (h[i / 2] & 0x08) << 1 |
                  (h[i / 2] & 0x04) << 3 |
                  (h[i / 2] & 0x02) << 5 |
                  (h[i / 2] & 0x01) << 7;

        sprintf(out + i, "%02x", c);
    }
    out[hash_len + 1] = '\0';
}

int hamming_distance(const std::string& a, const std::string& b) {
    int dist = 0;
    for (size_t i = 0; i < a.length(); i++) {
        dist += __builtin_popcount(std::stoi(std::string(1, a[i]), nullptr, 16) ^ std::stoi(std::string(1, b[i]), nullptr, 16));
    }
    return dist;
}

std::pair<int, int> match_query_in_video(
    const std::vector<std::string>& query_hashes,
    const std::vector<std::string>& source_hashes
    ) 
{
    int q_len = query_hashes.size();
    int s_len = source_hashes.size();
    int interval = 1;
    if (interval < 0)
        interval = 1;

    int best_match = -1;
    int min_diff = std::numeric_limits<int>::max();

    for (int i = 0; i < s_len - q_len; i++) {
        int total_diff = 0; 
        for (int j = 0; j < q_len; j += interval) {
            total_diff += hamming_distance(source_hashes[i + j], query_hashes[j]);
        }

        if (total_diff < min_diff) {
            min_diff = total_diff;
            best_match = i;
        }
    }

    return {best_match, min_diff};
}

PYBIND11_MODULE(cmatcher, m) {
    m.def("match_query_in_video", &match_query_in_video, "Get best match between query and video");
}
