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

typedef std::array<uint64_t, 4> HashBits;
typedef std::vector<HashBits> HashBitsVector;

class HashMatcher {
private:
    int coarse_unit;
    int coarse_interval;
    int max_query_length;
    HashBitsVector query_bits_coarse;
    HashBitsVector query_bits_fine;

    std::vector<HashBitsVector> dataset_bits_coarse_arr;
    std::vector<HashBitsVector> dataset_bits_fine_arr;

public:
    HashMatcher(int coarse_unit = 4, int coarse_interval = 3) 
        : coarse_unit(coarse_unit), coarse_interval(coarse_interval) {}

    void set_query(const std::vector<std::string>& query_hashes, int fps) {
        // Set coarse query bits
        query_bits_coarse.clear();
        for (size_t i = 0; i < query_hashes.size(); i += coarse_unit) {
            if (i < fps * 2) {
                std::array<uint64_t, 4> packed_bits;
                // Convert hex string to 4 uint64_t values
                for (size_t j = 0; j < 4; j++) {
                    std::string hex_chunk = query_hashes[i].substr(j * 16, 16);
                    packed_bits[j] = std::stoull(hex_chunk, nullptr, 16);
                }
                query_bits_coarse.push_back(packed_bits);
            }
        }

        // Set fine query bits
        query_bits_fine.clear();
        for (size_t i = 0; i < query_hashes.size() && i < fps * 5; i++) {
            std::array<uint64_t, 4> packed_bits;
            for (size_t j = 0; j < 4; j++) {
                std::string hex_chunk = query_hashes[i].substr(j * 16, 16);
                packed_bits[j] = std::stoull(hex_chunk, nullptr, 16);
            }
            query_bits_fine.push_back(packed_bits);
        }

        max_query_length = query_hashes.size();
    }

    void add_dataset(const std::vector<std::string>& dataset_hashes) {
        // Set coarse dataset bits
        HashBitsVector dataset_bits_coarse;
        for (size_t i = 0; i < dataset_hashes.size(); i += coarse_unit) {
            HashBits packed_bits;
            for (size_t j = 0; j < 4; j++) {
                std::string hex_chunk = dataset_hashes[i].substr(j * 16, 16);
                packed_bits[j] = std::stoull(hex_chunk, nullptr, 16);
            }
            dataset_bits_coarse.push_back(packed_bits);
        }
        dataset_bits_coarse_arr.push_back(dataset_bits_coarse);

        // Set fine dataset bits
        HashBitsVector dataset_bits_fine;
        for (size_t i = 0; i < dataset_hashes.size(); i++) {
            HashBits packed_bits;
            for (size_t j = 0; j < 4; j++) {
                std::string hex_chunk = dataset_hashes[i].substr(j * 16, 16);
                packed_bits[j] = std::stoull(hex_chunk, nullptr, 16);
            }
            dataset_bits_fine.push_back(packed_bits);
        }
        dataset_bits_fine_arr.push_back(dataset_bits_fine);
    }

    std::pair<int, int> match() {
        int best_start = -1;
        int best_score = std::numeric_limits<int>::max();
        int best_dataset_idx = -1;
        for (size_t i = 0; i < dataset_bits_coarse_arr.size(); i++) {
            std::pair<int, int> result = match_item(dataset_bits_coarse_arr[i], dataset_bits_fine_arr[i]);
            if (result.second < best_score) {
                best_score = result.second;
                best_start = result.first;
                best_dataset_idx = i;
            }
        }
        return {best_dataset_idx, best_start};
    }

    private:
        std::pair<int, int> match_item(HashBitsVector &dataset_bits_coarse, HashBitsVector &dataset_bits_fine) 
        {
            int d_len = dataset_bits_coarse.size();
            int q_len = query_bits_coarse.size();

            // Initialize scores and positions arrays
            std::vector<int> scores(3, std::numeric_limits<int>::max());
            std::vector<int> positions(3, -1);
            int worst_index = 0;

            int coarse_size = coarse_unit * coarse_interval;

            // Coarse search
            for (int i = 0; i < d_len - q_len + 1; i += coarse_interval) {
                int total_score = 0;

                // Compute Hamming distance for each frame
                for (int j = 0; j < q_len; j++) {
                    for (size_t k = 0; k < query_bits_coarse[j].size(); k++) {
                        total_score += __builtin_popcountll(
                            query_bits_coarse[j][k] ^ dataset_bits_coarse[i + j][k]
                        );
                    }
                }

                if (total_score < scores[worst_index]) {
                    scores[worst_index] = total_score;
                    positions[worst_index] = i * coarse_unit;
                    worst_index = (scores[0] > scores[1]) ? 0 : 1;
                }
            }

            // Sort positions
            std::sort(positions.begin(), positions.end());

            // Fine search
            int best_score = std::numeric_limits<int>::max();
            int best_start = -1;
            int prev_pos = -1;

            d_len = dataset_bits_fine.size();
            q_len = query_bits_fine.size();

            for (int pos : positions) {
                if (pos == -1) continue;

                int start_idx = std::max(0, pos - coarse_size);
                int end_idx = std::min(d_len - max_query_length + 1, pos + coarse_size);

                if (prev_pos != -1 && pos - prev_pos == coarse_size) {
                    start_idx = pos;
                }

                for (int i = start_idx; i < end_idx; i++) {
                    int total_score = 0;

                    // Compute Hamming distance for each frame
                    for (int j = 0; j < q_len; j++) {
                        for (size_t k = 0; k < query_bits_fine[j].size(); k++) {
                            total_score += __builtin_popcountll(
                                query_bits_fine[j][k] ^ dataset_bits_fine[i + j][k]
                            );
                        }
                    }

                    if (total_score < best_score) {
                        best_score = total_score;
                        best_start = i;
                    }
                }
                prev_pos = pos;
            }

            return {best_start, best_score};
        }
};


PYBIND11_MODULE(cmatcher, m) {
    py::class_<HashMatcher>(m, "HashMatcher")
        .def(py::init<int, int>())
        .def("set_query", &HashMatcher::set_query)
        .def("add_dataset", &HashMatcher::add_dataset)
        .def("match", &HashMatcher::match);
}
