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
#include <thread>

namespace py = pybind11;
namespace fs = std::filesystem;

using namespace cv;
using namespace std;

typedef std::array<uint64_t, 4> HashBits;
typedef std::vector<HashBits> HashBitsVector;

class HashMatcher {
private:
    int threads;
    int coarse_unit;
    int coarse_interval;
    int query_width;
    int max_query_length;
    HashBitsVector query_bits_coarse;
    HashBitsVector query_bits_fine;

    std::vector<HashBitsVector> dataset_bits_coarse_arr;
    std::vector<HashBitsVector> dataset_bits_fine_arr;
    std::vector<int> dataset_width_arr;

public:
    HashMatcher(int threads = 4, int coarse_unit = 4, int coarse_interval = 3) 
        : threads(threads), coarse_unit(coarse_unit), coarse_interval(coarse_interval) {}

    void set_query(const std::vector<std::string>& query_hashes, int fps, int width) {
        // Set coarse query bits
        query_width = width;
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

    void remove_dataset(int index) {
        if (index >= 0 && index < dataset_bits_coarse_arr.size()) {
            dataset_bits_coarse_arr.erase(dataset_bits_coarse_arr.begin() + index);
            dataset_bits_fine_arr.erase(dataset_bits_fine_arr.begin() + index);
        }
    }

    void add_dataset(const std::vector<std::string>& dataset_hashes, int width) {
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

        dataset_width_arr.push_back(width);
    }

    std::pair<int, int> match() {
        int best_start = -1;
        int best_score = std::numeric_limits<int>::max();
        int best_dataset_idx = -1;
        
        // Get number of available CPU cores
        unsigned int num_threads = threads;
        if (num_threads > std::thread::hardware_concurrency() || num_threads == 0)
            num_threads = std::thread::hardware_concurrency() / 2;
        if (num_threads == 0) 
            num_threads = 4;
        
        // Create thread pool
        std::vector<std::thread> threads;
        std::vector<std::pair<int, int>> results(num_threads, {-1, std::numeric_limits<int>::max()});
        std::vector<int> dataset_indices(num_threads, -1);
        
        // Split work among threads
        size_t items_per_thread = (dataset_bits_coarse_arr.size() + num_threads - 1) / num_threads;
        
        for (unsigned int t = 0; t < num_threads; t++) {
            threads.emplace_back([&, t]() {
                size_t start_idx = t * items_per_thread;
                size_t end_idx = std::min(start_idx + items_per_thread, dataset_bits_coarse_arr.size());
                
                for (size_t i = start_idx; i < end_idx; i++) {
                    if (dataset_width_arr[i] != query_width) {
                        continue;
                    }
                    std::pair<int, int> result = match_item(dataset_bits_coarse_arr[i], dataset_bits_fine_arr[i]);
                    if (result.second < results[t].second) {
                        results[t] = result;
                        dataset_indices[t] = i;
                    }
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Find best result across all threads
        for (unsigned int t = 0; t < num_threads; t++) {
            if (results[t].second < best_score) {
                best_score = results[t].second;
                best_start = results[t].first;
                best_dataset_idx = dataset_indices[t];
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
        .def(py::init<int, int, int>())
        .def("set_query", &HashMatcher::set_query)
        .def("add_dataset", &HashMatcher::add_dataset)
        .def("remove_dataset", &HashMatcher::remove_dataset)
        .def("match", &HashMatcher::match);
}
