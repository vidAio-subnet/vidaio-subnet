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


static void init() __attribute__((constructor));

void init() {
    fftw_make_planner_thread_safe();
}

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

std::pair<int, int> find_coarse_match(
    const std::vector<std::pair<int, std::string>>& source_hashes,
    const std::vector<std::pair<int, std::string>>& query_hashes,
    int tolerance
) {
    int best_match = -1;
    int min_diff = std::numeric_limits<int>::max();

    int interval = std::max((int)query_hashes.size() / 5, 10);
    std::vector<std::pair<int, std::string>> reduced_query;
    for (size_t i = 0; i < query_hashes.size(); i += interval) {
        reduced_query.push_back(query_hashes[i]);
    }

    for (size_t i = 0; i + reduced_query.size() * interval <= source_hashes.size(); i += interval) {
        int total_diff = 0;
        for (size_t j = 0; j < reduced_query.size(); j++) {
            total_diff += hamming_distance(
                source_hashes[i + j * interval].second,
                reduced_query[j].second
            );
        }

        if (total_diff < min_diff && total_diff <= tolerance * (int)reduced_query.size()) {
            min_diff = total_diff;
            best_match = source_hashes[i].first;
        }
    }

    return {best_match, min_diff};
}

std::pair<int, int> find_subsequence_position(
    const std::vector<std::pair<int, std::string>>& source_hashes,
    const std::vector<std::pair<int, std::string>>& query_hashes,
    int interval,
    int tolerance
) {
    int best_match = -1;
    int min_diff_total = std::numeric_limits<int>::max();

    // Reduce query
    std::vector<std::pair<int, std::string>> reduced_query;
    for (size_t i = 0; i < query_hashes.size(); i += interval) {
        reduced_query.push_back(query_hashes[i]);
    }
    int q_len = reduced_query.size();

    // Slide over source
    for (size_t i = 0; i + q_len * interval <= source_hashes.size(); i += interval) {
        int total_diff = 0;
        bool valid = true;

        for (size_t j = 0; j < q_len; ++j) {
            const std::string& src_hash = source_hashes[i + j * interval].second;
            const std::string& qry_hash = reduced_query[j].second;

            if (src_hash.length() != qry_hash.length()) {
                valid = false;
                break;
            }

            total_diff += hamming_distance(src_hash, qry_hash);
        }

        if (valid && total_diff < min_diff_total && total_diff <= tolerance * q_len) {
            min_diff_total = total_diff;
            best_match = source_hashes[i].first;
        }
    }

    return {best_match, min_diff_total};
}

std::string phash(const cv::Mat &input, const int hash_size, int highfreq_factor) {
    int img_size = hash_size * highfreq_factor;
    Mat im;
    resize(input, im, Size(img_size, img_size), 0, 0, INTER_AREA);
    double pixels[img_size * img_size];

    uchar *pixel = im.ptr(0);
    int endPixel = im.cols * im.rows;
    for (int i = 0; i < endPixel; i++) {
        pixels[i] = (double) pixel[i] / 255;
    }

    double dct_out[img_size * img_size];
    fftw_plan plan = fftw_plan_r2r_2d(
            img_size, img_size,
            pixels, dct_out,
            FFTW_REDFT10, FFTW_REDFT10, // DCT-II
            FFTW_ESTIMATE
    );
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    double dct_lowfreq[hash_size * hash_size];
    double sorted[hash_size * hash_size];

    int ptr_low = 0;
    int ptr = 0;
    for (int i = 0; i < hash_size; ++i) {
        for (int j = 0; j < hash_size; ++j) {
            dct_lowfreq[ptr_low] = dct_out[ptr];
            sorted[ptr_low] = dct_out[ptr];
            ptr_low += 1;
            ptr += 1;
        }
        ptr += (img_size - hash_size);
    }

    double med = median(sorted, hash_size * hash_size);

    std::string hashbit(hash_size * hash_size, '\0');
    std::string hashstr(hash_size * hash_size / 4, '\0');

    for (int i = 0; i < hash_size * hash_size; ++i) {
        set_bit_at((uchar*)hashbit.data() + i, 0, dct_lowfreq[i] > med); // set bit 0 of byte at position i
    }

    hash_to_hex_string_reversed((const uchar*)hashbit.data(), hashstr.data(), hash_size);
    return hashstr;
}

std::string dhash(const cv::Mat &input, int hash_size) {
    Mat im;
    resize(input, im, Size(hash_size + 1, hash_size), 0, 0, INTER_AREA);

    std::string hashbit(hash_size * hash_size, '\0');
    std::string hashstr(hash_size * hash_size / 4, '\0');

    int offset = 0;
    for (int i = 0; i < im.rows; ++i) {
        uchar *pixel = im.ptr(i);

        for (int j = 1; j < im.cols; ++j) {
            set_bit_at((uchar*)hashbit.data(), offset++, pixel[j] > pixel[j - 1]);
        }
    }

    hash_to_hex_string_reversed((const uchar*)hashbit.data(), hashstr.data(), hash_size);
    return hashstr;
}

std::pair<std::vector<std::pair<int, std::string>>, double> get_video_hashes(const std::string& video_path, int fine_interval = 10) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + video_path);
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    std::vector<std::pair<int, std::string>> hashes;

    cv::Mat frame;
    int frame_idx = 0;

    while (cap.read(frame)) {
        if (frame_idx % fine_interval == 0) {
            std::string hash = phash(frame, 8, 4);
            hashes.emplace_back(frame_idx, hash);
        }
        ++frame_idx;
    }

    cap.release();
    return {hashes, fps};
}

int compute_mae(const Mat& a, const Mat& b) {
    Mat diff;
    absdiff(a, b, diff);
    Scalar mae = mean(diff);
    return static_cast<int>(mae[0]);
}

int find_query_start_frame_fast(const string& original_path, const string& query_path, int& out_min_dist) {
    VideoCapture original_cap(original_path);
    VideoCapture query_cap(query_path);

    if (!original_cap.isOpened() || !query_cap.isOpened())
        return -1;

    int o_width = (int)original_cap.get(CAP_PROP_FRAME_WIDTH);
    int o_height = (int)original_cap.get(CAP_PROP_FRAME_HEIGHT);
    int q_width = (int)query_cap.get(CAP_PROP_FRAME_WIDTH);
    int q_height = (int)query_cap.get(CAP_PROP_FRAME_HEIGHT);

    if (o_width != q_width || o_height != q_height)
        return -1;

    double q_fps = query_cap.get(CAP_PROP_FPS);
    int q_total_frames = (int)query_cap.get(CAP_PROP_FRAME_COUNT);

    // Load query frames (downscaled grayscale)
    vector<Mat> query_frames;
    for (int i = 0; i < q_total_frames; ++i) {
        Mat frame, gray, small;
        if (!query_cap.read(frame)) break;
        resize(frame, small, Size(), 0.25, 0.25, INTER_AREA);
        cvtColor(small, gray, COLOR_BGR2GRAY);
        query_frames.push_back(gray);
    }

    int query_len = query_frames.size();
    if (query_len < 3) return -1;

    Mat q_start = query_frames.front();
    Mat q_mid = query_frames[query_len / 2];
    Mat q_end = query_frames.back();

    // Load all original frames in memory, downscaled
    vector<Mat> original_frames;
    while (true) {
        Mat frame, gray, small;
        if (!original_cap.read(frame)) break;
        resize(frame, small, Size(), 0.25, 0.25, INTER_AREA);
        cvtColor(small, gray, COLOR_BGR2GRAY);
        original_frames.push_back(gray);
    }

    int best_idx = -1;
    int min_dist = numeric_limits<int>::max();

    for (int i = 0; i <= (int)original_frames.size() - query_len; ++i) {
        int dist = 0;
        dist += compute_mae(original_frames[i], q_start);
        dist += compute_mae(original_frames[i + query_len / 2], q_mid);
        dist += compute_mae(original_frames[i + query_len - 1], q_end);

        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }

    out_min_dist = min_dist;
    return best_idx;
}

// bool run_ffmpeg_clip(const string& input, const string& output, double start, double duration) {
//     string cmd = "ffmpeg -loglevel quiet -y -ss " + to_string(start) +
//                  " -i \"" + input + "\" -t " + to_string(duration) +
//                  " -c:v libx264 -preset slow \"" + output + "\"";
//     return system(cmd.c_str()) == 0;
// }

// bool run_ffmpeg_resize(const string& input, const string& output, int height) {
//     string cmd = "ffmpeg -loglevel quiet -y -i \"" + input + "\" -vf scale=-1:" + to_string(height) +
//                  " -c:v libx264 -preset fast \"" + output + "\"";
//     return system(cmd.c_str()) == 0;
// }

bool run_ffmpeg_clip(const std::string& input, const std::string& output, double start, double duration) {
    std::string cmd = "ffmpeg -loglevel error -y -ss " + std::to_string(start) +
                      " -i \"" + input + "\" -t " + std::to_string(duration) +
                      " -c:v h264_nvenc -preset p1 -profile high -rc constqp -qp 0 -bf 0 -g 1 \"" + output + "\"";
    return system(cmd.c_str()) == 0;
}

bool run_ffmpeg_resize(const std::string& input, const std::string& output, int height) {
    std::string cmd = "ffmpeg -loglevel error -y -i \"" + input + "\" -vf scale=-1:" + std::to_string(height) +
                      " -c:v h264_nvenc -preset p1 -profile high -rc constqp -qp 0 -bf 0 -g 1 \"" + output + "\"";
    return system(cmd.c_str()) == 0;
}

bool convert_to_y4m(const std::string& input, const std::string& output) {
    std::string cmd = "ffmpeg -loglevel error -y -i \"" + input + "\" -pix_fmt yuv420p -f yuv4mpegpipe \"" + output + "\"";
    return system(cmd.c_str()) == 0;
}

float get_vmaf_score(const std::string& ref_y4m, const std::string& dist_y4m) {
    std::string xml_path = ref_y4m + ".xml";
    std::string cmd = "vmaf -r \"" + ref_y4m + "\" -d \"" + dist_y4m + "\" -out-fmt xml -o \"" + xml_path + "\"";
    if (system(cmd.c_str()) != 0) return -1.0f;

    std::ifstream f(xml_path);
    if (!f.is_open()) return -1.0f;

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    std::smatch match;
    std::regex vmaf_regex(R"REGEX(<metric[^>]*name="vmaf"[^>]*harmonic_mean="([0-9.]+)")REGEX");
    if (std::regex_search(content, match, vmaf_regex)) {
        fs::remove(xml_path);
        return std::stof(match[1].str());
    } else {
        throw std::runtime_error("VMAF score not found in XML output");
    }
}


std::string run_command(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

// float extract_vmaf_score(const std::string& xml_path) {
//     std::ifstream file(xml_path);
//     std::stringstream buffer;
//     buffer << file.rdbuf();
//     std::string content = buffer.str();

//     std::smatch match;
//     std::regex vmaf_regex("<metric name=\"vmaf\" harmonic_mean=\"([0-9.]+)\"");
//     if (std::regex_search(content, match, vmaf_regex)) {
//         return std::stof(match[1].str());
//     } else {
//         throw std::runtime_error("VMAF score not found in XML output");
//     }
// }

// float evaluate_vmaf_cpp(const std::string& ref_mp4, const std::string& dist_mp4) {
//     std::string ref_y4m = ref_mp4.substr(0, ref_mp4.find_last_of('.')) + ".y4m";
//     std::string dist_y4m = dist_mp4.substr(0, dist_mp4.find_last_of('.')) + ".y4m";
//     std::string xml_out = ref_y4m + ".xml";

//     std::string cmd1 = "ffmpeg -y -i " + ref_mp4 + " -pix_fmt yuv420p -f yuv4mpegpipe " + ref_y4m;
//     std::string cmd2 = "ffmpeg -y -i " + dist_mp4 + " -pix_fmt yuv420p -f yuv4mpegpipe " + dist_y4m;

//     run_command(cmd1);
//     run_command(cmd2);

//     std::string cmd_vmaf = "vmaf -r " + ref_y4m + " -d " + dist_y4m + " -out-fmt xml -o " + xml_out;
//     run_command(cmd_vmaf);

//     float score = extract_vmaf_score(xml_out);

//     std::remove(ref_y4m.c_str());
//     std::remove(dist_y4m.c_str());
//     std::remove(xml_out.c_str());

//     return score;
// }

float compute_vmaf(const string& query_path, const string& origin_path, const string& id, double start, double duration, int query_scale) {
    string upscale_path = id + "_upscaled.mp4";
    string downscale_path = "/dev/shm/" + id + "_tmp.mp4";
    string query_y4m = "/dev/shm/" + id + "_query.y4m";
    string dist_y4m = "/dev/shm/" + id + "_dist.y4m";

    if (!run_ffmpeg_clip(origin_path, upscale_path, start, duration)) return -1.0f;

    VideoCapture cap(upscale_path);
    if (!cap.isOpened()) return -1.0f;
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    cap.release();

    int downscaled_height = height / query_scale;
    if (!run_ffmpeg_resize(upscale_path, downscale_path, downscaled_height)) return -1.0f;

    if (!convert_to_y4m(query_path, query_y4m)) return -1.0f;
    if (!convert_to_y4m(downscale_path, dist_y4m)) return -1.0f;

    float score = get_vmaf_score(query_y4m, dist_y4m);

    fs::remove(query_y4m);
    fs::remove(dist_y4m);
    fs::remove(downscale_path);
    return score;
}

PYBIND11_MODULE(cmatcher, m) {
    m.def("find_coarse_match", &find_coarse_match, "Find coarse match between video hashes");
    m.def("find_subsequence_position", &find_subsequence_position, "Find coarse match between video hashes");
    m.def("get_video_hashes", &get_video_hashes, "Extract hashes and FPS from video");
    m.def("phash", &phash, "Extract phash");
    m.def("dhash", &dhash, "Extract dhash");
    m.def("find_query_start_frame",
          [](const std::string& original_path, const std::string& query_path) {
              int min_dist;
              int best_index = find_query_start_frame_fast(original_path, query_path, min_dist);
              return py::make_tuple(best_index, min_dist);
          },
          py::arg("original_path"),
          py::arg("query_path"),
          "Find the best matching start frame index between query and original video.");
    m.def("compute_vmaf", &compute_vmaf, "compute vmaf");
}
