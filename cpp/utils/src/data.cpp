#include "data.hpp"
#include <opencv2/opencv.hpp>

namespace data {
void normalize(cv::Mat& matrix, double mean, double std) {
    // matrix = (matrix / 255.0 - mean) / std;
    matrix = matrix / (255.0 * std) - mean / std;
}

}  // namespace data