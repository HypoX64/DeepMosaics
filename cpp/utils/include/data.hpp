#ifndef DATA_H
#define DATA_H
#include <opencv2/opencv.hpp>

namespace data {
void normalize(cv::Mat& matrix, double mean = 0.5, double std = 0.5);

}  // namespace data

#endif