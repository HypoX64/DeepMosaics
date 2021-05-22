#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <list>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "data.hpp"
#include "util.hpp"

int main() {
    std::string path = util::current_path();

    std::string net_path = "../res/models/mosaic_position.pth";
    std::string img_path = "../res/test_media/face/d.jpg";

    cv::Mat img = cv::imread(img_path);
    cv::resize(img, img, cv::Size(360, 360), 2);
    // img.convertTo(img, CV_32F);
    torch::Tensor img_tensor =
        torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.div(255);
    std::cout << img_tensor.sizes() << "\n";

    // end = clock();
    // dur = (double)(end - start);
    // printf("Use Time:%f\n", (dur / CLOCKS_PER_SEC));

    // std::string net_path = "../res/models/mosaic_position.pt";
    // torch::jit::script::Module net;
    // try{
    //     // if (!isfile(net_path)){
    //     //     std::cerr<<"model does not exist\n";
    //     // }

    //     net = torch::jit::load(net_path);
    // }
    // catch(const std::exception& e){
    //     std::cerr << "error loading the model\n";
    //     return -1;
    // }

    // torch::Tensor example = torch::ones({1,3,360,360});
    // torch::Tensor output = net.forward({example}).toTensor();
    // std::cout<<"ok"<<std::endl;
}