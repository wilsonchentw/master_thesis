#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv){

    int label;
    std::string path;
    std::ifstream fin("test_list");
    while(fin >> path >> label){
        cv::Mat image, norm_img;

        image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
        cv::resize(image, norm_img, cv::Size(128, 128));
    }
    fin.close();

    return 0;
}
