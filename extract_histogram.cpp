#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define NORM_SIZE 256
#define IMG_WIDTH NORM_SIZE
#define IMG_HEIGHT NORM_SIZE

using namespace cv;
using std::cout;
using std::endl;

Mat preprocessing(Mat image)
{
    Mat norm_image;

    // Resize shorter side to NORM_SIZE
    double scale = (double)NORM_SIZE/std::min(image.cols, image.rows);
    resize(image, norm_image, Size(), scale, scale);

    // Crop for central part of image
    Point ofs((norm_image.cols-IMG_WIDTH)/2, (norm_image.rows-IMG_HEIGHT)/2);
    norm_image = norm_image(Rect(ofs, Size(IMG_WIDTH, IMG_HEIGHT)));

    return norm_image;
}

int main(int argc, char **argv)
{
    int label;
    std::string path;
    std::fstream fin(argv[1], std::fstream::in);
    std::fstream fout(argv[2], std::fstream::out);

    fout.precision(4);
    while(fin >> path >> label){
        Mat image = imread(path, CV_LOAD_IMAGE_COLOR);
        Mat norm_image = preprocessing(image);
    }

    return 0;
}
