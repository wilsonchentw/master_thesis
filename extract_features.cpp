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

Mat normalizeCrop(Mat image)
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

Mat im2hist(Mat image)
{
    Mat hist;
    int dims=3, channels[]={0, 1, 2}, bins[]={16, 16, 16};
    float rgb_range[] = {0, 256};
    const float *hist_range[] = {rgb_range, rgb_range, rgb_range};

    // calculate histogram and normalize to 1 in summation
    calcHist(&image, 1, channels, Mat(), hist, dims, bins, hist_range);
    hist = hist / (IMG_WIDTH*IMG_HEIGHT);

    return hist;
}

void matToLibsvm(int label, Mat raw, std::fstream &fout){
    Mat m;
    raw.convertTo(m, CV_64F);

    fout << label;
    for( int i=0; i<m.total(); i++ ){
        if( m.at<double>(i) > 0 ){
            fout << " " << i+1 << ":" << m.at<double>(i);
        }
    }
    fout << endl;
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

        // normalize image
        Mat norm_image;
        resize(image, norm_image, Size(IMG_HEIGHT, IMG_WIDTH));
        norm_image = norm_image / 255;
        //norm_image = normalizeCrop(image);

        // calculate histogram
        //Mat hist = im2hist(norm_image);

        matToLibsvm(label, norm_image, fout);
    }


    return 0;
}
