#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define IMG_WIDTH 64
#define IMG_HEIGHT 64

int main(int argc, char **argv)
{
    int label;
    std::string path;
    std::fstream fin(argv[1], std::fstream::in);
    std::fstream fout(argv[2], std::fstream::out);

    fout.precision(4);
    while(fin >> path >> label){
        cv::Mat image, norm_img, patch;
        image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
        cv::resize(image, norm_img, cv::Size(IMG_HEIGHT, IMG_WIDTH));

        int w=4, h=4, step=2;
        for( int i=0; i+h<=IMG_HEIGHT; i+=step ){
            for( int j=0; j+w<=IMG_WIDTH; j+=step ){
                norm_img(cv::Range(i, i+h), cv::Range(j, j+w)).copyTo(patch);
                patch = patch.reshape(1,1);

                fout << label;
                for( int k=0; k<patch.total(); k++ ){
                    int value = patch.at<uchar>(k);
                    if( value!=0 )
                        fout << " " << k+1 << ":" << (double)value/255;
                }
                fout << std::endl;
            }
        }
        // std::cout << path << std::endl;
    }
    fin.close();

    return 0;
}
