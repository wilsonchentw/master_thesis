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

        int w=4, h=4, dx=2, dy=2;
        for(int y=0; y+h<=IMG_HEIGHT; y+=dy){
            for(int x=0; x+w<=IMG_WIDTH; x+=dx){
                patch = norm_img(cv::Range(y, y+h), cv::Range(x, x+w));

                // vectorize the patch
                if(!patch.isContinuous()){ patch = patch.clone(); }
                patch = patch.reshape(1, 1);

                // output for libsvm format
                fout << label;
                for( int idx=0; idx<patch.total(); idx++ ){
                    int value = patch.at<uchar>(idx);
                    if( value!=0 ){ 
                        fout << " " << idx+1 << ":" << (double)value/255; 
                    }
                }
                fout << std::endl;
            }
        }
    }
    fin.close();

    return 0;
}
