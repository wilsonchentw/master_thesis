#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define NORM_SIZE 256
#define IMG_WIDTH NORM_SIZE
#define IMG_HEIGHT NORM_SIZE
#define IMSHOW(m, t) {imshow(#m, m); waitKey(t);}

using namespace cv;
using std::cout;
using std::endl;
using std::vector;

Mat normalizeCrop(Mat &image)
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

Mat im2hist(Mat &image)
{
    Mat hist, hist_64F, h;
    int dims=1, channels[]={0}, bins[]={32};
    float rgb_range[] = {0, 256};
    const float *hist_range[] = {rgb_range};

    vector<Mat> images;
    split(image, images);
    for( int i=0; i<images.size(); i++ ){
        calcHist(&images[i], 1, channels, Mat(), h, dims, bins, hist_range);
        hist.push_back(h);
    }

    hist.convertTo(hist_64F, CV_64F, 1.0/image.total());
    return hist_64F;
}

void matToLibsvm(int label, Mat &m, std::fstream &fout){
    CV_Assert(m.channels()==1 && m.type()==CV_64F);

    MatIterator_<double> bgn = m.begin<double>();
    MatIterator_<double> end = m.end<double>();

    fout << label;
    for( MatIterator_<double> it=bgn; it!=end; ++it ){
        if( (*it) > 0 ){
            fout << " " << (it-bgn)+1 << ":" << (*it);
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

        /*** Normalize Image Size ***/
        //image = normalizeCrop(image);
        //resize(image, image, Size(IMG_HEIGHT, IMG_WIDTH));

        /*** Spatial Pyramid Matching ***/
        Mat concat_hist;
        for(int level=2; level>=2; level--){
            int w=image.cols>>level, h=image.rows>>level;
            for(int y=0, dy=h; y+dy<=image.rows; y+=dy){
                for(int x=0, dx=w; x+dx<=image.cols; x+=dx){
                    Mat patch = image(Range(y, y+dy), Range(x, x+dx));
                    Mat hist = im2hist(patch);
                    concat_hist.push_back(hist);
                }
            }
        }


        /*** Output to libsvm training data file ***/
        //Mat norm_image;
        //image.convertTo(norm_image, CV_64FC3, (double)1.0/255);
        //norm_image = norm_image.reshape(1);
        //matToLibsvm(label, norm_image, fout);
        matToLibsvm(label, concat_hist, fout);
    }

    return 0;
}
