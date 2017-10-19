//
// Created by dijiang on 10/11/17.
//

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "dbscan.h"
#include "iterative_dbscan.h"

DEFINE_string(input, "", "Input image file path");
DEFINE_string(output, "", "Output image file path");
DEFINE_string(input_dir, "", "Input image dir");
DEFINE_string(input_file, "", "Input image name list file");
DEFINE_string(output_dir, "", "Output image dir");
DEFINE_string(region_dir, "", "Output region image dir");
DEFINE_int32(min_sample, 3, "min_sample used in dbscan clustering");
DEFINE_double(eps, 1.5, "eps used in dbscan clustering");
DEFINE_double(roi_step, 0.2, "step for iterative dbscan clustering");
DEFINE_double(top, 0.0, "seed boundary top ratio");
DEFINE_double(bot, 0.4, "seed boundary bottom ratio");
DEFINE_int32(filter_thres, 500, "min points number in a cluster");
DEFINE_int32(width, 640, "image width");
DEFINE_int32(height, 300, "image height");
DEFINE_int32(bin_thres, 25, "binary threshold");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;
    FLAGS_logbufsecs  = 0; // output instantly
    FLAGS_max_log_size = 1024;

    std::ifstream fin(FLAGS_input_file.c_str());
    std::string line;

    LOG(INFO) << "construct DBSCAN" << std::flush;

//    DBSCAN lane_finder(FLAGS_min_sample, FLAGS_eps, FLAGS_bot);
    DBSCAN lane_finder(FLAGS_min_sample, FLAGS_eps, FLAGS_bot, FLAGS_top);
//    IterativeDBSCAN lane_finder(FLAGS_min_sample, FLAGS_eps);
//    lane_finder.SetROIStep(FLAGS_roi_step);
    lane_finder.SetFilterThreshold((unsigned int)FLAGS_filter_thres);

    clock_t start = clock();
    int img_cnt = 0;
    while (getline(fin, line)) {
        LOG(INFO) << "read " << line << std::flush;
        std::string ifname = FLAGS_input_dir + '/' + line;
        std::string ofname = FLAGS_output_dir + '/' + line;
        LOG(INFO) << ifname << std::flush;
        cv::Mat img = cv::imread(ifname);
        cv::Mat img_binary = cv::Mat::zeros(img.rows, img.cols, CV_8U);
        for (int i = 0; i < img.rows; i++)
            for (int j = 0; j < img.cols; j++) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                if (pixel[0] > FLAGS_bin_thres || pixel[1] > FLAGS_bin_thres || pixel[2] > FLAGS_bin_thres)
                    img_binary.at<uchar>(i, j) = 255;
            }

        LOG(INFO) << "setting image" << std::flush;
        lane_finder.SetImage(img_binary);
        lane_finder.Init();
        LOG(INFO) << "setting image done" << std::flush;


        LOG(INFO) << "run dbscan clustering and nearest neighbor classification" << std::flush;
        lane_finder.Run();
//    LOG(INFO) << "run dbscan clustering seeds";
//    lane_finder.ClusterSeeds();
//        LOG(INFO) << "run cluster filtering" << std::flush;
//        lane_finder.FilterCluster((unsigned int)FLAGS_filter_thres);

        cv::Mat output_img = lane_finder.GenVizImage();
        LOG(INFO) << "save image" << std::flush;
        LOG(INFO) << ofname << std::flush;
        cv::imwrite(ofname, output_img);
        img_cnt++;
    }

    double time_cost = (double)(clock() - start)/CLOCKS_PER_SEC*1000;

    LOG(INFO) << "time elapsed: " << time_cost
              << "\nnumber of images: " << img_cnt
              << "\naverage processing time: " << time_cost/img_cnt << std::flush;

    return 0;
}
