#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "basic.h"

class DBSCAN {
public:
    DBSCAN();
    DBSCAN(int min_sample, double eps);
    DBSCAN(int min_sample, double eps, double seed_bnd_bot);
    DBSCAN(int min_sample, double eps, double seed_bnd_bot, double seed_bnd_top);

    void Run();
    void SetImage(const cv::Mat& img);
    void Init();
    void ClusterSeeds();
    void ClassifyNonseeds();
    void FilterCluster(unsigned int thres);
    cv::Mat GenVizImage();


    void SetFilterThreshold(unsigned int thres) { filter_thres_ = thres; }


private:
    void ClassifyNonseedsRow(int i);

    void GenNeighborOffset();

    bool IsSeed(int row);
    bool IsValid(int row, int col);
    bool IsVisited(int row, int col);
    bool IsNoise(int row, int col);

    void Swap();

    void QueryNeighbors(int row, int col, std::vector<cv::Point>& neighbor);

    void ExpandCluster(uchar c, const std::vector<cv::Point>& neighbor);

    int min_sample_ = 3;
    double eps_ = 1.5;
    double seed_bnd_top_ratio_ = -1.0;
    double seed_bnd_bot_ratio_ = -1.0;
    int seed_bnd_top_ = -1;
    int seed_bnd_bot_ = -1;
    bool seed_region_ = false;
    bool swap_seed_region_ = false;
    bool swapped_ = true;

    int img_height_;
    int img_width_;
    unsigned int filter_thres_;

    cv::Mat img_;
    cv::Mat label_;
    cv::Mat tmp_label1_;     // used in nearest neighbor classification for non-seeds
    cv::Mat tmp_label2_;     // used in nearest neighbor classification for non-seeds
    std::vector<cv::Point> neighbor_offset_;

    uchar num_class_ = 0;

};
