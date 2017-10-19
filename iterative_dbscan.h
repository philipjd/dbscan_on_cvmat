
#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "basic.h"

class IterativeDBSCAN {
public:
    IterativeDBSCAN();
    IterativeDBSCAN(int min_sample, double eps);

    void SetROIStep(double step) { roi_step_ = step; }

    void Run();
    void SetImage(const cv::Mat& img);
    void ClusterROI();
    void ClassifyROI();
    void FilterCluster(unsigned int thres);
    cv::Mat GenVizImage();

private:
    void GenNeighborOffset();
    bool InitROI();

    bool IsROI(int row) { return (row >= roi_top_ && row < roi_bot_); }
    bool IsPreviousROI(int row) { return row >= roi_bot_; }
    bool IsValid(int row, int col);
    bool IsVisited(int row, int col) { return (label_.at<uchar>(row, col) > 0); }
    bool IsNoise(int row, int col) { return (label_.at<uchar>(row, col) == 255); }
    bool IterateROI();

    void QueryNeighbors(int row, int col, std::vector<cv::Point>& neighbor);

    void ExpandCluster(uchar c, const std::vector<cv::Point>& neighbor);

    int min_sample_ = 3;
    double eps_ = 1.5;
    double roi_step_ = 0.2;
    int roi_top_ = -1;
    int roi_bot_ = -1;
    bool roi_initial_ = false;

    int img_height_ = -1;
    int img_width_ = -1;

    cv::Mat img_;
    cv::Mat label_;
    cv::Mat tmp_label1_;     // used in nearest neighbor classification for non-seeds
    cv::Mat tmp_label2_;     // used in nearest neighbor classification for non-seeds
    std::vector<cv::Point> neighbor_offset_;

    uchar num_class_;

};

