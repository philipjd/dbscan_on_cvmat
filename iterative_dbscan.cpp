#include <iostream>
#include <set>
#include <map>
#include <cmath>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "iterative_dbscan.h"

IterativeDBSCAN::IterativeDBSCAN()
{
    GenNeighborOffset();
}

IterativeDBSCAN::IterativeDBSCAN(int min_sample, double eps)
        : min_sample_(min_sample), eps_(eps)
{
    GenNeighborOffset();
}

bool IterativeDBSCAN::InitROI() {
    if (img_width_ == -1 || img_height_ == -1) {
        LOG(ERROR) << "image size has not been set yet!" << std::flush;
        return false;
    }
    roi_bot_ = img_height_;
    roi_top_ = roi_bot_ - (int)(roi_step_ * img_height_);
    roi_initial_ = true;
    return true;
}

bool IterativeDBSCAN::IterateROI() {
    if (roi_initial_) {
        roi_initial_ = false;
        return true;
    }
    roi_bot_ = roi_top_;
    if (roi_bot_ <= 0)
        return false;
    roi_top_ = std::max(0, roi_bot_ - (int)(roi_step_ * img_height_));
    return true;
}

void IterativeDBSCAN::SetImage(const cv::Mat& img)
{
    img_ = img;

    LOG(INFO) << "image channel number: " << img_.channels() << std::flush << std::flush;
    LOG(INFO) << "image size: (" << img_.rows << ", " << img_.cols << ")" << std::flush;
    img_height_ = img_.rows;
    img_width_ = img_.cols;
    label_ = cv::Mat::zeros(img_height_, img_width_, CV_8U);

    InitROI();
    num_class_ = 0;

    tmp_label1_ = cv::Mat::zeros(img_height_, img_width_, CV_8U);
    tmp_label2_ = cv::Mat::zeros(img_height_, img_width_, CV_8U);
}

void IterativeDBSCAN::GenNeighborOffset()
{
    for (int i = -1*(int)eps_; i <= (int)eps_; i++)
        for (int j = -1*(int)eps_; j <= (int)eps_; j++) {
            if (i == 0 && j == 0)
                continue;
            double dist = sqrt(i*i + j*j);
            if (dist <= eps_) {
                neighbor_offset_.emplace_back(i, j);
            }
        }

//    LOG(INFO) << "neighbor offsets: ";
//    for (const auto& offset: neighbor_offset_)
//        LOG(INFO) << '(' << offset.x << ", " << offset.y << ')';
}

bool IterativeDBSCAN::IsValid(int row, int col) {
    if (row >= 0 && row < img_height_ && col >= 0 && col < img_width_)
        return (img_.at<uchar>(row, col) > 0);
    else
        return false;
}

void IterativeDBSCAN::QueryNeighbors(int row, int col, std::vector<cv::Point> &neighbor) {
    for (const auto& offset: neighbor_offset_) {
        int nrow = row + offset.x;
        int ncol = col + offset.y;
        if (IsValid(nrow, ncol))
            neighbor.emplace_back(nrow, ncol);
    }
}

void IterativeDBSCAN::ClusterROI() {
    clock_t start = clock();
    for (int i = roi_bot_ - 1; i >= roi_top_; i--)  // for current roi
        for (int j = 0; j < img_width_; j++) {
            if (!IsValid(i, j) || IsVisited(i, j))
                continue;
            std::vector<cv::Point> neighbors;
            QueryNeighbors(i, j, neighbors);
            if (neighbors.size() < min_sample_)
                label_.at<uchar>(i, j) = 255;  // mark as noise
            else {
                num_class_++;
                label_.at<uchar>(i, j) = num_class_;
                ExpandCluster(num_class_, neighbors);
            }
        }

    LOG(INFO) << "roi clustering cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000 << std::flush;
}

void IterativeDBSCAN::ExpandCluster(uchar c, const std::vector<cv::Point>& neighbor) {
    for (const auto& pt: neighbor) {
        if (IsNoise(pt.x, pt.y))
            label_.at<uchar>(pt.x, pt.y) = c;
        else if (!IsVisited(pt.x, pt.y) && IsROI(pt.x)) {
            label_.at<uchar>(pt.x, pt.y) = c;
            std::vector<cv::Point> recur_neighbors;
            QueryNeighbors(pt.x, pt.y, recur_neighbors);
            if (recur_neighbors.size() >= min_sample_)
                ExpandCluster(c, recur_neighbors);
        }
    }
}

void IterativeDBSCAN::ClassifyROI() {
    // nearest neighbor classification
    clock_t start = clock();
    for (int i = roi_bot_ - 1; i >= roi_top_; i--) {  // for current roi
        // first loop, left -> right
        for (int j = 0; j < img_width_; j++) {
            if (!IsValid(i, j))
                continue;
            std::vector<cv::Point> neighbors;
            QueryNeighbors(i, j, neighbors);
            uchar l = 0;
            bool first_n = true;
            for (const auto& pt: neighbors) {
                if (!IsValid(pt.x, pt.y))
                    continue;

                uchar tmp_l;
                if (IsPreviousROI(pt.x))
                    tmp_l = label_.at<uchar>(pt.x, pt.y);
                else
                    tmp_l = tmp_label1_.at<uchar>(pt.x, pt.y);

                if (tmp_l == 0)
                    continue;

                if (first_n) {
                    l = tmp_l;
                    first_n = false;
                } else if (l != tmp_l) {    // more than 1 label in neighborhood, we don't classify the point
                    l = 0;
                    break;
                }
            }

            tmp_label1_.at<uchar>(i, j) = l;
        }
        // second loop, right -> left
        for (int j = img_width_-1; j >= 0; j--) {
            if (!IsValid(i, j))
                continue;
            std::vector<cv::Point> neighbors;
            QueryNeighbors(i, j, neighbors);
            uchar l = 0;
            bool first_n = true;
            for (const auto& pt: neighbors) {
                if (!IsValid(pt.x, pt.y))
                    continue;

                uchar tmp_l;
                if (IsPreviousROI(pt.x))
                    tmp_l = label_.at<uchar>(pt.x, pt.y);
                else
                    tmp_l = tmp_label2_.at<uchar>(pt.x, pt.y);

                if (tmp_l == 0)
                    continue;

                if (first_n) {
                    l = tmp_l;
                    first_n = false;
                } else if (l != tmp_l) {    // more than 1 label in neighborhood, we don't classify the point
                    l = 0;
                    break;
                }
            }
            tmp_label2_.at<uchar>(i, j) = l;
            // assign label according to results of both first & second loops
            uchar la1 = tmp_label1_.at<uchar>(i, j);
            uchar final_la = 0;
            if (la1 == 0)
                final_la = l;
            else if (l == 0)
                final_la = la1;
            else if (la1 == l)
                final_la = l;
            label_.at<uchar>(i, j) = final_la;
        }
    }

    LOG(INFO) << "roi classification cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000 << std::flush;
}

void IterativeDBSCAN::Run() {
    clock_t start = clock();
    while ( IterateROI() ) {
        if ( num_class_ > 0 ) {
            ClassifyROI();
        }
        ClusterROI();
    }

    LOG(INFO) << "number of clusters: " << (int)num_class_ << std::flush;
    LOG(INFO) << "total cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000 << std::flush;
}

cv::Mat IterativeDBSCAN::GenVizImage() {
    cv::Mat viz_img = cv::Mat::zeros(img_height_, img_width_, CV_8UC3);

    for (int i = 0; i < img_height_; i++)
        for (int j = 0; j < img_width_; j++) {
            uchar la = label_.at<uchar>(i, j);
            if (la > 0 && la < 255) {
                cv::Vec3b color = COLORS[(la - 1)%11];
                viz_img.at<cv::Vec3b>(i, j) = COLORS[(la - 1)%11];
            }
        }

    return viz_img;
}

void IterativeDBSCAN::FilterCluster(unsigned int thres) {
    clock_t start = clock();
    unsigned int* label_cnt = new unsigned int[num_class_]();
    for (int i = 0; i < img_height_; i++)
        for (int j = 0; j < img_width_; j++) {
            uchar la = label_.data[img_width_ * i + j];
            if (la > 0 && la < 255) {
                label_cnt[la - 1] += 1;
            }
        }

    std::map<uchar, uchar> label_filter_map;
    uchar new_la = 0;
    for (uchar i = 0; i < num_class_; i++) {
        if (label_cnt[i] >= thres) {
            new_la++;
            label_filter_map[i + 1] = new_la;
        }
    }
    LOG(INFO) << (int)new_la << " clusters remained" << std::flush;
    LOG(INFO) << "filtered " << num_class_ - new_la << " clusters" << std::flush;

    for (int i = 0; i < img_height_; i++)
        for (int j = 0; j < img_width_; j++) {
            uchar la = label_.data[img_width_ * i + j];
            if (la > 0) {
                std::map<uchar, uchar>::const_iterator it = label_filter_map.find(la);
                if (it != label_filter_map.end())
                    label_.data[img_width_ * i + j] = it->second;
                else
                    label_.data[img_width_ * i + j] = 0;
            }
        }

    delete [] label_cnt;

    LOG(INFO) << "filtering cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000 << std::flush;
}

