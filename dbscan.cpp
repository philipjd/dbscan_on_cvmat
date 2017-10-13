#include <iostream>
#include <set>
#include <map>
#include <cmath>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "dbscan.h"

DBSCAN::DBSCAN()
{
    GenNeighborOffset();
}

DBSCAN::DBSCAN(int min_sample, double eps, int height, int width)
        : min_sample_(min_sample), eps_(eps),
          img_width_(width), img_height_(height)
{
    GenNeighborOffset();
    InitMat();
}

DBSCAN::DBSCAN(int min_sample, double eps, int height, int width,
               double seed_bnd_left, double seed_bnd_right, double seed_bnd_top, double seed_bnd_bot)
        : min_sample_(min_sample), eps_(eps), img_width_(width), img_height_(height),
          seed_bnd_left_ratio_(seed_bnd_left), seed_bnd_right_ratio_(seed_bnd_right),
          seed_bnd_top_ratio_(seed_bnd_top), seed_bnd_bot_ratio_(seed_bnd_bot)
{
    CHECK(seed_bnd_left_ratio_ >= 0 && seed_bnd_left_ratio_ <= 1.0) << "seed left bound should between [0, 1]!";
    CHECK(seed_bnd_right_ratio_ >= 0 && seed_bnd_right_ratio_ <= 1.0) << "seed right bound should between [0, 1]!";
    CHECK(seed_bnd_top_ratio_ >= 0 && seed_bnd_top_ratio_ <= 1.0) << "seed top bound should between [0, 1]!";
    CHECK(seed_bnd_bot_ratio_ >= 0 && seed_bnd_bot_ratio_ <= 1.0) << "seed bot bound should between [0, 1]!";
    seed_region_ = true;
    GenNeighborOffset();
    InitMat();
}

void DBSCAN::SetImage(const cv::Mat& img)
{
    img_ = img;
    CHECK(img_height_ == img_.rows && img_width_ == img_.cols) << "Image size incorrect: ("
                                        << img_.rows << ", " << img_.cols << ")\nexpected: ("
                                        << img_height_ << ", " << img_width_ << ")";
//    label_ = cv::Mat::zeros(img_height_, img_width_, CV_8U);
    label_ = cv::Scalar(0);
    LOG(INFO) << "image channel number: " << img_.channels();
    LOG(INFO) << "image size: (" << img_.rows << ", " << img_.cols << ")";
    if (seed_region_) {
        seed_bnd_left_ = (int)img_width_ * seed_bnd_left_ratio_;
        seed_bnd_right_ = (int)img_width_ * seed_bnd_right_ratio_;
        seed_bnd_top_ = (int)img_height_ * seed_bnd_top_ratio_;
        seed_bnd_bot_ = (int)img_height_ * seed_bnd_bot_ratio_;
        LOG(INFO) << "seed boundary: [(" << seed_bnd_left_ << ", " << seed_bnd_top_
                  << "), (" << seed_bnd_right_ << ", " << seed_bnd_bot_ << ")]";
//        tmp_label1_ = cv::Mat::zeros(seed_bnd_bot_ - seed_bnd_top_, seed_bnd_right_ - seed_bnd_left_, CV_8U);
//        tmp_label2_ = cv::Mat::zeros(seed_bnd_bot_ - seed_bnd_top_, seed_bnd_right_ - seed_bnd_left_, CV_8U);
        tmp_label1_ = cv::Scalar(0);
        tmp_label2_ = cv::Scalar(0);
    }
}

void DBSCAN::GenNeighborOffset()
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

void DBSCAN::InitMat() {
    label_.create(img_height_, img_width_, CV_8U);
    tmp_label1_.create(img_height_, img_width_, CV_8U);
    tmp_label2_.create(img_height_, img_width_, CV_8U);
    viz_img_.create(img_height_, img_width_, CV_8UC3);
}

bool DBSCAN::IsSeed(int row, int col) {
    if (!seed_region_)
        return true;
    if (row >= seed_bnd_top_ && row < seed_bnd_bot_ && col >= seed_bnd_left_ && col < seed_bnd_right_)
        return false;
    else
        return true;
}

bool DBSCAN::IsValid(int row, int col) {
    if (row >= 0 && row < img_height_ && col >= 0 && col < img_width_)
        return (img_.data[img_width_ * row + col] > 0);
    else
        return false;
}

bool DBSCAN::IsVisited(int row, int col) {
    return (label_.data[img_width_ * row + col] > 0);
}

bool DBSCAN::IsNoise(int row, int col) {
    return (label_.data[img_width_ * row + col] == 255);
}

void DBSCAN::QueryNeighbors(int row, int col, std::vector<cv::Point> &neighbor) {
    for (const auto& offset: neighbor_offset_) {
        int nrow = row + offset.x;
        int ncol = col + offset.y;
        if (IsValid(nrow, ncol))
            neighbor.emplace_back(nrow, ncol);
    }
}

void DBSCAN::ClusterSeeds() {
    clock_t start = clock();
    unsigned char c = 0;
    for (int i = 0; i < img_height_; i++)
        for (int j = 0; j < img_width_; j++) {
            if (!IsValid(i, j) || !IsSeed(i, j) || IsVisited(i, j))
                continue;
            std::vector<cv::Point> neighbors;
            QueryNeighbors(i, j, neighbors);
            if (neighbors.size() < min_sample_)
                label_.data[img_width_ * i + j] = 255;  // mark as noise
            else {
                c++;
                label_.data[img_width_ * i + j] = c;
                ExpandCluster(c, neighbors);
            }
        }

    num_class_ = c;

    LOG(INFO) << "seeds clustering cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000;
}

void DBSCAN::ExpandCluster(unsigned char c, const std::vector<cv::Point>& neighbor) {
    for (const auto& pt: neighbor) {
        if (IsNoise(pt.x, pt.y))
            label_.data[img_width_ * pt.x + pt.y] = c;
        else if (!IsVisited(pt.x, pt.y) && IsSeed(pt.x, pt.y)) {
            label_.data[img_width_ * pt.x + pt.y] = c;
            std::vector<cv::Point> recur_neighbors;
            QueryNeighbors(pt.x, pt.y, recur_neighbors);
            if (recur_neighbors.size() >= min_sample_)
                ExpandCluster(c, recur_neighbors);
        }
    }
}

void DBSCAN::Run() {
    clock_t start = clock();
    ClusterSeeds();
    if (seed_region_) {
        for (int i = seed_bnd_bot_+1; i >= seed_bnd_top_; i--) {
            // first loop, left -> right
            for (int j = seed_bnd_left_; j < seed_bnd_right_; j++) {
                if (!IsValid(i, j))
                    continue;
                std::vector<cv::Point> neighbors;
                QueryNeighbors(i, j, neighbors);
                unsigned char l = 0;
                bool first_n = true;
                for (const auto& pt: neighbors) {
                    if (!IsValid(pt.x, pt.y))
                        continue;

                    unsigned char tmp_l;
                    if (IsSeed(pt.x, pt.y))
                        tmp_l = label_.data[img_width_ * pt.x + pt.y];
                    else
                        tmp_l = tmp_label1_.data[(seed_bnd_right_ - seed_bnd_left_) * (pt.x - seed_bnd_top_)
                                                 + (pt.y - seed_bnd_left_)];
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
                tmp_label1_.data[(seed_bnd_right_ - seed_bnd_left_) * (i - seed_bnd_top_)
                                 + (j - seed_bnd_left_)] = l;
            }
            // second loop, right -> left
            for (int j = seed_bnd_right_-1; j >= seed_bnd_left_; j--) {
                if (!IsValid(i, j))
                    continue;
                std::vector<cv::Point> neighbors;
                QueryNeighbors(i, j, neighbors);
                unsigned char l = 0;
                bool first_n = true;
                for (const auto& pt: neighbors) {
                    if (!IsValid(pt.x, pt.y))
                        continue;

                    unsigned char tmp_l;
                    if (IsSeed(pt.x, pt.y))
                        tmp_l = label_.data[img_width_ * pt.x + pt.y];
                    else
                        tmp_l = tmp_label2_.data[(seed_bnd_right_ - seed_bnd_left_) * (pt.x - seed_bnd_top_)
                                                 + (pt.y - seed_bnd_left_)];
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
                tmp_label2_.data[(seed_bnd_right_ - seed_bnd_left_) * (i - seed_bnd_top_)
                                 + (j - seed_bnd_left_)] = l;
                // assign label according to results of both first & second loops
                unsigned char la1 = tmp_label1_.data[(seed_bnd_right_ - seed_bnd_left_) * (i - seed_bnd_top_)
                                                    + (j - seed_bnd_left_)];
                unsigned char final_la = 0;
                if (la1 == 0)
                    final_la = l;
                else if (l == 0)
                    final_la = la1;
                else if (la1 == l)
                    final_la = l;
                label_.data[img_width_ * i + j] = final_la;
            }
        }
    }

    LOG(INFO) << "total cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000;
}

cv::Mat DBSCAN::GenVizImage() {
//    viz_img_ = cv::Mat::zeros(img_height_, img_width_, CV_8UC3);
    viz_img_ = cv::Scalar(0, 0, 0);
    for (int i = 0; i < img_height_; i++)
        for (int j = 0; j < img_width_; j++) {
            unsigned char la = label_.data[img_width_ * i + j];
            if (la > 0 && la < 255) {
                cv::Scalar color = COLORS[(la - 1)%11];
                viz_img_.data[3 * (img_width_ * i + j)] = color[0];
                viz_img_.data[3 * (img_width_ * i + j) + 1] = color[1];
                viz_img_.data[3 * (img_width_ * i + j) + 2] = color[2];
            }
        }

    return viz_img_;
}

//cv::Mat DBSCAN::GetNonSeedRegion() {
//    region_ = cv::Mat::zeros(seed_bnd_bot_ - seed_bnd_top_, seed_bnd_right_ - seed_bnd_left_, CV_8U);
//    for (int i = 0; i < seed_bnd_bot_ - seed_bnd_top_; i++)
//        for (int j = 0; j < seed_bnd_right_ - seed_bnd_left_; j++) {
//            unsigned char la = img_.data[img_width_ * (i + seed_bnd_top_) + j + seed_bnd_left_];
//            region_.data[(seed_bnd_right_ - seed_bnd_left_) * i + j] = la;
//        }
//
//    return region_;
//}

void DBSCAN::FilterCluster(unsigned int thres) {
    clock_t start = clock();
    unsigned int* label_cnt = new unsigned int[num_class_]();
    for (int i = 0; i < img_height_; i++)
        for (int j = 0; j < img_width_; j++) {
            unsigned char la = label_.data[img_width_ * i + j];
            if (la > 0 && la < 255) {
                label_cnt[la - 1] += 1;
            }
        }

    std::map<unsigned char, unsigned char> label_filter_map;
    unsigned char new_la = 0;
    for (unsigned char i = 0; i < num_class_; i++) {
        if (label_cnt[i] >= thres) {
            new_la++;
            label_filter_map[i + 1] = new_la;
        }
    }

    for (int i = 0; i < img_height_; i++)
        for (int j = 0; j < img_width_; j++) {
            unsigned char la = label_.data[img_width_ * i + j];
            if (la > 0) {
                std::map<unsigned char, unsigned char>::const_iterator it = label_filter_map.find(la);
                if (it != label_filter_map.end())
                    label_.data[img_width_ * i + j] = it->second;
                else
                    label_.data[img_width_ * i + j] = 0;
            }
        }

    delete [] label_cnt;

    LOG(INFO) << "filtered " << num_class_ - new_la << " clusters";
    LOG(INFO) << "filtering cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000;
}

