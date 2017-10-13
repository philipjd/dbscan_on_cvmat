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

DBSCAN::DBSCAN(int min_sample, double eps)
        : min_sample_(min_sample), eps_(eps)
{
    GenNeighborOffset();
}

DBSCAN::DBSCAN(int min_sample, double eps,
               double seed_bnd_left, double seed_bnd_right, double seed_bnd_top, double seed_bnd_bot)
        : min_sample_(min_sample), eps_(eps),
          seed_bnd_left_ratio_(seed_bnd_left), seed_bnd_right_ratio_(seed_bnd_right),
          seed_bnd_top_ratio_(seed_bnd_top), seed_bnd_bot_ratio_(seed_bnd_bot)
{
    CHECK(seed_bnd_left_ratio_ >= 0 && seed_bnd_left_ratio_ <= 1.0) << "seed left bound should between [0, 1]!";
    CHECK(seed_bnd_right_ratio_ >= 0 && seed_bnd_right_ratio_ <= 1.0) << "seed right bound should between [0, 1]!";
    CHECK(seed_bnd_top_ratio_ >= 0 && seed_bnd_top_ratio_ <= 1.0) << "seed top bound should between [0, 1]!";
    CHECK(seed_bnd_bot_ratio_ >= 0 && seed_bnd_bot_ratio_ <= 1.0) << "seed bot bound should between [0, 1]!";
    seed_region_ = true;

    GenNeighborOffset();
}

void DBSCAN::SetImage(const cv::Mat& img)
{
    img_ = img;

    LOG(INFO) << "image channel number: " << img_.channels() << std::flush << std::flush;
    LOG(INFO) << "image size: (" << img_.rows << ", " << img_.cols << ")" << std::flush;
    img_height_ = img_.rows;
    img_width_ = img_.cols;
    label_ = cv::Mat::zeros(img_height_, img_width_, CV_8U);

    if (seed_region_) {
        seed_bnd_left_ = (int)img_width_ * seed_bnd_left_ratio_;
        seed_bnd_right_ = (int)img_width_ * seed_bnd_right_ratio_;
        seed_bnd_top_ = (int)img_height_ * seed_bnd_top_ratio_;
        seed_bnd_bot_ = (int)img_height_ * seed_bnd_bot_ratio_;
        LOG(INFO) << "seed boundary: [(" << seed_bnd_top_ << ", " << seed_bnd_left_
                  << "), (" << seed_bnd_bot_ << ", " << seed_bnd_right_ << ")]" << std::flush << std::flush;
        tmp_label1_ = cv::Mat::zeros(seed_bnd_bot_ - seed_bnd_top_, seed_bnd_right_ - seed_bnd_left_, CV_8U);
        tmp_label2_ = cv::Mat::zeros(seed_bnd_bot_ - seed_bnd_top_, seed_bnd_right_ - seed_bnd_left_, CV_8U);
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
        return (img_.at<uchar>(row, col) > 0);
    else
        return false;
}

bool DBSCAN::IsVisited(int row, int col) {
    return (label_.at<uchar>(row, col) > 0);
}

bool DBSCAN::IsNoise(int row, int col) {
    return (label_.at<uchar>(row, col) == 255);
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
    uchar c = 0;
    for (int i = 0; i < img_height_; i++)
        for (int j = 0; j < img_width_; j++) {
            if (!IsValid(i, j) || !IsSeed(i, j) || IsVisited(i, j))
                continue;
            std::vector<cv::Point> neighbors;
            QueryNeighbors(i, j, neighbors);
            if (neighbors.size() < min_sample_)
                label_.at<uchar>(i, j) = 255;  // mark as noise
            else {
                c++;
                label_.at<uchar>(i, j) = c;
                ExpandCluster(c, neighbors);
            }
        }

    num_class_ = c;

    LOG(INFO) << "number of clusters: " << (int)num_class_ << std::flush;
    LOG(INFO) << "seeds clustering cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000 << std::flush;
}

void DBSCAN::ExpandCluster(uchar c, const std::vector<cv::Point>& neighbor) {
    for (const auto& pt: neighbor) {
        if (IsNoise(pt.x, pt.y))
            label_.at<uchar>(pt.x, pt.y) = c;
        else if (!IsVisited(pt.x, pt.y) && IsSeed(pt.x, pt.y)) {
            label_.at<uchar>(pt.x, pt.y) = c;
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
        for (int i = seed_bnd_bot_ - 1; i >= seed_bnd_top_; i--) {
            // first loop, left -> right
            for (int j = seed_bnd_left_; j < seed_bnd_right_; j++) {
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
                    if (IsSeed(pt.x, pt.y))
                        tmp_l = label_.at<uchar>(pt.x, pt.y);
                    else
                        tmp_l = tmp_label1_.at<uchar>(pt.x - seed_bnd_top_, pt.y - seed_bnd_left_);

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

                tmp_label1_.at<uchar>(i - seed_bnd_top_, j - seed_bnd_left_) = l;
            }
            // second loop, right -> left
            for (int j = seed_bnd_right_-1; j >= seed_bnd_left_; j--) {
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
                    if (IsSeed(pt.x, pt.y))
//                        tmp_l = label_.data[img_width_ * pt.x + pt.y];
                        tmp_l = label_.at<uchar>(pt.x, pt.y);
                    else
                        tmp_l = tmp_label2_.at<uchar>(pt.x - seed_bnd_top_, pt.y - seed_bnd_left_);

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
                tmp_label2_.at<uchar>(i - seed_bnd_top_, j - seed_bnd_left_) = l;
                // assign label according to results of both first & second loops
                uchar la1 = tmp_label1_.at<uchar>(i - seed_bnd_top_, j - seed_bnd_left_);
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
    }

    LOG(INFO) << "total cost: " << (double)(clock() - start)/CLOCKS_PER_SEC*1000 << std::flush;
}

cv::Mat DBSCAN::GenVizImage() {
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

cv::Mat DBSCAN::GetNonSeedRegion() {
    cv::Mat region = cv::Mat::zeros(seed_bnd_bot_ - seed_bnd_top_, seed_bnd_right_ - seed_bnd_left_, CV_8U);
    for (int i = 0; i < seed_bnd_bot_ - seed_bnd_top_; i++)
        for (int j = 0; j < seed_bnd_right_ - seed_bnd_left_; j++) {
            uchar la = img_.at<uchar>(i + seed_bnd_top_, j + seed_bnd_left_);
            region.at<uchar>(i, j) = la;
        }

    return region;
}

void DBSCAN::FilterCluster(unsigned int thres) {
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
    LOG(INFO) << (int)num_class_ << " clusters remained" << std::flush;
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

