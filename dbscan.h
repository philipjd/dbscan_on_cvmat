#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

const cv::Vec3b COLORS[11] = {
    cv::Vec3b(0,0,255),
    cv::Vec3b(0,255,0),
    cv::Vec3b(255,0,0),
    cv::Vec3b(255,255,255),
    cv::Vec3b(255,255,0),
    cv::Vec3b(0,255,255),
    cv::Vec3b(255,0,255),
    cv::Vec3b(255,0,128),
    cv::Vec3b(255,64,255),
    cv::Vec3b(128,128,255),
    cv::Vec3b(255,128,128)
};

class DBSCAN {
public:
    DBSCAN();
    DBSCAN(int min_sample, double eps);
    DBSCAN(int min_sample, double eps,
           double seed_bnd_left, double seed_bnd_right, double seed_bnd_top, double seed_bnd_bot);

    void Run();
    void SetImage(const cv::Mat& img);
    void ClusterSeeds();
    void FilterCluster(unsigned int thres);
    cv::Mat GenVizImage();
    cv::Mat GetNonSeedRegion();


private:
    void GenNeighborOffset();

    bool IsSeed(int row, int col);
    bool IsValid(int row, int col);
    bool IsVisited(int row, int col);
    bool IsNoise(int row, int col);

    void QueryNeighbors(int row, int col, std::vector<cv::Point>& neighbor);

    void ExpandCluster(uchar c, const std::vector<cv::Point>& neighbor);

    int min_sample_ = 3;
    double eps_ = 1.5;
    double seed_bnd_left_ratio_ = -1.0;
    double seed_bnd_right_ratio_ = -1.0;
    double seed_bnd_top_ratio_ = -1.0;
    double seed_bnd_bot_ratio_ = -1.0;
    int seed_bnd_left_ = -1;
    int seed_bnd_right_ = -1;
    int seed_bnd_top_ = -1;
    int seed_bnd_bot_ = -1;
    bool seed_region_ = false;

    int img_height_;
    int img_width_;

    cv::Mat img_;
    cv::Mat label_;
    cv::Mat tmp_label1_;     // used in nearest neighbor classification for non-seeds
    cv::Mat tmp_label2_;     // used in nearest neighbor classification for non-seeds
    std::vector<cv::Point> neighbor_offset_;

    uchar num_class_;

};
