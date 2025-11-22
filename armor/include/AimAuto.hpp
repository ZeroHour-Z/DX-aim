#ifndef AIMAUTO
#define AIMAUTO
#include "globalParam.hpp"
#include "opencv2/core/mat.hpp"
#include "tracker.hpp"
#include <camera.hpp>
#include <chrono>
#include <cstdint>
#include <detector.hpp>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>

#ifdef APRILTAG
#include "ApriltagDetector.hpp"
#endif

class AimAuto
{
private:
    GlobalParam *gp;
    Detector *detector;
    Tracker *tracker;
    Armor last_armor;
    void pnp_solve(UnsolvedArmor &armor, Translator &ts, cv::Mat &src, Armor &tar, int number);
    void draw_armor_back(cv::Mat &src, Armor &armor, int number, cv::Scalar color = cv::Scalar(255, 255, 255));
    void optimizeYawZ(
        const std::vector<cv::Point3f>& objPoints,
        const std ::vector<cv::Point2f>& imgPoints,
        double known_x,
        double known_y,
        double known_z,
        double camera_yaw,
        double camera_pitch,
        double &yaw,
        const cv::Mat& K,
        const cv::Mat& dist
    );
public:
    AimAuto(GlobalParam *gp);
    ~AimAuto();
    void auto_aim(cv::Mat &src, Translator &ts, double dt,bool &have_armor);
};


#endif // AIMAUTO