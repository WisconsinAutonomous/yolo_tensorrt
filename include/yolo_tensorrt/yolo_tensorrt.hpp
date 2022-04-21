#pragma once

// ROS includes
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "wauto_perception_msgs/msg/roi_array.hpp"

// STL includes
#include <memory>

// Thirdparty includes
#include <image_transport/image_transport.hpp>
#include "opencv2/opencv.hpp"
#include "class_detector.h"

namespace vision_detector {

class Yolov5VisionDetector : public rclcpp::Node {
  public:
    explicit Yolov5VisionDetector(const rclcpp::NodeOptions& options);
    ~Yolov5VisionDetector();

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    BatchResult detect(const cv::Mat& img);

    void publish(const BatchResult& batch_result, const std_msgs::msg::Header& msg);

  private:
    bool m_debug;

    std::unique_ptr<Detector> m_detector;
    Config m_config;

    image_transport::Subscriber m_img_subscriber;

    rclcpp::Publisher<wauto_perception_msgs::msg::RoiArray>::SharedPtr m_rois_publisher;
    image_transport::Publisher m_debug_bb_publisher;
};

}  // namespace vision_detector
