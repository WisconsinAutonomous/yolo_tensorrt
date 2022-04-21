#include "yolov5_tensorrt_vision_detector/yolov5_tensorrt_vision_detector.hpp"

#include <memory>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    auto node = std::make_shared<vision_detector::Yolov5VisionDetector>(options);

    RCLCPP_INFO(node->get_logger(), "yolov5_vision_detector started up!");
    // actually run the node
    rclcpp::spin(node);  // should not return
    rclcpp::shutdown();
    return 0;
}
