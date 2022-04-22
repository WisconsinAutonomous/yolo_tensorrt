#include "yolo_tensorrt/yolo_tensorrt.hpp"

#include <cv_bridge/cv_bridge.h>

#include <exception>
#include <unordered_map>
#include <algorithm>

namespace vision_detector {

static std::unordered_map<std::string, Precision> inference_precision_map = {
    {"INT8", INT8}, {"INT", INT8}, {"HALF", FP16}, {"FP16", FP16}, {"FLOAT", FP32}, {"FP32", FP32},
};

static std::unordered_map<std::string, ModelType> net_type_map = {
    {"YOLOV3", YOLOV3}, {"V3", YOLOV3}, {"YOLOV4", YOLOV4}, {"V4", YOLOV4}, {"YOLOV5", YOLOV5}, {"V5", YOLOV5},
};

Yolov5VisionDetector::Yolov5VisionDetector(const rclcpp::NodeOptions& options)
    : rclcpp::Node("yolov5_vision_detector", options), m_detector(std::make_unique<Detector>()) {
    // Converts a param to uppercase and finds it's entry in the passed map
    auto to_enum = [&](const std::string& name, const std::string& def,
                       std::unordered_map<std::string, auto> map) -> auto {
        std::string str = this->declare_parameter<std::string>(name, def);
        std::transform(str.begin(), str.end(), str.begin(), ::toupper);
        if (map.count(str) == 0) {
            RCLCPP_WARN(this->get_logger(), "%s is set to %s, which is not recognized. Setting to %s.", name.c_str(), str.c_str(), def.c_str());
            return map[def];
        }
        return map[str];
    };

    m_config.file_model_cfg = this->declare_parameter<std::string>("file_model_cfg", "");
    m_config.file_model_weights = this->declare_parameter<std::string>("file_model_weights", "");
    m_config.detect_thresh = this->declare_parameter<float>("detect_thresh", 0.5);
    m_config.inference_precision = to_enum("precision", "FLOAT", inference_precision_map);
    m_config.gpu_id = 0;
    m_config.net_type = to_enum("network", "V5", net_type_map);
    m_config.calibration_image_list_file_txt =
        this->declare_parameter<std::string>("calibration_image_list_file_txt", "");

    m_debug = this->declare_parameter<bool>("debug", false);

    m_detector->init(m_config);

    using std::placeholders::_1;
    m_img_subscriber = image_transport::create_subscription(this, "~/input/image",
                                                            std::bind(&Yolov5VisionDetector::imageCallback, this, _1),
                                                            "raw", rmw_qos_profile_sensor_data);

    m_rois_publisher = this->create_publisher<art_perception_msgs::msg::RoiArray>("~/output/rois", 1);
    m_debug_bb_publisher = image_transport::create_publisher(this, "~/output/debug/bb");
}

Yolov5VisionDetector::~Yolov5VisionDetector() {}

void Yolov5VisionDetector::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat image;
    cv::cvtColor(cv_ptr->image, image, cv::COLOR_BayerRG2RGB);
    cv::flip(image, image, -1);
    auto batch_result = detect(image);
    publish(batch_result, msg->header);
}

BatchResult Yolov5VisionDetector::detect(const cv::Mat& img) {
    std::vector<BatchResult> batch_result(1);

    m_detector->detect(std::vector<cv::Mat>{img}, batch_result);

    if (m_debug) {
        static std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 56, 56),  cv::Scalar(255, 157, 151), cv::Scalar(255, 112, 31),  cv::Scalar(255, 178, 29),
            cv::Scalar(207, 210, 49), cv::Scalar(72, 249, 10),   cv::Scalar(146, 204, 23),  cv::Scalar(61, 219, 134),
            cv::Scalar(26, 147, 52),  cv::Scalar(0, 212, 187),   cv::Scalar(44, 153, 168),  cv::Scalar(0, 194, 255),
            cv::Scalar(52, 69, 147),  cv::Scalar(100, 115, 255), cv::Scalar(0, 24, 236),    cv::Scalar(132, 56, 255),
            cv::Scalar(82, 0, 133),   cv::Scalar(203, 56, 255),  cv::Scalar(255, 149, 200), cv::Scalar(255, 55, 199),
        };

        cv::Mat debug_img = img.clone();
        for (auto& r : batch_result[0]) {
            cv::rectangle(debug_img, r.rect, colors[r.id], 4);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
            cv::putText(debug_img, stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 1.5, cv::Scalar(255, 255, 255),
                        2);
        }

        cv::namedWindow("image", cv::WINDOW_NORMAL);
        cv::imshow("image", debug_img);
        cv::resizeWindow("image", 1280, 960);
        cv::waitKey(1);
    }

    /*
        std_msgs::msg::Header header;
        std::string encoding = "bgr8";
        header.frame_id = "map";
        auto cv_img = cv_bridge::CvImage(header, encoding, debug_img);
        m_debug_bb_publisher.publish(*cv_img.toImageMsg());
    */

    return batch_result[0];
}

void Yolov5VisionDetector::publish(const BatchResult& batch_result, const std_msgs::msg::Header& header) {
    auto roi_array = std::make_unique<art_perception_msgs::msg::RoiArray>();
    roi_array->header = header;

    roi_array->rois.reserve(batch_result.size());
    for (auto& result : batch_result) {
        art_perception_msgs::msg::Roi roi;

        geometry_msgs::msg::Point bl;
        bl.x = result.rect.tl().x;
        bl.y = result.rect.br().y;
        roi.bottom_left = bl;

        geometry_msgs::msg::Point tr;
        bl.y = result.rect.tl().y;
        bl.x = result.rect.br().x;
        roi.top_right = tr;

        art_perception_msgs::msg::ObjectClassification classification;
        classification.confidence = result.prob;
        classification.classification = result.id;
        roi.classification = classification;

        roi_array->rois.push_back(roi);
    }

    m_rois_publisher->publish(std::move(roi_array));
}

}  // namespace vision_detector

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(vision_detector::Yolov5VisionDetector)
