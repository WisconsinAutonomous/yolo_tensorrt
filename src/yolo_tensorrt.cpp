#include "yolo_tensorrt/yolo_tensorrt.hpp"

#include <cv_bridge/cv_bridge.h>

#include "wauto_perception_msgs/msg/object_classification.hpp"

#include <exception>
#include <unordered_map>
#include <algorithm>
#include <assert.h>

namespace vision_detector {

static std::unordered_map<std::string, Precision> inference_precision_map = {
    {"INT8", INT8}, {"INT", INT8}, {"HALF", FP16}, {"FP16", FP16}, {"FLOAT", FP32}, {"FP32", FP32},
};

static std::unordered_map<std::string, ModelType> net_type_map = {
    {"YOLOV3", YOLOV3}, {"V3", YOLOV3}, {"YOLOV4", YOLOV4}, {"V4", YOLOV4}, {"YOLOV5", YOLOV5}, {"V5", YOLOV5},
};

// HACK!!!!!
// TRAINING NEEDS TO OUTPUT THE ID THAT CORRESPONDS TO THE CORRECT CLASSIFICATION
static std::unordered_map<std::string, std::unordered_map<uint8_t, uint8_t>> class_id_map = {
    {"yolov5m6",
    {
        {0, wauto_perception_msgs::msg::ObjectClassification::WA_OBJECT_CLASSIFICATION_PEDESTRIAN},
        {1, wauto_perception_msgs::msg::ObjectClassification::WA_OBJECT_CLASSIFICATION_CAR},
        {2, wauto_perception_msgs::msg::ObjectClassification::WA_OBJECT_CLASSIFICATION_TRAFFIC_LIGHT},
        {3, wauto_perception_msgs::msg::ObjectClassification::WA_OBJECT_CLASSIFICATION_TRAFFIC_SIGN},
        {4, wauto_perception_msgs::msg::ObjectClassification::WA_OBJECT_CLASSIFICATION_DEER},
    }},
    {"barrels",
    {
        {0, wauto_perception_msgs::msg::ObjectClassification::WA_OBJECT_CLASSIFICATION_BARREL},
        {1, wauto_perception_msgs::msg::ObjectClassification::WA_OBJECT_CLASSIFICATION_BARRICADE},
    }}
};

Yolov5VisionDetector::Yolov5VisionDetector(const rclcpp::NodeOptions& options)
    : rclcpp::Node("yolov5_vision_detector", options) {
    // Converts a param to uppercase and finds it's entry in the passed map
    auto to_enum = [&](std::string& str, const std::string& def,
                       std::unordered_map<std::string, auto> map) -> auto {
        std::transform(str.begin(), str.end(), str.begin(), ::toupper);
        if (map.count(str) == 0) {
            RCLCPP_WARN(this->get_logger(), "Unrecognized parameter set to %s, which is not recognized. Setting to %s.", str.c_str(), def.c_str());
            return map[def];
        }
        return map[str];
    };
    
    // constants
    const auto GPU_ID = 0;

    // Declare as vectors so that we can have more than yolo_tensorrt inference
    m_name = this->declare_parameter<std::string>("name");

    auto config = this->declare_parameter<std::string>("config");
    auto weights = this->declare_parameter<std::string>("weights");
    auto threshold = this->declare_parameter<float>("threshold");
    auto precision = this->declare_parameter<std::string>("precision");
    auto network = this->declare_parameter<std::string>("network");
    auto calibration_images = this->declare_parameter<std::string>("calibration_images");

    auto detector = std::make_unique<Detector>();

    Config dconfig;
    dconfig.file_model_cfg = config;
    dconfig.file_model_weights = weights;
    dconfig.detect_thresh = threshold;
    dconfig.gpu_id = GPU_ID;
    dconfig.inference_precision = to_enum(precision, "HALF", inference_precision_map);
    dconfig.net_type = to_enum(network, "V5", net_type_map);
    dconfig.calibration_image_list_file_txt = calibration_images;

    detector->init(dconfig);
    m_detectors.push_back(std::move(detector));

    /*
    auto configs = this->declare_array_parameter<std::string>("configs");
    auto weights = this->declare_array_parameter<std::string>("weights");
    auto thresholds = this->declare_array_parameter<double>("thresholds");
    auto precisions = this->declare_array_parameter<std::string>("precisions");
    auto networks = this->declare_array_parameter<std::string>("networks");
    auto calibration_images = this->declare_array_parameter<std::string>("calibration_images");

    // All the lengths of the above parameters __must__ be the same!!
    assert(configs.size() == weights.size() && configs.size() == thresholds.size() && configs.size() == precisions.size() && configs.size() == networks.size() && configs.size() == calibration_images.size());
    assert(configs.size() > 0);

    // Create a detector/config for each
    for (std::size_t i = 0; i < configs.size(); i++) {
        auto detector = std::make_unique<Detector>();

        Config config;
        config.file_model_cfg = configs[i];
        config.file_model_weights = weights[i];
        config.detect_thresh = thresholds[i];
        config.gpu_id = GPU_ID;
        config.inference_precision = to_enum(precisions[i], "HALF", inference_precision_map);
        config.net_type = to_enum(networks[i], "V5", net_type_map);
        config.calibration_image_list_file_txt = calibration_images[i];

        detector->init(config);
        m_detectors.push_back(std::move(detector));
    }
    */

    using std::placeholders::_1;
    m_img_subscriber = image_transport::create_subscription(this, "~/input/image",
                                                            std::bind(&Yolov5VisionDetector::imageCallback, this, _1),
                                                            "raw", rmw_qos_profile_sensor_data);

    m_rois_publisher = this->create_publisher<wauto_perception_msgs::msg::RoiArray>("~/output/rois", 1);
} 
Yolov5VisionDetector::~Yolov5VisionDetector() {}

void Yolov5VisionDetector::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    const cv::Mat& image = cv_ptr->image;
    const std::vector<cv::Mat> image_array = {image};

    // Accumulate results/detections from each detector
    BatchResult batch_result;
    for (auto& detector : m_detectors) {
        std::vector<BatchResult> temp(1);
        detector->detect(image_array, temp);

        // concat the existing results and the new ones
        batch_result.insert(batch_result.end(), temp[0].begin(), temp[0].end());
    }

    publish(batch_result, msg->header);
}

void Yolov5VisionDetector::publish(const BatchResult& batch_result, const std_msgs::msg::Header& header) {
    auto roi_array = std::make_unique<wauto_perception_msgs::msg::RoiArray>();
    roi_array->header = header;

    roi_array->rois.reserve(batch_result.size());
    for (auto& result : batch_result) {
        wauto_perception_msgs::msg::Roi roi;

        geometry_msgs::msg::Point bl;
        bl.x = result.rect.tl().x;
        bl.y = result.rect.br().y;
        roi.bottom_left = bl;

        geometry_msgs::msg::Point tr;
        tr.y = result.rect.tl().y;
        tr.x = result.rect.br().x;
        roi.top_right = tr;

        wauto_perception_msgs::msg::ObjectClassification classification;
        classification.confidence = result.prob;
        classification.classification = class_id_map[m_name][result.id];
        roi.classification = classification;

        roi_array->rois.push_back(roi);
    }

    m_rois_publisher->publish(std::move(roi_array));
}

}  // namespace vision_detector

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(vision_detector::Yolov5VisionDetector)
