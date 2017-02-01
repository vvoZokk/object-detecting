//
//	Comparison of object detection methods
//	FeatureMatching class header file

#ifndef INCLUDE_FEATURE_MATCHING_HPP
#define INCLUDE_FEATURE_MATCHING_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "method.hpp"

class FeatureMatching : public Method {
protected:
	cv::Mat descriptors_object, descriptors_image;
public:
	FeatureMatching();
	~FeatureMatching();
	void setImages(cv::Mat image, cv::Mat object, cv::Point *answer);
	void setThreshold(double value);
	float recognize(int parameter);
};

#endif // INCLUDE_FEATURE_MATCHING_HPP
