//
//	Comparison of object detection methods
//	TemplateMatching class header file

#ifndef INCLUDE_TEMPLATE_MATCHING_HPP
#define INCLUDE_TEMPLATE_MATCHING_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include "method.hpp"

class TemplateMatching : public Method {
protected:
	int match_method;
	float begin_ratio, end_ratio;
	cv::Mat correlation;
public:
	TemplateMatching(int mode);
	~TemplateMatching();
	void setImages(cv::Mat image, cv::Mat object, cv::Point *answer);
	void setThreshold(double value);
	float recognize(int parameter);
};

#endif // INCLUDE_TEMPLATE_MATCHING_HPP
