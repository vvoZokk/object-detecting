//
//	Comparison of object detection methods
//	ContourMatching class header file

#ifndef INCLUDE_CONTOUR_MATCHING_HPP
#define INCLUDE_CONTOUR_MATCHING_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include "method.hpp"

class ContourMatching : public Method {
protected:
	int th_1, th_2;
	cv::Mat correlation;
public:
	ContourMatching(int th1, int th2);
	~ContourMatching();
	void setImages(cv::Mat image, cv::Mat object, cv::Point *answer);
	void setThreshold(double value);
	float recognize(int parameter); // blur radius
};

#endif // INCLUDE_CONTOUR_MATCHING_HPP
