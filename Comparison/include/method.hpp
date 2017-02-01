//
//	Comparison of object detection methods
//	Method abstract class header file

#ifndef INCLUDE_METHOD_HPP
#define INCLUDE_METHOD_HPP

#include <cstdio>
#include <opencv2/core/core.hpp>

class Method {
protected:
	float start_time, stop_time;
	double threshold;
	cv::Mat image, object;
	cv::Point *found_point;
public:
	virtual ~Method() {
		image.~Mat();
		object.~Mat();
	};
	virtual void setImages(cv::Mat image, cv::Mat object, cv::Point *answer) = 0;
	virtual void setThreshold(double value) = 0;
	virtual float recognize(int parameter) = 0;
	float stopTimer() {
		return 1000.0 * (stop_time - start_time) / CLOCKS_PER_SEC;
	};
};

#endif // INCLUDE_METHOD_HPP
