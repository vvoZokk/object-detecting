//
//	Comparison of object detection methods
//	Examination abstract class header file

#ifndef INCLUDE_EXAMINATION_HPP
#define INCLUDE_EXAMINATION_HPP

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "method.hpp"

struct Sample {
	std::string file_name;
	cv::Point point;
	int offset;
};

class Examination {
protected:
	float threshold;
	std::vector<Sample> *samples;
	cv::Mat *marker;
	Method *method;
public:
	virtual ~Examination() {};
	virtual void setSamples(cv::Mat *marker, std::vector<Sample> *samples) = 0;
	virtual void setThreshold(double value) = 0;
	virtual void examine(int from, int to, int step) = 0;
};

#endif // INCLUDE_EXAMINATION_HPP
