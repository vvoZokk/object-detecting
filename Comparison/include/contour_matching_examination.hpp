//
//	Comparison of object detection methods
//	ContourMatchingExamination class header file

#ifndef INCLUDE_CONTOUR_MATCHING_EXAMINATION_HPP
#define INCLUDE_CONTOUR_MATCHING_EXAMINATION_HPP

#include <iostream>
#include <opencv2/core/core.hpp>

#include "examination.hpp"
#include "contour_matching.hpp"

class ContourMatchingExamination : public Examination {
protected:
public:
	ContourMatchingExamination();
	~ContourMatchingExamination();
	void setSamples(cv::Mat *marker, std::vector<Sample> *samples);
	void setThreshold(double value);
	void examine(int from, int to, int step); // Canny threshold one
};

#endif // INCLUDE_CONTOUR_MATCHING_EXAMINATION_HPP
