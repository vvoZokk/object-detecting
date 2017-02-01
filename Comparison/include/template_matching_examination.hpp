//
//	Comparison of object detection methods
//	TemplateMatchingExamination class header file

#ifndef INCLUDE_TEMPLATE_MATCHING_EXAMINATION_HPP
#define INCLUDE_TEMPLATE_MATCHING_EXAMINATION_HPP

#include <iostream>
#include <opencv2/core/core.hpp>

#include "examination.hpp"
#include "template_matching.hpp"

class TemplateMatchingExamination : public Examination {
protected:
public:
	TemplateMatchingExamination();
	~TemplateMatchingExamination();
	void setSamples(cv::Mat *marker, std::vector<Sample> *samples);
	void setThreshold(double value);
	void examine(int from, int to, int step);
};

#endif // INCLUDE_TEMPLATE_MATCHING_EXAMINATION_HPP
