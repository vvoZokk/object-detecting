//
//	Comparison of object detection methods
//	FeatureMatchingExamination class header file

#ifndef INCLUDE_FEATURE_MATCHING_EXAMINATION_HPP
#define INCLUDE_FEATURE_MATCHING_EXAMINATION_HPP

#include <iostream>
#include <opencv2/core/core.hpp>

#include "examination.hpp"
#include "feature_matching.hpp"

class FeatureMatchingExamination : public Examination {
protected:
public:
	FeatureMatchingExamination();
	~FeatureMatchingExamination();
	void setSamples(cv::Mat *marker, std::vector<Sample> *samples);
	void setThreshold(double value);
	void examine(int from, int to, int step);
};

#endif // INCLUDE_FEATURE_MATCHING_EXAMINATION_HPP
