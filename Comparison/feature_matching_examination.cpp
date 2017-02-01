//
//	Comparison of object detection methods
//	FeatureMatchingExanimation class

#include "feature_matching_examination.hpp"

using namespace cv;

FeatureMatchingExamination::FeatureMatchingExamination() {
	samples = NULL;
}

FeatureMatchingExamination::~FeatureMatchingExamination() {
}

void FeatureMatchingExamination::setSamples(Mat *marker, std::vector<Sample> *samples) {
	this->samples = samples;
	this->marker = marker;
}

void FeatureMatchingExamination::setThreshold(double value) {
	threshold = value;
}

void FeatureMatchingExamination::examine(int from, int to, int step) {
	Mat image;
	Point answer;
	const char* format = " min Hessian value: %d, time: %.0f ms,\tfp error rate: %.2f, fn error rate: %.2f\n";

	if (samples == NULL) {
		std::cerr << "FeatureMatchingExamination: vector of samples is unset\n";
		return;
	}
	std::cout << "\nUsage feature matching, SURF detector" << std::endl;
	method = new FeatureMatching();
	for (int minHessian = from; minHessian <= to; minHessian += step) {
		int n_count = 0;	// negative sample count
		int fp_count = 0;	// false positive
		int fn_count = 0;	// false negative
		float mean_time = 0.0;
		for (int i = 0; i < samples->size(); i++) {
			Sample sample = (*samples)[i];
			image = imread(sample.file_name, 1);
			method->setImages(image, *marker, &answer);
			mean_time += method->recognize(minHessian);
			int x_error = abs(sample.point.x - answer.x);
			int y_error = abs(sample.point.y - answer.y);
			if (x_error > sample.offset || y_error > sample.offset / 2) {
				// incorrect regognition
				if (sample.offset > 1) {
					fp_count++;
				} else {
					n_count++;
					fn_count++;
				}
			} else {
				// correct recognition
				if (sample.offset == 1) {
					n_count++;
				}
			}
		}
		mean_time /= (*samples).size();
		float fp_error = float(fp_count) / ((*samples).size() - n_count);
		float fn_error = float(fn_count) / n_count;
		printf(format, minHessian, mean_time, fp_error, fn_error);
	}
	delete method;
}
