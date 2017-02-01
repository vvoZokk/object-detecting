//
//	Comparison of object detection methods
//	ContourMatchingExanimation class

#include "contour_matching_examination.hpp"

using namespace cv;

ContourMatchingExamination::ContourMatchingExamination() {
	samples = NULL;
}

ContourMatchingExamination::~ContourMatchingExamination() {
}

void ContourMatchingExamination::setSamples(Mat *marker, std::vector<Sample> *samples) {
	this->samples = samples;
	this->marker = marker;
}

void ContourMatchingExamination::setThreshold(double value) {
	threshold = value;
}

void ContourMatchingExamination::examine(int from, int to, int step) {
	Mat image;
	Point answer;
	const char* format = " blur raduis: %d, time: %.0f ms,\tfp error rate: %.2f, fn error rate: %.2f\n";

	if (samples == NULL) {
		std::cerr << "ContourMatchingExamination: vector of samples is unset\n";
		return;
	}
	// set Canny Edge Detector thresholds
	for (int th1 = 10; th1 < from; th1 += step) {
		for (int th2 = th1 + step; th2 < to; th2 += step) {
			method = new ContourMatching(th1, th2);
			//method->setThreshold(threshold); // not used
			std::cout << "\nUsage contour matching, Canny thresholds values " << th1;
			std::cout << " and " << th2 << std::endl;
			for (int radius = 0; radius < 7; radius++) {
				int n_count = 0;	// negative sample count
				int fp_count = 0;	// false positive
				int fn_count = 0;	// false negative
				float mean_time = 0.0;
				for (int i = 0; i < samples->size(); i++) {
					Sample sample = (*samples)[i];
					image = imread(sample.file_name, 1);
					method->setImages(image, *marker, &answer);
					mean_time += method->recognize(radius);
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
				printf(format, radius, mean_time, fp_error, fn_error);
			}

			delete method;
		}
	}
}
