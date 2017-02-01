//
//	Comparison of object detection methods
//	TemplateMatchingExanimation class

#include "template_matching_examination.hpp"

using namespace cv;

TemplateMatchingExamination::TemplateMatchingExamination() {
	samples = NULL;
}

TemplateMatchingExamination::~TemplateMatchingExamination() {
}

void TemplateMatchingExamination::setSamples(Mat *marker, std::vector<Sample> *samples) {
	this->samples = samples;
	this->marker = marker;
}

void TemplateMatchingExamination::setThreshold(double value) {
	threshold = value;
}

void TemplateMatchingExamination::examine(int from, int to, int step) {
	Mat image;
	Point answer;
	const char* format = " scaling count: %d, time: %.0f ms,\tfp error rate: %.2f, fn error rate: %.2f\n";

	if (samples == NULL) {
		std::cerr << "TemplateMatchingExamination: vector of samples is unset\n";
		return;
	}
	for (int mode = 3; mode < 6; mode++) {
		method = new TemplateMatching(mode);
		method->setThreshold(threshold);
		std::cout << "\nUsage template matching, method #" << mode;
		std::cout << ", threshold value " << threshold << std::endl;
		int parameter = from;
		while (parameter < to) {
			int n_count = 0;	// negative sample count
			int fp_count = 0;	// false positive
			int fn_count = 0;	// false negative
			float mean_time = 0.0;
			for (int i = 0; i < samples->size(); i++) {
				Sample sample = (*samples)[i];
				image = imread(sample.file_name, 1);
				method->setImages(image, *marker, &answer);
				mean_time += method->recognize(parameter);
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
			printf(format, parameter, mean_time, fp_error, fn_error);

			parameter += step;
		}
		delete method;
	}
}
