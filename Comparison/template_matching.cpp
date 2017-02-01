//
//	Comparison of object detection methods
//	TemplateMatching class

#include "template_matching.hpp"

using namespace cv;

TemplateMatching::TemplateMatching(int mode) : match_method(mode) {
	// from 20% to 100% size
	begin_ratio = 0.2;
	end_ratio = 1.0;
}

TemplateMatching::~TemplateMatching() {
	correlation.~Mat();
}

void TemplateMatching::setImages(Mat image, cv::Mat object, cv::Point *answer) {
	this->image = image;
	this->object = object;
	found_point = answer;
}

void TemplateMatching::setThreshold(double value) {
	threshold = value;
}

float TemplateMatching::recognize(int parameter) {
 	int correlation_cols, correlation_rows;
 	float time = 0.0;
	double min_value, max_value, best_value;
	Point min_point, max_point, best_point;
	Mat test_object;

	cvtColor(object, object, CV_BGR2GRAY);
	start_time = clock();
	cvtColor(image, image, CV_BGR2GRAY);
	stop_time = clock();
	time = stopTimer();

	if (match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
		best_value = 1.0e100;
	} else {
		best_value = -1.0e100;
	}
	float step = (end_ratio - begin_ratio) / parameter;
	float ratio = begin_ratio;
	for (int i = 0; i < parameter; i++) {
		ratio += step;
	 	correlation_cols =  image.cols - ratio * object.cols + 1;
		correlation_rows = image.rows - ratio * object.rows + 1;
		test_object.create(ratio * object.rows, ratio * object.cols, CV_32SC3);

		start_time = clock();
		correlation.create(correlation_rows, correlation_cols, CV_BGR2GRAY);
		resize(object, test_object, test_object.size());

		matchTemplate(image, test_object, correlation, match_method);
		//normalize(correlation, correlation, 0, 1, NORM_MINMAX, -1, Mat());

		minMaxLoc(correlation, &min_value, &max_value, &min_point, &max_point, Mat());
		if (match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
			if (best_value > min_value) {
				best_value = min_value;
				best_point.x = min_point.x + test_object.cols / 2;
				best_point.y = min_point.y + test_object.rows / 2;
			}
		} else {
			if (best_value < max_value) {
				best_value = max_value;
				best_point.x = max_point.x + test_object.cols / 2;
				best_point.y = max_point.y + test_object.rows / 2;
			}
		}
		stop_time = clock();
		time += stopTimer();
	}

	switch (match_method) {
		case CV_TM_SQDIFF:
			*found_point = best_point;
			break;
		case CV_TM_SQDIFF_NORMED:
			if (best_value <= 1.0 - threshold) {
				*found_point = best_point;
			} else {
				found_point->x = -1;
				found_point->y = -1;
			}
			break;
		case CV_TM_CCORR:
			*found_point = best_point;
			break;
		case CV_TM_CCORR_NORMED:
			if (best_value >= threshold) {
				*found_point = best_point;
			} else {
				found_point->x = -1;
				found_point->y = -1;
			}
			break;
		case CV_TM_CCOEFF:
			*found_point = best_point;
			break;
		case CV_TM_CCOEFF_NORMED:
			if (best_value >= threshold) {
				*found_point = best_point;
			} else {
				found_point->x = -1;
				found_point->y = -1;
			}
			break;
		default:
			break;
	}
	return time;
}
