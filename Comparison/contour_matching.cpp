//
//	Comparison of object detection methods
//	ContourMatching class

#include "contour_matching.hpp"

using namespace cv;

ContourMatching::ContourMatching(int th1, int th2) : th_1(th1), th_2(th2) {
}

ContourMatching::~ContourMatching() {
	correlation.~Mat();
}

void ContourMatching::setImages(Mat image, cv::Mat object, cv::Point *answer) {
	this->image = image;
	this->object = object;
	found_point = answer;
}

void ContourMatching::setThreshold(double value) {
	threshold = value;
}

float ContourMatching::recognize(int parameter) {
 	float time = 0;
 	std::vector<std::vector<cv::Point> > image_contours, object_contours;
	std::vector<cv::Vec4i> hierarchy;

	cvtColor(object, object, CV_BGR2GRAY);
	Canny(object, object, th_1, th_2);
	findContours(object, object_contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	start_time = clock();
	cvtColor(image, image, CV_BGR2GRAY);
	if (parameter) {
		cv::blur(image, image, cv::Size(parameter, parameter));
	}
	Canny(image, image, th_1, th_2);
	findContours(image, image_contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	int target_i = -1;
	double best = -1.0;
	for (int k = 0; k < image_contours.size(); k++) {
		if (hierarchy[k][2] == -1) continue;
		RotatedRect rect = minAreaRect(image_contours[k]);
		if (rect.size.height > 15
			&& rect.angle < 15
			&& rect.angle > -15
			&& rect.size.width / rect.size.height > 1.25
			&& rect.size.width / rect.size.height < 2.75
			&& contourArea(image_contours[k]) > rect.size.width * rect.size.height * 0.75) {
			int j = hierarchy[k][2];
			double sum = 0;
			int count = 0;
			while(j != -1) {
				Rect b_rect = boundingRect(image_contours[j]);
				if (b_rect.width > 10.0 && b_rect.height > 0.5 * rect.size.height) {
					double max = -1.0;
					for (int i = 0; i < object_contours.size(); i++) {
						double match = matchShapes(image_contours[j], object_contours[i], CV_CONTOURS_MATCH_I3, 0);
						if (match > max) max = match;
					}
					sum += max;
					count++;
				}
				j = hierarchy[j][0];
			}
			if (count == 0) continue;
			sum /= count;
			if (sum > best) {
				best = sum;
				target_i = k;
			}

		}
		if (target_i != -1) {
			Rect r = boundingRect(image_contours[target_i]);
			found_point->x = r.x + r.width / 2;
			found_point->y = r.y + r.height / 2;
		} else {
			found_point->x = -1;
			found_point->y = -1;
		}
	}

	stop_time = clock();

	return time + stopTimer();
}
