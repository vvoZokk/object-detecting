//
//	Comparison of object detection methods
//	FeatureMatching class

#include "feature_matching.hpp"

using namespace cv;

FeatureMatching::FeatureMatching() {
}

FeatureMatching::~FeatureMatching() {
}

void FeatureMatching::setImages(Mat image, cv::Mat object, cv::Point *answer) {
	this->image = image;
	this->object = object;
	found_point = answer;
}

void FeatureMatching::setThreshold(double value) {
	threshold = value;
}

float FeatureMatching::recognize(int parameter) {
 	float time = 0;
	std::vector<KeyPoint> keypoints_object, keypoints_image;
	std::vector<DMatch> matches;
	//std::vector<DMatch> best_matches;
	std::vector<Point2f> points_object, points_image;
	std::vector<Point2f> object_corners(4);
	std::vector<Point2f> image_corners(4);
	SurfFeatureDetector detector(parameter);
	SurfDescriptorExtractor extractor;
	FlannBasedMatcher matcher;

	cvtColor(object, object, CV_BGR2GRAY);
	detector.detect(object, keypoints_object);
	extractor.compute(object, keypoints_object, descriptors_object);
	object_corners[0] = cvPoint(0,0);
	object_corners[1] = cvPoint(object.cols, 0);
	object_corners[2] = cvPoint(object.cols, object.rows);
	object_corners[3] = cvPoint(0, object.rows);

	start_time = clock();
	cvtColor(image, image, CV_BGR2GRAY);

	detector.detect(image, keypoints_image);
	extractor.compute(image, keypoints_image, descriptors_image);
	matcher.match(descriptors_object, descriptors_image, matches);

	/*double max = 0.0;
	for (int i = 0; i < descriptors_object.rows; i++) {
		if (matches[i].distance > max) {
			max = matches[i].distance;
		}
	}
	for (int i = 0; i < descriptors_object.rows; i++) {
  		if (matches[i].distance < threshold * max) {
  			best_matches.push_back(matches[i]);
  		}
  }*/
	if (matches.size() < 4) {
		found_point->x = -1;
		found_point->y = -1;
		stop_time = clock();

		return time + stopTimer();
	}
	for (int i = 0; i < matches.size(); i++) {
		points_object.push_back(keypoints_object[matches[i].queryIdx].pt);
		points_image.push_back(keypoints_image[matches[i].trainIdx].pt);
	}
	Mat H = findHomography(points_object, points_image, CV_RANSAC);
	perspectiveTransform(object_corners, image_corners, H);

	Rect r = boundingRect(image_corners);
	if (r.width > 1.5 * r.height && r.width < 2.5 * r.height) {
		found_point->x = r.x + r.width / 2;
		found_point->y = r.y + r.height / 2;
	} else {
		found_point->x = -1;
		found_point->y = -1;
	}
	stop_time = clock();

	return time + stopTimer();
}
