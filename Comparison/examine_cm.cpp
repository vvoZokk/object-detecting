//
//	Comparison of object detection methods
//	Examine contour matching, main file

#include <fstream>
#include <opencv2/core/core.hpp>

#include "contour_matching_examination.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	if (argc != 3) {
		cout << "Incorrect argument count" << endl;
		cout << "Usage: "<< argv[0] <<" MARKER_IMAGE SAMPLE_LIST";
		cout << endl;
		return 1;
	}

	Examination *cm = new ContourMatchingExamination();
	vector<Sample> samples;
	Mat marker = imread(argv[1], 1);
	ifstream list(argv[2]);
	Sample line;
	while (list >> line.file_name >> line.point.x >> line.point.y >> line.offset) {
		samples.push_back(line);
	}
	cm->setSamples(&marker, &samples);
	//cm->setThreshold(0.85); // not used
	cm->examine(60, 100, 10);

	delete cm;

	return 0;
}
