//
//	Comparison of object detection methods
//	Generator of samples

#define SAMPLE_COUNT 120
#define C_RANGE 30

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

struct Foreground {
	std::string file_name;
	int position_x;
	int position_y;
	int offset;
};

int main(int argc, char** argv) {

	std::srand(time(NULL));
	std::vector<Foreground> fg_list;
	std::vector<std::string> bg_list;
	Mat foreground, background, fg, bg, mask, result;
	int static count = 0;
	char result_file_name[20];

	if (argc != 3) {
		printf("Incorrect argument count\n");
		return 1;
	}
	std::ifstream fg_file(argv[1]);
	Foreground line;
	while (fg_file >> line.file_name >> line.position_x >> line.position_y >> line.offset) {
		fg_list.push_back(line);
	}
	std::ifstream bg_file(argv[2]);
	std::string file;
	while (bg_file >> file) {
		bg_list.push_back(file);
	}

	while (count < SAMPLE_COUNT) {
		for (int i = 0; i < fg_list.size(); i++) {
			foreground = imread(fg_list[i].file_name, 1);
			cv::cvtColor(foreground, mask, CV_BGR2GRAY);
			threshold(mask, mask, 0, 255, THRESH_BINARY);
			for (int j = 0; j < bg_list.size(); j++) {
				background = imread(bg_list[j], 1);

				foreground.copyTo(fg);
				background.copyTo(bg);

				// resize background image
				Rect rect(0, 0, fg.cols, fg.rows);
				if (bg.rows > 2 * fg.rows && bg.cols > 2 * fg.cols) {
					float r = 0.5 + 0.5 * rand() / RAND_MAX;
					rect.x = r * bg.cols / 4;
					rect.y = r * bg.rows / 4;
					r = 0.8 + 0.2 * rand() / RAND_MAX;
					rect.width = fg.cols / r;
					rect.height = fg.rows / r;
					bg = bg(rect);
					resize(bg, bg, fg.size());
				} else {
					bg = bg(rect);
				}

				// change colors
				int half = C_RANGE / 2;
				fg += Scalar(half - rand() % C_RANGE, half - rand() % C_RANGE, half - rand() % C_RANGE);
				bg += Scalar(half - rand() % C_RANGE, half - rand() % C_RANGE, half - rand() % C_RANGE);

				// merge image
				bitwise_and(fg, fg, fg, mask);
				bitwise_not(mask, mask);
				bitwise_and(bg, bg, bg, mask);
				add(bg, fg, fg, mask);
				bitwise_not(mask, mask);

				// change contrast
				fg.convertTo(result, -1, 0.25 + 1.5 * rand() / RAND_MAX, 0);

				// write result
				std::vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
				compression_params.push_back(95);
				sprintf(result_file_name, "./samples/%05d.jpg", count);
				printf("%s %d %d %d\n",
					result_file_name,
					fg_list[i].position_x,
					fg_list[i].position_y,
					fg_list[i].offset);
				try {
					imwrite(result_file_name, result, compression_params);
				}
				catch (std::runtime_error& ex) {
					fprintf(stderr, "Exception converting image: %s\n", ex.what());
					return 1;
				}
				count++;
			}
		}
	}
	return 0;
}
