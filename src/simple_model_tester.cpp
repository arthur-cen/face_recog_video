#include <mutex>
#include <thread>
#include <opencv2/core/core.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include <pirvs.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <dirent.h>

using namespace cv;
using namespace cv::face;
using namespace std;

int main(int argc, const char *argv[]) 
{
	if (argc < 3) {
		printf("Not enough input argument.\nUsage:\n%s <path/to/test/data/> <path/to/recognizer/model> \n", argv[0]);
    	return -1;
  	}

  	const string test_path = string(argv[1]);
  	const string model_path = string(argv[2]);
  	
 //  	DIR *dir;			
 //  	struct dirent *ent;

 //  	if ((dir = opendir (test_path.c_str())) != NULL) {
 //  	/* print all the files and directories within directory */
 //  		while ((ent = readdir (dir)) != NULL) {
 //    		printf ("%s\n", ent->d_name);
 //  		}
 //  		closedir (dir);
	// } else {
 //  		/* could not open directory */
 //  		perror ("");
 //  		return EXIT_FAILURE;
	// }

	std::vector<cv::String> fn;
	glob(test_path + "*.png", fn, false);
	cv::Ptr<cv::face::FisherFaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
	// model->load<cv::face::FisherFaceRecognizer>(model_path); // Try model read, write
	model->read(model_path); 
	size_t count = fn.size();
	for (size_t i=0; i < count; i++) {
		int pred = model->predict(imread(fn[i], IMREAD_GRAYSCALE));//TODO try using read gray scales
		printf("pred = %d\n", pred);
	}
}