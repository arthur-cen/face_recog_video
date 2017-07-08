// #include "./helper.h"
#include <stdio.h>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/face.hpp>
using namespace std;
using namespace cv;
using namespace cv::face;

int dataLoad(const string& datacsv_path, const string& filepath, std::vector<cv::Mat>& images, std::vector<int>& labels, std::vector<string> labels_txt,char seperator = ',')
{	
	std::ifstream file;
	file.open(datacsv_path.c_str());
	if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        cout << error_message << endl;
        CV_Error(CV_StsBadArg, error_message);
        exit(1);
    }
    string line, filename, classlabel;
    bool read_label = true;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, classlabel, seperator);
        getline(liness, filename);
        if ( read_label && !filename.empty() && !classlabel.empty()) {
        	labels_txt.push_back(filename); //File name here actually stores name of the person.
        }
        if( !read_label && !filename.empty() && !classlabel.empty()) {
        	// cout << filepath.c_str() + filename << endl;
        	images.push_back(imread(filepath.c_str() + filename, IMREAD_GRAYSCALE)); //TODO replace string concat to path join later
            labels.push_back(atoi(classlabel.c_str()));
        }
        if ( !filename.empty() && !classlabel.empty() && classlabel == "#" || filename == "#")
        {	
        	// cout << "Reading filenames" << endl;
        	read_label = false;
        }
    }
}

int main(int argc, const char *argv[]) {
	//Get the path to the csv file that stores data information
	string const MODEL_NAME = "face_recog_model.xml";

	if (argc != 4) {
        cout << "usage: " << argv[0] << " </path/to/train/data> </path/to/model/directory/>" << endl;
        cout << "\t </path/to/train/data/> -- Path to input training image storage, include slash at the end eg. ../data/" << endl;
        cout << "\t </path/to/test/data/> -- Path to input test image storage, include slash at the end eg. ../data_test/" << endl;        
        cout << "\t </path/to/model/directory/> -- Path to model stroage directory, include slash at the end eg. ../model/" << endl;
        exit(1);
    }
    string input_path = string(argv[1]);
    string test_path = string(argv[2]);
    string model_path = string(argv[3]);	
    // Expected file path
	// string input_path = "../data/data.csv"; // TODO: Need Read From Input
	// string model_path = "./models/"; //TODO: Need to Read from input line
    string fn_csv = input_path + "data.csv";
    string test_fn_csv = test_path + "data.csv";
	char seperator = ',';
	// These vectors holds the images and the corresponding labels
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	std::vector<string> labels_txt;
	// Read in the data
	try
	{
	    dataLoad(fn_csv, input_path, images, labels, labels_txt, seperator);
	} catch (cv::Exception& e) {
		cerr << "Error process data \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// exit the program, debug on your own
		exit(1);
	}

	//Load Cascade Object Detector and detect faces in the images
	//and crop out faces from image
	//TODO: Move this part to train()
	//TODO: Implement train update functionality
	//Train the Eigenface Recognizer 
	int num_components = 10;
	double threshold = 10.0;
	cv::Ptr<cv::face::FisherFaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
	model->train(images, labels);
	cout << "Trained Model" << endl;
	// These vectors holds the images and the corresponding labels
	std::vector<cv::Mat> images_test;
	std::vector<int> labels_test;
	std::vector<string> labels_txt_test;
	try
	{
		dataLoad(test_fn_csv, test_path, images_test, labels_test, labels_txt_test, seperator);
	} catch (cv::Exception& e) {
		cerr << "Error process data \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// exit the program, debug on your own
		exit(1);
	}
	cout << "Start Testing" << endl;
	int prediction;

	std::vector<cv::String> fn;
	glob(test_path + "*.png", fn, false);
	size_t count = fn.size();
	for (size_t i=0; i < count; i++) {
		cv::Mat img = imread(fn[i], IMREAD_GRAYSCALE);
		int pred = model->predict(img);
		printf("%d\n", pred);
	}
	//Save the model to the working directory
	cout << "Saving model" << endl;
	model->write(model_path + MODEL_NAME);
	return(1);
}