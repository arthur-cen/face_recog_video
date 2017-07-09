/**
 * Copyright 2017 PerceptIn
 *
 * This End-User License Agreement (EULA) is a legal agreement between you
 * (the purchaser) and PerceptIn regarding your use of
 * PerceptIn Robotics Vision System (PIRVS), including PIRVS SDK and
 * associated documentation (the "Software").
 *
 * IF YOU DO NOT AGREE TO ALL OF THE TERMS OF THIS EULA, DO NOT INSTALL,
 * USE OR COPY THE SOFTWARE.
 */

// Don't use  GCC CXX11 ABI by default
#ifndef _GLIBCXX_USE_CXX11_ABI
#define _GLIBCXX_USE_CXX11_ABI 0
#endif

#include <mutex>
#include <thread>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include <pirvs.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
std::shared_ptr<PIRVS::PerceptInDevice> gDevice = NULL;
using namespace cv;
using namespace std;
/**
 * Gracefully exit when CTRL-C is hit
 */
void exit_handler(int s){
  if (gDevice != NULL) {
    gDevice->StopDevice();
  }
  cv::destroyAllWindows();
  exit(1);
}

/**
 * online_features visualizes the 2d features + 3d points detected from a device.
 * online_features is also an useful tool for finding the best exposure value
 * for SLAM.
 */

// Callback function for OpenCV trackbar to set exposure of the device.
void ExposureTrackBarCallback(int value, void *ptr) {
  std::shared_ptr<PIRVS::PerceptInDevice>* device_ptr =
      static_cast<std::shared_ptr<PIRVS::PerceptInDevice>*>(ptr);
  if (!device_ptr) {
    return;
  }
  (*device_ptr)->SetExposure(value);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Not enough input argument.\nUsage:\n%s <path/to/data/csv> <path/to/cascade/model> <path/to/recognizer/model> \n", argv[0]);
    return -1;
  }
  const string data_csv = string(argv[1]);
  const string haar_cascade = string(argv[2]);
  const string model_path = string(argv[3]);
  // install SIGNAL handler
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  // Create an initial state for feature detection + matching + triangulation.
  std::shared_ptr<PIRVS::FeatureState> state;
  // if (!PIRVS::InitFeatureState(file_calib, &state)){
  //   printf("Failed to InitFeatureState.\n");
  //   return -1;
  // }

  // Create an interface to stream the PerceptIn V1 device.
  if (!PIRVS::CreatePerceptInV1Device(&gDevice) || !gDevice) {
    printf("Failed to create device.\n");
    return -1;
  }
  // Start streaming from the device.
  if (!gDevice->StartDevice()) {
    printf("Failed to start device.\n");
    return -1;
  }
  cv::Mat img_2d;
  // Add a trackbar to the window to tune the exposure of the stereo camera.
  uint32_t exposure_value_u;
  if (!gDevice->GetExposure(&exposure_value_u)) {
    printf("Failed to get exposure.\n");
  }

  int exposure_value = exposure_value_u;
  cv::namedWindow("Detected faces");
  cv::createTrackbar("Exposure", "Detected faces", &exposure_value, 2000,
                     ExposureTrackBarCallback, &gDevice);

  cv::CascadeClassifier face_cascade;
  face_cascade.load(haar_cascade);
  cv::Ptr<cv::face::FisherFaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
  printf("Loaded FisherFaceRecognizer\n");
  model->read(model_path);

  //Read tag information from data.csv
  std::ifstream file;
  std::vector<string> names;
  file.open(data_csv.c_str());
  if (!file) {
        std::string error_message = "No valid input file was given, please check the given filename.";
        printf("%s\n", error_message.c_str());
        CV_Error(CV_StsBadArg, error_message);
        exit(1);
    }
    string line, digit_label, text_label;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, digit_label, ',');
        getline(liness, text_label);
        if (!digit_label.empty() && !text_label.empty() && digit_label == "#" || text_label == "#")
        { 
          break;
        }
        if (!digit_label.empty() && !text_label.empty()) {
          printf("%s\n", text_label.c_str());
          names.push_back(text_label); //File name here actually stores name of the person.
        }
    }

  if (model.empty())
  {
  	printf("FisherFaceRecognizer is empty\n");
  	exit(1);
  }
  cv::Mat img_left;
  std::vector<cv::Rect> faces;
  model->setThreshold(7000.0); //Manually Set the threshold of the model here
  // Stream data from the device and update the feature state.
  printf("Capturing Data From camera ...\n");
  while (1) {
    // Get the newest data from the device.
    // Note, it could be either an ImuData or a StereoData.
    std::shared_ptr<const PIRVS::Data> data;
    if (!gDevice->GetData(&data)) {
      continue;
    }
    // RunFeature only accept StereoData.
    std::shared_ptr<const PIRVS::StereoData> stereo_data =
        std::dynamic_pointer_cast<const PIRVS::StereoData>(data);
    if (!stereo_data) {
      continue;
    }

    // if (PIRVS::Draw2dFeatures(stereo_data, state, &img_2d)) {
    //   cv::imshow("Detected features", img_2d);
    // }
    stereo_data->img_l.copyTo(img_left);
    face_cascade.detectMultiScale(img_left, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    for( int i = 0; i < faces.size(); i++ )
    {   
        // cv::Rect cropArea(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        cv::Mat face_recog;
        // img_disp(cropArea).copyTo(face_recog);
        img_left(faces[i]).copyTo(face_recog);
        // cv::cvtColor(face_recog, face_recog, COLOR_BGR2GRAY);
        cv::equalizeHist(face_recog, face_recog);
        cv::resize(face_recog, face_recog, cv::Size(350, 350));

        // printf("Predicting Face\n");
        int prediction = model->predict(face_recog);
        rectangle(img_left, faces[i], Scalar( 255, 0, 255 ), 1);
        string box_text = "Unknown";
        if (prediction >= 0)
        {
          box_text = names[prediction];
        }
        
        // Calculate the position for annotated text (make sure we don't
        // put illegal values in there):
        int pos_x = std::max(faces[i].x - 10, 0);
        int pos_y = std::max(faces[i].y - 10, 0);
        // putText(img_disp, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        putText(img_left, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar( 255, 0, 255 ), 2.0);
    }
    cv::imshow("Detected faces", img_left);
    // Press ESC to stop.
    char key = cv::waitKey(20); //Change wait key to change the display time for the image
    if (key == 27) {
      printf("Stopped.\n");
      break;
    }
  }

  gDevice->StopDevice();
  cv::destroyAllWindows();

  return 0;
}
