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
  if (argc < 3) {
    printf("Not enough input argument.\nUsage:\n%s <path/to/cascade/model> <path/to/recognizer/model> \n", argv[0]);
    return -1;
  }
  const string haar_cascade = string(argv[1]);
  const string model_path = string(argv[2]);
  // install SIGNAL handler
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  // Create an initial state for feature detection + matching + triangulation.
  // std::shared_ptr<PIRVS::FeatureState> state;
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

  // TODO check if exposure track bar works or not
  // Add a trackbar to the window to tune the exposure of the stereo camera.
  // uint32_t exposure_value_u;
  // if (!gDevice->GetExposure(&exposure_value_u)) {
  //   printf("Failed to get exposure.\n");
  // }

  // int exposure_value = exposure_value_u;
  // namedWindow("Detected face", WINDOW_AUTOSIZE);
  // cv::createTrackbar("Exposure", "Detected Face", &exposure_value, 2000,
  //                    ExposureTrackBarCallback, &gDevice);

  cv::CascadeClassifier face_cascade;
  face_cascade.load(haar_cascade);
  //TODO change, availabel for user setting
  int num_components = 10;
  double threshold = 10.0;
  cv::Ptr<cv::face::EigenFaceRecognizer> model = cv::face::EigenFaceRecognizer::create(num_components, threshold);
  printf("Loaded EigenFaceRecognizer\n");
  // model->read<cv::face::EigenFaceRecognizer>(model_path);
  model->load<cv::face::EigenFaceRecognizer>(model_path);

  if (model.empty())
  {
  	printf("EigenFaceRecognizer is empty\n");
  	exit(1);
  }
  cv::Mat img_left;
  // cv::Mat img_right;
  std::vector<cv::Rect> faces;
  // cv::Size sz_l;
  // cv::Size sz_r;

  // Mat img_disp;
  // Mat left;
  // Mat right;
  // Mat face_recog;
  // cv::Rect cropArea;
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
    printf("Data Captured\n");
    // Get left image in the stereo data
    // sz_l = stereo_data->img_l.size();
    // sz_r = stereo_data->img_r.size();
    // cv::Mat img_disp(sz_l.height, sz_l.width+sz_r.width, CV_8UC3);
    // cv::Mat left(img_disp, cv::Rect(0, 0, sz_l.width, sz_r.height));
    // stereo_data->img_l.copyTo(left);
    // cv::Mat right(img_disp, cv::Rect(sz_l.width, 0, sz_r.width, sz_r.height));
    // stereo_data->img_r.copyTo(right);
    stereo_data->img_l.copyTo(img_left);
    // face_cascade.detectMultiScale(img_disp, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    face_cascade.detectMultiScale(img_left, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    // if (faces.size() > 0) {
    //   printf("Detected faces\n");
    // } else {
    //   printf("Face not Detected\n");
    // }
    
    for( int i = 0; i < faces.size(); i++ )
    {   
        // cv::Rect cropArea(faces[i].x, faces[i].y, 256, 256);
        // Mat face_recog;
        // img_disp(cropArea).copyTo(face_recog);
        // img_left(cropArea).copyTo(face_recog);
        // cv::equalizeHist(face_recog, face_recog);
        // printf("Predicting Face\n");
        // int prediction = model->predict(face_recog);
        //TODO map prediction back to label
        rectangle(img_left, faces[i], CV_RGB(255, 0, 0), 1);
        string box_text = "Prediction = Arthur";
        
        // Calculate the position for annotated text (make sure we don't
        // put illegal values in there):
        int pos_x = std::max(faces[i].x - 10, 0);
        int pos_y = std::max(faces[i].y - 10, 0);
        // putText(img_disp, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        putText(img_left, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
    }
    cv::imshow("Detected Face", img_left);
    // Press ESC to stop.
    char key = cv::waitKey(20); //Change wait key to change the display time for the image
    if (key == 27) {
      printf("Stopped.\n");
      break;
    }
  }

  // printf("Final exposure value is %d.\n", exposure_value);
  gDevice->StopDevice();
  cv::destroyAllWindows();

  return 0;
}
