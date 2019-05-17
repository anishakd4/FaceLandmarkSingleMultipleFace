#include<dlib/image_processing.h>
#include<dlib/image_processing/frontal_face_detector.h>
#include<dlib/opencv.h>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>

using namespace std;
using namespace dlib;

//draw points on the image
void drawPoints(cv::Mat &image, full_object_detection landmarks){
    for(int i=0; i<landmarks.num_parts(); i++){
        cv::circle(image, cv::Point(landmarks.part(i).x(), landmarks.part(i).y()), 3, cv::Scalar(0, 255, 255), -1);
    }
}

int main(){
    //get face detector
    frontal_face_detector faceDetector = get_frontal_face_detector();

    //define landmark detector
    shape_predictor landmarkDetector;

    //load the face landmark model
    deserialize("../dlibAndModel/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    //Read images with single and mulitple persons
    cv::Mat imageSingle = cv::imread("../assets/anish.jpg");
    cv::Mat imageMultiple = cv::imread("../assets/anish2.jpg");

    //create images clone to work with
    cv::Mat imageSingleClone=imageSingle.clone();
    cv::Mat imageMultipleClone=imageMultiple.clone();

    //convert opencv image format to dlib image format
    cv_image<bgr_pixel> dlibSingleImage(imageSingleClone);
    cv_image<bgr_pixel> dlibMultipleImage(imageMultipleClone);

    //detect faces in the images
    std::vector<rectangle> facesSingle = faceDetector(dlibSingleImage);
    std::vector<rectangle> facesMultiple = faceDetector(dlibMultipleImage);

    //vector to store face landmarks for all the faces
    std::vector<full_object_detection> faceLandmarksSingle;
    std::vector<full_object_detection> faceLandmarksMultiple;

    //Loop over all the faces detected and find face landmarks for all of them
    for(int i=0; i<facesSingle.size(); i++){
        full_object_detection landmarks = landmarkDetector(dlibSingleImage, facesSingle[i]);
        faceLandmarksSingle.push_back(landmarks);
        drawPoints(imageSingleClone, landmarks);
    }
    for(int i=0; i<facesMultiple.size(); i++){
        full_object_detection landmarks = landmarkDetector(dlibMultipleImage, facesMultiple[i]);
        faceLandmarksMultiple.push_back(landmarks);
        drawPoints(imageMultipleClone, landmarks);
    }

    //create windows to display windows
    cv::namedWindow("single person image", cv::WINDOW_NORMAL);
    cv::namedWindow("single face landmarks", cv::WINDOW_NORMAL);
    cv::namedWindow("multiple person image", cv::WINDOW_NORMAL);
    cv::namedWindow("multiple face landmarks", cv::WINDOW_NORMAL);

    //display images
    cv::imshow("single person image", imageSingle);
    cv::imshow("single face landmarks", imageSingleClone);
    cv::imshow("multiple person image", imageMultiple);
    cv::imshow("multiple face landmarks", imageMultipleClone);

    //press esc to exit the program
    cv::waitKey(0);
    
    //close all the opened windows
    cv::destroyAllWindows();

    return 0;
}