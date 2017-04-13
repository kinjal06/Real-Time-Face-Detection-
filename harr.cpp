/*The code includes functions to capture image, query the captured images, gray scale conversion, 
edge detection and drawing ellipse on the frame for detecting face and circle on the frame for 
detecting eyes.*/
 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>

 using namespace std;
 using namespace cv;

 /** Function Definition*/
 void detectOutput( Mat frame );// detecting the face & features and displaying them

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Face-Detected Frame";
 RNG rng(12345);


 int main( int argc, const char** argv )
 {
   CvCapture* capture; // pointer for capture
   Mat frame; // matrix for each frame captured

   //Loading the Haar cascades and error handling
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //Opencv function to read the video
   capture = cvCaptureFromCAM( -1 );
   if( capture )//check if capture occured
   {
     while( true )
     {	// Change the resolution to 640*480 for reduced latency
		cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 640 );
		cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 480 );
		frame = cvQueryFrame( capture );// continuous video capture

	   // Apply Haar classifier to each frame captured
       if( !frame.empty() )
       { detectOutput( frame ); }
       else
       { printf("Empty Capture"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }
   }
   return 0;
 }

/* function to detect the face & features and displaying them */
void detectOutput( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );//grayscale conversion
  equalizeHist( frame_gray, frame_gray );

  //Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
	// outline the face with a green ellipse
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //outline the eyes with a red circle for each face in the image
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( frame, center, radius, Scalar(0, 0, 255 ), 4, 8, 0 );
     }
  }
  //output stream
  imshow( window_name, frame );
  //output stream can be saved as well using imwrite
 }
