 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>
 #include <unistd.h>
 #include <sys/types.h>
 #include <errno.h>
 #include <stdio.h>
 #include <sys/wait.h>
 #include <stdlib.h>
 #include <pthread.h>
 #define NUM_THREADS 2
 
 using namespace std;
 using namespace cv;

 /** Function Declaration */
 void* detectAndDisplay_face( void* args);
 void* detectAndDisplay_eyes( void* args);

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);
 vector <Mat> imageQueue;
 static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


 int main( int argc, const char** argv )
 {
   CvCapture* capture;
   Mat frame;

   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };


   
   capture = cvCaptureFromCAM( -1 );
   if( capture )
   {
	   
     while( true )
     {
		frame = cvQueryFrame( capture );
		if(!frame.empty())
		{
			pthread_t thid[NUM_THREADS];
			// Create the threads
			pthread_create(&thid[0], NULL, detectAndDisplay_face, NULL);
			//pthread_create(&thid[1], NULL, detectAndDisplay_eyes, NULL);
			//
			pthread_join(thid[0],NULL);
			//pthread_join(thid[1],NULL);
	          	
		}
		else{printf("EMPTY FRAME");break;}
	}
	}
	return 0;
}

void* detectAndDisplay_face( void* args)
{
  std::vector<Rect> faces;
  Mat frame_gray;
  Mat frame;
  pthread_mutex_lock(&mutex);
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );


  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( frame, center, radius, Scalar(0, 0, 255 ), 4, 8, 0 );
     }
  }
imshow(window_name, frame);
 pthread_mutex_unlock(&mutex);


}


