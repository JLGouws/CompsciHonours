#include <opencv2/videoio/videoio_c.h>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;

int main(int argc, char **argv)
{
  CvCapture *in_vid = cvCreateCameraCapture(0);
  cvNamedWindow ("window");
  IplImage *frame;
  while(true)
  {
    frame = cvQueryFrame(in_vid);
    cvShowImage("window", frame);
  }
  return 0;
}
