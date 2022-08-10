#ifndef OPENCV_TLD_FERN
#define OPENCV_TLD_FERN

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <vector>
#include <set>
#include <map>
#include <list>

using namespace std;
namespace cv
{
	namespace tld
	{
    const double VARIANCE_THRESHOLD = 0.5;
		class TLDFern
		{
		public:
      TLDFern();
      ~TLDFern();

			void init();
			void update();
			void evaluate();
			bool detect();
			void getPatterns();

			static void computeIntegralImages(const Mat& img, Mat_<double>& intImgP, Mat_<double>& intImgP2){ integral(img, intImgP, intImgP2, CV_64F); }
			static inline bool patchVariance(Mat_<double>& intImgP, Mat_<double>& intImgP2, double *originalVariance, Point pt, Size size);

      private:
        static double thrN;
        static int nBBOX;
        static int mBBOX;
        static int nTREES;
        static int nFEAT;
        static int nSCALE;
        static int iHEIGHT;
        static int iWIDTH;
        static int *BBOX;
        static int *OFF;
        static double *IIMG;
        static double *IIMG2;
        static vector<vector <double> > WEIGHT;
        static vector<vector <int> > nP;
        static vector<vector <int> > nN;
        const static int BBOX_STEP = 7;
        const static int nBIT = 1; // number of bits per feature
        void generateFeatures(int nTrees, int nFeat, double *features);

    };
	}
}

#endif
