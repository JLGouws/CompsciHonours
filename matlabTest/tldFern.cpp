#include "tldFern.hpp"

#include <opencv2/core/utility.hpp>

using namespace std;
namespace cv
{
	namespace tld
	{
    TLDFern::TLDFern() {
      nTREES    = 10;
      nFEAT     = 13;
      thrN      = 0.5 * nTREES;
      //nSCALE    = ;

      double *x = (double*) malloc(4 * nTREES * nFEAT * sizeof(double));
      generateFeatures(nTREES, nFEAT, x);

      for (int i = 0; i<nTREES; i++) { 
        WEIGHT.push_back(vector<double>(pow(2.0,nBIT*nFEAT), 0));
        nP.push_back(vector<int>(pow(2.0,nBIT*nFEAT), 0));
        nN.push_back(vector<int>(pow(2.0,nBIT*nFEAT), 0));
      } 
 
      for (int i = 0; i<nTREES; i++)
        for (unsigned int j = 0; j < WEIGHT[i].size(); j++) {
          WEIGHT[i].at(j) = 0;
          nP[i].at(j) = 0;
          nN[i].at(j) = 0;
        }
      free(x);
    }
    TLDFern::~TLDFern(){}

    void TLDFern::update(){}
    void TLDFern::evaluate(){}
    bool TLDFern::detect(){
      return false;
    }
    void TLDFern::getPatterns() {}

		// Computes the variance of subimage given by box, with the help of two integral
		// images intImgP and intImgP2 (sum of squares), which should be also provided.
		bool TLDFern::patchVariance(Mat_<double>& intImgP, Mat_<double>& intImgP2, double *originalVariance, Point pt, Size size)
		{
			int x = (pt.x), y = (pt.y), width = (size.width), height = (size.height);
			CV_Assert(0 <= x && (x + width) < intImgP.cols && (x + width) < intImgP2.cols);
			CV_Assert(0 <= y && (y + height) < intImgP.rows && (y + height) < intImgP2.rows);
			double p = 0, p2 = 0;
			double A, B, C, D;

			A = intImgP(y, x);
			B = intImgP(y, x + width);
			C = intImgP(y + height, x);
			D = intImgP(y + height, x + width);
			p = (A + D - B - C) / (width * height);

			A = intImgP2(y, x);
			B = intImgP2(y, x + width);
			C = intImgP2(y + height, x);
			D = intImgP2(y + height, x + width);
			p2 = (A + D - B - C) / (width * height);

			return ((p2 - p * p) > VARIANCE_THRESHOLD * *originalVariance);
		}

    double randEx() {
      return (1 + rand()) / (1 + (unsigned) RAND_MAX);
    }

    void generateFeatures(int nTrees, int nFeat, double *features) {
      srand(42);
      double SHIFT  = 0.2,
             SCALE  = 1,
             OFF    = SHIFT,
             r[4][72],        //second index should me malloc'ed for general use
             l[4][72],
             b[4][72],
             t[4][72];

      int count = 0;
      vector<pair<int, int>> indices = vector<pair<int, int>>();
      for (int i = 0; i <= (int) (1 / SHIFT); i++)
        for (int j = 0; j <= (int) (1 / SHIFT); j++) {
          r[0][count] = l[0][count] = b[0][count] = t[0][count] = i * SHIFT;
          r[1][count] = l[1][count] = b[1][count] = t[1][count] = j * SHIFT;
          r[2][36 + count] = l[2][36 + count] 
            = b[2][36 + count] = t[2][36 + count] = (i + 0.5) * SHIFT;
          r[3][36 + count] = l[3][36 + count] 
            = b[3][36 + count] = t[3][36 + count] = (j + 0.5) * SHIFT;
          count++;
          r[2][count] += SCALE * randEx() + OFF; 
          r[2][36 + count] += SCALE * randEx() + OFF;
          l[2][count] -= SCALE * randEx() + OFF; 
          l[2][36 + count] -= SCALE * randEx() + OFF;
          t[3][count] -= SCALE * randEx() + OFF; 
          t[3][36 + count] -= SCALE * randEx() + OFF;
          b[3][count] += SCALE * randEx() + OFF; 
          b[3][36 + count] += SCALE * randEx() + OFF;
          if (0 < i && i < (int) (1 / SHIFT) && 0 < j && j < (int) (1 / SHIFT)) {
            indices.push_back(pair<int, int>(count, 0));
            indices.push_back(pair<int, int>(count, 1));
            indices.push_back(pair<int, int>(count, 2));
            indices.push_back(pair<int, int>(count, 3));
          }
        }
      }

	}
}
