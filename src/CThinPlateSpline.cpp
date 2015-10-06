/*
* CThinPlateSpline.cpp
*
*  Created on: 24.01.2010
*      Author: schmiedm
*/

//#define _CRT_SECURE_NO_DEPRECATE

#include <vector>
#include "mex.h"
//#include "opencv2/opencv.hpp" //test
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "CThinPlateSpline.h"

using namespace cv;

CThinPlateSpline::CThinPlateSpline() {
}

CThinPlateSpline::CThinPlateSpline(const std::vector<Point2d>& pS, const std::vector<Point2d>& pD)
{
	if(pS.size() == pS.size())
	{
		pSrc = pS;
		pDst = pD;
	}
}

CThinPlateSpline::~CThinPlateSpline() {
}

void CThinPlateSpline::addCorrespondence(const Point2d& pS, const Point2d& pD)
{
	pSrc.push_back(pS);
	pDst.push_back(pD);
	
}

void CThinPlateSpline::setCorrespondences(const std::vector<Point2d>& pS, const std::vector<Point2d>& pD)
{
	pSrc = pS;
	pDst = pD;
}

double CThinPlateSpline::fktU(const Point2d& p1, const Point2d& p2) 
{
	double r2 = pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2);

	if (r2 == 0)
		return 0.0;
	else 
	{
		//double r = sqrt(r2);

		return (r2 * log(r2));
	}
}

void CThinPlateSpline::computeSplineCoeffs(std::vector<Point2d>& iPIn, std::vector<Point2d>& iiPIn, float lambda,const TPS_INTERPOLATION tpsInter)
{

	std::vector<Point2d>* iP = NULL;
	std::vector<Point2d>*	iiP = NULL;

	if(tpsInter == FORWARD_WARP)
	{
		iP = &iPIn;
		iiP = &iiPIn;
	}
	else if(tpsInter == BACK_WARP)
	{
		iP = &iiPIn;
		iiP = &iPIn;
	}

	//get number of corresponding points
	int dim = 2;
	int n = iP->size();

	//Initialize mathematical datastructures
	Mat_<float> V(dim,n+dim+1,0.0);
	Mat_<float> P(n,dim+1,1.0);
	Mat_<float> K = (K.eye(n,n)*lambda);
	Mat_<float> L(n+dim+1,n+dim+1,0.0);

	// fill up K und P matrix
	std::vector<Point2d>::iterator itY;
	std::vector<Point2d>::iterator itX;

	int y = 0;
	for (itY = iP->begin(); itY != iP->end(); ++itY, y++) {
		int x = 0;
		for (itX = iP->begin(); itX != iP->end(); ++itX, x++) {
			if (x != y) {
				K(y, x) = (float)fktU(*itY, *itX);
			}
		}
		P(y,1) = (float)itY->x;
		P(y,2) = (float)itY->y;
	}

	Mat Pt;
	transpose(P,Pt);

	// insert K into L
	Rect range = Rect(0, 0, n, n);
	Mat Lr(L,range);
	K.copyTo(Lr);


	// insert P into L
	range = Rect(n, 0, dim + 1, n);
	Lr = Mat(L,range);
	P.copyTo(Lr);

	// insert Pt into L
	range = Rect(0,n,n,dim+1);
	Lr = Mat(L,range);
	Pt.copyTo(Lr);

	// fill V array
	std::vector<Point2d>::iterator it;
	int u = 0;

	for(it = iiP->begin(); it != iiP->end(); ++it)
	{
		V(0,u) = (float)it->x;
		V(1,u) = (float)it->y;
		u++;
	}

	// transpose V
	Mat Vt;
	transpose(V,Vt);

	cMatrix = Mat_<float>(n+dim+1,dim,0.0);

	// invert L
	Mat invL;
	invert(L,invL,DECOMP_LU);

	//multiply(invL,Vt,cMatrix);
	cMatrix = invL * Vt;

	//compensate for rounding errors
	/*for(int row = 0; row < cMatrix.rows; row++)
	{
		for(int col = 0; col < cMatrix.cols; col++)
		{
			double v = cMatrix(row,col);
			if(v > (-1.0e-006) && v < (1.0e-006) )
			{
				cMatrix(row,col) = 0.0;
			}
		}
	}*/
}


Point2d CThinPlateSpline::interpolate_forward_(const Point2d& p)
{
	Point2d interP;
	std::vector<Point2d>* pList = &pSrc;

	int k1 = cMatrix.rows - 3;
	int kx = cMatrix.rows - 2;
	int ky = cMatrix.rows - 1;

	double a1 = 0, ax = 0, ay = 0, cTmp = 0, uTmp = 0, tmp_i = 0, tmp_ii = 0;

	for (int i = 0; i < 2; i++) {
		a1 = cMatrix(k1,i);
		ax = cMatrix(kx,i);
		ay = cMatrix(ky,i);

		tmp_i = a1 + ax * p.x + ay * p.y;

		for (int j = 0; j < (int)pSrc.size(); j++) {
			cTmp = cMatrix(j,i);
			uTmp = fktU( (*pList)[j], p);

			tmp_i = tmp_i + (cTmp * uTmp);
		}

		if (i == 0) {
			interP.x = (tmp_i);
		}
		if (i == 1) {

			interP.y = (tmp_i);
		}
	}

	return interP;
}
Point2d CThinPlateSpline::interpolate_back_(const Point2d& p)
{
	Point2d interP;
	std::vector<Point2d>* pList = &pDst;

	int k1 = cMatrix.rows - 3;
	int kx = cMatrix.rows - 2;
	int ky = cMatrix.rows - 1;

	double a1 = 0, ax = 0, ay = 0, cTmp = 0, uTmp = 0, tmp_i = 0, tmp_ii = 0;

	for (int i = 0; i < 2; i++) {
		a1 = cMatrix(k1,i);
		ax = cMatrix(kx,i);
		ay = cMatrix(ky,i);

		tmp_i = a1 + ax * p.x + ay * p.y;
		tmp_ii = 0;

		for (int j = 0; j < (int)pSrc.size(); j++) {
			cTmp = cMatrix(j,i);
			uTmp = fktU( (*pList)[j], p);

			tmp_ii = tmp_ii + (cTmp * uTmp);
		}

		if (i == 0) {
			interP.x = (tmp_i + tmp_ii);
		}
		if (i == 1) {
			interP.y = (tmp_i + tmp_ii);
		}
	}

	return interP;
}

Point2d CThinPlateSpline::interpolate(const Point2d& p, const TPS_INTERPOLATION tpsInter)
{
	if(tpsInter == BACK_WARP)
	{
		return interpolate_back_(p);
	}
	else if(tpsInter == FORWARD_WARP)
	{
		return interpolate_forward_(p);
	}
	else
	{
		return interpolate_back_(p);
	}
	
}

void CThinPlateSpline::warpImage(const Mat& src, Mat& dst, float lambda, const int interpolation,const TPS_INTERPOLATION tpsInter)
{
	Size size = src.size();
	dst = Mat(size,src.type());

	// only compute the coefficients new if they weren't already computed
	// or there had been changes to the points
	if(tpsInter == BACK_WARP)
	{
		computeSplineCoeffs(pSrc,pDst,lambda,tpsInter);
	}
	else if(tpsInter == FORWARD_WARP)
	{
		computeSplineCoeffs(pSrc,pDst,lambda,tpsInter);
	}
	
	computeMaps(size,mapx,mapy,tpsInter);

	remap(src,dst,mapx,mapy,interpolation);
	//computeMaps_forward(size,mapx,mapy,tpsInter);
    //remap_forward(src,dst,mapx,mapy);
}

void CThinPlateSpline::remap_forward(const Mat& src, Mat& dst, Mat& mx, Mat&my)
{
	int temp_mx = 0;
	int temp_my = 0;
	double temp_src = 0;

    for (int row = 0; row < dst.rows-1; row++) {
		for (int col = 0; col < dst.cols-1; col++) {
			//dst(row, col) = 1;
			temp_mx = mx.at<float>(row,col);
			temp_my = my.at<float>(row,col);
			if (col == 120)
				int a = 0 ;
			if (temp_mx >src.cols-1  || temp_my >src.rows-1  || temp_mx<0 || temp_my <0)
			{
				dst.at<double>(row,col) = 0;
			}
			else
			{
				temp_src = src.at<double>(temp_my,temp_mx);
				dst.at<double>(row,col) = temp_src;
			}
		}

	};
}

void CThinPlateSpline::getMaps(Mat& mx, Mat& my)
{
	mx = mapx;
	my = mapy;
}

void CThinPlateSpline::computeMaps(const Size& dstSize, Mat_<float>& mx, Mat_<float>& my,const TPS_INTERPOLATION tpsInter)
{
	mx = Mat_<float>(dstSize);
	my = Mat_<float>(dstSize);

	Point p(0, 0);
	Point2d intP(0, 0);
	
	for (int row = 0; row < dstSize.height; row++) {
		for (int col = 0; col < dstSize.width; col++) {
			p = Point(row, col);
			if (p.y==1 && p.x == 1)
				int a = 0;
			intP = interpolate(p,tpsInter);
			mx(row, col) = intP.y;
			my(row, col) = intP.x;
		}
	}
}

void CThinPlateSpline::computeMaps_forward(const Size& dstSize, Mat_<float>& mx, Mat_<float>& my,const TPS_INTERPOLATION tpsInter)
{
	mx = Mat_<int>(dstSize);
	my = Mat_<int>(dstSize);

	Point p(0, 0);
	Point2d intP(0, 0);
	
	for (int row = 0; row < dstSize.height; row++) {
		for (int col = 0; col < dstSize.width; col++) {
			p = Point(row, col);
			if (p.y==1 && p.x == 1)
				int a = 0;
			intP = interpolate(p,tpsInter);
			mx(row, col) = (int)intP.y;
			my(row, col) = (int)intP.x;
		}
	}
}