#include "mexopencv.hpp"
#include "CThinPlateSpline.h"
#include "math.h"
#include <vector>

void mexFunction( int nlhs, mxArray *plhs[],
int nrhs, const mxArray *prhs[] )
{
/*/ Check arguments
if (nlhs!=1 || nrhs!=1)
    mexErrMsgIdAndTxt("myfunc:invalidArgs","Wrong number of arguments");
 */
    cv::Mat_<double> img_src = MxArray(prhs[0]).toMat();
	cv::Mat img_dst;
    
    cv::Mat landmarksP= MxArray(prhs[1]).toMat();
    cv::Mat landmarksS= MxArray(prhs[2]).toMat();
    std::vector<cv::Point2d> Zp;
    std::vector<cv::Point2d> Zs;
    for (int i = 0; i < landmarksP.rows; i++)
    {
        std::vector<double> rowP;    
        landmarksP.row(i).copyTo(rowP);
        cv::Point2d tempP(rowP[0]-1,rowP[1]-1);
		//cv::Point2d tempP(rowP[0],rowP[1]);
        Zp.push_back(tempP);
        
        std::vector<double> rowS;    
        landmarksS.row(i).copyTo(rowS);
        cv::Point2d tempS(rowS[0]-1,rowS[1]-1);
		//cv::Point2d tempS(rowS[0],rowS[1]);
        Zs.push_back(tempS);
    }
    
    //cv::Mat img = cv::imread("X:\Bachelorarbeit Inés Fandos\matlab code\Semmler 35\pict000.jpg",0);
    //img.convertTo(img,CV_64FC1);

    
    CThinPlateSpline tps(Zp, Zs);
       
    tps.warpImage(img_src, img_dst, 0.0, INTER_LANCZOS4, FORWARD_WARP); //INTER_NEAREST INTER_LINEAR INTER_CUBIC INTER_LANCZOS4

   
    
    plhs[0] = MxArray(img_dst);

}