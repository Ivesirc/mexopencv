#include "mexopencv.hpp"
#include "math.h"
#include "opencv2/shape/shape_transformer.hpp" //Für cv::Ptr und cv::ThinPlateSplineShapeTransformer
#include <vector>


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    std::vector<cv::Point2f> SPoints, TPoints;
    std::vector<cv::DMatch> good_matches;
    cv::Mat SPicture, TPicture;
    
    MxArray SArray(prhs[0]); //Funktioniert mit Array
    MxArray TArray(prhs[1]);
    SPicture = MxArray(prhs[2]).toMat();
    
 	SPoints = MxArrayToVectorPoint<float>(SArray);
    TPoints = MxArrayToVectorPoint<float>(TArray);
    for (int i =0; i<SPoints.size(); i++) good_matches.push_back(cv::DMatch(i, i, 0));
    
    // Apply TPS
    cv::Ptr<cv::ThinPlateSplineShapeTransformer> mytps = cv::createThinPlateSplineShapeTransformer(0);
    mytps->estimateTransformation(SPoints, TPoints, good_matches); 

    mytps->warpImage(SPicture, TPicture);
    
    plhs[0] = MxArray(TPicture);
}