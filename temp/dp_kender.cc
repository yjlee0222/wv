/*=================================================================
* dp_kender.cc
*=================================================================*/

#include <math.h>
#include "mex.h"

#define NEGINF -1000000.0

void mexFunction(int nlhs, mxArray *plhs[], 
				 int nrhs, const mxArray *prhs[])
// rhs (input):
// sceneDist (m x m matrix) 
// k (scalar)
// lhs (output):
// costMat (m x k matrix)
// indexMat (m x k matrix)
{
        
    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
		mexErrMsgTxt("2 input arguments required.");
    } 
    if (nlhs != 2){
		mexErrMsgTxt("2 output arguments required.");
    }
    
	// parameters
    mwSize m = mxGetM(prhs[0]);  // get number of rows
    size_t k = mxGetScalar(prhs[1]);

	// output
    plhs[0] = mxCreateDoubleMatrix(m, k, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(m, k, mxREAL);
	
	// set output and input pointers
	double* costMat = mxGetPr(plhs[0]);
    double* indexMat = mxGetPr(plhs[1]);    
    double* sceneDist = mxGetPr(prhs[0]);	

    for (size_t i=0; i<m; i++) {
        costMat[i] = 0.0;           
	}
    
    double val = 0.0;
    double ind = 0.0;
    double maxVal = NEGINF; // in place of neg infinity
    double thisVal = 0.0;
    double maxInd = 0.0;

    for (size_t t=1; t<k; t++) {
        for (size_t j=t; j<=m-k+t; j++) {  
            maxVal = NEGINF;
            for (size_t i=t-1; i<=j-1; i++) {
                thisVal = sceneDist[j*m+i] + costMat[(t-1)*m+i];
                if (thisVal > maxVal) {
                    maxVal = thisVal;
                    maxInd = i;
                }            
            }
            costMat[t*m+j] = maxVal;
            indexMat[t*m+j] = maxInd+1; // since matlab indices start at 1
        }
	}   

}
 
