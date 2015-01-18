/*=================================================================
* extract_pyra_feat.cc
*=================================================================*/


#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], 
				 int nrhs, const mxArray *prhs[])
// rhs (input):
// featPyra (nr x nc x featdim, single matrix) 
// h (scalar) 
// w (scalar) 
// num_region (scalar)
//
// lhs (output):
// featMat (h*w*featdim x num_region, single matrix)
// featPos (2 x num_region, uint16 matrix)
{        
    /* Check for proper number of input and output arguments */    
    if (nrhs != 4) {
		mexErrMsgTxt("4 input arguments required.");
    } 
    if (nlhs != 2){
		mexErrMsgTxt("2 output arguments required.");
    }
    
	// parameters
    const mwSize *dims = mxGetDimensions(prhs[0]);  // get number of dims
    size_t h = mxGetScalar(prhs[1]);
    size_t w = mxGetScalar(prhs[2]);
    size_t num_region = mxGetScalar(prhs[3]);

    size_t numRow = dims[0];
    size_t numCol = dims[1];
    size_t featDim = dims[2];
    size_t featLength = h*w*featDim;

	// output
    plhs[0] = mxCreateNumericMatrix(featLength, num_region, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(2, num_region, mxUINT16_CLASS, mxREAL);
	
	// set output and input pointers
    float *featPyra = (float *)mxGetData(prhs[0]);
	float *featMat = (float *)mxGetData(plhs[0]);
    unsigned *featPos = (unsigned *)mxGetData(plhs[1]);        


    // code
    int region = 0;
    int dimCount, ndx;
    for (size_t nn=0; nn<numRow-h+1; nn++) {
        for (size_t mm=0; mm<numCol-w+1; mm++) {              
            dimCount = 0;
            for (size_t kk=0; kk<featDim; kk++) {
                for (size_t jj=mm; jj<mm+w-1; jj++) {
                    for (size_t ii=nn; ii<nn+h-1; ii++) {
                        ndx = kk*featDim + jj*numCol + ii;
                        featMat[region*featLength+dimCount] = featPyra[ndx];
                        dimCount++;
                    }
                }
            }
            featPos[region*2+1] = nn;
            featPos[region*2+2] = mm;

            region++;
        }
	}
}
 
