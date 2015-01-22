/*=================================================================
* extract_pyra_feat.cc
*=================================================================*/


#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], 
				 int nrhs, const mxArray *prhs[])
// rhs (input):
// featPyra (nr x nc x feat_dim, single matrix) 
// h (scalar) 
// w (scalar) 
// num_region (scalar)
//
// lhs (output):
// featMat (h*w*feat_dim x num_region, single matrix)
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
    int h = mxGetScalar(prhs[1]);
    int w = mxGetScalar(prhs[2]);
    int num_region = mxGetScalar(prhs[3]);

    int num_row = dims[0];
    int num_col = dims[1];
    int feat_dim = dims[2];
    int feat_length = h*w*feat_dim;

    //mexPrintf("h=%d, w=%d, num_region=%d, num_row=%d, num_col=%d, feat_dim=%d, feat_length=%d\n", h, w, num_region, num_row, num_col, feat_dim, feat_length);


	// output
    plhs[0] = mxCreateNumericMatrix(feat_length, num_region, mxSINGLE_CLASS, mxREAL);    
    plhs[1] = mxCreateNumericMatrix(2, num_region, mxUINT16_CLASS, mxREAL);
	//plhs[1] = mxCreateNumericMatrix(2, num_region, mxSINGLE_CLASS, mxREAL);
	
    //const mwSize *dims_out = mxGetDimensions(plhs[0]);
    //mexPrintf("h=%d, w=%d\n", dims_out[0], dims_out[1]);

	// set output and input pointers
    float *featPyra = (float *)mxGetPr(prhs[0]);
	float *featMat = (float *)mxGetPr(plhs[0]);      
    unsigned short *featPos = (unsigned short *)mxGetPr(plhs[1]);   
    //float *featPos = (float *)mxGetPr(plhs[1]);        


    // code
    int region = 0;
    int dimCount, ndx;
    for (int nn=0; nn<num_row-h+1; nn++) {
        for (int mm=0; mm<num_col-w+1; mm++) {          
            dimCount = 0;
            for (int kk=0; kk<feat_dim; kk++) {
                for (int jj=mm; jj<mm+w; jj++) {
                    for (int ii=nn; ii<nn+h; ii++) {
                        ndx = kk*(num_row*num_col) + jj*num_row + ii;
                        featMat[region*feat_length+dimCount] = featPyra[ndx];
                        dimCount++;
                        //mexPrintf("dimCount=%d, kk=%d, jj=%d, ii=%d\n",dimCount,kk,jj,ii);
                    }
                }
            }
            featPos[region*2] = nn+1;
            featPos[region*2+1] = mm+1;

            region++;
        }
	}
}
 
