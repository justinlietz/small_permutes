#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cutt.h>

//
// Error checking wrapper for cutt
//
#define cuttCheck(stmt) do {                                 \
  cuttResult err = stmt;                            \
  if (err != CUTT_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)

size_t idx(int i, int j, int k, int l, int* dims){
  size_t index = l*dims[0]*dims[1]*dims[2]
                +k*dims[0]*dims[1]
                +j*dims[0]
                +i;
  return index;
}

int equal_4tensors(double* data1, double*data2, int dims[4]){
  double diff;
  double tol = 1.e-10;
  for(int d3 = 0; d3 < dims[3]; d3++){
  for(int d2 = 0; d2 < dims[2]; d2++){
  for(int d1 = 0; d1 < dims[1]; d1++){
  for(int d0 = 0; d0 < dims[0]; d0++){
    size_t index = idx(d0,d1,d2,d3,dims);
    diff = fabs(data1[index] - data2[index]);
    if( diff > tol ){
      printf("diff at: %d,%d,%d,%d: %zu\n", d0,d1,d2,d3,index);
      return 0;
    }
  }
  }
  }
  }
  return 1;
}


void load_4tensor(double* data, int dims[4]){
  for(int d3 = 0; d3 < dims[3]; d3++){
  for(int d2 = 0; d2 < dims[2]; d2++){
  for(int d1 = 0; d1 < dims[1]; d1++){
  for(int d0 = 0; d0 < dims[0]; d0++){
    size_t index = idx(d0,d1,d2,d3,dims);
    /* printf("%d,%d,%d,%d: %zu\n", d0,d1,d2,d3,index); */
    data[index] = (double)d0*d1/d3+d2;
  }
  }
  }
  }
}

void print_4tensor(double* data, int dims[4]){
  for(int d3 = 0; d3 < dims[3]; d3++){
  for(int d2 = 0; d2 < dims[2]; d2++){
  for(int d1 = 0; d1 < dims[1]; d1++){
  for(int d0 = 0; d0 < dims[0]; d0++){
    size_t index = idx(d0,d1,d2,d3,dims);
    printf("%d,%d,%d,%d: %zu\n", d0,d1,d2,d3,index);
  }
  }
  }
  }
}

void transpose_4tensor(double* idata, double* odata, int dims[4], int perm[4]){
  int outdims[4];
  int outidx[4];
  int inidx[4];
  for( int i = 0; i<4; i++){
    outdims[i] = dims[ perm[i] ];
  }
  printf("outdims: %d %d %d %d\n",outdims[0], outdims[1], outdims[2], outdims[3]);
  for(int d3 = 0; d3 < dims[3]; d3++){
  for(int d2 = 0; d2 < dims[2]; d2++){
  for(int d1 = 0; d1 < dims[1]; d1++){
  for(int d0 = 0; d0 < dims[0]; d0++){
    inidx[0] = d0;
    inidx[1] = d1;
    inidx[2] = d2;
    inidx[3] = d3;
    for( int i = 0; i<4; i++){
      outidx[i] = inidx[ perm[i] ];
    }

    size_t iindex = idx(d0,d1,d2,d3,dims);
    size_t oindex = idx(outidx[0],outidx[1],outidx[2],outidx[3],outdims);
    odata[oindex] = idata[iindex];
  }
  }
  }
  }
}

int main() {

  // Four dimensional tensor
  // Transpose (31, 549, 2, 3) -> (3, 31, 2, 549)
  /* int dim[4] = {31, 549, 2, 3}; */
  /* int odim[4] = {3, 31, 2, 549}; */
  int dim[4] = {310, 5490, 2, 3};
  int odim[4] = {3, 310, 2, 5490};

  int permutation[4] = {3, 0, 2, 1};
  int nElems = 1;
  int nBytes;
  for(int i=0; i<4; i++){
    nElems = nElems*dim[i];
  }
  nBytes = sizeof(double)*nElems;
  printf("nBytes: %f Gb\n", nBytes/1.e9);

  /* .... input and output data is setup here ... */
  // double* idata : size product(dim)
  // double* odata : size product(dim)
  double* idata;
  double* ref_data;
  double* cutt_odata;
  double* d_idata;
  double* d_cutt_odata;

  idata = (double*)malloc(nBytes);
  ref_data = (double*)malloc(nBytes);
  cutt_odata = (double*)malloc(nBytes);
  cudaMalloc((void**)&d_idata,nBytes);
  cudaMalloc((void**)&d_cutt_odata,nBytes);


  load_4tensor(idata,dim);
  transpose_4tensor(idata,ref_data,dim,permutation);
  printf("nElems: %zu\n",nElems);

  cudaMemcpy(d_idata,idata,nBytes,cudaMemcpyHostToDevice);


  // Option 1: Create plan on NULL stream and choose implementation based on heuristics
  cuttHandle plan;
  cuttCheck(cuttPlan(&plan, 4, dim, permutation, sizeof(double), 0));

  // Option 2: Create plan on NULL stream and choose implementation based on performance measurements
  // cuttCheck(cuttPlanMeasure(&plan, 4, dim, permutation, sizeof(double), 0, idata, odata));

  // Execute plan
  cuttCheck(cuttExecute(plan, d_idata, d_cutt_odata));
  cudaMemcpy(cutt_odata,d_cutt_odata,nBytes,cudaMemcpyDeviceToHost);

  int ans;
  ans = equal_4tensors(ref_data, cutt_odata, odim);
  printf("ans: %d\n", ans);

  /* ... do stuff with your output and deallocate data ... */

  // Destroy plan
  cuttCheck(cuttDestroy(plan));

  return 0;
}


