/* diffusion2d.c: sequential version of 2d diffusion.
 * The length of the side of the square is 1. Initially entire square
 * is 100 degrees, but edges are held at 0 degrees.
 *
 * Author: Stephen F. Siegel
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <gd.h>
#include <cuda_runtime.h>
#define MAXCOLORS 256

/* Constants: the following should be defined at compilation:
 *
 *       M = initial temperature at center
 *  NSTEPS = number of time steps
 *   WSTEP = write frame every this many steps
 *      NX = number of points in x direction, including endpoints
 *       K = D*dt/(dx*dx)
 * 
 */

/* Global variables */
int nx = NX;              /* number of discrete points including endpoints */
double m = M;             /* initial temperature of rod */
double k = K;             /* D*dt/(dx*dx) */
int nsteps = NSTEPS;      /* number of time steps */
double dx;                /* distance between two grid points: 1/(nx-1) */
double **u[2];            /* temperature function */
double *u_storage[2];     /* storage for u */
double * device_u_storage[2];
FILE *file;               /* file containing animated GIF */
gdImagePtr im, previm;    /* pointers to consecutive GIF images */
int *colors;              /* colors we will use */
int framecount = 0;       /* number of animation frames written */


/* init: initializes global variables. */
void init() {
  int i, j, t;
  
  printf("Diffusion2d with k=%f, M=%f, nx=%d, nsteps=%d\n",
	 k, m, nx, nsteps);
  fflush(stdout);
  assert(k>0 && k<.5);
  assert(m>=0);
  assert(nx>=2);
  assert(nsteps>=1);
  dx = 1.0/(nx-1);
  for (t=0; t<2; t++) {
    u_storage[t] = (double*)malloc(nx*nx*sizeof(double));
    assert(u_storage[t]);
    u[t] = (double**)malloc(nx*sizeof(double*));
    assert(u[t]);
    for (i=0; i<nx; i++)
      u[t][i]=&u_storage[t][i*nx];
    for (i=1; i<nx-1; i++)
      for (j=1; j<nx-1; j++)
	u[t][i][j] = m;
#if BOUNDARY == 2
    for (i=1; i<nx-1; i++) {
      u[t][i][0] = u[t][0][i] = 0.0;
      u[t][i][nx-1] = u[t][nx-1][i] = m;
    }
    u[t][0][0] = 0.0;
    u[t][nx-1][nx-1] = m;
    u[t][nx-1][0] = u[t][0][nx-1] = m/2;
#else
    for (i=1; i<nx-1; i++)
      u[t][i][0] = u[t][i][nx-1] = u[t][0][i] = u[t][nx-1][i] = 0.0;
    u[t][0][0] = u[t][0][nx-1] = u[t][nx-1][0] = u[t][nx-1][nx-1] = 0.0;
#endif
  }
  file = fopen("./parout/out.gif", "wb");
  assert(file);
  colors = (int*)malloc(MAXCOLORS*sizeof(int));
  assert(colors);
}

/* write_plain: write current data to plain text file and stdout */
void write_plain(int time) {
  FILE *plain;
  char filename[50];
  char command[50];
  int i,j;
  
  sprintf(filename, "./parout/out_%d", time);
  plain = fopen(filename, "w");
  assert(plain);
  for (i=nx-1; i>=0; i--) {
    for (j=0; j<nx; j++) {
      fprintf(plain, "%8.2f", u[time%2][i][j]);
    }
    fprintf(plain, "\n");
  }
  fprintf(plain, "\n");
  fclose(plain);
  sprintf(command, "cat %s", filename);
  system(command);
}

/* write_frame: add a frame to animation */
void write_frame(int time) {
  int i,j;
  
  im = gdImageCreate(nx*PWIDTH,nx*PWIDTH);
  if (time == 0) {
    for (j=0; j<MAXCOLORS; j++)
      colors[j] = gdImageColorAllocate (im, j, 0, MAXCOLORS-j-1); 
    /* (im, j,j,j); gives gray-scale image */
    gdImageGifAnimBegin(im, file, 1, -1);
  } else {
    gdImagePaletteCopy(im, previm);
  }
  for (i=0; i<nx; i++) {
    for (j=0; j<nx; j++) {
      int color = (int)(u[time%2][i][j]*MAXCOLORS/M);

      assert(color >= 0);
      if (color >= MAXCOLORS) color = MAXCOLORS-1;
      gdImageFilledRectangle(im, j*PWIDTH, i*PWIDTH,
			     (j+1)*PWIDTH-1, (i+1)*PWIDTH-1, colors[color]);
    }
  }
  if (time == 0) {
    gdImageGifAnimAdd(im, file, 0, 0, 0, 0, gdDisposalNone, NULL);
  } else {
    // Following is necessary due to bug in gd.
    // There must be at least one pixel difference between
    // two consecutive frames.  So I keep flipping one pixel.
    // gdImageSetPixel (gdImagePtr im, int x, int y, int color);
    gdImageSetPixel(im, 0, 0, framecount%2);
    gdImageGifAnimAdd(im, file, 0, 0, 0, 5, gdDisposalNone, previm /*NULL*/);
    gdImageDestroy(previm);
  }
  previm=im;
  im=NULL;
#ifdef DEBUG
  write_plain(time);
#endif
  framecount++;
}

/* updates u for next time step. */

__global__ void update(double * u_prev, double * u_curr){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    //border
    if(i > 0 && i < NX-1 && j > 0 && j < NX-1){
      u_curr[NX*i + j] = u_prev[NX*i + j] + 
	K*(u_prev[NX*(i+1) + j] +
	   u_prev[NX*(i-1) + j] + u_prev[NX*i + (j+1)] + u_prev[NX*i + (j-1)] -
	   4*u_prev[NX*i + j]);
    }
}

/* main: executes simulation, creates one output file for each time
 * step */
int main(int argc,char *argv[]) {
  
  init();
  write_frame(0);
  
  for (int t = 0; t < 2; t++) {
    cudaMalloc((void**)&device_u_storage[t], NX*NX*sizeof(double));
    cudaMemcpy(device_u_storage[t], u_storage[t], NX*NX*sizeof(double), cudaMemcpyHostToDevice);
  }
  
  int checkerSize = min(NX, 32);
  dim3 threadsPerBlock(checkerSize, checkerSize);
  double blocks1 = (double)(NX)/(double)(threadsPerBlock.x);
  int blocks2 = NX/threadsPerBlock.x;
  if(blocks1 > blocks2) blocks2++;
  dim3 numBlocks(blocks2, blocks2);

  for (int i = 1; i <= nsteps; i++) {
    if(i%2 == 0)
      update<<<numBlocks, threadsPerBlock>>>(device_u_storage[0], device_u_storage[1]);
    else
      update<<<numBlocks, threadsPerBlock>>>(device_u_storage[1], device_u_storage[0]);
    if (WSTEP!=0 && i%WSTEP==0){
      for(int t = 0; t < 2; t++){ 
	cudaError_t err = cudaMemcpy(u_storage[t], device_u_storage[t], NX*NX*sizeof(double), cudaMemcpyDeviceToHost);
	if(err != 0)
	  printf("ERR %s\n", cudaGetErrorString(err));
      }
      write_frame(i);
    }
  }
  gdImageDestroy(previm);
  gdImageGifAnimEnd(file);
  fclose(file);
  free(colors);
  for (int t=0; t<2; t++) {
    free(u_storage[t]);
    free(u[t]);
    cudaFree(device_u_storage);
  }
  return 0;
}
