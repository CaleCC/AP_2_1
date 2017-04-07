#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"
#include <vector>
#include <algorithm>
#define NUM_THREADS 128

//
//  benchmarking program
//

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(particle_t * particles, int n, int bins_row, int* counter,double binSize)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int step = blockDim.x * gridDim.x;

  for(int i = tid; i < n; i+=step){
      particle_t p = particles[i];//use local variables here maybe faster
      p.ax = p.ay = 0;
      int x = floor(p.x/binSize);
      int y = floor(p.y/binSize);

      //determine the neighbours
      int x_start = -1;
      int x_end = 1;
      int y_start = -1;
      int y_end = 1;
      if(x == 0) x_start = 0;
      if(x == bins_row-1) x_end = 0;
      if(y == 0) y_start = 0;;
      if(y == bins_row - 1) y_end = 0;
      //calculate force
      for(int xx=x_start;xx<=x_end;xx++){
        for(int yy=y_start;yy<=y_end;yy++){
          int loc = (x+xx)+(y+yy)*bins_row;
          //printf("loc: %d\n",loc);
          for(int m = counter[loc-1];m<counter[loc];m++){
            //printf("m %d\n",m);
            apply_force_gpu(p,particles[m]);
          }
        }
      }
      particles[i].ax = p.ax;//copy back to shared memory
      particles[i].ay = p.ay;

  }
}

/*__global__ void move_gpu (particle_t * particles,
                          int n,
                          double size,
                          int* counter,
                          int bins_row,
                          particle_t *bin_seperate_p,
                          int* return_counter,
                          int off_set)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int step = blockDim.x*gridDim.x;
  //if(tid >= bins_row*bins_row) return;
  for(int i = tid; i < bins_row*bins_row; i+=step){
    for(int j = 0; j < counter[i];j++){

      if(i*off_set == 0){
        printf("particles %d x %f  y %f\n",j,bin_seperate_p[i*off_set+j].x, bin_seperate_p[i*off_set+j].y );
      }
      particle_t * p = &bin_seperate_p[i*off_set+j];
      //
      //  slightly simplified Velocity Verlet integration
      //  conserves energy better than explicit Euler method
      //
      p->vx += p->ax * dt;
      p->vy += p->ay * dt;
      p->x  += p->vx * dt;
      p->y  += p->vy * dt;

      //
      //  bounce from walls
      //
      while( p->x < 0 || p->x > size )
      {
          p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
          p->vx = -(p->vx);
      }
      while( p->y < 0 || p->y > size )
      {
          p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
          p->vy = -(p->vy);
      }
      particles[return_counter[0]] = *p;
      if(return_counter[0] == 0){
        printf("particles0  x %f  y %f\n",p->x, p->y );
      }

      atomicAdd(return_counter,1);
    }
  }

}*/
__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x*blockDim.x;
  for(int i = tid;i<n;i+=stride){
    particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }
  }

}
//count the particles in each bin
__global__ void countParticles(particle_t *d_particles,int n,int* counter, double binSize, int bins_row){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    for(int i = threadId; i < n; i+=step){
      //cal which bin the particle belongs to
      int x = floor(d_particles[i].x / binSize);
      int y = floor(d_particles[i].y / binSize);
      atomicAdd(counter+x + y * bins_row, 1);
    }
}
//use new array to keep the reorganized particles
__global__ void putParticles(particle_t *d_particles,int n,int* counter, double binSize, int bins_row, particle_t *bin_seperate_p){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    for(int i = threadId; i < n; i+=step){
      int x = floor(d_particles[i].x / binSize);
      int y = floor(d_particles[i].y / binSize);
      int loc = x+y*bins_row;
      int particle_loc = atomicSub(counter+loc, 1);
      //the correct location of the particle is particle_loc
      bin_seperate_p[particle_loc - 1] = d_particles[i];
    }
}
__global__ void copy_back(particle_t* bin_seperate_p,particle_t* d_particles,int n, int bins_row,int* counter, int off_set, int* return_counter){
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  for(int i=threadId;i < bins_row*bins_row;i+=step){
    for(int j = 0; j< counter[threadId];j++){
      d_particles[return_counter[0]] = bin_seperate_p[i*off_set+j];
      atomicAdd(return_counter,1);
    }
  }
}

//used to calculate the prefix sum
__global__ void upTree(int *device_output, int len, int interval)
{
    int index = ((blockIdx.x * blockDim.x + threadIdx.x) + 1) * interval - 1;
    if(index <= len){
        device_output[index] += device_output[index - interval / 2];
    }

}

__global__ void downTree(int *device_output, int len, int interval)
{
    //set last element zero when first called
    if(interval >= len){
        device_output[len - 1] = 0;
    }

    int index = ((blockIdx.x * blockDim.x + threadIdx.x) + 1) * interval - 1;
    if(index < len){
        int temp = device_output[index];
        device_output[index] += device_output[index - interval / 2];
        device_output[index - interval / 2] = temp;
     }

}


int powerFloor(int len)
{
	int rounded = 1;
	while (rounded * 2 < len)
	{
		rounded *= 2;
	}
	return rounded;
}






int main( int argc, char **argv )
{
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t *d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    //bins size and number of bins
    //
    int bins_row;
    double binSize;
    binSize = cutoff*5;//maybe larger size?

    double size = sqrt( density * n );

    bins_row = ceil(size / binSize);
    printf("size = %f binSize = %f bins_row = %d\n",size,binSize,bins_row);

    int bin_num = bins_row * bins_row;
    //cudamalloc another shared memory to store the particles seperated by bins
    //
    particle_t * bin_seperate_p;
    //int off_set = 5;//assume the max number of particles in a bin is off_set
    cudaMalloc((void **) &bin_seperate_p,  n * sizeof(particle_t));

    //a counter to keep the number of particles in each bin
    int* counter;
    //use counter to keep a copy of the prefix sum
    int* counter2;
    int powerf = powerFloor(bin_num+1);
    int powerc = powerf * 2;
    cudaMalloc((void **) &counter, (powerc)* sizeof(int));
    cudaMalloc((void **) &counter2, (bin_num+1)* sizeof(int));

    cudaMemset(counter, 0, powerc * sizeof(int));//set counter to zero
    cudaMemset(counter2, 0, (bin_num+1) * sizeof(int));//set counter to zero
    int* h_counter = (int*)malloc(bin_num*sizeof(int));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );
    //printf("start steps \n");
    //for( int step = 0; step < NSTEPS; step++ )
    for( int step = 0; step <  NSTEPS; step++ )
    {
        //compute the number of blocks
        int blks =(n + NUM_THREADS - 1) / NUM_THREADS;
        //printf("new setp bigins \n");
        //count the number of particles in each bins
        cudaMemset(counter, 0, powerc * sizeof(int));//set counter to zero
        cudaMemset(counter2, 0, (bin_num+1) * sizeof(int));//set counter to zero

        //
        //count number of particles in each bin
        //
        countParticles<<<blks,NUM_THREADS>>>(d_particles,n,counter,binSize, bins_row);


        //cuda calculate the prefix sum
        //cudaMemcpy(h_counter,counter,bin_num*sizeof(int),cudaMemcpyDeviceToHost);
        //////////////////////////////////////////////////////////////////////////

        // in-place algorithm is used, we did not allocate device_input
        // Up Tree

        for(int interval = 2; interval <= powerf; interval *= 2){
            int numBlocks = (powerc / interval + NUM_THREADS - 1) / NUM_THREADS;
            upTree<<<numBlocks, NUM_THREADS>>>(counter, powerc, interval);

        }



        //down tree
        for(int interval = powerf * 2; interval >= 1; interval /= 2){
            int numBlocks = (powerc / interval + NUM_THREADS - 1) / NUM_THREADS;
            downTree<<<numBlocks, NUM_THREADS>>>(counter, powerc, interval);
        }



        // Wait for all instances to finish
        cudaThreadSynchronize();
        cudaMemcpy(counter2, counter, (bin_num+1)*sizeof(int), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(h_counter,counter+1,bin_num*sizeof(int),cudaMemcpyDeviceToHost);
        // for(int i = 0;i<bin_num;i++){
        //   printf("counter 1 bin %d number %d\n",i,h_counter[i]);
        // }
        // cudaMemcpy(h_counter,counter2+1,bin_num*sizeof(int),cudaMemcpyDeviceToHost);
        // for(int i = 0;i<bin_num;i++){
        //   printf("counter 2 bin %d number %d\n",i,h_counter[i]);
        // }


        ////////////////////////////////////////////////////////////////////////////

        putParticles<<<blks,NUM_THREADS>>>(d_particles,n,counter+1,binSize, bins_row,bin_seperate_p);

        //
        //  compute forces
        //
        //sent prefix sum value to counter again for force computation
        //cudaMemcpy(counter,h_counter,bin_num*sizeof(int),cudaMemcpyHostToDevice);
        std::swap(d_particles,bin_seperate_p);
	      compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n, bins_row, counter2+1,binSize);
        //
        //  move particles
        //
        move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
          cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
          save( fsave, n, particles);
	      }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    cudaFree(bin_seperate_p);
    cudaFree(counter);
    cudaFree(counter2);
    if( fsave )
        fclose( fsave );

    return 0;
}
