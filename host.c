/****************************************************************************
 *               University of North Carolina Charlotte                     *
 *                        MobileNet V1 CNN                                  *
 *                        				                                    *
 *                                                                          *
 *                                                                          *
 *   Author:    1. Kaustubh Manohar Mhatre                                  *
 *              2. Ushma Bharucha                                           *
 *   Date: 08 June 2019														*
 ****************************************************************************/

/****************************************************************************
* Includes																	*
*****************************************************************************/
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>
#include <time.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

int err;
int layer_count = 0;

cl_device_id device_id;             // compute device id 
cl_context context;                 // compute context
cl_command_queue commands;          // compute command queue
cl_program program;                 // compute program
cl_kernel kernel0;            // compute kernel for standard convolution
cl_kernel kernel1;           // compute kernel for depthwise convolution

cl_mem d_filter; //filter
cl_mem d_output; //output image
cl_event myevent; //profiling event
cl_ulong start; //time start
cl_ulong end; //time stop
cl_float kernelExecTimeNs;
cl_uint dev_cnt = 0;
cl_platform_id platform_ids[100];
clock_t t; 
long LoadOpenCLKernel(char const* path, char **buf)
{
	FILE  *fp;
	size_t fsz;
	long   off_end;
	int    rc;

	/* Open the file */
	fp = fopen(path, "r");
	if( NULL == fp ) {
		return -1L;
	}

	/* Seek to the end of the file */
	rc = fseek(fp, 0L, SEEK_END);
	if( 0 != rc ) {
		return -1L;
	}

	/* Byte offset to the end of the file (size) */
	if( 0 > (off_end = ftell(fp)) ) {
		return -1L;
	}
	fsz = (size_t)off_end;

	/* Allocate a buffer to hold the whole file */
	*buf = (char *) malloc( fsz+1);
	if( NULL == *buf ) {
		return -1L;
	}

	/* Rewind file pointer to start of file */
	rewind(fp);

	/* Slurp file into buffer */
	if( fsz != fread(*buf, 1, fsz, fp) ) {
		free(*buf);
		return -1L;
	}

	/* Close the file */
	if( EOF == fclose(fp) ) {
		free(*buf);
		return -1L;
	}


	/* Make sure the buffer is NUL-terminated, just in case */
	(*buf)[fsz] = '\0';

	/* Return the file size */
	return (long)fsz;
}

int openClDeviceConfig(){

	printf("Initializing OpenCL device...\n"); 

	clGetPlatformIDs(0, 0, &dev_cnt);
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	
	// Connect to a compute device
	int gpu = 1;
	//err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    //FPGA-Change
    err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);	
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

}

int openClCreateContext() {
	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}
}

int openClCreateKernel() {
	
	// Create the compute program from the source file
	char *KernelSource;
    size_t lFileSize;
	//lFileSize = LoadOpenCLKernel("kernel.cl", &KernelSource);
    //FPGA-Change    
    lFileSize = LoadOpenCLKernel("./xclbin/pipesTest.sw_emu.xilinx_aws-vu9p-f1-04261818_dynamic_5_0.xclbin", &KernelSource);
	if( lFileSize < 0L ) {
		perror("File read failed");
		return 1;
	}

	//program = clCreateProgramWithSource(context, 1, (const char **) &KernelSource, NULL, &err);
    //FPGA-Change
    //program = clCreateProgramWithSource(context, 1, &device_id , &size_var, ( const char** ) & KernelSource, NULL, &err);
    program = clCreateProgramWithBinary(context, 1, &device_id , &lFileSize, (const unsigned char **) & KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel for standard convolution
	kernel0 = clCreateKernel(program, "kernel0", &err);
	if (!kernel0 || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Create the compute kernel for depthwise convolution
	kernel1 = clCreateKernel(program, "kernel1", &err);
	if (!kernel1 || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}
}

int main(int argc, char** argv) {
	

	unsigned char* op_fm_0 = (unsigned char*) malloc(1000 * sizeof(unsigned char)); //output feature map for layer 0
	int i,j,k;
	t = clock();
	openClDeviceConfig();
	openClCreateContext();
	openClCreateKernel();
	cl_mem d_image_r; //R channel

	unsigned char* ipdata = (unsigned char*) malloc(1000* sizeof(unsigned char)); //R channel

	//Create buffer for device
	d_image_r = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 1000*sizeof(unsigned char), ipdata, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1000*sizeof(unsigned char), NULL, &err);

	if (!d_image_r || !d_output )
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_r, CL_TRUE, 0, 1000*sizeof(unsigned char), ipdata, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 

	err = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&d_image_r);
    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_output);

	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}
	/******************************Kernel 0*********************************/
	size_t localWorkSize[1], globalWorkSize[1];
	localWorkSize[0] = 1;
	globalWorkSize[0] = 10;
	err = clEnqueueNDRangeKernel(commands, kernel0, 1, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
/******************************Kernel 1*********************************/
    
	//size_t localWorkSize[1], globalWorkSize[1];
	localWorkSize[0] = 1;
	globalWorkSize[0] = 10;
	err = clEnqueueNDRangeKernel(commands, kernel1, 1, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}

	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, 1000*sizeof(unsigned char), op_fm_0, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}
	// unsigned char* output_proper = (unsigned char*) malloc(HEIGHT_1 * WIDTH_1 * IP_FM_1 * sizeof(unsigned char)); 
	// arrangOutput(opfm, output_proper, 0);
	// FILE *write_ptr;

	// write_ptr = fopen("test.npy","wb");  // w for write, b for binary
	// fwrite(output_proper,HEIGHT_1*WIDTH_1*IP_FM_1,1,write_ptr);
     
	//Get kernel execution time
	printf("Kernel Execution time for Layer %d: %f\n", layer_count, kernelExecTimeNs/1000000000);

	// printf("Data for Layer %d\n", layer_count);
    for(i = 0;i <1000;i++)
    {
        printf("%d\t",op_fm_0[i]);
    }
	// for (k = 0; k < 32; k++){
	// 	printf("Layer No: %d\n",k);
	// 	for (j = 100; j < 112; j++){
	// 		for(i = 100; i < 112; i++){
	// 			printf("%d\t", opfm[(j*112+i) + (k*112*112)]);
	// 		}
	// 		printf("\n");
	// 	}
    // printf("\n");
	// }	


	clReleaseMemObject(d_output);
	clReleaseMemObject(d_filter);
	clReleaseProgram(program);
	clReleaseKernel(kernel0);
	clReleaseKernel(kernel1);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	return 0;
}
