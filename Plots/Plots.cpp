

#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <ctime>
#include <limits>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <vector>
#include <set>
#include <unordered_map>
#include <CL/cl.h>
#include <Windows.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


#define LOG_LEVEL 0
#define NUM_CHARS 30
#define ALPHABETS 29
#define MFCC_DIM 39
#define MU 20.0
#define BETA 1.5
#define PARTITION 3
#define ALIGNMENT 4096

#define HID_DIM 1792
#define GPU_WORKGROUP 56
#define SLM_SIZE 64

#define BLOCK 32

typedef struct bias_vector
{
	int len;
	float *ARRAY;
}BiasVector;


// Structure denoting buffer information for generic CPU/GPU partitioning of OpenCL Kernels

typedef struct buffer_info
{
	float * host_ptr;
	cl_mem buffer_complete; // Complete OpenCL Buffer for CPU and GPU
	cl_mem buffer_CPU; // Complete OpenCL buffer only for CPU device 
	cl_mem buffer_GPU; //Complete OpenCL buffer for GPU device
	// Complete OpenCL sub buffers for CPU and GPU device
	cl_mem _b_cpu; 
	cl_mem _b_gpu;
	
	size_t size[2];
	size_t datasize_CPU;
	size_t datasize_GPU;
	size_t _d_cpu;
	size_t _d_gpu;
	int offset;
	int partition;
	int dimension;

}BufferPair;


// Structure denoting information for weight matrix used in neural networks

typedef struct weight_matrix
{
	int num_rows;
	int num_cols;
	float *MAT;
}WeightMatrix;


// Data structures required 

typedef struct buffer_info * buffer;
//Function prototypes required 

//void read_model2(RNN &model, FILE *model_file);
struct weight_matrix *alloc_weight_matrix(int num_rows, int num_cols);
struct bias_vector* alloc_bias_vector(int len);
void cl_init_data_structures(cl_context &context, vector<cl_command_queue> &queues, cl_device_id* &devices);
cl_kernel cl_build_kernel(const char*kernel_source, const char* kernel_name, cl_context &context, cl_device_id*& device_type);
void rand_init_weight(weight_matrix *W);
void rand_init_bias(bias_vector *b);
void cl_create_and_write_buffer(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info &buffers, float *host_array, int size[], int dimension, int partition);
void cl_set_kernel_arg_matvecmul(cl_context &context, vector<cl_command_queue>& cmdq, vector <cl_kernel> kernel, struct buffer_info &W, struct buffer_info &b, struct buffer_info &h, int partition, int hid_dim);
void cl_create_buffer(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info &buffers, int size[], int dimension, int partition);
void cl_enqueue_nd_range_kernel(vector <cl_command_queue> &cmdq, vector<cl_kernel> &kernel, int partition, int dimension, size_t global_size[], size_t local_size[], bool soft = false, int num_rows = 0, int num_cols = 0);
void release_buf(struct buffer_info buf);
void removePartition(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info & buffer, int partition);
void restorePartition(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info & buffer, int partition);

#define F_MAX 2.0
#define F_MIN -2.0

// Function for generating a random floating point number

float rand_float()
{
	return F_MIN + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (F_MAX - F_MIN)));
}

// Function for generating a random bias vector

void rand_init_bias(bias_vector *b)
{
	int len = b->len;
	float randval;
	for (int i = 0; i < len; i++)
	{
		randval = rand_float();
		b->ARRAY[i] = randval;
	}

}

// Function for generating a random weight matrix

void rand_init_weight(weight_matrix *W)
{
	int len = (W->num_rows)*(W->num_cols);
	float randval;
	for (int i = 0; i < len; i++)
	{
		randval = rand_float();
		W->MAT[i] = randval;
	}

}

// Function for allocating a bias vector of length len

struct bias_vector* alloc_bias_vector(int len)
{
	struct bias_vector *b;
	b = (struct bias_vector*)malloc(sizeof(struct bias_vector));
	float *B;
	B = (float*)_aligned_malloc(len*sizeof(float), ALIGNMENT);
	b->len = len;
	b->ARRAY = B;
	return b;
}

// Function for releasing memory of buffer structures 

void release_buf(struct buffer_info buf)
{
	/*Release OpenCL buffers after use*/
	clReleaseMemObject(buf.buffer_complete);
	return;
}

// Function for allocating memory for a weight matrix of dimension num_rows*num_cols

struct weight_matrix *alloc_weight_matrix(int num_rows, int num_cols)
{
	struct weight_matrix *W;
	W = (struct weight_matrix*)malloc(sizeof(struct weight_matrix));
	float *A;
	A = (float *)_aligned_malloc(num_rows*num_cols*sizeof(float), ALIGNMENT);
	W->num_rows = num_rows;
	W->num_cols = num_cols;
	W->MAT = A;
	return W;
}

// Function for printing values of weight matrix

void print_matrix(struct weight_matrix *W)
{
	/*Matrixx printer for debugging*/
	int num_rows = W->num_rows;
	int num_cols = W->num_cols;
	for (int r = 0; r < num_rows; r++)
	{
		for (int c = 0; c < num_cols; c++)
			printf("%f ", W->MAT[r*num_cols + c]);
		printf("\n");
	}
	return;
}



/*OpenCL API Structures and Functions */

// Function for printing error message specific to OpenCL

void check(cl_int status, const char* str)
{
	if (status != CL_SUCCESS)
	{
		printf("Failed: %s. Error %d\n", str, status);
		exit(EXIT_FAILURE);
	}
}

char g_device_name[128];

//Function for obtaining Interl OpenCL platform id 

cl_platform_id GetIntelOCLPlatform()
{
	cl_platform_id pPlatforms[10] = { 0 };
	char pPlatformName[128] = { 0 };

	cl_uint uiPlatformsCount = 0;
	cl_int err = clGetPlatformIDs(10, pPlatforms, &uiPlatformsCount);
	for (cl_uint ui = 0; ui < uiPlatformsCount; ++ui)
	{
		err = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
		if (err != CL_SUCCESS)
		{
			printf("ERROR: Failed to retreive platform vendor name.\n");
			return NULL;
		}

		if (!strcmp(pPlatformName, "Intel(R) OpenCL"))
			return pPlatforms[ui];
	}

	return NULL;
}

// Function for checking whether OpenCL CPU device is present for given platform id

bool IsCPUDevicePresented(cl_platform_id id)
{
	cl_uint num = 0;
	clGetDeviceIDs(id, CL_DEVICE_TYPE_CPU, 0, NULL, &num);
	return num != 0;
}
// Function for checking whether OpenCL GPU device is present for given platform id

bool IsGPUDevicePresented(cl_platform_id id)
{
	cl_uint num = 0;
	clGetDeviceIDs(id, CL_DEVICE_TYPE_GPU, 0, NULL, &num);
	return num != 0;
}

// Function for synchronizing CPU and GPU execution in the event of partitioning using OpenCL events

void cl_synchronize(cl_event &event_CPU, cl_event &event_GPU, int partition)
{
	/*Wait for command queue to process kernels*/
	if (partition != 10)
	{
		clWaitForEvents(1, &event_CPU);
		clReleaseEvent(event_CPU);

	}

	if (partition != 0)
	{
		clWaitForEvents(1, &event_GPU);
		clReleaseEvent(event_GPU);
	}


	return;

}

// Primary function for initializing platforms, devices and respective OpenCL command queues 

void cl_init_data_structures(cl_context &context, vector<cl_command_queue> &queues, cl_device_id* &devices)
{
	cl_int err;
	cl_platform_id platform_id;
	platform_id = GetIntelOCLPlatform();
	//printf("Platform ID %d\n", platform_id);
	if (LOG_LEVEL > 0)
	{

		if (IsCPUDevicePresented(platform_id))
			printf("CPU device present!\n");
		if (IsGPUDevicePresented(platform_id))
			printf("GPU device present!\n");
	}

	cl_uint deviceCount;
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	check(err, "Obtaining device ids");
	devices = (cl_device_id*)malloc(sizeof(cl_device_id)*deviceCount);
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
	// create a single context for all devices
	//printf("Creating context\n");
	context = clCreateContext(NULL, deviceCount, devices, NULL, NULL, &err);
	check(err, "Creating context");
	// for each device create a separate queue
	//cl_command_queue* queues = new cl_command_queue[deviceCount];
	for (int i = 0; i < deviceCount; i++)
	{
		err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, g_device_name, NULL);
		printf("Using device %s...\n", g_device_name);
		queues.push_back(clCreateCommandQueue(context, devices[i], 0, &err));
		check(err, "Creating command queue");
	}
}

//Function to compile an OpenCL kernel stored in kernel_source for an OpenCL context

cl_program cl_compile_program(const char* kernel_source, cl_context &context)
{
	cl_int err;
	ifstream f(kernel_source);
	stringstream sbuffer;
	sbuffer << f.rdbuf();
	string kernel_src = sbuffer.str();
	const char *program_source = kernel_src.c_str();
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&program_source, NULL, &err);
	check(err, "Creating program with source");
	return program;
}
// Function to build a device specific binary for an OpenCL kernel stored in kernel_source

cl_kernel cl_build_kernel(const char*kernel_source, const char* kernel_name, cl_context &context, cl_device_id*& device_type)
{
	cl_int status;
	cl_program program;
	program = cl_compile_program(kernel_source, context);
	cl_kernel kernel;
	status = clBuildProgram(program, 1, device_type, NULL, NULL, NULL);
	check(status, "Building program");
	kernel = clCreateKernel(program, kernel_name, &status);
	check(status, "Creating Kernel");
	return kernel;
}

//Function for creating and writing data to OpenCL buffers specific to a buffer_info object
// The function takes as arguments a vector of command queues where each element denotes a command queue for a specific device
// Writing is done using the CL_MEM_USE_HOST_PTR flag set, so that the zero copy feature may be used (in order to avoid data transfer overhead)
void cl_create_and_write_buffer(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info &buffers, float *host_array, int size[], int dimension, int partition)
{
	cl_event CPU_event, GPU_event;
	int total_size = 1;
	buffers.dimension = dimension;
	buffers.partition = partition;
	buffers.host_ptr = host_array;


	for (int i = 0; i < dimension; i++)
	{
		buffers.size[i] = size[i];
		total_size *= buffers.size[i];
	}

	size_t datasize = total_size*sizeof(float);
	cl_int status;
	if (partition == 0)
	{

		buffers.datasize_CPU = datasize;
		buffers.datasize_GPU = 0;

		buffers.offset = 0;
		buffers.buffer_GPU = NULL;
		buffers.buffer_CPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, datasize, host_array, &status);
		check(status, "Creating buffer for CPU ... Parition Class 0");
		buffers.buffer_complete = buffers.buffer_CPU;
	}
	else if (partition == 10)
	{
		buffers.datasize_CPU = 0;
		buffers.datasize_GPU = datasize;

		buffers.offset = 0;
		buffers.buffer_CPU = NULL;
		buffers.buffer_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, datasize, host_array, &status);
		check(status, "Creating buffer for GPU ... Parition Class 10");
		buffers.buffer_complete = buffers.buffer_GPU;
	}
	else
	{

		buffers.buffer_complete = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, datasize, host_array, &status);
		int size_gpu = ((size[0] * partition / BLOCK) / 10)*BLOCK;
		int size_cpu = size[0] - size_gpu;
		size_t datasize_CPU = (total_size / size[0])*size_cpu*sizeof(float);
		size_t datasize_GPU = (total_size / size[0])*size_gpu*sizeof(float);
		buffers.datasize_CPU = datasize_CPU;
		buffers.datasize_GPU = datasize_GPU;

		cl_buffer_region CPU_Region = { 0, datasize_CPU };
		cl_buffer_region GPU_Region = { datasize_CPU, datasize_GPU };

		buffers.offset = (total_size / size[0])*size_cpu;
		buffers.buffer_CPU = clCreateSubBuffer(buffers.buffer_complete, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &CPU_Region, &status);
		buffers.buffer_GPU = clCreateSubBuffer(buffers.buffer_complete, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &GPU_Region, &status);

		buffers._b_cpu = buffers.buffer_CPU;
		buffers._b_gpu = buffers.buffer_GPU;
		buffers._d_gpu = buffers.datasize_GPU;
		buffers._d_cpu = buffers.datasize_CPU;

	}

	return;


}


// Modifies buffer_info object by merging CPU and GPU specific buffers to one single buffer
void removePartition(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info & buffer, int partition)
/*Remove partition for matrix multiplication*/
{
	size_t datasize = buffer.datasize_CPU + buffer.datasize_GPU;
	buffer.datasize_CPU = buffer.datasize_GPU = datasize;
	buffer.buffer_GPU = buffer.buffer_CPU = buffer.buffer_complete;
	//buffer.offset = 0;
}
// Modifies buffer_info object so that the buffer can be partitioned across the CPU and the GPU devices

void restorePartition(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info & buffer, int partition)
/*Restore partition after matrix multiplication*/
{
	buffer.datasize_GPU = buffer._d_gpu;
	buffer.datasize_CPU = buffer._d_cpu;
	buffer.buffer_CPU = buffer._b_cpu;
	buffer.buffer_GPU = buffer._b_gpu;
}

// Function for reading OpenCL buffers specific to an buffer_info object

void cl_read_buffer(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info &buffers, int size[], int dimension, int partition, float *& host)
{

	host = (float *)clEnqueueMapBuffer(cmdq[1], buffers.buffer_complete, CL_FALSE, CL_MAP_READ, 0, buffers.datasize_CPU + buffers.datasize_GPU, 0, NULL, NULL, NULL);
	clFinish(cmdq[1]);
	return;

}



// Unmap buffers
void unmap_buffers(vector<cl_command_queue>& cmdq, buffer_info buf, int partition)
{
	clEnqueueUnmapMemObject(cmdq[1], buf.buffer_complete, (void *)buf.host_ptr, 0, NULL, NULL);
	clFinish(cmdq[1]);
}

// Function for printing out contents of buffer_info object

void debug_print(cl_context &context, vector<cl_command_queue>& cmdq, buffer_info buf, int partition)
{
	int tot_size = 1;
	for (int i = 0;i < buf.dimension;i++)
		tot_size *= buf.size[i];

	float * host = (float *)malloc(sizeof(float) * tot_size);
	int sz[2] = { buf.size[0],buf.size[1] };
	cl_read_buffer(context, cmdq, buf, sz, buf.dimension, partition, host);

	printf("\n");
	for (int i = 0;i < tot_size;i++)
		printf("%E ", host[i]);
	printf("\n\n");
	unmap_buffers(cmdq, buf, partition);
	return;

}
//Function for finishing operations in OpenCL command queues for CPU and GPU devices 
void finish_queues(vector<cl_command_queue>& cmdq)
{
	clFinish(cmdq[1]);
	clFinish(cmdq[0]);
}

// Function for creating OpenCL buffers for a buffer_info object 

void cl_create_buffer(cl_context &context, vector<cl_command_queue>& cmdq, struct buffer_info &buffers, int size[], int dimension, int partition)
{
	int total_size = 1;
	buffers.dimension = dimension;
	buffers.partition = partition;


	for (int i = 0; i < dimension; i++)
	{
		buffers.size[i] = size[i];
		total_size *= buffers.size[i];
	}

	size_t datasize = total_size*sizeof(float);
	size_t aligned_size = datasize;


	/* We allocate the host memory using _aligned_malloc instead of letting the runtime do it*/

	if (aligned_size % 64 != 0)							//Datasize must be a multiple of 64 bytes
		aligned_size += 64 - (aligned_size % 64);



	//float * buf = (float *) _aligned_malloc(aligned_size, ALIGNMENT);
	//buffers.host_ptr = buf;
	cl_int status;
	if (partition == 0)
	{
		buffers.datasize_CPU = datasize;
		buffers.datasize_GPU = 0;

		buffers.offset = 0;
		buffers.buffer_GPU = NULL;
		buffers.buffer_CPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, datasize, NULL, &status);
		check(status, "Creating buffer for CPU ... Partition Class 0");
		buffers.buffer_complete = buffers.buffer_CPU;

	}
	else if (partition == 10)
	{
		//printf("Total size %d\n", total_size);

		buffers.datasize_CPU = 0;
		buffers.datasize_GPU = datasize;


		//printf("Size of data: %d\n", datasize);
		buffers.offset = 0;
		buffers.buffer_CPU = NULL;
		buffers.buffer_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, datasize, NULL, &status);
		check(status, "Creating buffer for GPU ... Partition Class 10");
		buffers.buffer_complete = buffers.buffer_GPU;

	}
	else
	{

		buffers.buffer_complete = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, datasize, NULL, &status);
		int size_gpu = (size[0] * partition) / 10;
		int size_cpu = size[0] - size_gpu;
		size_t datasize_CPU = (total_size / size[0])*size_cpu*sizeof(float);
		size_t datasize_GPU = (total_size / size[0])*size_gpu*sizeof(float);
		buffers.datasize_CPU = datasize_CPU;
		buffers.datasize_GPU = datasize_GPU;

		cl_buffer_region CPU_Region = { 0, datasize_CPU };
		cl_buffer_region GPU_Region = { datasize_CPU, datasize_GPU };

		buffers.offset = (total_size / size[0])*size_cpu;
		buffers.buffer_CPU = clCreateSubBuffer(buffers.buffer_complete, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &CPU_Region, &status);
		buffers.buffer_GPU = clCreateSubBuffer(buffers.buffer_complete, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &GPU_Region, &status);
		buffers._b_cpu = buffers.buffer_CPU;
		buffers._b_gpu = buffers.buffer_GPU;
		buffers._d_gpu = buffers.datasize_GPU;
		buffers._d_cpu = buffers.datasize_CPU;


	}


	return;


}

// Function for setting kernel arugments of OpenCL kernel for vector addition for CPU and GPU devices 

void cl_set_kernel_arg_vecadd(cl_context &context, vector<cl_command_queue>& cmdq, vector <cl_kernel> kernel, struct buffer_info &x, struct buffer_info &b, struct buffer_info &h, size_t size[], int dimension, float relu, int partition)
{
	cl_int status;
	if (partition == 0)
	{
		status = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&(x.buffer_CPU));
		check(status, "Setting Kernel Arugment 1");
		status = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&(b.buffer_CPU));
		check(status, "Setting Kernel Arugment 2");
		status = clSetKernelArg(kernel[1], 2, sizeof(cl_int), (void *)&(h.buffer_CPU));
		check(status, "Setting Kernel Arugment 3");
		status = clSetKernelArg(kernel[1], 3, sizeof(cl_mem), (void *)&(size[0]));
		check(status, "Setting Kernel Arugment 4");
		status = clSetKernelArg(kernel[1], 4, sizeof(float), (void *)&(relu));
		check(status, "Setting Kernel Arugment 5");


	}
	else if (partition == 10)
	{
		status = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&(x.buffer_GPU));
		check(status, "Setting Kernel Arugment 1");
		status = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&(b.buffer_GPU));
		check(status, "Setting Kernel Arugment 2");
		status = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&(h.buffer_GPU));
		check(status, "Setting Kernel Arugment 3");
		status = clSetKernelArg(kernel[0], 3, sizeof(cl_mem), (void *)&(size[0]));
		check(status, "Setting Kernel Arugment 4");
		status = clSetKernelArg(kernel[0], 4, sizeof(float), (void *)&(relu));
		check(status, "Setting Kernel Arugment 5");

	}
	else
	{
		int len_GPU = (h.size[0] * partition) / 10;
		int len_CPU = h.size[0] - len_GPU;
		// Setting CPU arguments 
		status = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&(x.buffer_CPU));
		check(status, "Setting Kernel Arugment 1");
		status = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&(b.buffer_CPU));
		check(status, "Setting Kernel Arugment 2");
		status = clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void *)&(h.buffer_CPU));
		check(status, "Setting Kernel Arugment 3");
		status = clSetKernelArg(kernel[1], 3, sizeof(cl_int), (void *)&len_CPU);
		check(status, "Setting Kernel Arugment 4");
		status = clSetKernelArg(kernel[1], 4, sizeof(float), (void *)&(relu));
		check(status, "Setting Kernel Arugment 5");



		// Setting GPU arguments

		status = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&(x.buffer_GPU));
		check(status, "Setting Kernel Arugment 1");
		status = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&(b.buffer_GPU));
		check(status, "Setting Kernel Arugment 2");
		status = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *)&(h.buffer_GPU));
		check(status, "Setting Kernel Arugment 3");
		status = clSetKernelArg(kernel[0], 3, sizeof(cl_int), (void *)&len_GPU);
		check(status, "Setting Kernel Arugment 4");
		status = clSetKernelArg(kernel[0], 4, sizeof(float), (void *)&(relu));
		check(status, "Setting Kernel Arugment 5");

	}

}

//Function for setting arguments of OpenCL matrix vector multiplicatione kernel for CPU and GPU devices 

void cl_set_kernel_arg_matvecmul(cl_context &context, vector<cl_command_queue>& cmdq, vector <cl_kernel> kernel, struct buffer_info &W, struct buffer_info &b, struct buffer_info &h, int partition, int hid_dim = HID_DIM)
{
	cl_int status;
	if (partition == 0)
	{
		status = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&(W.buffer_CPU));
		check(status, "Setting Kernel Arugment 1");
		status = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&(b.buffer_CPU));
		check(status, "Setting Kernel Arugment 2");
		status = clSetKernelArg(kernel[1], 2, sizeof(cl_uint), (void *)&(W.size[1]));
		check(status, "Setting Kernel Arugment 3");
		status = clSetKernelArg(kernel[1], 3, sizeof(cl_uint), (void *)&(W.size[0]));
		check(status, "Setting Kernel Arugment 4");
		status = clSetKernelArg(kernel[1], 4, sizeof(cl_mem), (void *)&(h.buffer_CPU));
		check(status, "Setting Kernel Arugment 5");

		status = clSetKernelArg(kernel[1], 5, SLM_SIZE*sizeof(float), NULL);

	}
	else if (partition == 10)
	{

		status = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&(W.buffer_GPU));
		check(status, "Setting Kernel Arugment 1");
		status = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&(b.buffer_GPU));
		check(status, "Setting Kernel Arugment 2");
		status = clSetKernelArg(kernel[0], 2, sizeof(cl_uint), (void *)&(W.size[1]));
		check(status, "Setting Kernel Arugment 3");
		status = clSetKernelArg(kernel[0], 3, sizeof(cl_uint), (void *)&(W.size[0]));
		check(status, "Setting Kernel Arugment 4");
		status = clSetKernelArg(kernel[0], 4, sizeof(cl_mem), (void *)&(h.buffer_GPU));
		check(status, "Setting Kernel Arugment 5");

		status = clSetKernelArg(kernel[0], 5, SLM_SIZE*sizeof(float), NULL);
	}
	else
	{

		int num_rows_GPU = ((W.size[0] * partition / BLOCK) / 10)*BLOCK;
		int num_rows_CPU = W.size[0] - num_rows_GPU;
		// Setting CPU arguments 
		status = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&(W.buffer_CPU));
		check(status, "Setting Kernel Arugment 1");
		status = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&(b.buffer_CPU));
		check(status, "Setting Kernel Arugment 2");
		status = clSetKernelArg(kernel[1], 2, sizeof(cl_uint), (void *)&(W.size[1]));
		check(status, "Setting Kernel Arugment 3");
		status = clSetKernelArg(kernel[1], 3, sizeof(cl_uint), (void *)&(num_rows_CPU));
		check(status, "Setting Kernel Arugment 4");
		status = clSetKernelArg(kernel[1], 4, sizeof(cl_mem), (void *)&(h.buffer_CPU));
		check(status, "Setting Kernel Arugment 5");

		status = clSetKernelArg(kernel[1], 5, SLM_SIZE *sizeof(float), NULL);


		// Setting GPU arguments

		status = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&(W.buffer_GPU));
		check(status, "Setting Kernel Arugment 1");
		status = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&(b.buffer_GPU));
		check(status, "Setting Kernel Arugment 2");
		status = clSetKernelArg(kernel[0], 2, sizeof(cl_uint), (void *)&(W.size[1]));
		check(status, "Setting Kernel Arugment 3");
		status = clSetKernelArg(kernel[0], 3, sizeof(cl_uint), (void *)&(num_rows_GPU));
		check(status, "Setting Kernel Arugment 4");
		status = clSetKernelArg(kernel[0], 4, sizeof(cl_mem), (void *)&(h.buffer_GPU));
		check(status, "Setting Kernel Arugment 5");

		status = clSetKernelArg(kernel[0], 5, SLM_SIZE *sizeof(float), NULL);
	}

}



//Function for launching kernels (complete CPU execution, complete GPU execution, partitioned mode of execution)


void cl_enqueue_nd_range_kernel(vector <cl_command_queue> &cmdq, vector<cl_kernel> &kernel, int partition, int dimension, size_t global_size[], size_t local_size[], bool soft, int num_rows, int num_cols)
{
	cl_int status;
	if (partition == 0)
	{
		status = clEnqueueNDRangeKernel(cmdq[1], kernel[1], dimension, 0, global_size, local_size, 0, NULL, NULL);
		check(status, "Launching kernel for complete CPU");
		clFlush(cmdq[1]);
	}
	else if (partition == 10)
	{
		status = clEnqueueNDRangeKernel(cmdq[0], kernel[0], dimension, 0, global_size, local_size, 0, NULL, NULL);
		check(status, "Launching kernel for complete GPU");
		clFlush(cmdq[0]);
	}
	else
	{
		size_t global_size_CPU[2], global_size_GPU[2];
		for (int i = 0; i < dimension; i++)
		{
			global_size_CPU[i] = global_size[i];
			global_size_GPU[i] = global_size[i];
		}
		if (!soft)
		{
			size_t size_gpu = ((global_size[0] * partition / BLOCK) / 10)*BLOCK;
			size_t size_cpu = global_size[0] - size_gpu;
			global_size_CPU[0] = size_cpu;
			global_size_GPU[0] = size_gpu;
		}
		else
		{
			size_t size_gpu = ((num_rows * partition) / 10) * num_cols;
			size_t size_cpu = global_size[0] - size_gpu;
			global_size_CPU[0] = size_cpu;
			global_size_GPU[0] = size_gpu;
		}

		size_t local_size_CPU[2] = { 1,1 };
		status = clEnqueueNDRangeKernel(cmdq[0], kernel[0], dimension, 0, global_size_GPU, local_size, 0, NULL, NULL);
		check(status, "Launching kernel for mixed GPU");
		clFlush(cmdq[0]);

		status = clEnqueueNDRangeKernel(cmdq[1], kernel[1], dimension, 0, global_size_CPU, local_size_CPU, 0, NULL, NULL);
		check(status, "Launching kernel for mixed CPU");
		clFlush(cmdq[1]);
	}





}




// Run test on Matrix-Vector multiplication Kernel and print results to a file
void test_kernel(int hidden_size, int locsize, cl_context context, vector<cl_command_queue> queues, vector <cl_kernel> gemv, int partition, ofstream& outfile)
{
	long long perfCounter = 0;
	size_t lsize[2] = { locsize,1 };
	for (int i = 0; i < 10; i++)
	{
		weight_matrix * w = alloc_weight_matrix(hidden_size, hidden_size);
		rand_init_weight(w);
		bias_vector * b = alloc_bias_vector(hidden_size);
		rand_init_bias(b);
		struct buffer_info w_buf, b_buf, out_buf;
		int sz[2] = { w->num_rows,w->num_cols };
		cl_create_and_write_buffer(context, queues, w_buf, w->MAT, sz, 2, partition);
		cl_create_and_write_buffer(context, queues, b_buf, b->ARRAY, sz, 1, partition);
		cl_create_buffer(context, queues, out_buf, sz, 1, partition);

		if (partition > 0 && partition < 10)
			removePartition(context, queues, b_buf, partition);

		high_resolution_clock::time_point start = high_resolution_clock::now();
		cl_set_kernel_arg_matvecmul(context, queues, gemv, w_buf, b_buf, out_buf, partition, hidden_size);
		//cl_event weight_event_CPU, weight_event_GPU;
		cl_enqueue_nd_range_kernel(queues, gemv, partition, 1, w_buf.size, lsize);
		clFinish(queues[0]);
		clFinish(queues[1]);
		high_resolution_clock::time_point end = high_resolution_clock::now();


		perfCounter += duration_cast<microseconds> (end - start).count();
		release_buf(w_buf);
		release_buf(b_buf);
		release_buf(out_buf);
		_aligned_free(w->MAT);
		_aligned_free(b->ARRAY);
		free(w);
		free(b);

	}
	
	outfile << hidden_size << "," << locsize << "," << perfCounter << endl;
	return;
}




//Run Tests on matrix-vector multiplication kernel and print results to stdout
void test_k(int hidden_size, int locsize, cl_context context, vector<cl_command_queue> queues, vector <cl_kernel> gemv, int partition, ofstream& outfile)
{
	long long perfCounter = 0;
	size_t lsize[2] = { locsize,1 };

		weight_matrix * w = alloc_weight_matrix(hidden_size, hidden_size);
		rand_init_weight(w);
		bias_vector * b = alloc_bias_vector(hidden_size);
		rand_init_bias(b);
		struct buffer_info w_buf, b_buf, out_buf;
		int sz[2] = { w->num_rows,w->num_cols };

		cl_create_and_write_buffer(context, queues, w_buf, w->MAT, sz, 2, partition);
		cl_create_and_write_buffer(context, queues, b_buf, b->ARRAY, sz, 1, partition);
		cl_create_buffer(context, queues, out_buf, sz, 1, partition);


		if (partition > 0 && partition < 10)
			removePartition(context, queues, b_buf, partition);

		
		cl_set_kernel_arg_matvecmul(context, queues, gemv, w_buf, b_buf, out_buf, partition, hidden_size);

		cl_enqueue_nd_range_kernel(queues, gemv, partition, 1, w_buf.size, lsize);
		//high_resolution_clock::time_point end1 = high_resolution_clock::now();
		clFinish(queues[0]);
		clFinish(queues[1]);



		//high_resolution_clock::time_point start1 = high_resolution_clock::now();
		high_resolution_clock::time_point start = high_resolution_clock::now();
		cl_enqueue_nd_range_kernel(queues, gemv, partition, 1, w_buf.size, lsize);
		//high_resolution_clock::time_point end1 = high_resolution_clock::now();
		clFinish(queues[0]);
		clFinish(queues[1]);
		high_resolution_clock::time_point end = high_resolution_clock::now();


		perfCounter = duration_cast<nanoseconds> (end - start).count();
		//ec += duration_cast<microseconds> (end1 - start1).count();
		release_buf(w_buf);
		release_buf(b_buf);
		release_buf(out_buf);
		_aligned_free(w->MAT);
		_aligned_free(b->ARRAY);
		free(w);
		free(b);


	cout << "(" << hidden_size << ", " << locsize << ") - " << perfCounter << endl;
	//cout << partition << " (" << hidden_size << ", " << locsize << ") - " << ec << endl;
	return;
}


int main(int argc, char* argv[])
{
	int partition = PARTITION;
	if (argc > 1) partition = atoi(argv[1]);

	
	cl_context context;
	vector<cl_command_queue> queues;
	cl_device_id *devices;
	cl_init_data_structures(context, queues, devices);
	vector <cl_kernel> gemv,blank,load,add,mult,barriers;
	
	for (int i = 0; i <= 1; i++)
	{
		cl_device_id *device = &devices[i];
		cl_kernel kernel_gemv;
		kernel_gemv = cl_build_kernel("gemv.cl", "gemv", context, device);

		gemv.push_back(kernel_gemv);
		

	}
	cout << "Initialized " << endl;
	
	ofstream outfile;
	outfile.open("timing.txt");

	test_k(1024, 16, context, queues, gemv, 0, outfile);
	test_k(512, 16, context, queues, gemv, 0, outfile);
	test_k(3072, 16, context, queues, gemv, 0, outfile);
	test_k(4096, 16, context, queues, gemv, 0, outfile);
	test_k(2048, 16, context, queues, gemv, 0, outfile);
	//test_k(4096, 32, context, queues, gemv, 10, outfile);

	outfile.close();
	system("pause");
	return 0;
}