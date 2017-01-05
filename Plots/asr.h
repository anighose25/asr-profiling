#pragma once

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





typedef struct buffer_info
{
	float * host_ptr;
	cl_mem buffer_complete;
	cl_mem buffer_CPU;
	cl_mem _b_cpu;
	cl_mem _b_gpu;
	cl_mem buffer_GPU;
	size_t size[2];
	size_t datasize_CPU;
	size_t datasize_GPU;
	size_t _d_cpu;
	size_t _d_gpu;
	int offset;
	int partition;
	int dimension;

}BufferPair;



typedef struct weight_matrix
{
	int num_rows;
	int num_cols;
	float *MAT;
}WeightMatrix;



typedef struct rnn_structure
{
	int num_layers;
	std::vector <struct bias_vector*> h_init;
	std::vector <std::string> layer_type;
	std::vector <struct weight_matrix*> W;
	std::vector <struct bias_vector*>b;

}RNN;


struct string_info
{
	string s;
	double pb;
	double pnb;
	double ptot;

};

struct string_compare
{
	bool operator()(const struct string_info &first, const struct string_info &second)
	{
		double ptot_f = first.pb + first.pnb;
		double ptot_s = second.pb + second.pnb;
		double f_size = (double)first.s.length();
		double s_size = (double)second.s.length();
		return(ptot_f*pow(f_size, BETA) > ptot_s*pow(s_size, BETA));
	}

};

typedef struct buffer_info * buffer;
typedef multiset <struct string_info, string_compare> z_set;
typedef unordered_map <string, float *> my_map;
typedef unordered_map <string, struct buffer_info *> act_map;
typedef unordered_map <string, int> strings;

void read_model2(RNN &model, FILE *model_file);
struct weight_matrix *alloc_weight_matrix(int num_rows, int num_cols);
struct bias_vector* alloc_bias_vector(int len);
void generate_model(RNN &model, int hidden_size);
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