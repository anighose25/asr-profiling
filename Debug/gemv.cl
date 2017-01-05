__kernel void gemv(const __global float4* M,
	const __global float4* V,
	uint width, uint height,
	__global float* W,
	__local float* partialDotProduct)
{
	// Each work-group handles as many matrix rows as necessary


	for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {

		// Row pointer
		const __global float4* row = M + (y * width/4);

		// Each work-item accumulates as many products as necessary
		// into local variable "sum"
		float4 sum = (float4) (0.0f);

		for (uint x = get_local_id(0); x < width/4; x += get_local_size(0))
			sum = fma(row[x],V[x],sum);
		
		
		// Each partial dot product is stored in shared memory
		partialDotProduct[get_local_id(0)] = sum.x + sum.y + sum.z + sum.w; 

		// Perform parallel reduction to add each work-item's
		// partial dot product together

		for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {

			// Synchronize to make sure each work-item is done updating
			// shared memory; this is necessary because work-items read
			// results that have been written by other work-items
			barrier(CLK_LOCAL_MEM_FENCE);

			// Only the first work-items in the work-group add elements together
			if (get_local_id(0) < stride) {

				// Add two elements from the "partialDotProduct" array
				// and store the result in partialDotProduct[index]
				partialDotProduct[get_local_id(0)] += partialDotProduct[get_local_id(0) + stride];
			}
		}

		// Write the result of the reduction to global memory
		if (get_local_id(0) == 0)
			W[y] = partialDotProduct[0];

	}
}

__kernel void gemv_no_memAccess(const __global float4* M,
	const __global float4* V,
	uint width, uint height,
	__global float* W,
	__local float* partialDotProduct)
{
	// Each work-group handles as many matrix rows as necessary


	for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {

		// Row pointer
		const __global float4* row = M + (y * width/4);

		// Each work-item accumulates as many products as necessary
		// into local variable "sum"
		float4 sum = (float4) (0.0f);

		for (uint x = get_local_id(0); x < width/4; x += get_local_size(0))
			sum = fma(1.0,2.0,sum);
		
		
		// Each partial dot product is stored in shared memory
		partialDotProduct[get_local_id(0)] = sum.x + sum.y + sum.z + sum.w; 

		// Perform parallel reduction to add each work-item's
		// partial dot product together

		for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {

			// Synchronize to make sure each work-item is done updating
			// shared memory; this is necessary because work-items read
			// results that have been written by other work-items
			barrier(CLK_LOCAL_MEM_FENCE);

			// Only the first work-items in the work-group add elements together
			if (get_local_id(0) < stride) {

				// Add two elements from the "partialDotProduct" array
				// and store the result in partialDotProduct[index]
				partialDotProduct[get_local_id(0)] += partialDotProduct[get_local_id(0) + stride];
			}
		}

		// Write the result of the reduction to global memory
		if (get_local_id(0) == 0)
			W[y] = partialDotProduct[0];

	}
}


__kernel void gemv_blank(const __global float4* M,
	const __global float4* V,
	uint width, uint height,
	__global float* W,
	__local float* partialDotProduct)
{
	
}



__kernel void gemv_no_SIMD(const __global float* M,
	const __global float* V,
	uint width, uint height,
	__global float* W,
	__local float* partialDotProduct)
{
	// Each work-group handles as many matrix rows as necessary
	for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {

		// Row pointer
		const __global float* row = M + y * width;

		// Each work-item accumulates as many products as necessary
		// into local variable "sum"
		float sum = 0;
		for (uint x = get_local_id(0); x < width; x += get_local_size(0))
			sum += row[x] * V[x];

		// Each partial dot product is stored in shared memory
		partialDotProduct[get_local_id(0)] = sum;

		// Perform parallel reduction to add each work-item's
		// partial dot product together
		for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {

			// Synchronize to make sure each work-item is done updating
			// shared memory; this is necessary because work-items read
			// results that have been written by other work-items
			barrier(CLK_LOCAL_MEM_FENCE);

			// Only the first work-items in the work-group add elements together
			if (get_local_id(0) < stride) {

				// Add two elements from the "partialDotProduct" array
				// and store the result in partialDotProduct[index]
				partialDotProduct[get_local_id(0)] += partialDotProduct[get_local_id(0) + stride];
			}
		}

		// Write the result of the reduction to global memory
		if (get_local_id(0) == 0)
			W[y] = partialDotProduct[0];

		// Synchronize to make sure the first work-item is done with
		// reading partialDotProduct
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
