#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>
#include <vector>

#include "math_functions.cuh"
#include "render_data.cuh"


__device__ bool can_expand_cuboid(uint8_t* grid, int3 chunk_pos, int min_x, int max_x, int min_y, int max_y, int min_z, int max_z)
{
	if (min_x < 0 || max_x >= CHUNK_SIZE || min_y < 0 || max_y >= CHUNK_SIZE || min_z < 0 || max_z >= CHUNK_SIZE)
	{
		return false;
	}

	for (int z = min_z; z <= max_z; ++z)
	{
		for (int y = min_y; y <= max_y; ++y)
		{
			for (int x = min_x; x <= max_x; ++x)
			{
				int3 block_pos = { chunk_pos.x * CHUNK_SIZE + x, chunk_pos.y * CHUNK_SIZE + y, chunk_pos.z * CHUNK_SIZE + z };

				int block_index = grid_linear_pos(block_pos);

				if (grid[block_index] != 0)
				{
					return false;
				}
			}
		}
	}

	return true;
}

__global__ void generate_chunk_distance_field(Scene scene, int3 chunk_pos)
{
	int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_z = blockIdx.z * blockDim.z + threadIdx.z;


	int3 block_pos = { chunk_pos.x * CHUNK_SIZE + thread_x, chunk_pos.y * CHUNK_SIZE + thread_y, chunk_pos.z * CHUNK_SIZE + thread_z };

	int block_index = grid_linear_pos(block_pos);

	uint8_t block_id = scene.grid[block_index];


	if (block_id == 0)
	{
		int min_x = thread_x;
		int max_x = thread_x;
		int min_y = thread_y;
		int max_y = thread_y;
		int min_z = thread_z;
		int max_z = thread_z;

		bool can_expanded = true;

		while (can_expanded)
		{
			can_expanded = false;

			if (can_expand_cuboid(scene.grid, chunk_pos, max_x + 1, max_x + 1, min_y, max_y, min_z, max_z))
			{
				++max_x;
				can_expanded = true;
			}

			if (can_expand_cuboid(scene.grid, chunk_pos, min_x - 1, min_x - 1, min_y, max_y, min_z, max_z))
			{
				--min_x;
				can_expanded = true;
			}

			if (can_expand_cuboid(scene.grid, chunk_pos, min_x, max_x, max_y + 1, max_y + 1, min_z, max_z))
			{
				++max_y;
				can_expanded = true;
			}

			if (can_expand_cuboid(scene.grid, chunk_pos, min_x, max_x, min_y - 1, min_y - 1, min_z, max_z))
			{
				--min_y;
				can_expanded = true;
			}

			if (can_expand_cuboid(scene.grid, chunk_pos, min_x, max_x, min_y, max_y, max_z + 1, max_z + 1))
			{
				++max_z;
				can_expanded = true;
			}

			if (can_expand_cuboid(scene.grid, chunk_pos, min_x, max_x, min_y, max_y, min_z - 1, min_z - 1))
			{
				--min_z;
				can_expanded = true;
			}
		}

		uint32_t right_dist = max(abs(max_x - thread_x) - 1, 0);
		uint32_t left_dist = max(abs(min_x - thread_x) - 1, 0);
		uint32_t up_dist = max(abs(max_y - thread_y) - 1, 0);
		uint32_t down_dist = max(abs(min_y - thread_y) - 1, 0);
		uint32_t front_dist = max(abs(max_z - thread_z) - 1, 0);
		uint32_t back_dist = max(abs(min_z - thread_z) - 1, 0);

		uint32_t distances = 0;

		distances |= right_dist << 0;
		distances |= left_dist << 5;
		distances |= up_dist << 10;
		distances |= down_dist << 15;
		distances |= front_dist << 20;
		distances |= back_dist << 25;

		distances |= 1 << 30;

		surf3Dwrite(distances, scene.distance_field_surface, block_pos.x * sizeof(uint32_t), block_pos.y, block_pos.z);
	}
	else
	{
		surf3Dwrite(0, scene.distance_field_surface, block_pos.x * sizeof(uint32_t), block_pos.y, block_pos.z);
	}
}

void create_acceleration_structure(Scene& scene)
{
	{
		cudaArray_t distance_field_array;

		cudaExtent extent = make_cudaExtent(WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z);
		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uint32_t>();

		cudaMalloc3DArray(&distance_field_array, &channel_desc, extent, cudaArraySurfaceLoadStore);

		cudaResourceDesc resource_desc = {};
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = distance_field_array;

		cudaCreateSurfaceObject(&scene.distance_field_surface, &resource_desc);

		cudaTextureDesc texture_desc = {};
		texture_desc.readMode = cudaReadModeElementType;
		texture_desc.normalizedCoords = 0;
		texture_desc.filterMode = cudaFilterModePoint;
		texture_desc.addressMode[0] = cudaAddressModeClamp;
		texture_desc.addressMode[1] = cudaAddressModeClamp;
		texture_desc.addressMode[2] = cudaAddressModeClamp;
		cudaCreateTextureObject(&scene.distance_field_texture, &resource_desc, &texture_desc, nullptr);
	}
	
	{
		dim3 grid(4, 4, 4);
		dim3 block(CHUNK_SIZE / 4, CHUNK_SIZE / 4, CHUNK_SIZE / 4);

		for (int z = 0; z < WORLD_CHUNKS_Z; ++z)
		{
			for (int y = 0; y < WORLD_CHUNKS_Y; ++y)
			{
				for (int x = 0; x < WORLD_CHUNKS_X; ++x)
				{
					int3 chunk_pos = { x, y, z };

					generate_chunk_distance_field<<<grid, block>>>(scene, chunk_pos);
				}
			}
		}
	}

	std::cout << "The distance field generated successfully." << '\n';
}