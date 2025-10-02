#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>
#include <random>

#include "math_functions.cuh"
#include "stb_image/stb_image.h"

#define SCREEN_W 1920
#define SCREEN_H 1080

#define SCREEN_CELL_W 3
#define SCREEN_CELL_H 3

#define PROBE_CACHE_SIZE 50000000


#define RENDER_DISTANCE 512

#define WORLD_SIZE_X 1024
#define WORLD_SIZE_Y 128
#define WORLD_SIZE_Z 1024

#define WORLD_SIZE (WORLD_SIZE_X * WORLD_SIZE_Y * WORLD_SIZE_Z)

#define CHUNK_SIZE 32

#define WORLD_CHUNKS_X (WORLD_SIZE_X / CHUNK_SIZE)
#define WORLD_CHUNKS_Y (WORLD_SIZE_Y / CHUNK_SIZE)
#define WORLD_CHUNKS_Z (WORLD_SIZE_Z / CHUNK_SIZE)

#define BLOCK_COUNT 256

#define MAX_COLOR_CHANNEL 255.0f


__host__ __device__ Vec3D world_space(Vec3D pos)
{
	return { pos.x - WORLD_SIZE_X / 2, pos.y - WORLD_SIZE_Y / 2, pos.z - WORLD_SIZE_Z / 2 };
}

__host__ __device__ int3 world_space(int3 pos)
{
	return { pos.x - WORLD_SIZE_X / 2, pos.y - WORLD_SIZE_Y / 2, pos.z - WORLD_SIZE_Z / 2 };
}

__host__ __device__ Vec3D grid_space(Vec3D pos)
{
	return { pos.x + WORLD_SIZE_X / 2, pos.y + WORLD_SIZE_Y / 2, pos.z + WORLD_SIZE_Z / 2 };
}

__host__ __device__ int3 grid_space(int3 pos)
{
	return { pos.x + WORLD_SIZE_X / 2, pos.y + WORLD_SIZE_Y / 2, pos.z + WORLD_SIZE_Z / 2 };
}

__host__ __device__ int grid_linear_pos(Vec3D pos) // expects pos to be in grid space
{
	return int(pos.z) * WORLD_SIZE_Y * WORLD_SIZE_X + int(pos.y) * WORLD_SIZE_X + int(pos.x);
}

__host__ __device__ int grid_linear_pos(int3 pos) // expects pos to be in grid space
{
	return pos.z * WORLD_SIZE_Y * WORLD_SIZE_X + pos.y * WORLD_SIZE_X + pos.x;
}

__host__ __device__ int normal_index(Vec3D normal)
{
	return 5 * (normal.x > 0) + 4 * (normal.x < 0) + 3 * (normal.y > 0) + 2 * (normal.y < 0) + (normal.z > 0);
}


struct Probe
{
	uint64_t key;
	Vec3D accumulated_light;
	uint32_t sample_count;
	uint32_t frame_id;
	uint32_t update_id;
};

__device__ inline uint32_t spatial_hash(uint3 p, uint32_t normal_id)
{
	const uint32_t p1 = 73856093u;
	const uint32_t p2 = 19349663u;
	const uint32_t p3 = 83492791u;
	const uint32_t p4 = 2654435761u;

	uint32_t hash = normal_id * p4;

	hash ^= (p.x + WORLD_SIZE_X) * p1;
	hash ^= (p.y + WORLD_SIZE_Y) * p2;
	hash ^= (p.z + WORLD_SIZE_Z) * p3;

	return hash;
}

__device__ int claim_cache_index(Vec3D point, Vec3D normal, uint32_t frame_id, Probe* probes, uint32_t probe_resolution)
{
	uint32_t normal_id = normal_index(normal);

	uint3 discretized_point = {
		__float2int_rd(point.x * probe_resolution) + (WORLD_SIZE_X / 2) * probe_resolution,
		__float2int_rd(point.y * probe_resolution) + (WORLD_SIZE_Y / 2) * probe_resolution,
		__float2int_rd(point.z * probe_resolution) + (WORLD_SIZE_Z / 2) * probe_resolution
	};

	uint64_t key = 0;

	key |= uint64_t(discretized_point.x & ((1 << 20) - 1));
	key |= uint64_t(discretized_point.y & ((1 << 20) - 1)) << 20;
	key |= uint64_t(discretized_point.z & ((1 << 20) - 1)) << 40;
	key |= uint64_t(normal_id + 1) << 60;

	int hash = spatial_hash(discretized_point, normal_id) % PROBE_CACHE_SIZE;

	for (int i = 0; i < 8; ++i)
	{
		Probe& probe = probes[hash];

		uint64_t previous_key = atomicCAS(&(probe.key), 0, key);

		if (previous_key == 0 || previous_key == key)
		{
			return hash;
		}

		uint32_t previous_frame_id = atomicExch(&(probe.frame_id), frame_id);
		
		if ((frame_id + 1 - previous_frame_id) > 128)
		{
			probe.accumulated_light = ZERO_VEC3D;
			probe.sample_count = 0;
			probe.key = key;

			return hash;
		}

		hash = (hash + 1) % PROBE_CACHE_SIZE;
	}

	return hash;
}


struct Texture
{
	Vec4D* buffer;
	int width = 0;
	int height = 0;
};

void make_texture(Texture& texture, const std::string& file_path)
{
	//stbi_set_flip_vertically_on_load(1);

	int bits_per_pixel = 0;

	int texture_channel_count = 4;

	uint8_t* image_buffer = stbi_load(file_path.c_str(), &texture.width, &texture.height, &bits_per_pixel, texture_channel_count);


	int pixel_count = texture.width * texture.height;

	int texture_bytes = pixel_count * sizeof(Vec4D);

	Vec4D* temporary_buffer = (Vec4D*)malloc(texture_bytes);


	for (int i = 0; i < pixel_count; ++i)
	{
		int offset = i * texture_channel_count;

		float r = image_buffer[offset + 0] / MAX_COLOR_CHANNEL;
		float g = image_buffer[offset + 1] / MAX_COLOR_CHANNEL;
		float b = image_buffer[offset + 2] / MAX_COLOR_CHANNEL;
		float a = image_buffer[offset + 3] / MAX_COLOR_CHANNEL;

		temporary_buffer[i] = { r, g, b, a };
	}

	free(image_buffer);


	cudaMalloc(&texture.buffer, texture_bytes);
	cudaMemcpy(texture.buffer, temporary_buffer, texture_bytes, cudaMemcpyHostToDevice);


	free(temporary_buffer);
}


struct Player
{
	uint32_t frame_id;
	uint32_t update_id;

	Vec3D position;
	float yaw, pitch;
	Matrix3D rotation_matrix;

	Player(Vec3D position, float yaw, float pitch)
	{
		this->frame_id = 0;
		this->update_id = 0;

		this->position = position;
		this->yaw = yaw;
		this->pitch = pitch;

		rotation_matrix = create_rotation_matrix(yaw, pitch);
	}
};


struct Sample
{
	Vec3D albedo;
	float emittance;
	Vec3D position;
	Vec3D normal;
	Vec3D tangent;
	Vec3D bitangent;
};


struct G_Buffer
{
	Texture blue_noise_0;
	Texture blue_noise_1;
	Texture blue_noise_2;

	Vec3D* albedo;
	Vec3D* direct_light;
	int* updated_probes;
	uint32_t* random_seeds;

	G_Buffer()
	{
		make_texture(blue_noise_0, "textures/blue_noise_0.png");
		make_texture(blue_noise_1, "textures/blue_noise_1.png");
		make_texture(blue_noise_2, "textures/blue_noise_2.png");

		Vec3D* initial_frame = (Vec3D*)malloc(SCREEN_W * SCREEN_H * sizeof(Vec3D));
		int* initial_probes = (int*)malloc(SCREEN_W * SCREEN_H * sizeof(int));
		uint32_t* initial_seeds = (uint32_t*)malloc(SCREEN_W * SCREEN_H * sizeof(uint32_t));

		std::mt19937 generator;
		std::uniform_int_distribution<uint32_t> distribution(0, UINT32_MAX);

		for (int i = 0; i < SCREEN_W * SCREEN_H; ++i)
		{
			initial_frame[i] = ZERO_VEC3D;
			initial_probes[i] = -1;
			initial_seeds[i] = distribution(generator);
		}

		cudaMalloc(&albedo, SCREEN_W * SCREEN_H * sizeof(Vec3D));
		cudaMemcpy(albedo, initial_frame, SCREEN_W * SCREEN_H * sizeof(Vec3D), cudaMemcpyHostToDevice);

		cudaMalloc(&direct_light, SCREEN_W * SCREEN_H * sizeof(Vec3D));
		cudaMemcpy(direct_light, initial_frame, SCREEN_W * SCREEN_H * sizeof(Vec3D), cudaMemcpyHostToDevice);

		cudaMalloc(&updated_probes, SCREEN_W * SCREEN_H * sizeof(int));
		cudaMemcpy(updated_probes, initial_probes, SCREEN_W * SCREEN_H * sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc(&random_seeds, SCREEN_W * SCREEN_H * sizeof(uint32_t));
		cudaMemcpy(random_seeds, initial_seeds, SCREEN_W * SCREEN_H * sizeof(uint32_t), cudaMemcpyHostToDevice);
	}
};

__device__ float random_unilateral(G_Buffer g_buffer, int pixel_index)
{
	uint32_t state = g_buffer.random_seeds[pixel_index] * 747796405u + 2891336453u;
	uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	uint32_t random_unsigned_int = (word >> 22u) ^ word;

	g_buffer.random_seeds[pixel_index] = random_unsigned_int;

	return float(random_unsigned_int) / UINT32_MAX;
}


struct Block
{
	Texture texture;
	float emittance;
};

struct Scene
{
	Vec3D sun_color;
	Vec3D sun_direction;

	uint8_t* grid;
	Block* blocks;

	cudaTextureObject_t distance_field_texture;
	cudaSurfaceObject_t distance_field_surface;

	Probe* probes;
};