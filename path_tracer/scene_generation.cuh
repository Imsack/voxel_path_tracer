#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>
#include <vector>

#include "render_data.cuh"
#include "acceleration_structure.cuh"
#include "game_loop.cuh"


inline void add_block(uint8_t* grid, int3 position, uint16_t index)
{
	int pos = grid_linear_pos(position);

	grid[pos] = index;
}

inline void add_cuboid(uint8_t* grid, int3 start, int3 end, uint16_t index)
{
	for (int z = start.z; z <= end.z; ++z)
	{
		for (int y = start.y; y <= end.y; ++y)
		{
			for (int x = start.x; x <= end.x; ++x)
			{
				add_block(grid, { x, y, z }, index);
			}
		}
	}
}

inline void remove_block(uint8_t* grid, int3 position)
{
	int pos = grid_linear_pos(position);

	grid[pos] = 0;
}

inline void remove_cuboid(uint8_t* grid, int3 start, int3 end)
{
	for (int z = start.z; z <= end.z; ++z)
	{
		for (int y = start.y; y <= end.y; ++y)
		{
			for (int x = start.x; x <= end.x; ++x)
			{
				remove_block(grid, { x, y, z });
			}
		}
	}
}


float fade(float t)
{
	return t * t * (3.0f - 2.0f * t);
}

uint32_t wang_hash(int x, int y, int z)
{
	uint32_t h = 2166136261u;

	h ^= (uint32_t)x * 374761393u;
	h = (h << 13) | (h >> 19);

	h ^= (uint32_t)y * 668265263u;
	h = (h << 11) | (h >> 21);

	h ^= (uint32_t)z * 1274126177u;
	h = (h << 15) | (h >> 17);

	h ^= h >> 16;
	h *= 0x7feb352du;
	h ^= h >> 15;
	h *= 0x846ca68bu;
	h ^= h >> 16;

	return h;
}

Vec2D gradient(int x, int y)
{
	int index = wang_hash(x, y, 0) % 8;

	static const Vec2D gradient_lookup_table[8] =
	{
		{ 1.0, 0.0 },
		{ -1.0, 0.0 },
		{ 0.0, 1.0 },
		{ 0.0,-1.0 },
		{ 0.70710678,  0.70710678 },
		{ -0.70710678,  0.70710678 },
		{ 0.70710678, -0.70710678 },
		{ -0.70710678, -0.70710678 }
	};

	return gradient_lookup_table[index];
}

float perlin2D(Vec2D p)
{
	int x0 = floorf(p.x);
	int y0 = floorf(p.y);
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	Vec2D grad00 = gradient(x0, y0);
	Vec2D grad01 = gradient(x0, y1);
	Vec2D grad10 = gradient(x1, y0);
	Vec2D grad11 = gradient(x1, y1);

	Vec2D offset00 = { p.x - x0, p.y - y0 };
	Vec2D offset01 = { p.x - x0, p.y - y1 };
	Vec2D offset10 = { p.x - x1, p.y - y0 };
	Vec2D offset11 = { p.x - x1, p.y - y1 };

	float dot00 = dot(grad00, offset00);
	float dot01 = dot(grad01, offset01);
	float dot10 = dot(grad10, offset10);
	float dot11 = dot(grad11, offset11);

	float u = fade(p.x - x0);
	float v = fade(p.y - y0);

	float dot0 = lerp(dot00, dot10, u);
	float dot1 = lerp(dot01, dot11, u);

	float dot = lerp(dot0, dot1, v);

	return dot;
}

Vec3D gradient(int x, int y, int z)
{
	int index = wang_hash(x, y, z) % 12;

	static const Vec3D gradient_lookup_table[12] =
	{
		{  1,  1,  0 }, { -1,  1,  0 }, {  1, -1,  0 }, { -1, -1,  0 },
		{  1,  0,  1 }, { -1,  0,  1 }, {  1,  0, -1 }, { -1,  0, -1 },
		{  0,  1,  1 }, {  0, -1,  1 }, {  0,  1, -1 }, {  0, -1, -1 }
	};

	return gradient_lookup_table[index];
}

float perlin3D(Vec3D p)
{
	int x0 = floorf(p.x);
	int y0 = floorf(p.y);
	int z0 = floorf(p.z);
	int x1 = x0 + 1;
	int y1 = y0 + 1;
	int z1 = z0 + 1;

	Vec3D grad000 = gradient(x0, y0, z0);
	Vec3D grad001 = gradient(x0, y0, z1);
	Vec3D grad010 = gradient(x0, y1, z0);
	Vec3D grad011 = gradient(x0, y1, z1);
	Vec3D grad100 = gradient(x1, y0, z0);
	Vec3D grad101 = gradient(x1, y0, z1);
	Vec3D grad110 = gradient(x1, y1, z0);
	Vec3D grad111 = gradient(x1, y1, z1);

	Vec3D offset000 = { p.x - x0, p.y - y0, p.z - z0 };
	Vec3D offset001 = { p.x - x0, p.y - y0, p.z - z1 };
	Vec3D offset010 = { p.x - x0, p.y - y1, p.z - z0 };
	Vec3D offset011 = { p.x - x0, p.y - y1, p.z - z1 };
	Vec3D offset100 = { p.x - x1, p.y - y0, p.z - z0 };
	Vec3D offset101 = { p.x - x1, p.y - y0, p.z - z1 };
	Vec3D offset110 = { p.x - x1, p.y - y1, p.z - z0 };
	Vec3D offset111 = { p.x - x1, p.y - y1, p.z - z1 };

	float dot000 = dot(grad000, offset000);
	float dot001 = dot(grad001, offset001);
	float dot010 = dot(grad010, offset010);
	float dot011 = dot(grad011, offset011);
	float dot100 = dot(grad100, offset100);
	float dot101 = dot(grad101, offset101);
	float dot110 = dot(grad110, offset110);
	float dot111 = dot(grad111, offset111);

	float u = fade(p.x - x0);
	float v = fade(p.y - y0);
	float w = fade(p.z - z0);

	float dot00 = lerp(dot000, dot100, u);
	float dot01 = lerp(dot001, dot101, u);
	float dot10 = lerp(dot010, dot110, u);
	float dot11 = lerp(dot011, dot111, u);

	float dot0 = lerp(dot00, dot10, v);
	float dot1 = lerp(dot01, dot11, v);

	float dot = lerp(dot0, dot1, w);

	return dot;
}


uint16_t get_rock_type(int x, int y, int z)
{
	uint32_t hash = wang_hash(x, y, z);

	if (y < 32 && hash % 8192 == 0)
	{
		return 6;
	}

	if (hash % 4096 == 1)
	{
		return 5;
	}
	
	if (hash % 512 == 2)
	{
		return 4;
	}

	return 3;
}

void generate_hills(uint8_t* grid)
{
	for (int z = 0; z < WORLD_SIZE_Z; ++z)
	{
		for (int x = 0; x < WORLD_SIZE_X; ++x)
		{
			Vec2D p = { float(x), float(z) };

			float noise0 = 32.0f * perlin2D(p / 128.0f);
			float noise1 = 8.0f * perlin2D(p / 32.0f);
			float noise2 = 2.0f * perlin2D(p / 8.0f);

			int hill_level = (noise0 + noise1 + noise2) + 64.0f;

			int rock_level = (noise0 + noise1) + 60.0f;

			int y = 0;

			for(; y < rock_level; ++y)
			{
				uint16_t rock_type = get_rock_type(x, y, z);

				if (y < WORLD_SIZE_Y)
				{
					add_block(grid, { x, y, z }, rock_type);
				}
			}

			for (; y < hill_level; ++y)
			{
				if (y < WORLD_SIZE_Y)
				{
					add_block(grid, { x, y, z }, 2);
				}
			}

			if (y < WORLD_SIZE_Y)
			{
				add_block(grid, { x, y, z }, 1);
			}
		}
	}
}

void generate_mountains(uint8_t* grid)
{
	for (int z = 0; z < WORLD_SIZE_Z; ++z)
	{
		for (int x = 0; x < WORLD_SIZE_X; ++x)
		{
			Vec2D p = { float(x), float(z) };

			float height_factor = tanhf(10.0f * (perlin2D(p / 256.0f) - 0.25f));

			float noise0 = 64.0f * perlin2D(p / 32.0f);
			float noise1 = 8.0f * perlin2D(p / 8.0f);

			int mountain_level = (noise0 + noise1 + 80.0f) * height_factor;

			for (int y = 0; y < mountain_level; ++y)
			{
				uint16_t rock_type = get_rock_type(x, y, z);

				if (y < WORLD_SIZE_Y)
				{
					add_block(grid, { x, y, z }, rock_type);
				}
			}
		}
	}
}

void generate_caves(uint8_t* grid)
{
	for (int z = 0; z < WORLD_SIZE_Z; ++z)
	{
		for (int y = 0; y < WORLD_SIZE_Y; ++y)
		{
			for (int x = 0; x < WORLD_SIZE_X; ++x)
			{
				Vec3D p = { float(x), float(y), float(z) };

				float noise0 = perlin3D(p / 256.0f);
				float noise1 = perlin3D(p / 128.0f);
				float noise2 = perlin3D(p / 64.0f);
				float noise3 = perlin3D(p / 32.0f);

				if ((noise0 > 0.1f && abs(noise1 + noise2) < 0.05f) || noise3 > 0.5f)
				{
					remove_block(grid, { x, y, z });
				}
			}
		}
	}
}

void plant_oak_tree(uint8_t* grid, int3 pos)
{
	int height = wang_hash(pos.x, 0, pos.z) % 8 + 8;

	int radius = height / 2;

	int3 center = { pos.x, pos.y + height, pos.z };

	for (int z = center.z - radius; z < center.z + radius; ++z)
	{
		for (int y = center.y - radius; y < center.y + radius; ++y)
		{
			for (int x = center.x - radius; x < center.x + radius; ++x)
			{
				if (x > 0 && x < WORLD_SIZE_X && y > 0 && y < WORLD_SIZE_Y && z > 0 && z < WORLD_SIZE_Z)
				{
					float distance = magnitude({ float(x - center.x), float(y - center.y), float(z - center.z) });

					if (distance < radius && wang_hash(x, y, z) % 2 == 0)
					{
						add_block(grid, { x, y, z }, 8);
					}
				}
			}
		}
	}

	for (int y = pos.y; y < pos.y + height; ++y)
	{
		if (y < WORLD_SIZE_Y)
		{
			add_block(grid, { pos.x, y, pos.z }, 7);
		}
	}
}

void plant_birch_tree(uint8_t* grid, int3 pos)
{
	int height = wang_hash(pos.x, 0, pos.z) % 8 + 16;

	int radius = height / 4;

	int3 center = { pos.x, pos.y + height, pos.z };

	for (int z = center.z - radius; z < center.z + radius; ++z)
	{
		for (int y = center.y - radius; y < center.y + radius; ++y)
		{
			for (int x = center.x - radius; x < center.x + radius; ++x)
			{
				if (x > 0 && x < WORLD_SIZE_X && y > 0 && y < WORLD_SIZE_Y && z > 0 && z < WORLD_SIZE_Z)
				{
					float distance = magnitude({ float(x - center.x), float(y - center.y), float(z - center.z) });

					if (distance < radius && wang_hash(x, y, z) % 2 == 0)
					{
						add_block(grid, { x, y, z }, 10);
					}
				}
			}
		}
	}

	for (int y = pos.y; y < pos.y + height; ++y)
	{
		if (y < WORLD_SIZE_Y)
		{
			add_block(grid, { pos.x, y, pos.z }, 9);
		}
	}
}

void generate_oak_trees(uint8_t* grid)
{
	for (int z = 0; z < WORLD_SIZE_Z; ++z)
	{
		for (int x = 0; x < WORLD_SIZE_X; ++x)
		{
			if (wang_hash(x, 0, z) % 256 == 0)
			{
				for (int y = 0; y < 100; ++y)
				{
					int block_index = grid[grid_linear_pos(int3{ x, y, z })];

					if (block_index == 1)
					{
						plant_oak_tree(grid, { x, y + 1, z });

						break;
					}
				}
			}
		}
	}
}

void generate_birch_trees(uint8_t* grid)
{
	for (int z = 0; z < WORLD_SIZE_Z; ++z)
	{
		for (int x = 0; x < WORLD_SIZE_X; ++x)
		{
			if (wang_hash(x, 0, z) % 1024 == 1)
			{
				for (int y = 0; y < 100; ++y)
				{
					int block_index = grid[grid_linear_pos(int3{ x, y, z })];

					if (block_index == 1)
					{
						plant_birch_tree(grid, { x, y + 1, z });

						break;
					}
				}
			}
		}
	}
}

void generate_world_floor(uint8_t* grid)
{
	add_cuboid(grid, { 0, 0, 0 }, { WORLD_SIZE_X - 1, 2, WORLD_SIZE_Z - 1 }, 12);
}


inline void generate_scene(Scene& scene)
{
	scene.sun_direction = normalize(Vec3D{ 0.0f, 1.0f, 1.0f });
	scene.sun_color = sun_color(scene.sun_direction);

	Texture blue_noise_0;
	make_texture(blue_noise_0, "textures/blue_noise_0.png");
	Texture blue_noise_1;
	make_texture(blue_noise_1, "textures/blue_noise_1.png");
	Texture blue_noise_2;
	make_texture(blue_noise_2, "textures/blue_noise_2.png");

	Texture grass;
	make_texture(grass, "textures/grass.png");
	Texture dirt;
	make_texture(dirt, "textures/dirt.png");
	Texture stone;
	make_texture(stone, "textures/stone.png");
	Texture coal;
	make_texture(coal, "textures/coal.png");
	Texture copper;
	make_texture(copper, "textures/copper.png");
	Texture uranium;
	make_texture(uranium, "textures/uranium.png");
	Texture oak;
	make_texture(oak, "textures/oak.png");
	Texture oak_leaves;
	make_texture(oak_leaves, "textures/oak_leaves.png");
	Texture birch;
	make_texture(birch, "textures/birch.png");
	Texture birch_leaves;
	make_texture(birch_leaves, "textures/birch_leaves.png");
	Texture planks;
	make_texture(planks, "textures/planks.png");
	Texture magma;
	make_texture(magma, "textures/magma.png");


	Probe* probes = (Probe*)malloc(PROBE_CACHE_SIZE * sizeof(Probe));

	for (int i = 0; i < PROBE_CACHE_SIZE; ++i)
	{
		probes[i] = { 0, ZERO_VEC3D, 0, 0, 0 };
	}

	uint8_t* grid = (uint8_t*)malloc(WORLD_SIZE * sizeof(uint8_t));

	for (int i = 0; i < WORLD_SIZE; ++i)
	{
		grid[i] = 0;
	}


	Block* blocks = (Block*)malloc(BLOCK_COUNT * sizeof(Block));

	blocks[1] = { grass, 0 };
	blocks[2] = { dirt, 0 };
	blocks[3] = { stone, 0 };
	blocks[4] = { coal, 0 };
	blocks[5] = { copper, 0 };
	blocks[6] = { uranium, 0.5 };
	blocks[7] = { oak, 0 };
	blocks[8] = { oak_leaves, 0 };
	blocks[9] = { birch, 0 };
	blocks[10] = { birch_leaves, 0 };
	blocks[11] = { planks, 0 };
	blocks[12] = { magma, 1.0 };
 

	generate_hills(grid);

	generate_mountains(grid);

	generate_caves(grid);

	generate_oak_trees(grid);

	generate_birch_trees(grid);

	generate_world_floor(grid);


	{
		cudaMalloc(&scene.grid, WORLD_SIZE * sizeof(uint8_t));

		cudaMemcpy(scene.grid, grid, WORLD_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);


		cudaMalloc(&scene.blocks, BLOCK_COUNT * sizeof(Block));

		cudaMemcpy(scene.blocks, blocks, BLOCK_COUNT * sizeof(Block), cudaMemcpyHostToDevice);


		cudaMalloc(&scene.probes, PROBE_CACHE_SIZE * sizeof(Probe));

		cudaMemcpy(scene.probes, probes, PROBE_CACHE_SIZE * sizeof(Probe), cudaMemcpyHostToDevice);
	}

	std::cout << "The scene generated successfully." << '\n';


	create_acceleration_structure(scene);
}