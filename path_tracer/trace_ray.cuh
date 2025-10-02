#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>

#include "render_data.cuh"


__device__ Vec3D sample_atmosphere(Vec3D ray_direction, Scene scene)
{
    Vec3D sun_dir = normalize(scene.sun_direction);

    // Sun height: 1 = zenith, 0 = horizon, negative = below
    float sun_height = sun_dir.y;

    // View-sun alignment
    float cos_theta = dot(ray_direction, sun_dir);

    // Colors
    Vec3D day_zenith = { 0.1f, 0.3f, 0.9f };
    Vec3D sunset_tint = { 0.95f, 0.45f, 0.15f };
    Vec3D night_sky = { 0.002f, 0.004f, 0.01f }; // very dark, close to black-blue

    // Base day sky: smooth horizon darkening
    float horizon_factor = smoothstep(-0.2f, 1.0f, ray_direction.y);
    Vec3D day_sky = lerp(day_zenith * 0.15f, day_zenith, horizon_factor);

    // Sunset band: Gaussian falloff near horizon
    float sunset_band = expf(-powf(sun_height * 6.0f, 2.0f));
    float sun_facing = smoothstep(0.0f, 1.0f, cos_theta);
    float sunset_strength = sunset_band * sun_facing;
    Vec3D sunset_sky = sunset_tint * sunset_strength;

    // Day-night factor: sharper fade to night
    float day_night_factor = smoothstep(-0.5f, 0.25f, sun_height);

    // Combine
    Vec3D sky_color = lerp(night_sky, day_sky + sunset_sky, day_night_factor);

    return sky_color;
}


__device__ Vec3D sample_sky(Vec3D ray_direction, Scene scene)
{
    Vec3D atmosphere_color = sample_atmosphere(ray_direction, scene);

    float sun_sharpness = 45000.0f;

    return scene.sun_color * expf(sun_sharpness * (dot(ray_direction, scene.sun_direction) - 1.0f)) + atmosphere_color;
}


__device__ Vec4D sample_texture(Texture texture, Vec2D uv_coordinates)
{
    int pixel_index = int(uv_coordinates.y * texture.height) * texture.width + (uv_coordinates.x * texture.width);

    Vec4D color = texture.buffer[pixel_index];

    return color;
}


__device__ bool trace_ray(Ray ray, Scene scene, Vec3D bounds_center, Sample* sample)
{
    bounds_center = grid_space(bounds_center);

    float min_x = fmaxf(0, floorf(bounds_center.x - RENDER_DISTANCE));
    float min_y = fmaxf(0, floorf(bounds_center.y - RENDER_DISTANCE));
    float min_z = fmaxf(0, floorf(bounds_center.z - RENDER_DISTANCE));
    float max_x = fminf(WORLD_SIZE_X, ceilf(bounds_center.x + RENDER_DISTANCE));
    float max_y = fminf(WORLD_SIZE_Y, ceilf(bounds_center.y + RENDER_DISTANCE));
    float max_z = fminf(WORLD_SIZE_Z, ceilf(bounds_center.z + RENDER_DISTANCE));


    float reciprocal_ray_x = 1.0f / ray.direction.x;
    float reciprocal_ray_y = 1.0f / ray.direction.y;
    float reciprocal_ray_z = 1.0f / ray.direction.z;

    float sign_x = (ray.direction.x >= 0) * 2 - 1;
    float sign_y = (ray.direction.y >= 0) * 2 - 1;
    float sign_z = (ray.direction.z >= 0) * 2 - 1;


#define TRAVERSAL_OFFSET 0.0001f

    float sign_offset_x = float(ray.direction.x >= 0) + TRAVERSAL_OFFSET * sign_x;
    float sign_offset_y = float(ray.direction.y >= 0) + TRAVERSAL_OFFSET * sign_y;
    float sign_offset_z = float(ray.direction.z >= 0) + TRAVERSAL_OFFSET * sign_z;

#undef TRAVERSAL_OFFSET

    float scale_x = 0;
    float scale_y = 0;
    float scale_z = 0;

    float scale = 0;


    while (true)
    {
        Vec3D grid_pos = grid_space(ray.position);

        if (grid_pos.x < min_x || grid_pos.x >= max_x || grid_pos.y < min_y || grid_pos.y >= max_y || grid_pos.z < min_z || grid_pos.z >= max_z)
        {
            return false;
        }


        uint32_t distances = tex3D<uint32_t>(scene.distance_field_texture, grid_pos.x, grid_pos.y, grid_pos.z);

        if (distances == 0)
        {
            int block_id = scene.grid[grid_linear_pos(grid_pos)];

            Block block = scene.blocks[block_id];


            Vec3D normal;
            Vec3D tangent;
            Vec3D bitangent;


            if (scale == scale_x)
            {
                normal = { -sign_x, 0, 0 };
                tangent = { 0, 0, 1 };
                bitangent = { 0, 1, 0 };
            }
            else if (scale == scale_y)
            {
                normal = { 0, -sign_y, 0 };
                tangent = { 1, 0, 0 };
                bitangent = { 0, 0, 1 };
            }
            else if (scale == scale_z)
            {
                normal = { 0, 0, -sign_z };
                tangent = { 1, 0, 0 };
                bitangent = { 0, 1, 0 };
            }


            Vec3D position_in_block = { frac(ray.position.x), frac(ray.position.y), frac(ray.position.z) };

            Vec2D uv_coordinates = { dot(position_in_block, tangent), dot(position_in_block, bitangent) };

            if (normal.x < 0 || normal.z > 0)
            {
                uv_coordinates.u = 1.0f - uv_coordinates.u;
            }


            Vec4D texture_sample = sample_texture(block.texture, uv_coordinates);

            sample->albedo = { texture_sample.r, texture_sample.g, texture_sample.b };
            sample->emittance = texture_sample.a < 1.0f ? 0.0f : block.emittance;

#define INTERSECTION_OFFSET 0.0002f
            sample->position = ray.position + INTERSECTION_OFFSET * normal;
#undef INTERSECTION_OFFSET

            sample->normal = normal;
            sample->tangent = tangent;
            sample->bitangent = bitangent;

            return true;
        }


        int right_dist = distances & 0b11111;
        int left_dist = (distances >> 5) & 0b11111;
        int up_dist = (distances >> 10) & 0b11111;
        int down_dist = (distances >> 15) & 0b11111;
        int front_dist = (distances >> 20) & 0b11111;
        int back_dist = (distances >> 25) & 0b11111;

        float dist_x = ray.direction.x >= 0 ? right_dist : -left_dist;
        float dist_y = ray.direction.y >= 0 ? up_dist : -down_dist;
        float dist_z = ray.direction.z >= 0 ? front_dist : -back_dist;


        scale_x = (sign_offset_x - frac(grid_pos.x) + dist_x) * reciprocal_ray_x;
        scale_y = (sign_offset_y - frac(grid_pos.y) + dist_y) * reciprocal_ray_y;
        scale_z = (sign_offset_z - frac(grid_pos.z) + dist_z) * reciprocal_ray_z;

        scale = fminf(scale_x, fminf(scale_y, scale_z));


        ray.position += ray.direction * scale;
    }
}


__device__ Vec3D trace_sun_ray(Vec3D position, Vec3D normal, Scene scene, Vec3D bounds_center)
{
    Ray ray = { position, scene.sun_direction };

    Sample sample;

    if (dot(ray.direction, normal) <= 0.0f || trace_ray(ray, scene, bounds_center, &sample))
    {
        return ZERO_VEC3D;
    }
    else
    {
        return scene.sun_color * dot(ray.direction, normal);
    }
}