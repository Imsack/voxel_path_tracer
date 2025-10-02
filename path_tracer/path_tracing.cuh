#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>

#include "render_data.cuh"
#include "trace_ray.cuh"


__device__ Vec3D diffuse_direction(float random_unilateral_0, float random_unilateral_1, Vec3D normal, Vec3D tangent, Vec3D bitangent)
{
    float sin_phi;
    float cos_phi;

    fast_sin_cos(random_unilateral_0 * TAU, &sin_phi, &cos_phi);


    float sqrt_random_unilateral = fast_sqrt(random_unilateral_1);

    return sqrt_random_unilateral * cos_phi * tangent + fast_sqrt(1.0f - random_unilateral_1) * normal + sqrt_random_unilateral * sin_phi * bitangent;
}


__device__ void draw_pixel(cudaSurfaceObject_t screen_cuda_surface_object, Vec3D color, int pixel_x, int pixel_y)
{
    //color = aces_tone_mapping(color);

    color = gamma_correction(color);


    color *= MAX_COLOR_CHANNEL;

    color.x = fminf(color.x, MAX_COLOR_CHANNEL);
    color.y = fminf(color.y, MAX_COLOR_CHANNEL);
    color.z = fminf(color.z, MAX_COLOR_CHANNEL);

    surf2Dwrite(make_uchar4(color.x, color.y, color.z, MAX_COLOR_CHANNEL), screen_cuda_surface_object, pixel_x * sizeof(uchar4), pixel_y);
}


__device__ Vec4D sample_blue_noise(Texture blue_noise, int x, int y)
{
    x = x % blue_noise.width;
    y = y % blue_noise.height;


    int pixel_number = y * blue_noise.width + x;

    Vec4D color = blue_noise.buffer[pixel_number];


    const float phi = 1.6180339887498948482;

    blue_noise.buffer[pixel_number] = { frac(color.x + phi), frac(color.y + phi), frac(color.z + phi), frac(color.w + phi) };


    return color;
}


__device__ uint32_t choose_probe_resolution(Vec3D p1, Vec3D p2, float random_unilateral)
{
    float distance = manhattan_distance(p1, p2);

    distance += random_unilateral * distance;


    uint32_t probe_resolution = 64;

    if (distance > 2.0f)
    {
        probe_resolution /= 2;
    }

    if (distance > 4.0f)
    {
        probe_resolution /= 2;
    }

    if (distance > 8.0f)
    {
        probe_resolution /= 2;
    }

    if (distance > 16.0f)
    {
        probe_resolution /= 2;
    }

    if (distance > 32.0f)
    {
        probe_resolution /= 2;
    }

    if (distance > 64.0f)
    {
        probe_resolution /= 2;
    }

    return probe_resolution;
}

__device__ void update_probe(Probe& probe, Player player)
{ 
    probe.frame_id = player.frame_id;

    Vec3D accumulated_light = probe.accumulated_light;
    uint32_t sample_count = probe.sample_count;

    if (atomicExch(&(probe.update_id), player.update_id) != player.update_id)
    {
        atomicAdd(&(probe.accumulated_light.r), -accumulated_light.r);
        atomicAdd(&(probe.accumulated_light.g), -accumulated_light.g);
        atomicAdd(&(probe.accumulated_light.b), -accumulated_light.b);
        atomicSub(&(probe.sample_count), sample_count);
    }
}

__device__ void accumulate_probe(Probe& probe, Vec3D outgoing_light, Player player)
{
    while (probe.update_id != player.update_id);

    atomicAdd(&(probe.accumulated_light.r), outgoing_light.r);
    atomicAdd(&(probe.accumulated_light.g), outgoing_light.g);
    atomicAdd(&(probe.accumulated_light.b), outgoing_light.b);
    atomicAdd(&(probe.sample_count), 1);
}


__device__ void trace_path(G_Buffer g_buffer, Scene scene, Player player, int pixel_x, int pixel_y)
{
    int pixel_index = pixel_y * SCREEN_W + pixel_x;


    Vec3D primary_direction = normalize({ (pixel_x + 0.5f - SCREEN_W / 2) / float(SCREEN_W / 2), (pixel_y + 0.5f - SCREEN_H / 2) / float(SCREEN_W / 2), 1.0f });
    primary_direction = player.rotation_matrix * primary_direction;

    Ray ray = { player.position, primary_direction };


    Vec3D albedo = { 1.0f, 1.0f, 1.0f };
    Vec3D direct_light = ZERO_VEC3D;
    Vec3D outgoing_light = ZERO_VEC3D;
    Vec3D throughput = { 1.0f, 1.0f, 1.0f };


    Vec4D blue_noise_0 = sample_blue_noise(g_buffer.blue_noise_0, pixel_x, pixel_y);
    Vec4D blue_noise_1 = sample_blue_noise(g_buffer.blue_noise_1, pixel_x, pixel_y);


    Sample sample;

    if (trace_ray(ray, scene, player.position, &sample))
    {
        albedo = sample.albedo;

        direct_light += sample.emittance * throughput;

        
        Vec3D intersection = sample.position;

        Vec3D normal = sample.normal;


        uint32_t probe_resolution = choose_probe_resolution(player.position, intersection, blue_noise_0.w);

        Vec3D jitter = Vec3D{ blue_noise_0.x - 0.5f, blue_noise_0.y - 0.5f, blue_noise_0.z - 0.5f } / float(probe_resolution);

        int probe_index = claim_cache_index(intersection + jitter, normal, player.frame_id, scene.probes, probe_resolution);


        Probe& probe = scene.probes[probe_index];

        update_probe(probe, player);


        ray = { sample.position, diffuse_direction(blue_noise_1.x, blue_noise_1.y, sample.normal, sample.tangent, sample.bitangent)};

        direct_light += trace_sun_ray(ray.position, sample.normal, scene, player.position) * throughput;

        if (trace_ray(ray, scene, player.position, &sample))
        {
            throughput *= sample.albedo;

            outgoing_light += sample.emittance * throughput;


            ray = { sample.position, diffuse_direction(blue_noise_1.z, blue_noise_1.w, sample.normal, sample.tangent, sample.bitangent) };

            outgoing_light += trace_sun_ray(ray.position, sample.normal, scene, player.position) * throughput;

            if (trace_ray(ray, scene, player.position, &sample))
            {
                throughput *= sample.albedo;

                outgoing_light += sample.emittance * throughput;
            }
            else
            {
                outgoing_light += sample_atmosphere(ray.direction, scene) * throughput;
            }
        }
        else
        {
            outgoing_light += sample_atmosphere(ray.direction, scene) * throughput;
        }


        accumulate_probe(probe, outgoing_light, player);

        g_buffer.updated_probes[pixel_index] = probe_index;
    }
    else
    {
        direct_light = sample_sky(ray.direction, scene);

        g_buffer.updated_probes[pixel_index] = -1;
    }

    g_buffer.albedo[pixel_index] = albedo;
    g_buffer.direct_light[pixel_index] = direct_light;
}


__global__ void path_tracing(cudaSurfaceObject_t screen_cuda_surface_object, G_Buffer g_buffer, Scene scene, Player player)
{
    const int screen_cell_x = (blockIdx.x * blockDim.x + threadIdx.x) * SCREEN_CELL_W;
    const int screen_cell_y = (blockIdx.y * blockDim.y + threadIdx.y) * SCREEN_CELL_H;

    for (int y = screen_cell_y; y < screen_cell_y + SCREEN_CELL_H; ++y)
    {
        for (int x = screen_cell_x; x < screen_cell_x + SCREEN_CELL_W; ++x)
        {
            trace_path(g_buffer, scene, player, x, y);
        }
    }
}


__global__ void regulate_probes(G_Buffer g_buffer, Scene scene, Player player)
{
    const int screen_cell_x = (blockIdx.x * blockDim.x + threadIdx.x) * SCREEN_CELL_W;
    const int screen_cell_y = (blockIdx.y * blockDim.y + threadIdx.y) * SCREEN_CELL_H;

    for (int y = screen_cell_y; y < screen_cell_y + SCREEN_CELL_H; ++y)
    {
        for (int x = screen_cell_x; x < screen_cell_x + SCREEN_CELL_W; ++x)
        {
            int probe_index = g_buffer.updated_probes[y * SCREEN_W + x];

            if (probe_index != -1)
            {
                Probe& probe = scene.probes[probe_index];

                if (atomicExch(&(probe.frame_id), player.frame_id + 1) == player.frame_id)
                {
                    const int max_sample_count = 4096;

                    if (probe.sample_count > max_sample_count)
                    {
                        probe.accumulated_light *= (max_sample_count / float(probe.sample_count));
                        probe.sample_count = max_sample_count;
                    }
                }
            }
        }
    }
}


__global__ void render_frame(cudaSurfaceObject_t screen_cuda_surface_object, G_Buffer g_buffer, Scene scene)
{
    const int screen_cell_x = (blockIdx.x * blockDim.x + threadIdx.x) * SCREEN_CELL_W;
    const int screen_cell_y = (blockIdx.y * blockDim.y + threadIdx.y) * SCREEN_CELL_H;

    for (int y = screen_cell_y; y < screen_cell_y + SCREEN_CELL_H; ++y)
    {
        for (int x = screen_cell_x; x < screen_cell_x + SCREEN_CELL_W; ++x)
        {
            int pixel_index = y * SCREEN_W + x;

            Vec3D albedo = g_buffer.albedo[pixel_index];

            Vec3D outgoing_light = g_buffer.direct_light[pixel_index];


            int probe_index = g_buffer.updated_probes[y * SCREEN_W + x];

            if (probe_index != -1)
            {
                Probe& probe = scene.probes[probe_index];

                outgoing_light += probe.accumulated_light / probe.sample_count;
            }


            outgoing_light = outgoing_light * albedo;


            draw_pixel(screen_cuda_surface_object, outgoing_light, x, y);
        }
    }
}