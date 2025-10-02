#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>

#include "render_data.cuh"
#include "scene_generation.cuh"
#include "game_loop.cuh"
#include "path_tracing.cuh"


uint32_t screen_GL_texture;
cudaGraphicsResource_t screen_cuda_resource;

void render(G_Buffer& g_buffer, Scene& scene, Player player)
{
    cudaGraphicsMapResources(1, &screen_cuda_resource);

    cudaArray_t screen_cuda_array;
    cudaGraphicsSubResourceGetMappedArray(&screen_cuda_array, screen_cuda_resource, 0, 0);

    cudaResourceDesc screen_cuda_array_resource_desc = {};
    screen_cuda_array_resource_desc.resType = cudaResourceTypeArray;
    screen_cuda_array_resource_desc.res.array.array = screen_cuda_array;

    cudaSurfaceObject_t screen_cuda_surface_object;
    cudaCreateSurfaceObject(&screen_cuda_surface_object, &screen_cuda_array_resource_desc);

    // render here

    dim3 grid(20, 45, 1);
    dim3 block(32, 8, 1);


    path_tracing<<<grid, block>>>(screen_cuda_surface_object, g_buffer, scene, player);

    regulate_probes<<<grid, block>>>(g_buffer, scene, player);

    render_frame<<<grid, block>>>(screen_cuda_surface_object, g_buffer, scene);


    cudaDestroySurfaceObject(screen_cuda_surface_object);

    cudaGraphicsUnmapResources(1, &screen_cuda_resource);

    cudaStreamSynchronize(0);

    glBindTexture(GL_TEXTURE_2D, screen_GL_texture);

    glBegin(GL_QUADS);

    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);

    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glFinish();
}

int main()
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(SCREEN_W, SCREEN_H, "Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSwapInterval(0);

    // copied initialization code
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &screen_GL_texture);

    glBindTexture(GL_TEXTURE_2D, screen_GL_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCREEN_W, SCREEN_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsGLRegisterImage(&screen_cuda_resource, screen_GL_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);


    // setup render data

    Player player({ 0.1f, 1.5f, 0.1f }, 0, 0);

    G_Buffer g_buffer = G_Buffer();

    Scene scene;

    generate_scene(scene);


    float time = 0;
    int frame_count = 0;

    float time_step = 0;

    while (!glfwWindowShouldClose(window))
    {
        double initial_time = glfwGetTime();

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        input(scene, player, time_step, window);
        // simulate(scene, player, time_step);
        render(g_buffer, scene, player);

        ++player.frame_id;

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();


        double final_time = glfwGetTime();

        time_step = final_time - initial_time;

        time += time_step;
        ++frame_count;

        if (time > 1.0)
        {
            std::cout << "FPS: " << frame_count << '\n';
            std::cout << "ms/frame: " << 1000.0 / frame_count << '\n';

            time = 0;
            frame_count = 0;
        }
    }

    glfwTerminate();

    return 0;
}