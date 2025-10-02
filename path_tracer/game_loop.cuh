#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>

#include "math_functions.cuh"
#include "render_data.cuh"

void input(Scene& scene, Player& player, float time_step, GLFWwindow* window)
{
	float movement_speed = time_step * 4.0f;

	Vec3D motion_vector = ZERO_VEC3D;

	if (glfwGetKey(window, GLFW_KEY_W))
	{
		motion_vector = player.rotation_matrix * Vec3D{ 0.0f, 0.0f, 1.0f } * movement_speed;
	}
	if (glfwGetKey(window, GLFW_KEY_A))
	{
		motion_vector = player.rotation_matrix * Vec3D{ -1.0f, 0.0f, 0.0f } * movement_speed;
	}
	if (glfwGetKey(window, GLFW_KEY_S))
	{
		motion_vector = player.rotation_matrix * Vec3D{ 0.0f, 0.0f, -1.0f } * movement_speed;
	}
	if (glfwGetKey(window, GLFW_KEY_D))
	{
		motion_vector = player.rotation_matrix * Vec3D{ 1.0f, 0.0f, 0.0f } * movement_speed;
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE))
	{
		motion_vector = Vec3D{ 0.0f, 1.0f, 0.0f } * movement_speed;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT))
	{
		motion_vector = Vec3D{ 0.0f, -1.0f, 0.0f } * movement_speed;
	}

	player.position += motion_vector;


	float rotation_speed = time_step;

	if (glfwGetKey(window, GLFW_KEY_LEFT))
	{
		player.yaw -= rotation_speed;
	}
	if (glfwGetKey(window, GLFW_KEY_RIGHT))
	{
		player.yaw += rotation_speed;
	}
	if (glfwGetKey(window, GLFW_KEY_UP))
	{
		player.pitch -= rotation_speed;
	}
	if (glfwGetKey(window, GLFW_KEY_DOWN))
	{
		player.pitch += rotation_speed;
	}

	player.rotation_matrix = create_rotation_matrix(player.yaw, player.pitch);
}

Vec3D sun_color(Vec3D sun_direction)
{
	float t = clamp(sun_direction.y * 0.5f + 0.5f, 0.0f, 1.0f);

	Vec3D sun_color = lerp(Vec3D{ 1.0f, 0.4f, 0.2f }, Vec3D{ 1.0f, 1.0f, 0.9f }, t);

	return sun_color;
}

void simulate(Scene& scene, Player& player, float time_step)
{
	Matrix3D sun_rotation_matrix = create_rotation_matrix(0.0f, -0.005f * time_step);

	scene.sun_direction = sun_rotation_matrix * scene.sun_direction;

	scene.sun_color = sun_color(scene.sun_direction);
}