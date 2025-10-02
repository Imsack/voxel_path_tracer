#pragma once
#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <math.h>

#define PI 3.1415926536f
#define TAU (PI * 2)
#define HALF_PI (PI / 2)

#define ZERO_VEC3D { 0.0f, 0.0f, 0.0f }


__host__ __device__ float square(float x)
{
	return x * x;
}

__host__ __device__ float sign(float x)
{
	return (x >= 0) ? 1.0f : -1.0f;
}

__host__ __device__ float clamp(float x, float lower, float upper)
{
	return fmaxf(lower, fminf(upper, x));
}

__host__ __device__ int clamp(int x, int lower, int upper)
{
	return fmaxf(lower, fminf(upper, x));
}

__host__ __device__ float frac(float x)
{
	return x - floor(x);
}

__host__ __device__ float lerp(float a, float b, float t)
{
	return a + (b - a) * t;
}

__device__ float fast_sqrt(float x)
{
	return __fsqrt_rd(x);
}

__device__ float fast_sin(float x)
{
	return __sinf(x);
}

__device__ float fast_cos(float x)
{
	return __cosf(x);
}

__device__ void fast_sin_cos(float x, float* sin, float* cos)
{
	__sincosf(x, sin, cos);
}

__device__ float fast_reciprocal(float x)
{
	return __frcp_ru(x);
}

__device__ float fast_divide(float x, float y)
{
	return __fdividef(x, y);
}

__device__ float fast_reciprocal_sqrt(float x)
{
	return __frsqrt_rn(x);
}

__device__ float smoothstep(float a, float b, float t)
{
	t = fminf(fmaxf((t - a) / (b - a), 0.0f), 1.0f);

	return t * t * (3.0f - 2.0f * t);
}

__device__ float gamma_correction(float x)
{
	float output = 12.92 * x;

	if (x > 0.0031308)
	{
		output = 1.055 * __powf(x, 1.0 / 2.4) - 0.055;
	}

	return output;
}

__device__ float aces_curve(float x)
{
	return (x * (x + 0.0245786f) - 0.000090537f) / (x * (0.983729f * x + 0.4329510f) + 0.238081f);
}


struct Vec2D
{
	union
	{
		float x, u;
	};

	union
	{
		float y, v;
	};
};

__host__ __device__ Vec2D operator + (Vec2D v1, Vec2D v2)
{
	return { v1.x + v2.x, v1.y + v2.y };
}

__host__ __device__ Vec2D operator - (Vec2D v1, Vec2D v2)
{
	return { v1.x - v2.x, v1.y - v2.y };
}

__host__ __device__ Vec2D operator * (Vec2D v, float s)
{
	return { v.x * s, v.y * s };
}

__host__ __device__ Vec2D operator * (float s, Vec2D v)
{
	return { v.x * s, v.y * s };
}

__host__ __device__ Vec2D operator * (Vec2D v1, Vec2D v2)
{
	return { v1.x * v2.x, v1.y * v2.y };
}

__host__ __device__ Vec2D operator / (Vec2D v, float s)
{
	return v * (1.0f / s);
}

__host__ __device__ Vec2D operator / (Vec2D v1, Vec2D v2)
{
	return { v1.x / v2.x, v1.y / v2.y };
}

__host__ __device__ float dot(Vec2D v1, Vec2D v2)
{
	return v1.x * v2.x + v1.y * v2.y;
}


struct Vec3D
{
	union
	{
		float x, r;
	};

	union
	{
		float y, g, theta;
	};

	union
	{
		float z, b, phi;
	};
};

__host__ __device__ Vec3D operator + (Vec3D v1, Vec3D v2)
{
	return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

__host__ __device__ Vec3D operator - (Vec3D v1, Vec3D v2)
{
	return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

__host__ __device__ Vec3D operator - (Vec3D v)
{
	return { -v.x, -v.y, -v.z };
}

__host__ __device__ Vec3D operator * (Vec3D v, float s)
{
	return { v.x * s, v.y * s, v.z * s };
}

__host__ __device__ Vec3D operator * (float s, Vec3D v)
{
	return { v.x * s, v.y * s, v.z * s };
}

__host__ __device__ Vec3D operator * (Vec3D v1, Vec3D v2)
{
	return { v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
}

__host__ __device__ Vec3D operator / (Vec3D v, float s)
{
	return v * (1.0f / s);
}

__host__ __device__ Vec3D operator / (Vec3D v1, Vec3D v2)
{
	return { v1.x / v2.x, v1.y / v2.y, v1.z / v2.z };
}

__host__ __device__ void operator += (Vec3D& v1, Vec3D v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;
}

__host__ __device__ void operator -= (Vec3D& v1, Vec3D v2)
{
	v1.x -= v2.x;
	v1.y -= v2.y;
	v1.z -= v2.z;
}

__host__ __device__ void operator *= (Vec3D& v1, Vec3D v2)
{
	v1.x *= v2.x;
	v1.y *= v2.y;
	v1.z *= v2.z;
}

__host__ __device__ void operator *= (Vec3D& v1, float s)
{
	v1.x *= s;
	v1.y *= s;
	v1.z *= s;
}

__host__ __device__ bool operator == (Vec3D v1, Vec3D v2)
{
	return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
}

__host__ __device__ bool operator != (Vec3D v1, Vec3D v2)
{
	return !(v1 == v2);
}

__host__ __device__ uint4 operator + (uint4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uchar4 a, uchar4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uint4 a, uchar4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uchar4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ void scale(Vec3D* v, float s)
{
	v->x *= s;
	v->y *= s;
	v->z *= s;
}

__host__ __device__ float dot(Vec3D v1, Vec3D v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ Vec3D cross(Vec3D v1, Vec3D v2)
{
	return { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
}

__host__ __device__ float magnitude(Vec3D v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float fast_magnitude(Vec3D v)
{
	return __fsqrt_rd(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ float magnitude_squared(Vec3D v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ Vec3D normalize(Vec3D v)
{
	float reciprocalMagnitude = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	return { v.x * reciprocalMagnitude, v.y * reciprocalMagnitude, v.z * reciprocalMagnitude };
}

__host__ __device__ void normalize(Vec3D* v)
{
	float reciprocalMagnitude = 1.0f / sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);

	v->x *= reciprocalMagnitude;
	v->y *= reciprocalMagnitude;
	v->z *= reciprocalMagnitude;
}

__device__ Vec3D fast_normalize(Vec3D v)
{
	float reciprocalMagnitude = __frsqrt_rn(v.x * v.x + v.y * v.y + v.z * v.z);

	return { v.x * reciprocalMagnitude, v.y * reciprocalMagnitude, v.z * reciprocalMagnitude };
}

__device__ void fast_normalize(Vec3D* v)
{
	float reciprocalMagnitude = __frsqrt_rn(v->x * v->x + v->y * v->y + v->z * v->z);

	v->x *= reciprocalMagnitude;
	v->y *= reciprocalMagnitude;
	v->z *= reciprocalMagnitude;
}

__host__ __device__ Vec3D lerp(Vec3D v1, Vec3D v2, float t)
{
	return v1 + t * (v2 - v1);
}

__device__ Vec3D proj(Vec3D v1, Vec3D v2)
{
	return fast_divide(dot(v1, v2), magnitude_squared(v2)) * v2;
}

__host__ __device__ float manhattan_distance(Vec3D v1, Vec3D v2)
{
	return fmaxf(abs(v2.x - v1.x), fmaxf(abs(v2.y - v1.y), abs(v2.z - v1.z)));
}

__host__ __device__ int manhattan_distance(int3 v1, int3 v2)
{
	return max(abs(v2.x - v1.x), max(abs(v2.y - v1.y), abs(v2.z - v1.z)));
}

__host__ __device__ Vec3D cartesian_to_spherical(Vec3D p)
{
	float r = magnitude(p);

	float theta = atan2(p.z, p.x);
	float phi = acos(p.y / r);

	return { r, theta, phi };
}

__device__ Vec3D gamma_correction(Vec3D color)
{
	return { gamma_correction(color.r), gamma_correction(color.g), gamma_correction(color.b) };
}


struct Vec4D
{
	union
	{
		float x, r;
	};

	union
	{
		float y, g;
	};

	union
	{
		float z, b;
	};

	union
	{
		float w, a;
	};
};


struct Ray
{
	Vec3D position;
	Vec3D direction;
};


struct Matrix3D
{
	Vec3D i_hat;
	Vec3D j_hat;
	Vec3D k_hat;
};

__host__ __device__ Vec3D operator * (Matrix3D m, Vec3D v)
{
	return v.x * m.i_hat + v.y * m.j_hat + v.z * m.k_hat;
}

__host__ __device__ Matrix3D operator * (Matrix3D m2, Matrix3D m1)
{
	return { m2 * m1.i_hat, m2 * m1.j_hat, m2 * m1.k_hat };
}

inline Matrix3D create_rotation_matrix(float yaw, float pitch)
{
	Matrix3D y_axis_rotation =
	{
		{ cos(yaw), 0.0f, -sin(yaw) },
		{ 0.0f, 1.0f, 0.0f },
		{ sin(yaw), 0.0f, cos(yaw) }
	};

	Matrix3D x_axis_rotation =
	{
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, cos(pitch), sin(pitch) },
		{ 0.0f, -sin(pitch), cos(pitch) }
	};

	return y_axis_rotation * x_axis_rotation;
}

__device__ Vec3D aces_tone_mapping(Vec3D color)
{
	Matrix3D input_matrix =
	{
		{ 0.59719f, 0.07600f, 0.02840f },
		{ 0.35458f, 0.90834f, 0.13383f },
		{ 0.04823f, 0.01566f, 0.83777f }
	};

	Matrix3D output_matrix =
	{
		{ 1.60475f, -0.10208f, -0.00327f },
		{ -0.53108f,  1.10813f, -0.07276f },
		{ -0.07367f, -0.00605f,  1.07602f }
	};

	color = input_matrix * color;
	color = { aces_curve(color.x), aces_curve(color.y), aces_curve(color.z) };
	color = output_matrix * color;

	return color;
}