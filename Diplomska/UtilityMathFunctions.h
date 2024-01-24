#pragma once
#ifndef UTILITYMATHFUNCTIONS_H
#define UTILITYMATHFUNCTIONS_H

#include <cuda_runtime.h>

using namespace std;

constexpr auto PI = 3.14159265358979323846;

__forceinline__ __device__ __host__ float uintAsFloat(unsigned int x) {
    union { float f; unsigned int i; } var;
    var.i = x;
    return var.f;
}

__forceinline__ __device__ __host__ float floatAsUint(float x) {
    union { float f; unsigned int i; } var;
    var.f = x;
    return var.i;
}

__forceinline__ __device__ __host__ double2 operator-(const double2& a, const double2& b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

__forceinline__ __device__ __host__ float toRadians(float degrees) {
    return degrees * (PI / 180);
}

__forceinline__ __device__ __host__ float3 make_float3(float s) {
	return make_float3(s, s, s);
}

__forceinline__ __device__ __host__ float3 make_float3(const float4& s) {
    return make_float3(s.x, s.y, s.z);
}

__forceinline__ __device__ __host__ float3 make_float3(const float2& s, float a) {
    return make_float3(s.x, s.y, a);
}

__forceinline__ __device__ __host__ float4 make_float4(const float3& a, float b) {
    return make_float4(a.x, a.y, a.z, b);
}

__forceinline__ __device__ __host__ float4 make_float4(const float2& a, float b, float c) {
    return make_float4(a.x, a.y, b, c);
}

__forceinline__ __device__ __host__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ __host__ float3 operator+(const float3& a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __device__ __host__ float3 operator-(const float3& s) {
	return make_float3(-s.x, -s.y, -s.z);
}

__forceinline__ __device__ __host__ float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __device__ __host__ float2 operator-(const float2& a, float b) {
    return make_float2(a.x - b, a.y - b);
}

__forceinline__ __device__ __host__ float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ __host__ float3 operator*(const float3& a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__ __host__ float3 operator*(const float3& a, float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ __host__ float4 operator*(const float4& a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__forceinline__ __device__ __host__ float4 operator*(float b, const float4& a) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__forceinline__ __device__ __host__ float3 operator*(float b, const float3& a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__ __host__ float2 operator*(float s, const float2& a) {
    return make_float2(a.x * s, a.y * s);
}

__forceinline__ __device__ __host__ float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__forceinline__ __device__ __host__ float2 operator/(const float2& a, float b) {
    return make_float2(a.x / b, a.y / b);
}

__forceinline__ __device__ __host__ void operator*=(float3& a, const float s) {
	a.x *= s;
	a.y *= s;
	a.z *= s;
}

__forceinline__ __device__ __host__ void operator*=(float3& a, float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__forceinline__ __device__ __host__ void operator*=(float4& a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    a.w *= s;
}

__forceinline__ __device__ __host__ void operator+=(float3& a, const float3& s) {
	a.x += s.x;
	a.y += s.y;
	a.z += s.z;
}

__forceinline__ __device__ __host__ void operator-=(float3& a, const float3& s) {
	a.x -= s.x;
	a.y -= s.y;
	a.z -= s.z;
}

__forceinline__ __device__ __host__ float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ __host__ float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__forceinline__ __device__ __host__ float3 cross(const float3& a, const float3& b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__forceinline__ __device__ __host__ float3 normalize(const float3& s) {
	float invLen = 1.0f / sqrtf(dot(s, s));
	return s * invLen;
}

__forceinline__ __device__ __host__ float4 normalize(const float4& s) {
    float invLen = 1.0f / sqrtf(dot(s, s));
    return s * invLen;
}

__forceinline__ __device__ __host__ float length(const float3& s) {
	return sqrtf(dot(s, s));
}

__forceinline__ __device__ __host__ float lengthSquared(const float3& s) {
    return dot(s, s);
}

__forceinline__ __device__ __host__ float maximum(float a, float b) {
    return a > b ? a : b;
}

__forceinline__ __device__ __host__ float minimum(float a, float b) {
    return a < b ? a : b;
}

__forceinline__ __device__ __host__ unsigned char maximum(unsigned char a, unsigned char b) {
    return a > b ? a : b;
}

__forceinline__ __device__ __host__ unsigned char minimum(unsigned char a, unsigned char b) {
    return a < b ? a : b;
}

__forceinline__ __device__ __host__ float clamp(float s, float minVal, float maxVal) {
    return minimum(maximum(s, minVal), maxVal);
}

__forceinline__ __device__ __host__ float3 clamp(const float3& s, float minVal, float maxVal) {
    return make_float3(
        minimum(maximum(s.x, minVal), maxVal),
        minimum(maximum(s.y, minVal), maxVal),
        minimum(maximum(s.z, minVal), maxVal)
    );
}

__forceinline__ __device__ __host__ uchar4 clamp(const uchar4& s, unsigned char minVal, unsigned char maxVal) {
    return make_uchar4(
        minimum(maximum(s.x, minVal), maxVal),
        minimum(maximum(s.y, minVal), maxVal),
        minimum(maximum(s.z, minVal), maxVal),
        minimum(maximum(s.w, minVal), maxVal)
    );
}

__forceinline__ __device__ __host__ float smoothstep(float leftEgde, float rightEdge, float input) {
    input = clamp((input - leftEgde) / (rightEdge - leftEgde), 0.0f, 1.0f);
    return input * input * (3.0f - 2.0f * input);
}

__forceinline__ __device__ __host__ float3 lerp(const float3& vec1, const float3& vec2, float t) {
    return vec1 + t * (vec2 - vec1);
}

struct mat4 {
    float m11, m12, m13, m14;
    float m21, m22, m23, m24;
    float m31, m32, m33, m34;
    float m41, m42, m43, m44;

    __host__ __device__ __forceinline__ float4 operator*(const float4& v) const {
        float4 ret;
        ret.x = m11 * v.x + m12 * v.y + m13 * v.z + m14 * v.w;
        ret.y = m21 * v.x + m22 * v.y + m23 * v.z + m24 * v.w;
        ret.z = m31 * v.x + m32 * v.y + m33 * v.z + m34 * v.w;
        ret.w = m41 * v.x + m42 * v.y + m43 * v.z + m44 * v.w;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator*(const float f) const {
        mat4 ret;
        ret.m11 = m11 * f; ret.m12 = m12 * f; ret.m13 = m13 * f; ret.m14 = m14 * f;
        ret.m21 = m21 * f; ret.m22 = m22 * f; ret.m23 = m23 * f; ret.m24 = m24 * f;
        ret.m31 = m31 * f; ret.m32 = m32 * f; ret.m33 = m33 * f; ret.m34 = m34 * f;
        ret.m41 = m41 * f; ret.m42 = m42 * f; ret.m43 = m43 * f; ret.m44 = m44 * f;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator/(const float f) const {
        mat4 ret;
        ret.m11 = m11 / f; ret.m12 = m12 / f; ret.m13 = m13 / f; ret.m14 = m14 / f;
        ret.m21 = m21 / f; ret.m22 = m22 / f; ret.m23 = m23 / f; ret.m24 = m24 / f;
        ret.m31 = m31 / f; ret.m32 = m32 / f; ret.m33 = m33 / f; ret.m34 = m34 / f;
        ret.m41 = m41 / f; ret.m42 = m42 / f; ret.m43 = m43 / f; ret.m44 = m44 / f;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator+(const mat4& other) const {
        mat4 ret;
        ret.m11 = m11 + other.m11; ret.m12 = m12 + other.m12; ret.m13 = m13 + other.m13; ret.m14 = m14 + other.m14;
        ret.m21 = m21 + other.m21; ret.m22 = m22 + other.m22; ret.m23 = m23 + other.m23; ret.m24 = m24 + other.m24;
        ret.m31 = m31 + other.m31; ret.m32 = m32 + other.m32; ret.m33 = m33 + other.m33; ret.m34 = m34 + other.m34;
        ret.m41 = m41 + other.m41; ret.m42 = m42 + other.m42; ret.m43 = m43 + other.m43; ret.m44 = m44 + other.m44;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator-(const mat4& other) const {
        mat4 ret;
        ret.m11 = m11 - other.m11; ret.m12 = m12 - other.m12; ret.m13 = m13 - other.m13; ret.m14 = m14 - other.m14;
        ret.m21 = m21 - other.m21; ret.m22 = m22 - other.m22; ret.m23 = m23 - other.m23; ret.m24 = m24 - other.m24;
        ret.m31 = m31 - other.m31; ret.m32 = m32 - other.m32; ret.m33 = m33 - other.m33; ret.m34 = m34 - other.m34;
        ret.m41 = m41 - other.m41; ret.m42 = m42 - other.m42; ret.m43 = m43 - other.m43; ret.m44 = m44 - other.m44;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator*(const mat4& other) const {
        mat4 ret;
        ret.m11 = m11 * other.m11 + m12 * other.m21 + m13 * other.m31 + m14 * other.m41;
        ret.m12 = m11 * other.m12 + m12 * other.m22 + m13 * other.m32 + m14 * other.m42;
        ret.m13 = m11 * other.m13 + m12 * other.m23 + m13 * other.m33 + m14 * other.m43;
        ret.m14 = m11 * other.m14 + m12 * other.m24 + m13 * other.m34 + m14 * other.m44;

        ret.m21 = m21 * other.m11 + m22 * other.m21 + m23 * other.m31 + m24 * other.m41;
        ret.m22 = m21 * other.m12 + m22 * other.m22 + m23 * other.m32 + m24 * other.m42;
        ret.m23 = m21 * other.m13 + m22 * other.m23 + m23 * other.m33 + m24 * other.m43;
        ret.m24 = m21 * other.m14 + m22 * other.m24 + m23 * other.m34 + m24 * other.m44;

        ret.m31 = m31 * other.m11 + m32 * other.m21 + m33 * other.m31 + m34 * other.m41;
        ret.m32 = m31 * other.m12 + m32 * other.m22 + m33 * other.m32 + m34 * other.m42;
        ret.m33 = m31 * other.m13 + m32 * other.m23 + m33 * other.m33 + m34 * other.m43;
        ret.m34 = m31 * other.m14 + m32 * other.m24 + m33 * other.m34 + m34 * other.m44;

        ret.m41 = m41 * other.m11 + m42 * other.m21 + m43 * other.m31 + m44 * other.m41;
        ret.m42 = m41 * other.m12 + m42 * other.m22 + m43 * other.m32 + m44 * other.m42;
        ret.m43 = m41 * other.m13 + m42 * other.m23 + m43 * other.m33 + m44 * other.m43;
        ret.m44 = m41 * other.m14 + m42 * other.m24 + m43 * other.m34 + m44 * other.m44;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 transpose() const {
        mat4 ret;
        ret.m11 = m11; ret.m12 = m21; ret.m13 = m31; ret.m14 = m41;
        ret.m21 = m12; ret.m22 = m22; ret.m23 = m32; ret.m24 = m42;
        ret.m31 = m13; ret.m32 = m32; ret.m33 = m33; ret.m34 = m43;
        ret.m41 = m14; ret.m42 = m24; ret.m43 = m43; ret.m44 = m44;
        return ret;
    }

    __host__ __device__ __forceinline__ static mat4 createRotationX(float radians) {
        mat4 ret;
        ret.m11 = 1; ret.m12 = 0; ret.m13 = 0; ret.m14 = 0;
        ret.m21 = 0; ret.m22 = cos(radians); ret.m23 = -sin(radians); ret.m24 = 0;
        ret.m31 = 0; ret.m32 = sin(radians); ret.m33 = cos(radians); ret.m34 = 0;
        ret.m41 = 0; ret.m42 = 0; ret.m43 = 0; ret.m44 = 1;
        return ret;
    }

    __host__ __device__ __forceinline__ static mat4 createRotationY(float radians) {
        mat4 ret;
        ret.m11 = cos(radians); ret.m12 = 0; ret.m13 = sin(radians); ret.m14 = 0;
        ret.m21 = 0; ret.m22 = 1; ret.m23 = 0; ret.m24 = 0;
        ret.m31 = -sin(radians); ret.m32 = 0; ret.m33 = cos(radians); ret.m34 = 0;
        ret.m41 = 0; ret.m42 = 0; ret.m43 = 0; ret.m44 = 1;
        return ret;
    }

    __host__ __device__ __forceinline__ static mat4 createRotationZ(float radians) {
        mat4 ret;
        ret.m11 = cos(radians); ret.m12 = -sin(radians); ret.m13 = 0; ret.m14 = 0;
        ret.m21 = sin(radians); ret.m22 = cos(radians); ret.m23 = 0; ret.m24 = 0;
        ret.m31 = 0; ret.m32 = 0; ret.m33 = 1; ret.m34 = 0;
        ret.m41 = 0; ret.m42 = 0; ret.m43 = 0; ret.m44 = 1;
        return ret;
    }

    __host__ __device__ __forceinline__ static mat4 createTranslation(float3 translation) {
        mat4 ret;
        ret.m11 = 1; ret.m12 = 0; ret.m13 = 0; ret.m14 = translation.x;
        ret.m21 = 0; ret.m22 = 1; ret.m23 = 0; ret.m24 = translation.y;
        ret.m31 = 0; ret.m32 = 0; ret.m33 = 1; ret.m34 = translation.z;
        ret.m41 = 0; ret.m42 = 0; ret.m43 = 0; ret.m44 = 1;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 inverse() const {
        float t11 = m23 * m34 * m42 - m24 * m33 * m42 + m24 * m32 * m43 - m22 * m34 * m43 - m23 * m32 * m44 + m22 * m33 * m44;
        float t12 = m14 * m33 * m42 - m13 * m34 * m42 - m14 * m32 * m43 + m12 * m34 * m43 + m13 * m32 * m44 - m12 * m33 * m44;
        float t13 = m13 * m24 * m42 - m14 * m23 * m42 + m14 * m22 * m43 - m12 * m24 * m43 - m13 * m22 * m44 + m12 * m23 * m44;
        float t14 = m14 * m23 * m32 - m13 * m24 * m32 - m14 * m22 * m33 + m12 * m24 * m33 + m13 * m22 * m34 - m12 * m23 * m34;

        float det = m11 * t11 + m21 * t12 + m31 * t13 + m41 * t14;
        float idet = 1.0f / det;

        mat4 ret;

        ret.m11 = t11 * idet;
        ret.m21 = (m24 * m33 * m41 - m23 * m34 * m41 - m24 * m31 * m43 + m21 * m34 * m43 + m23 * m31 * m44 - m21 * m33 * m44) * idet;
        ret.m31 = (m22 * m34 * m41 - m24 * m32 * m41 + m24 * m31 * m42 - m21 * m34 * m42 - m22 * m31 * m44 + m21 * m32 * m44) * idet;
        ret.m41 = (m23 * m32 * m41 - m22 * m33 * m41 - m23 * m31 * m42 + m21 * m33 * m42 + m22 * m31 * m43 - m21 * m32 * m43) * idet;

        ret.m12 = t12 * idet;
        ret.m22 = (m13 * m34 * m41 - m14 * m33 * m41 + m14 * m31 * m43 - m11 * m34 * m43 - m13 * m31 * m44 + m11 * m33 * m44) * idet;
        ret.m32 = (m14 * m32 * m41 - m12 * m34 * m41 - m14 * m31 * m42 + m11 * m34 * m42 + m12 * m31 * m44 - m11 * m32 * m44) * idet;
        ret.m42 = (m12 * m33 * m41 - m13 * m32 * m41 + m13 * m31 * m42 - m11 * m33 * m42 - m12 * m31 * m43 + m11 * m32 * m43) * idet;

        ret.m13 = t13 * idet;
        ret.m23 = (m14 * m23 * m41 - m13 * m24 * m41 - m14 * m21 * m43 + m11 * m24 * m43 + m13 * m21 * m44 - m11 * m23 * m44) * idet;
        ret.m33 = (m12 * m24 * m41 - m14 * m22 * m41 + m14 * m21 * m42 - m11 * m24 * m42 - m12 * m21 * m44 + m11 * m22 * m44) * idet;
        ret.m43 = (m13 * m22 * m41 - m12 * m23 * m41 - m13 * m21 * m42 + m11 * m23 * m42 + m12 * m21 * m43 - m11 * m22 * m43) * idet;

        ret.m14 = t14 * idet;
        ret.m24 = (m13 * m24 * m31 - m14 * m23 * m31 + m14 * m21 * m33 - m11 * m24 * m33 - m13 * m21 * m34 + m11 * m23 * m34) * idet;
        ret.m34 = (m14 * m22 * m31 - m12 * m24 * m31 - m14 * m21 * m32 + m11 * m24 * m32 + m12 * m21 * m34 - m11 * m22 * m34) * idet;
        ret.m44 = (m12 * m23 * m31 - m13 * m22 * m31 + m13 * m21 * m32 - m11 * m23 * m32 - m12 * m21 * m33 + m11 * m22 * m33) * idet;

        return ret;
    }

    __host__ __device__ __forceinline__ static mat4 createScale(float3 scale) {
        mat4 ret;
        ret.m11 = scale.x; ret.m12 = 0; ret.m13 = 0; ret.m14 = 0;
        ret.m21 = 0; ret.m22 = scale.y; ret.m23 = 0; ret.m24 = 0;
        ret.m31 = 0; ret.m32 = 0; ret.m33 = scale.z; ret.m34 = 0;
        ret.m41 = 0; ret.m42 = 0; ret.m43 = 0; ret.m44 = 1;
        return ret;
    }
    
    __host__ __device__ __forceinline__ void zero() {
        m11 = 0; m12 = 0; m13 = 0; m14 = 0;
        m21 = 0; m22 = 0; m23 = 0; m24 = 0;
        m31 = 0; m32 = 0; m33 = 0; m34 = 0;
        m41 = 0; m42 = 0; m43 = 0; m44 = 0;
    }
    
    __host__ __device__ __forceinline__ void identity() {
        m11 = 1; m12 = 0; m13 = 0; m14 = 0;
        m21 = 0; m22 = 1; m23 = 0; m24 = 0;
        m31 = 0; m32 = 0; m33 = 1; m34 = 0;
        m41 = 0; m42 = 0; m43 = 0; m44 = 1;
    }
    
    __host__ __device__ __forceinline__ mat4& operator*=(const float f) { return *this = *this * f; }
    __host__ __device__ __forceinline__ mat4& operator/=(const float f) { return *this = *this / f; }
    __host__ __device__ __forceinline__ mat4& operator+=(const mat4& m) { return *this = *this + m; }
    __host__ __device__ __forceinline__ mat4& operator-=(const mat4& m) { return *this = *this - m; }
    __host__ __device__ __forceinline__ mat4& operator*=(const mat4& m) { return *this = *this * m; }
};

__host__ __device__ __forceinline__ float randomFloat(unsigned int* state) {
    *state = *state * 747796405 + 2891336453;
    unsigned int result = ((*state >> ((*state >> 28) + 4)) ^ *state) * 277803737;
    result = (result >> 22) ^ result;
    return result / 4294967295.0f;
}

__host__ __device__ __forceinline__ float randomFloatNormalDistribution(unsigned int* state) {
    float theta = 2 * PI * randomFloat(state);
    float rho = sqrt(-2 * log(randomFloat(state)));
    return rho * cos(theta);
}

__host__ __device__ __forceinline__ float3 randomDirection(unsigned int* state) {
    float x = randomFloatNormalDistribution(state);
    float y = randomFloatNormalDistribution(state);
    float z = randomFloatNormalDistribution(state);
    return normalize(make_float3(x, y, z));
}

#endif // !UTILITYMATHFUNCTIONS_H