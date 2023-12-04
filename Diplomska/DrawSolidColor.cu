#include <optix/optix.h>
#include "Parameters.h"
#include <curand.h>
#include <curand_kernel.h>
extern "C" {
	__constant__ Params params;
}
/*
static __forceinline__ __device__ float3 operator*(const float3& a, const float s) {
	return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float3 operator*(const float s, const float3& a) {
	return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float2 operator*(const float s, const float2& a) {
	return make_float2(a.x * s, a.y * s);
}

static __forceinline__ __device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float2 operator-(const float2& a, const float b) {
	return make_float2(a.x - b, a.y - b);
}

static __forceinline__ __device__ float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 normalize(const float3& v) {
	float invLen = 1.0f / sqrtf(dot(v, v));
	return v * invLen;
}
*/

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction, float2& offset) {
	origin = params.camPosition;

	const mat4 inverseProjection = params.inverseProjection;
	const mat4 inverseView = params.inverseView;
	const float2 d = 2.0f * make_float2(
		(static_cast<float>(idx.x) + offset.x) / static_cast<float>(dim.x),
		(static_cast<float>(idx.y) + offset.y) / static_cast<float>(dim.y)
	) - 1.0f;

	float4 homogeniousDeviceCoords = make_float4(d.x, -d.y, 1.0f, 1.0f);
	float4 viewSpace = inverseProjection * homogeniousDeviceCoords;
	float4 worldSpace = inverseView * make_float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

	direction = make_float3(normalize(worldSpace));
}

static __forceinline__ __device__ void setPayload(float3 p) {
	optixSetPayload_0(p.x * 255);
	optixSetPayload_1(p.y * 255);
	optixSetPayload_2(p.z * 255);
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	unsigned int pixelIndex = idx.y * params.width + idx.x;
	int samplesPerPixel = 3;

	uint4 colorAccumulator;
	colorAccumulator.x = 0;
	colorAccumulator.y = 0;
	colorAccumulator.z = 0;

	for (int i = 0; i < samplesPerPixel; i++) {
		float2 pixelOffset = make_float2(-0.5 + randomFloat(params.rngState, pixelIndex), -0.5 + randomFloat(params.rngState, pixelIndex));
		float3 rayOrigin, rayDirection;

		computeRay(idx, dim, rayOrigin, rayDirection, pixelOffset);

		unsigned int p0, p1, p2;
		optixTrace(params.handle, rayOrigin, rayDirection, 0.0f, 100, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2);
		colorAccumulator.x += p0;
		colorAccumulator.y += p1;
		colorAccumulator.z += p2;
	}

	uchar4 pixelColor;
	pixelColor.x = colorAccumulator.x / samplesPerPixel;
	pixelColor.y = colorAccumulator.y / samplesPerPixel;
	pixelColor.z = colorAccumulator.z / samplesPerPixel;
	pixelColor.w = 255u;

	params.image[pixelIndex] = pixelColor;
	//params.debug_buffer[pixelIndex] = colorAccumulator;
}
/*
extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	unsigned int pixelIndex = idx.y * params.width + idx.x;

	uchar4 pixelColor;
	pixelColor.x = randomFloat(params.rngState, pixelIndex) * 255;
	pixelColor.y = randomFloat(params.rngState, pixelIndex) * 255;
	pixelColor.z = randomFloat(params.rngState, pixelIndex) * 255;
	pixelColor.w = 255u;

	params.image[pixelIndex] = pixelColor;
}
*/
extern "C" __global__ void __miss__ms() {
	MissData* missData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

	float ratio = (float) optixGetLaunchIndex().y / (float) params.height;

	float3 newColor = ratio * (missData->bottomColor) + (1 - ratio) * (missData->topColor);

	setPayload(newColor);
}

extern "C" __global__ void __closesthit__ch() {
	HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
	//setPayload(rt_data->matColor);
}

extern "C" __global__ void __closesthit__red() {
	HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
	setPayload(make_float3(1.0f, 0.0f, 0.0f));
}

extern "C" __global__ void __closesthit__green() {
	HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
	setPayload(make_float3(1.0f, 1.0f, 0.0f));
}

extern "C" __global__ void __closesthit__blue() {
	HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
	setPayload(make_float3(1.0f, 1.0f, 1.0f));
}