#include <optix/optix.h>
#include "Parameters.h"
#include <curand.h>
#include <curand_kernel.h>
extern "C" {
	__constant__ Params params;
}

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
	unsigned int rngState = idx.y * params.width + idx.x + params.frameIndex;
	int samplesPerPixel = 3;

	uint4 colorAccumulator;
	colorAccumulator.x = 0;
	colorAccumulator.y = 0;
	colorAccumulator.z = 0;

	for (int i = 0; i < samplesPerPixel; i++) {
		float2 pixelOffset = make_float2(-0.5 + randomFloat(&rngState), -0.5 + randomFloat(&rngState));
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
	HitgroupData* rt_data = (HitgroupData*)optixGetSbtDataPointer();
	setPayload(rt_data->color);
}