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
	optixSetPayload_0(floatAsUint(p.x));
	optixSetPayload_1(floatAsUint(p.y));
	optixSetPayload_2(floatAsUint(p.z));
}

static __forceinline__ __device__ void setPayload(float3 p, float3 n) {
	optixSetPayload_0(floatAsUint(p.x));
	optixSetPayload_1(floatAsUint(p.y));
	optixSetPayload_2(floatAsUint(p.z));

	optixSetPayload_3(floatAsUint(n.x));
	optixSetPayload_4(floatAsUint(n.y));
	optixSetPayload_5(floatAsUint(n.z));
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	unsigned int pixelIndex = idx.y * params.width + idx.x;

	float3 pixelColor;
	float3 normal;

	float2 pixelOffset = make_float2(0.0f, 0.0f);
	float3 rayOrigin, rayDirection;

	computeRay(idx, dim, rayOrigin, rayDirection, pixelOffset);

	unsigned int p0, p1, p2, p3, p4, p5;
	optixTrace(params.handle, rayOrigin, rayDirection, 0.0f, 100, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0, 1, 0, p0, p1, p2, p3, p4, p5);

	pixelColor.x = uintAsFloat(p0);
	pixelColor.y = uintAsFloat(p1);
	pixelColor.z = uintAsFloat(p2);

	normal.x = uintAsFloat(p3);
	normal.y = uintAsFloat(p4);
	normal.z = uintAsFloat(p5);

	params.image[pixelIndex] = pixelColor;
	params.normals[pixelIndex] = normal;
}

extern "C" __global__ void __miss__ms() {
	MissData* missData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

	const float3 rayDirection = optixGetWorldRayDirection();

	float skyGradientT = pow(smoothstep(0.0f, 0.4f, rayDirection.y), 0.35f);
	float3 skyGradient = lerp(missData->skyColorHorizon, missData->skyColorZenith, skyGradientT);

	float groundToSkyT = smoothstep(-0.01f, 0.0f, rayDirection.y);

	float sun = pow(maximum(0, dot(rayDirection, -missData->sunDirection)), missData->sunFocus) * missData->sunIntensity;
	float sunMask = groundToSkyT >= 1;

	float3 light = lerp(missData->groundColor, skyGradient, groundToSkyT) + sun * sunMask;

	float3 normal = make_float3(0.0f, 0.0f, 0.0f);
	setPayload(light, normal);
}

extern "C" __global__ void __closesthit__ch() {
	HitgroupData* rt_data = (HitgroupData*)optixGetSbtDataPointer();

	float3 hitVertices[3];
	optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), optixGetRayTime(), hitVertices);
	float3 normal = normalize(cross(hitVertices[1] - hitVertices[0], hitVertices[2] - hitVertices[0]));
	setPayload(rt_data->color, normal);
}