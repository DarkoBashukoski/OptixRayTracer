#include <optix/optix.h>
#include "Parameters.h"
#include <curand.h>
#include <curand_kernel.h>

struct RayPayload {
	float3 color;
	float3 emittedLight;
	float3 origin;
	float3 direction;
	float done;
	unsigned int seed;
};

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

static __forceinline__ __device__ void storePayload(RayPayload payload) {
	optixSetPayload_0(floatAsUint(payload.color.x));
	optixSetPayload_1(floatAsUint(payload.color.y));
	optixSetPayload_2(floatAsUint(payload.color.z));

	optixSetPayload_3(floatAsUint(payload.emittedLight.x));
	optixSetPayload_4(floatAsUint(payload.emittedLight.y));
	optixSetPayload_5(floatAsUint(payload.emittedLight.z));

	optixSetPayload_6(floatAsUint(payload.origin.x));
	optixSetPayload_7(floatAsUint(payload.origin.y));
	optixSetPayload_8(floatAsUint(payload.origin.z));

	optixSetPayload_9(floatAsUint(payload.direction.x));
	optixSetPayload_10(floatAsUint(payload.direction.y));
	optixSetPayload_11(floatAsUint(payload.direction.z));

	optixSetPayload_12(floatAsUint(payload.done));
	optixSetPayload_13(payload.seed);
}

static __forceinline__ __device__ RayPayload loadPayloadCH() {
	RayPayload payload = {};
	payload.seed = optixGetPayload_13();
	return payload;
}

static __forceinline__ __device__ RayPayload loadPayloadMS() {
	RayPayload payload = {};
	return payload;
}

static __forceinline__ __device__ void trace(float3 origin, float3 direction, RayPayload& payload) {
	unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13;
	p13 = payload.seed;
	optixTrace(params.handle, origin, direction, 0.0001f, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0, 1, 0, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);

	payload.color = make_float3(uintAsFloat(p0), uintAsFloat(p1), uintAsFloat(p2));
	payload.emittedLight = make_float3(uintAsFloat(p3), uintAsFloat(p4), uintAsFloat(p5));
	payload.origin = make_float3(uintAsFloat(p6), uintAsFloat(p7), uintAsFloat(p8));
	payload.direction = make_float3(uintAsFloat(p9), uintAsFloat(p10), uintAsFloat(p11));
	payload.done = uintAsFloat(p12);
	payload.seed = p13;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	unsigned int pixelIndex = idx.y * params.width + idx.x;
	unsigned int rngSeed = (idx.y * params.width + idx.x) + params.frameIndex * 719393;

	int maxBounces = 5;
	int samplesPerPixel = 1; 

	int i = samplesPerPixel;

	do {
		float2 pixelOffset = make_float2(-0.5 + randomFloat(&rngSeed), -0.5 + randomFloat(&rngSeed));
		float3 rayOrigin, rayDirection;

		computeRay(idx, dim, rayOrigin, rayDirection, pixelOffset);

		float3 rayColor = make_float3(1.0f);
		float3 rayLight = make_float3(0.0f);

		for (int i = 0; i < maxBounces; i++) {
			RayPayload payload = {};
			payload.seed = rngSeed;

			trace(rayOrigin, rayDirection, payload);

			if (payload.done == 1.0f) {
				rayLight += payload.emittedLight * rayColor;
				break;
			}

			rayLight += payload.emittedLight * rayColor;
			rayColor *= payload.color;

			rayDirection = payload.direction;
			rayOrigin = payload.origin;
		}

		if (params.frameIndex == 1) {
			params.colorAccumulator[pixelIndex] = rayLight;
		}
		else {
			params.colorAccumulator[pixelIndex] += rayLight;
		}

	} while (--i);

	float3 pixelFloat = clamp(params.colorAccumulator[pixelIndex] / params.frameIndex, 0.0f, 1.0f);

	uchar4 pixelColor;
	pixelColor.x = pixelFloat.x * 255u;
	pixelColor.y = pixelFloat.y * 255u;
	pixelColor.z = pixelFloat.z * 255u;
	pixelColor.w = 255u;
	params.image[pixelIndex] = pixelColor;
}

extern "C" __global__ void __miss__ms() {
	RayPayload payload = loadPayloadMS();
	MissData* missData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

	const float3 rayDirection = optixGetWorldRayDirection();

	float skyGradientT = pow(smoothstep(0.0f, 0.4f, rayDirection.y), 0.35f);
	float3 skyGradient = lerp(missData->skyColorHorizon, missData->skyColorZenith, skyGradientT);

	float groundToSkyT = smoothstep(-0.01f, 0.0f, rayDirection.y);

	float sun = pow(maximum(0, dot(rayDirection, -missData->sunDirection)), missData->sunFocus) * missData->sunIntensity;
	float sunMask = groundToSkyT >= 1;

	float3 light = lerp(missData->groundColor, skyGradient, groundToSkyT) + sun * sunMask;

	payload.emittedLight = light;
	payload.done = 1.0f;

	storePayload(payload);
}

extern "C" __global__ void __closesthit__ch() {
	RayPayload payload = loadPayloadCH();
	HitgroupData* hitData = (HitgroupData*)optixGetSbtDataPointer();

	const float3 rayDirection = optixGetWorldRayDirection();
	const float3 rayOrigin = optixGetWorldRayOrigin();

	float3 hitVertices[3];
	optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), optixGetRayTime(), hitVertices);
	float3 normal = normalize(cross(hitVertices[1] - hitVertices[0], hitVertices[2] - hitVertices[0]));

	payload.origin = rayOrigin + optixGetRayTmax() * rayDirection;
	/*
	float3 diffuseDirection = randomDirection(&payload.seed);
	if (dot(normal, diffuseDirection) < 0) {
		diffuseDirection = -diffuseDirection;
	}
	*/
	float3 diffuseDirection = normalize(normal + randomDirection(&payload.seed));

	float3 specularDirection = rayDirection - 2 * dot(rayDirection, normal) * normal;

	payload.direction = lerp(specularDirection, diffuseDirection, hitData->roughness);

	payload.color = hitData->color;
	payload.emittedLight = hitData->emissionColor * hitData->emissionPower;
	payload.done = 0.0f;

	storePayload(payload);
}