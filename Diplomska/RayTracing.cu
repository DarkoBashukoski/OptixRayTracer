#include <optix/optix.h>
#include "Parameters.h"
#include <curand.h>
#include <curand_kernel.h>

struct RayPayload {
	float3 color;
	float3 emittedLight;
	float3 origin;
	float3 direction;
	float3 normal;
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

	optixSetPayload_12(floatAsUint(payload.normal.x));
	optixSetPayload_13(floatAsUint(payload.normal.y));
	optixSetPayload_14(floatAsUint(payload.normal.z));

	optixSetPayload_15(floatAsUint(payload.done));
	optixSetPayload_16(payload.seed);
}

static __forceinline__ __device__ RayPayload loadPayload() {
	RayPayload payload = {};

	payload.color = make_float3(uintAsFloat(optixGetPayload_0()), uintAsFloat(optixGetPayload_1()), uintAsFloat(optixGetPayload_2()));
	payload.emittedLight = make_float3(uintAsFloat(optixGetPayload_3()), uintAsFloat(optixGetPayload_4()), uintAsFloat(optixGetPayload_5()));
	payload.origin = make_float3(uintAsFloat(optixGetPayload_6()), uintAsFloat(optixGetPayload_7()), uintAsFloat(optixGetPayload_8()));
	payload.direction = make_float3(uintAsFloat(optixGetPayload_9()), uintAsFloat(optixGetPayload_10()), uintAsFloat(optixGetPayload_11()));
	payload.normal = make_float3(uintAsFloat(optixGetPayload_12()), uintAsFloat(optixGetPayload_13()), uintAsFloat(optixGetPayload_14()));
	payload.done = uintAsFloat(optixGetPayload_15());
	payload.seed = optixGetPayload_16();

	return payload;
}

static __forceinline__ __device__ void trace(RayPayload& payload) {
	unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16;
	p0 = floatAsUint(payload.color.x);
	p1 = floatAsUint(payload.color.y);
	p2 = floatAsUint(payload.color.z);
	p3 = floatAsUint(payload.emittedLight.x);
	p4 = floatAsUint(payload.emittedLight.y);
	p5 = floatAsUint(payload.emittedLight.z);
	p6 = floatAsUint(payload.origin.x);
	p7 = floatAsUint(payload.origin.y);
	p8 = floatAsUint(payload.origin.z);
	p9 = floatAsUint(payload.direction.x);
	p10 = floatAsUint(payload.direction.y);
	p11 = floatAsUint(payload.direction.z);
	p12 = floatAsUint(payload.normal.x);
	p13 = floatAsUint(payload.normal.y);
	p14 = floatAsUint(payload.normal.z);
	p15 = floatAsUint(payload.done);
	p16 = payload.seed;

	optixTrace(params.handle, payload.origin, payload.direction, 0.0001f, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0, 1, 0, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);

	payload.color = make_float3(uintAsFloat(p0), uintAsFloat(p1), uintAsFloat(p2));
	payload.emittedLight = make_float3(uintAsFloat(p3), uintAsFloat(p4), uintAsFloat(p5));
	payload.origin = make_float3(uintAsFloat(p6), uintAsFloat(p7), uintAsFloat(p8));
	payload.direction = make_float3(uintAsFloat(p9), uintAsFloat(p10), uintAsFloat(p11));
	payload.normal = make_float3(uintAsFloat(p12), uintAsFloat(p13), uintAsFloat(p14));
	payload.done = uintAsFloat(p15);
	payload.seed = p16;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	unsigned int pixelIndex = idx.y * params.width + idx.x;
	float2 pixelOffset = make_float2(0.0f, 0.0f);
	RayPayload payload = {};
	payload.seed = pixelIndex;
	float3 totalLight = make_float3(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < params.samplesPerPixel; i++) {
		payload.color = make_float3(1.0f);
		payload.emittedLight = make_float3(0.0f);
		payload.done = 0.0f;
		computeRay(idx, dim, payload.origin, payload.direction, pixelOffset);
		pixelOffset = make_float2(-0.5f + randomFloat(&payload.seed), -0.5f + randomFloat(&payload.seed));

		for (int j = 0; j < params.maxDepth; j++) {
			trace(payload);

			if (payload.done == 1.0f) {
				break;
			}
		}

		totalLight += payload.emittedLight;
	}

	params.image[pixelIndex] = totalLight / params.samplesPerPixel;
}

extern "C" __global__ void __closesthit__ch() {
	RayPayload payload = loadPayload();
	HitgroupData* hitData = (HitgroupData*)optixGetSbtDataPointer();
	/*
	float3 hitVertices[3];
	optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), optixGetRayTime(), hitVertices);
	payload.normal = normalize(cross(hitVertices[1] - hitVertices[0], hitVertices[2] - hitVertices[0]));
	*/
	unsigned int primitiveIndex = optixGetPrimitiveIndex();
	uint3 normalIndexTriplet = hitData->vertexNormalIndices[primitiveIndex];

	float3 normal0 = hitData->vertexNormals[normalIndexTriplet.x];
	float3 normal1 = hitData->vertexNormals[normalIndexTriplet.y];
	float3 normal2 = hitData->vertexNormals[normalIndexTriplet.z];

	float2 barycentrics = optixGetTriangleBarycentrics();
	float alpha = 1.0f - barycentrics.x - barycentrics.y;

	payload.normal = normal0 * alpha + normal1 * barycentrics.x + normal2 * barycentrics.y;

	payload.origin = payload.origin + optixGetRayTmax() * payload.direction;

	float3 diffuseDirection = normalize(payload.normal + randomDirection(&payload.seed));
	float3 specularDirection = payload.direction - 2 * dot(payload.direction, payload.normal) * payload.normal;
	bool isSpecualarBounce = hitData->metallic >= randomFloat(&payload.seed);
	payload.direction = normalize(lerp(specularDirection, diffuseDirection, hitData->roughness * (1 - isSpecualarBounce)));

	payload.emittedLight += (hitData->emissionColor * hitData->emissionPower) * payload.color;
	float3 color = lerp(hitData->color, make_float3(1.0f, 1.0f, 1.0f), isSpecualarBounce);
	payload.color *= color;

	payload.done = hitData->emissionPower > 0.0f ? 1.0f : 0.0f;

	storePayload(payload);
}

extern "C" __global__ void __miss__ms() {
	RayPayload payload = loadPayload();
	MissData* missData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
	
	float skyGradientT = pow(smoothstep(0.0f, 0.4f, payload.direction.y), 0.35f);
	float3 skyGradient = lerp(missData->skyColorHorizon, missData->skyColorZenith, skyGradientT);

	float groundToSkyT = smoothstep(-0.01f, 0.0f, payload.direction.y);

	float sun = pow(maximum(0, dot(payload.direction, -missData->sunDirection)), missData->sunFocus) * missData->sunIntensity;
	float sunMask = groundToSkyT >= 1;

	float3 light = lerp(missData->groundColor, skyGradient, groundToSkyT) + sun * sunMask;

	payload.emittedLight += make_float3(0.0f, 0.0f, 0.0f);
	payload.normal = make_float3(0.0f, 0.0f, 0.0f);
	payload.done = 1.0f;

	storePayload(payload);
}