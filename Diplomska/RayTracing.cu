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
	float prepForDenoiser;
	unsigned int seed;
};

extern "C" {
	__constant__ Params params;
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction, float2& offset) {
	origin = params.camPosition;

	const float2 d = 2.0f * make_float2(
		(static_cast<float>(idx.x) + offset.x) / static_cast<float>(dim.x),
		(static_cast<float>(idx.y) + offset.y) / static_cast<float>(dim.y)
	) - 1.0f;

	float4 homogeniousDeviceCoords = make_float4(d.x, -d.y, 1.0f, 1.0f);
	float4 viewSpace = params.inverseProjection * homogeniousDeviceCoords;
	float4 worldSpace = params.inverseView * make_float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

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
	optixSetPayload_14(payload.prepForDenoiser);
}

static __forceinline__ __device__ RayPayload loadPayload() {
	RayPayload payload = {};

	payload.color = make_float3(uintAsFloat(optixGetPayload_0()), uintAsFloat(optixGetPayload_1()), uintAsFloat(optixGetPayload_2()));
	payload.emittedLight = make_float3(uintAsFloat(optixGetPayload_3()), uintAsFloat(optixGetPayload_4()), uintAsFloat(optixGetPayload_5()));
	payload.origin = make_float3(uintAsFloat(optixGetPayload_6()), uintAsFloat(optixGetPayload_7()), uintAsFloat(optixGetPayload_8()));
	payload.direction = make_float3(uintAsFloat(optixGetPayload_9()), uintAsFloat(optixGetPayload_10()), uintAsFloat(optixGetPayload_11()));
	payload.done = uintAsFloat(optixGetPayload_12());
	payload.seed = optixGetPayload_13();
	payload.prepForDenoiser = uintAsFloat(optixGetPayload_14());

	return payload;
}

static __forceinline__ __device__ void trace(RayPayload& payload) {
	unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14;
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
	p12 = floatAsUint(payload.done);
	p13 = payload.seed;
	p14 = floatAsUint(payload.prepForDenoiser);

	optixTrace(params.handle, payload.origin, payload.direction, 0.0001f, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0, 1, 0, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);

	payload.color = make_float3(uintAsFloat(p0), uintAsFloat(p1), uintAsFloat(p2));
	payload.emittedLight = make_float3(uintAsFloat(p3), uintAsFloat(p4), uintAsFloat(p5));
	payload.origin = make_float3(uintAsFloat(p6), uintAsFloat(p7), uintAsFloat(p8));
	payload.direction = make_float3(uintAsFloat(p9), uintAsFloat(p10), uintAsFloat(p11));
	payload.done = uintAsFloat(p12);
	payload.seed = p13;
	payload.prepForDenoiser = p14;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	unsigned int pixelIndex = idx.y * params.width + idx.x;
	float2 pixelOffset = make_float2(0.0f, 0.0f);
	RayPayload payload = {};
	payload.seed = pixelIndex + params.frameIndex * 719393;
	payload.prepForDenoiser = 1.0f;

	float3 totalLight = make_float3(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < params.samplesPerPixel; i++) {
		payload.color = make_float3(1.0f);
		payload.emittedLight = make_float3(0.0f);
		payload.done = 0.0f;
		computeRay(idx, dim, payload.origin, payload.direction, pixelOffset);
		pixelOffset = make_float2(-0.5f + randomFloat(&payload.seed), -0.5f + randomFloat(&payload.seed));

		for (int j = 0; j < params.maxDepth; j++) {
			trace(payload);
			payload.prepForDenoiser = 0.0f;

			if (payload.done == 1.0f) {
				break;
			}
		}

		totalLight += payload.emittedLight;
	}

	float3 averageLight = totalLight / params.samplesPerPixel;
	float weight = 1.0f / (params.frameIndex + 1);

	params.image[pixelIndex] = params.image[pixelIndex] * (1.0f - weight) + averageLight * weight;
}

extern "C" __global__ void __closesthit__ch() {
	RayPayload payload = loadPayload();
	HitgroupData* hitData = (HitgroupData*)optixGetSbtDataPointer();
	
	unsigned int primitiveIndex = optixGetPrimitiveIndex();
	uint3 normalIndexTriplet = hitData->vertexNormalIndices[primitiveIndex];

	float3 normal0 = hitData->vertexNormals[normalIndexTriplet.x];
	float3 normal1 = hitData->vertexNormals[normalIndexTriplet.y];
	float3 normal2 = hitData->vertexNormals[normalIndexTriplet.z];

	float2 barycentrics = optixGetTriangleBarycentrics();
	float alpha = 1.0f - barycentrics.x - barycentrics.y;

	float3 normal = normal0 * alpha + normal1 * barycentrics.x + normal2 * barycentrics.y;

	float3 hitPoint = payload.origin + optixGetRayTmax() * payload.direction;
	payload.origin = hitPoint;

	float3 diffuseDirection = normalize(normal + randomDirection(&payload.seed));
	float3 specularDirection = payload.direction - 2 * dot(payload.direction, normal) * normal;
	bool isSpecualarBounce = hitData->metallic >= randomFloat(&payload.seed);
	payload.direction = normalize(lerp(diffuseDirection, specularDirection, (1 - hitData->roughness) * isSpecualarBounce));

	payload.emittedLight += (hitData->emissionColor * hitData->emissionPower) * payload.color;
	float3 color = lerp(hitData->color, make_float3(1.0f, 1.0f, 1.0f), isSpecualarBounce);
	payload.color *= color;

	payload.done = hitData->emissionPower > 0.0f ? 1.0f : 0.0f;

	if (payload.prepForDenoiser) {
		const uint3 idx = optixGetLaunchIndex();
		const uint3 dim = optixGetLaunchDimensions();
		unsigned int pixelIndex = idx.y * params.width + idx.x;
		mat4 transformationMatrix = {};
		optixGetObjectToWorldTransformMatrix((float*) &transformationMatrix);

		params.normals[pixelIndex] = normal;
		params.albedo[pixelIndex] = hitData->color;

		float4 currentPosition = params.projectionMatrix * params.viewMatrix * transformationMatrix * make_float4(hitPoint, 1.0f);
		float4 previousPosition = params.previousProjection * params.previousView * transformationMatrix * make_float4(hitPoint, 1.0f);
		params.flow[pixelIndex] = (make_float2(currentPosition.x, currentPosition.y) / currentPosition.w) - (make_float2(previousPosition.x, previousPosition.y) / previousPosition.w);
	}

	storePayload(payload);
}

static __forceinline__ __device__ float reflectance(float cosTheta, float refractionRatio) {
	//Schlick's approximation
	float r0 = (1 - refractionRatio) / (1 + refractionRatio);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosTheta), 5);
}

extern "C" __global__ void __closesthit__dielectric() {
	RayPayload payload = loadPayload();
	HitgroupData* hitData = (HitgroupData*)optixGetSbtDataPointer();

	unsigned int primitiveIndex = optixGetPrimitiveIndex();
	uint3 normalIndexTriplet = hitData->vertexNormalIndices[primitiveIndex];

	float3 normal0 = hitData->vertexNormals[normalIndexTriplet.x];
	float3 normal1 = hitData->vertexNormals[normalIndexTriplet.y];
	float3 normal2 = hitData->vertexNormals[normalIndexTriplet.z];

	float2 barycentrics = optixGetTriangleBarycentrics();
	float alpha = 1.0f - barycentrics.x - barycentrics.y;

	float3 normal = normal0 * alpha + normal1 * barycentrics.x + normal2 * barycentrics.y;

	float ir = 1.0f; //TODO switch to variable
	float refractionRatio = ir; //TODO add check for front face

	float cosTheta = dot(-payload.direction, normal);
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

	bool cannotReflect = refractionRatio * sinTheta > 1.0f;

	float3 hitPoint = payload.origin + optixGetRayTmax() * payload.direction;
	payload.origin = hitPoint;

	if (cannotReflect || reflectance(cosTheta, refractionRatio) > randomFloat(&payload.seed)) {
		payload.direction = payload.direction - 2 * dot(payload.direction, normal) * normal; //reflect
	} else {
		float3 rOutPerp = refractionRatio * (payload.direction + cosTheta * normal);
		float3 rOutParallel = -sqrt(fabs(1.0f - lengthSquared(rOutPerp))) * normal;
		payload.direction = rOutPerp + rOutParallel; // refract
	}

	payload.done = 0.0f;

	if (payload.prepForDenoiser) {
		const uint3 idx = optixGetLaunchIndex();
		const uint3 dim = optixGetLaunchDimensions();
		unsigned int pixelIndex = idx.y * params.width + idx.x;
		mat4 transformationMatrix = {};
		optixGetObjectToWorldTransformMatrix((float*)&transformationMatrix);

		params.normals[pixelIndex] = normal;
		params.albedo[pixelIndex] = hitData->color;

		float4 currentPosition = params.projectionMatrix * params.viewMatrix * transformationMatrix * make_float4(hitPoint, 1.0f);
		float4 previousPosition = params.previousProjection * params.previousView * transformationMatrix * make_float4(hitPoint, 1.0f);
		params.flow[pixelIndex] = (make_float2(currentPosition.x, currentPosition.y) / currentPosition.w) - (make_float2(previousPosition.x, previousPosition.y) / previousPosition.w);
	}

	storePayload(payload);
}

extern "C" __global__ void __miss__ms() {
	RayPayload payload = loadPayload();
	MissData* missData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

	payload.emittedLight += make_float3(0.0f, 0.0f, 0.0f);
	payload.done = 1.0f;

	if (payload.prepForDenoiser) {
		const uint3 idx = optixGetLaunchIndex();
		const uint3 dim = optixGetLaunchDimensions();
		unsigned int pixelIndex = idx.y * params.width + idx.x;

		params.normals[pixelIndex] = make_float3(0.0f);
		params.albedo[pixelIndex] = payload.emittedLight;
		params.flow[pixelIndex] = make_float2(0.0f, 0.0f);
	}

	storePayload(payload);
}