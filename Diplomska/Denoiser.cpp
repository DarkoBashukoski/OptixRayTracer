#include "Denoiser.h"

Denoiser::Denoiser(OptixDeviceContext context, CUstream _stream, int _width, int _height) : stream(_stream), width(_width), height(_height) {
	OptixDenoiserOptions denoiserOptions = {};
	denoiserOptions.guideAlbedo = 0;
	denoiserOptions.guideNormal = 0;

	OPTIX_CHECK(optixDenoiserCreate(context, OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &denoiser));
	OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiserSizes));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&denoiserStateBuffer), denoiserSizes.stateSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&denoiserScratchBuffer), denoiserSizes.withoutOverlapScratchSizeInBytes));

	OPTIX_CHECK(optixDenoiserSetup(denoiser, stream, width, height, denoiserStateBuffer, denoiserSizes.stateSizeInBytes, denoiserScratchBuffer, denoiserSizes.withoutOverlapScratchSizeInBytes));

	denoiserParams.blendFactor = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&denoiserParams.hdrIntensity), sizeof(float)));
}

void Denoiser::launch(float3* dInputData, float3* dOutputData, float3* dNormalData, float3* dAlbedoData) {
	OptixDenoiserLayer layer = {};

	layer.input.data = reinterpret_cast<CUdeviceptr>(dInputData);
	layer.input.width = width;
	layer.input.height = height;
	layer.input.rowStrideInBytes = width * sizeof(float3);
	layer.input.pixelStrideInBytes = sizeof(float3);
	layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	layer.output.data = reinterpret_cast<CUdeviceptr>(dOutputData);
	layer.output.width = width;
	layer.output.height = height;
	layer.output.rowStrideInBytes = width * sizeof(float3);
	layer.output.pixelStrideInBytes = sizeof(float3);
	layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	OptixDenoiserGuideLayer guideLayer = {};
	/*
	guideLayer.albedo.data = reinterpret_cast<CUdeviceptr>(dAlbedoData);
	guideLayer.albedo.width = width;
	guideLayer.albedo.height = height;
	guideLayer.albedo.rowStrideInBytes = width * sizeof(float3);
	guideLayer.albedo.pixelStrideInBytes = sizeof(float3);
	guideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	guideLayer.normal.data = reinterpret_cast<CUdeviceptr>(dNormalData);
	guideLayer.normal.width = width;
	guideLayer.normal.height = height;
	guideLayer.normal.rowStrideInBytes = width * sizeof(float3);
	guideLayer.normal.pixelStrideInBytes = sizeof(float3);
	guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
	*/
	OPTIX_CHECK(optixDenoiserComputeIntensity(
		denoiser,
		stream,
		&layer.input,
		denoiserParams.hdrIntensity,
		denoiserScratchBuffer,
		denoiserSizes.withoutOverlapScratchSizeInBytes
	));

	OPTIX_CHECK(optixDenoiserInvoke(
		denoiser,
		stream,
		&denoiserParams,
		denoiserStateBuffer,
		denoiserSizes.stateSizeInBytes,
		&guideLayer,
		&layer,
		1,
		0,
		0,
		denoiserScratchBuffer,
		denoiserSizes.withoutOverlapScratchSizeInBytes
	));
}