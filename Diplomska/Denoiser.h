#pragma once
#ifndef DENOISER_H
#define DENOISER_H

#include <optix/optix_types.h>
#include <optix/optix_stubs.h>
#include <cuda_runtime.h>
#include "ErrorChecks.h"

class Denoiser {
private:
	int width;
	int height;
	CUstream stream;
	OptixDenoiser denoiser = nullptr;
	CUdeviceptr denoiserStateBuffer;
	CUdeviceptr denoiserScratchBuffer;
	OptixDenoiserSizes denoiserSizes = {};
	OptixDenoiserParams denoiserParams = {};
public:
	Denoiser(OptixDeviceContext context, CUstream _stream, int _width, int _height);
	void launch(float3* dInputData, float3* dOutputData, float3* dNormalData, float3* dAlbedoData, float2* dFlowData);
};

#endif // !DENOISER_H
