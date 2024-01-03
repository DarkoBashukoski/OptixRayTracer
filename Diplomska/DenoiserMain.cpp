#include <optix/optix.h>
#include <optix/optix_stack_size.h>
#include <optix/optix_stubs.h>

#include "CudaOutputBuffer.h"

#include <cuda_runtime.h>
#include "ErrorChecks.h"

int main() {
	OptixDeviceContext context;
	CUDA_CHECK(cudaFree(0));
	CUcontext cuCtx = 0;
	OPTIX_CHECK(optixInit());

	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;

	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

	char* outLog = (char*)malloc(sizeof(char) * 2048);


}