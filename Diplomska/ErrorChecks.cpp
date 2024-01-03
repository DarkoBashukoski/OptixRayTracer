#include "ErrorChecks.h"

void CUDA_CHECK(cudaError_t error) {
	if (error != cudaError::cudaSuccess) {
		cout << "Cuda error: " << error << endl;
	}
}

void OPTIX_CHECK(OptixResult error) {
	if (error != OptixResult::OPTIX_SUCCESS) {
		cout << "Optix error: " << error << endl;
	}
}

void NVRTC_CHECK(nvrtcResult error) {
	if (error != nvrtcResult::NVRTC_SUCCESS) {
		cout << "NVRTC error: " << error << endl;
	}
}

void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
	cerr << "[" << setw(2) << level << "][" << setw(12) << tag << "]: " << message << "\n";
}
