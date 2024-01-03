#pragma once
#ifndef ERRORCHECKS_H
#define ERRORCHECKS_H

#include <driver_types.h>
#include <iostream>
#include <optix/optix_types.h>
#include <nvrtc.h>
#include <chrono>
#include <iomanip>

using namespace std;

void CUDA_CHECK(cudaError_t error);
void OPTIX_CHECK(OptixResult error);
void NVRTC_CHECK(nvrtcResult error);
void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */);

#endif // !ERRORCHECKS_H

