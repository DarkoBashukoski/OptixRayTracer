#include <optix/optix.h>
#include <optix/optix_function_table_definition.h>
#include <optix/optix_stack_size.h>
#include <optix/optix_stubs.h>

#include "CudaOutputBuffer.h"

#include <cuda_runtime.h>

#include "DisplayManager.h"
#include "Renderer.h"
#include "Camera.h"
#include "Timer.h"
#include "OptixShaderCompiler.h"
#include "ErrorChecks.h"
#include "OptixManager.h"
#include "RawModel.h"

using namespace std;

int main2() {
	DisplayManager* displayManager = DisplayManager::getInstance();
	Renderer renderer = Renderer();
	OptixManager* optixManager = OptixManager::getInstance();

	int width = displayManager->getWidth();
	int height = displayManager->getHeight();

	RawModel cornellBox = RawModel(optixManager->getContext(), "CornellBox.json");
	//RawModel cube = RawModel(optixManager->getContext(), "BasicCube.json");
	RawModel knight = RawModel(optixManager->getContext(), "Knight.json");

	Entity e1 = Entity(&cornellBox, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
	//Entity e2 = Entity(&cube, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 45.0f, 0.0f), make_float3(0.5f, 1.0f, 0.5f));
	//Entity e3 = Entity(&cube, make_float3(-0.8f, 0.0f, 0.0f), make_float3(0.0f, 30.0f, 0.0f), make_float3(0.5f, 0.5f, 0.5f));
	Entity e4 = Entity(&knight, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, -45.0f, 0.0f), make_float3(0.4f, 0.4f, 0.4f));

	optixManager->addEntity(&e1);
	//optixManager->addEntity(&e2);
	//optixManager->addEntity(&e3);
	optixManager->addEntity(&e4);

	optixManager->buildIas();
	optixManager->buildSbt();

	CudaOutputBuffer<uchar4> output_buffer(width, height);
	CudaOutputBuffer<float3> colorAccumulator(width, height);

	CUstream stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	float3 camPos = make_float3(0.0f, 0.0f, 5.0f);
	float3 camRot = make_float3(0.0f, 0.0f, 0.0f);

	Camera camera = Camera(camPos, camRot, displayManager->getWindow());

	Params params;
	params.image = output_buffer.map();
	params.colorAccumulator = colorAccumulator.map();
	params.width = width;
	params.height = height;
	params.handle = optixManager->getIasHandle();
	params.frameIndex = 0;

	CUdeviceptr d_param;
	
	while (!displayManager->isCloseRequested()) {
		if (camera.update()) {
			params.frameIndex = 1;
		}
		else {
			params.frameIndex++;
		}
		params.camPosition = camera.getPosition();
		params.projectionMatrix = camera.getProjectionMatrix();
		params.viewMatrix = camera.getViewMatrix();
		params.inverseProjection = camera.getProjectionMatrix().inverse();
		params.inverseView = camera.getViewMatrix().inverse();

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(params), cudaMemcpyHostToDevice));
		OptixShaderBindingTable sbt = optixManager->getSbt();
		OPTIX_CHECK(optixLaunch(optixManager->getPipeline(), stream, d_param, sizeof(Params), &sbt, width, height, 1));

		CUDA_CHECK(cudaDeviceSynchronize());
		output_buffer.unmap();
		colorAccumulator.unmap();
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));

		renderer.render(output_buffer);
		displayManager->updateDisplay();
		Timer::getInstance()->update();
		cout << Timer::getInstance()->getFPS() << endl;
	}

	displayManager->closeDisplay();
}