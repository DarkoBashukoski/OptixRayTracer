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
#include "Denoiser.h"

using namespace std;

int main() {
	DisplayManager* displayManager = DisplayManager::getInstance();
	Renderer renderer = Renderer();
	OptixManager* optixManager = OptixManager::getInstance();

	CUstream stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	int width = displayManager->getWidth();
	int height = displayManager->getHeight();

	Denoiser denoiser = Denoiser(optixManager->getContext(), stream, width, height);

	//RawModel cornellBox = RawModel(optixManager->getContext(), "CornellBox.json");
	//RawModel cube = RawModel(optixManager->getContext(), "BasicCube.json");
	//RawModel knight = RawModel(optixManager->getContext(), "Knight.json");
	RawModel stormtrooper = RawModel(optixManager->getContext(), "Stormtrooper.json");
	RawModel stormtrooperFrame = RawModel(optixManager->getContext(), "StormtrooperFrame.json");
	RawModel stormtrooperBox = RawModel(optixManager->getContext(), "StormtrooperBox.json");
	RawModel stormtrooperLights = RawModel(optixManager->getContext(), "StormtrooperLights.json");

	//Entity e1 = Entity(&cornellBox, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
	//Entity e2 = Entity(&cube, make_float3(0.1f, 0.0f, 0.0f), make_float3(0.0f, 45.0f, 0.0f), make_float3(0.5f, 1.0f, 0.5f));
	//Entity e3 = Entity(&cube, make_float3(-1.0f, 0.0f, 0.0f), make_float3(0.0f, -30.0f, 0.0f), make_float3(0.5f, 0.5f, 0.5f));
	Entity e4 = Entity(&stormtrooper, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
	Entity e5 = Entity(&stormtrooperFrame, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
	Entity e6 = Entity(&stormtrooperBox, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
	Entity e7 = Entity(&stormtrooperLights, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));

	//optixManager->addEntity(&e1);
	//optixManager->addEntity(&e2);
	//optixManager->addEntity(&e3);
	optixManager->addEntity(&e4);
	optixManager->addEntity(&e5);
	optixManager->addEntity(&e6);
	optixManager->addEntity(&e7);

	optixManager->buildIas();
	optixManager->buildSbt();

	CudaOutputBuffer<float3> imageBuffer(width, height);
	CudaOutputBuffer<float3> normalBuffer(width, height);
	CudaOutputBuffer<float3> albedoBuffer(width, height);
	CudaOutputBuffer<float3> denoisedBuffer(width, height);

	float3 camPos = make_float3(0.0f, 1.0f, 5.0f);
	float3 camRot = make_float3(0.0f, 0.0f, 0.0f);

	Camera camera = Camera(camPos, camRot, displayManager->getWindow());

	Params params;
	params.image = imageBuffer.map();
	params.normals = normalBuffer.map();
	params.albedo = albedoBuffer.map();
	params.width = width;
	params.height = height;
	params.handle = optixManager->getIasHandle();
	params.samplesPerPixel = 3;
	params.maxDepth = 5;

	CUdeviceptr d_param;

	bool useDenoiser = true;
	bool accumulate = false;
	
	while (!displayManager->isCloseRequested()) {
		Timer::getInstance()->update();

		camera.update();
		params.camPosition = camera.getPosition();
		params.projectionMatrix = camera.getProjectionMatrix();
		params.viewMatrix = camera.getViewMatrix();
		params.inverseProjection = camera.getProjectionMatrix().inverse();
		params.inverseView = camera.getViewMatrix().inverse();

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(params), cudaMemcpyHostToDevice));
		OptixShaderBindingTable sbt = optixManager->getSbt();

		uint64_t renderStart = Timer::getInstance()->getTime();
		OPTIX_CHECK(optixLaunch(optixManager->getPipeline(), stream, d_param, sizeof(Params), &sbt, width, height, 1));
		CUDA_CHECK(cudaDeviceSynchronize());
		uint64_t renderTime = Timer::getInstance()->getTime() - renderStart;

		uint64_t denoiseStart = Timer::getInstance()->getTime();
		if (useDenoiser) {
			denoiser.launch(params.image, denoisedBuffer.map(), params.normals, params.albedo);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		uint64_t denoiseTime = Timer::getInstance()->getTime() - denoiseStart;

		imageBuffer.unmap();
		normalBuffer.unmap();
		albedoBuffer.unmap();
		denoisedBuffer.unmap();

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("My name is window, ImGUI window");
		ImGui::Text("FPS: %f", Timer::getInstance()->getFPS());
		ImGui::Text("Frame time: %fms", 1000.0f / Timer::getInstance()->getFPS());
		ImGui::Text("Render time: %fms", renderTime / 1000.0f);
		ImGui::Text("Denoise time: %fms", denoiseTime / 1000.0f);
		ImGui::SliderInt("Samples per pixel", &params.samplesPerPixel, 1, 32);
		ImGui::SliderInt("Max depth", &params.maxDepth, 1, 32);
		ImGui::Checkbox("Use denoiser", &useDenoiser);
		ImGui::Checkbox("Accumulate light", &accumulate);
		ImGui::End();
		
		renderer.render(useDenoiser ? denoisedBuffer : imageBuffer);
		displayManager->updateDisplay();
	}
	displayManager->closeDisplay();

	return 0;
}