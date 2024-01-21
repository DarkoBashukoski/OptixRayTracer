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
#include "Scene.h"

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

	Scene scene = Scene(optixManager->getContext(), "StormtrooperScene.json");
	optixManager->addEntities(scene.getEntities());

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
	params.frameIndex = 0;
	params.samplesPerPixel = 3;
	params.maxDepth = 5;

	CUdeviceptr d_param;

	bool useDenoiser = true;
	bool accumulate = false;

	uint32_t triangleCount = optixManager->getTriangleCount();
	
	while (!displayManager->isCloseRequested()) {
		Timer::getInstance()->update();
		if (camera.update()) {
			params.frameIndex = 0;
		} else if (accumulate) {
			params.frameIndex += 1;
		}

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

		ImGui::Begin("Settings and Information");
		ImGui::Text("Triangle count: %d", triangleCount);
		ImGui::Text("FPS: %f", Timer::getInstance()->getFPS());
		ImGui::Text("Frame time: %fms", 1000.0f / Timer::getInstance()->getFPS());
		ImGui::Text("Render time: %fms", renderTime / 1000.0f);
		ImGui::Text("Denoise time: %fms", denoiseTime / 1000.0f);
		ImGui::SliderInt("Samples per pixel", &params.samplesPerPixel, 1, 32);
		ImGui::SliderInt("Max depth", &params.maxDepth, 1, 32);
		ImGui::Checkbox("Use denoiser", &useDenoiser);
		ImGui::Checkbox("Accumulate light", &accumulate);
		ImGui::Text("Frame index: %d", params.frameIndex);
		ImGui::End();
		
		renderer.render(useDenoiser ? denoisedBuffer : imageBuffer);
		displayManager->updateDisplay();
	}
	displayManager->closeDisplay();

	return 0;
}