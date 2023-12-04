#pragma once
#ifndef CUDAOUTPUTBUFFER_H
#define CUDAOUTPUTBUFFER_H

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

template <typename T>
class CudaOutputBuffer {
private:
	int deviceIndex = 0;
	int width = 0;
	int height = 0;

	cudaGraphicsResource* cudaResource = nullptr;
	GLuint pixelBufferObject;

	std::vector<T> hostPixels;
	T* devicePixels = nullptr;
	T* hostCopyPixels = nullptr;

	CUstream stream = 0u;

	void setCurrentGPU();
	void ensureMinimumSize(int _width, int _height);
public:
	CudaOutputBuffer(int _width, int _height);
	~CudaOutputBuffer();

	void setStream(CUstream _stream);

	void resize(int _width, int _height);

	T* map();
	void unmap();

	int getWidth();
	int getHeight();

	GLuint getPixelBufferObject();
	void deletePixelBufferObject();
	T* getHostPointer();
};

template <typename T>
void CudaOutputBuffer<T>::setCurrentGPU() {
	cudaSetDevice(deviceIndex);
}

template<typename T>
void CudaOutputBuffer<T>::ensureMinimumSize(int _width, int _height) {
	if (_width < 1) {width = 1;}
	if (_height < 1) {height = 1;}
}

template <typename T>
CudaOutputBuffer<T>::CudaOutputBuffer(int _width, int _height) {
	ensureMinimumSize(_width, _height);

	int currentDevice;
	int isDisplayDevice;

	cudaGetDevice(&currentDevice);
	cudaDeviceGetAttribute(&isDisplayDevice, cudaDevAttrKernelExecTimeout, currentDevice);

	if (!isDisplayDevice) {
		throw std::invalid_argument("Current graphics device is not the same as the display device.");
	}

	resize(_width, _height);
}

template <typename T>
CudaOutputBuffer<T>::~CudaOutputBuffer() {
	setCurrentGPU();
	cudaGraphicsUnregisterResource(cudaResource);

	if (pixelBufferObject != 0) {
		glBindBuffer(GL_ARRAY_BUFFER, 0u);
		glDeleteBuffers(1, &pixelBufferObject);
	}
}

template <typename T>
void CudaOutputBuffer<T>::setStream(CUstream _stream) {
	stream = _stream;
}

template <typename T>
void CudaOutputBuffer<T>::resize(int _width, int _height) {
	ensureMinimumSize(_width, _height);

	if (width == _width && height == _height) {
		return;
	}

	width = _width;
	height = _height;

	setCurrentGPU();

	glGenBuffers(1, &pixelBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, pixelBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(T) * width * height, nullptr, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0u);

	cudaGraphicsGLRegisterBuffer(&cudaResource, pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);

	if (!hostPixels.empty()) {
		hostPixels.resize(width * height);
	}
}

template <typename T>
T* CudaOutputBuffer<T>::map() {
	setCurrentGPU();

	size_t bufferSize = 0u;
	cudaGraphicsMapResources(1, &cudaResource, stream);
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devicePixels), &bufferSize, cudaResource);

	return devicePixels;
}

template <typename T>
void CudaOutputBuffer<T>::unmap() {
	setCurrentGPU();

	cudaGraphicsUnmapResources(1, &cudaResource, stream);
}

template <typename T>
int CudaOutputBuffer<T>::getWidth() {
	return width;
}

template <typename T>
int CudaOutputBuffer<T>::getHeight() {
	return height;
}

template <typename T>
GLuint CudaOutputBuffer<T>::getPixelBufferObject() {
	if (pixelBufferObject == 0u) {
		glGenBuffers(1, &pixelBufferObject);
	}

	return pixelBufferObject;
}

template <typename T>
void CudaOutputBuffer<T>::deletePixelBufferObject() {
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &pixelBufferObject);
	pixelBufferObject = 0;
}

template <typename T>
T* CudaOutputBuffer<T>::getHostPointer() {
	hostPixels.resize(width * height);

	setCurrentGPU();
	cudaMemcpy(static_cast<void*>(hostPixels.data()), map(), width * height * sizeof(T), cudaMemcpyDeviceToHost);
	unmap();

	return hostPixels.data();
}

#endif // !CUDAOUTPUTBUFFER_H