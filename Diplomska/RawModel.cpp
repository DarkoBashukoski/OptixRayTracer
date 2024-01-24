#include "RawModel.h"

void RawModel::parseModelData(json modelData) {
	vertexCount = modelData["vertexCount"];
	string vertexString = modelData["vertices"].template get<string>();
	vertexNormalCount = modelData["vertexNormalCount"];
	string vertexNormalsString = modelData["vertexNormals"].template get<string>();
	triangleCount = modelData["triangleCount"];
	string indexString = modelData["indices"].template get<string>();
	string vertexNormalIndicesString = modelData["vertexNormalIndices"];

	vertices = (float3*)malloc(vertexCount * sizeof(float3)); //moras free nekogas
	vertexNormals = (float3*)malloc(vertexNormalCount * sizeof(float3)); //morat free nekogas
	indices = (uint3*)malloc(triangleCount * sizeof(uint3)); //moras free nekogas
	vertexNormalIndices = (uint3*)malloc(triangleCount * sizeof(uint3)); //moras free nekogas

	string str;

	stringstream s = stringstream(vertexString);
	for (int i = 0; i < vertexCount; i++) {
		float3 vertex;

		getline(s, str, ';');
		vertex.x = stof(str);
		getline(s, str, ';');
		vertex.y = stof(str);
		getline(s, str, ';');
		vertex.z = stof(str);

		*(vertices + i) = vertex;
	}
	s.clear();

	s = stringstream(vertexNormalsString);
	for (int i = 0; i < vertexNormalCount; i++) {
		float3 vertexNormal;

		getline(s, str, ';');
		vertexNormal.x = stof(str);
		getline(s, str, ';');
		vertexNormal.y = stof(str);
		getline(s, str, ';');
		vertexNormal.z = stof(str);

		*(vertexNormals + i) = vertexNormal;
	}
	s.clear();

	s = stringstream(indexString);
	for (int i = 0; i < triangleCount; i++) {
		uint3 indexTriplet;

		getline(s, str, ';');
		indexTriplet.x = stoi(str);
		getline(s, str, ';');
		indexTriplet.y = stoi(str);
		getline(s, str, ';');
		indexTriplet.z = stoi(str);

		*(indices + i) = indexTriplet;
	}
	s.clear();

	s = stringstream(vertexNormalIndicesString);
	for (int i = 0; i < triangleCount; i++) {
		uint3 indexTriplet;

		getline(s, str, ';');
		indexTriplet.x = stoi(str);
		getline(s, str, ';');
		indexTriplet.y = stoi(str);
		getline(s, str, ';');
		indexTriplet.z = stoi(str);

		*(vertexNormalIndices + i) = indexTriplet;
	}
	s.clear();

	size_t vertexNormalsSize = sizeof(float3) * vertexNormalCount;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dVertexNormals), vertexNormalsSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dVertexNormals), vertexNormals, vertexNormalsSize, cudaMemcpyHostToDevice));

	size_t vertexNormalIndicesSize = sizeof(uint3) * triangleCount;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dVertexNormalIndices), vertexNormalIndicesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dVertexNormalIndices), vertexNormalIndices, vertexNormalIndicesSize, cudaMemcpyHostToDevice));

	//mojs free da dodajs ovde na host data, mozno e da trebit cuda sync check
}

void RawModel::parseMaterialData(json materialData){
	material = Material();

	material.shaderId = materialData["shaderId"];
	material.color.x = materialData["color"]["red"];
	material.color.y = materialData["color"]["green"];
	material.color.z = materialData["color"]["blue"];
	material.emissionColor.x = materialData["color"]["red"];
	material.emissionColor.y = materialData["color"]["green"];
	material.emissionColor.z = materialData["color"]["blue"];
	material.roughness = materialData["roughness"];
	material.metallic = materialData["metallic"];
	material.emissionPower = materialData["emissionPower"];
}

RawModel::RawModel(OptixDeviceContext _context, string fileName) : context(_context) {
	ifstream f(fileName);
	json data = json::parse(f);

	parseModelData(data["modelData"]);
	parseMaterialData(data["material"]);
	buildGas();
}

void RawModel::buildGas() {
	const size_t verticesSize = sizeof(float3) * vertexCount;
	CUdeviceptr dVertices = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dVertices), verticesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dVertices), vertices, verticesSize, cudaMemcpyHostToDevice));

	const size_t indicesSize = sizeof(uint3) * triangleCount;
	CUdeviceptr dIndexBuffer = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIndexBuffer), indicesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dIndexBuffer), indices, indicesSize, cudaMemcpyHostToDevice));

	const size_t matSize = sizeof(uint32_t);
	CUdeviceptr dMaterialIndeces = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMaterialIndeces), matSize));
	CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dMaterialIndeces), 0, matSize));

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	uint32_t* triangleInputFlags = (uint32_t*)malloc(sizeof(uint32_t));
	for (int i = 0; i < 1; i++) {
		triangleInputFlags[i] = OPTIX_GEOMETRY_FLAG_NONE;
	}

	OptixBuildInput triangleInput = {};
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	triangleInput.triangleArray.numVertices = static_cast<uint32_t>(vertexCount);
	triangleInput.triangleArray.vertexBuffers = &dVertices;
	triangleInput.triangleArray.indexBuffer = dIndexBuffer;
	triangleInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(triangleCount);
	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexStrideInBytes = sizeof(uint3);
	triangleInput.triangleArray.flags = triangleInputFlags;
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.sbtIndexOffsetBuffer = dMaterialIndeces;
	triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
	triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

	OptixAccelBufferSizes gasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &triangleInput, 1, &gasBufferSizes));
	CUdeviceptr dTempBufferGas;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTempBufferGas), gasBufferSizes.tempSizeInBytes));

	CUdeviceptr dTempBufferCompactedGas;
	size_t compactedSizeOffset = ((gasBufferSizes.outputSizeInBytes + 8ull - 1) / 8ull) * 8ull;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTempBufferCompactedGas), compactedSizeOffset + 8));

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = reinterpret_cast<CUdeviceptr>(reinterpret_cast<char*>(dTempBufferCompactedGas) + compactedSizeOffset);

	OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &triangleInput, 1, dTempBufferGas, gasBufferSizes.tempSizeInBytes, dTempBufferCompactedGas, gasBufferSizes.outputSizeInBytes, &gasHandle, &emitProperty, 1));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dVertices)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dIndexBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dMaterialIndeces)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dTempBufferGas)));
	free(reinterpret_cast<void*>(triangleInputFlags));
}

OptixTraversableHandle RawModel::getGasHandle() {
	return gasHandle;
}

Material& RawModel::getMaterial() {
	return material;
}

CUdeviceptr RawModel::getDeviceVertexNormals() {
	return dVertexNormals;
}

CUdeviceptr RawModel::getDeviceVertexNormalIndices() {
	return dVertexNormalIndices;
}

uint32_t RawModel::getTriangleCount() {
	return triangleCount;
}
