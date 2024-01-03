#include "RawModel.h"

void RawModel::parseModelData(json modelData) {
	vertexCount = modelData["vertexCount"];
	string vertexString = modelData["vertices"].template get<string>();
	triangleCount = modelData["triangleCount"];
	string indexString = modelData["indices"].template get<string>();
	string meterialIndexString = modelData["materialIndices"];

	vertices = (float3*)malloc(vertexCount * sizeof(float3)); //moras free nekogas
	indices = (uint3*)malloc(triangleCount * sizeof(uint3)); //moras free nekogas
	materialIndices = (uint32_t*)malloc(triangleCount * sizeof(uint32_t)); //moras free nekogas

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

	s = stringstream(meterialIndexString);
	for (int i = 0; i < triangleCount; i++) {
		uint32_t index;

		getline(s, str, ';');
		index = stoi(str);

		*(materialIndices + i) = index;
	}
	s.clear();
}

void RawModel::parseMaterialData(json materialData){
	materialCount = materialData.size();
	materials = vector<Material>();
	
	for (int i = 0; i < materialCount; i++) {
		json matData = materialData[i];

		Material mat = Material();
		mat.color.x = matData["color"]["red"];
		mat.color.y = matData["color"]["green"];
		mat.color.z = matData["color"]["blue"];
		mat.emissionColor.x = matData["color"]["red"];
		mat.emissionColor.y = matData["color"]["green"];
		mat.emissionColor.z = matData["color"]["blue"];
		mat.roughness = matData["roughness"];
		mat.metallic = matData["metallic"];
		mat.emissionPower = matData["emissionPower"];

		materials.push_back(mat);
	}
}

RawModel::RawModel(OptixDeviceContext _context, string fileName) : context(_context) {
	ifstream f(fileName);
	json data = json::parse(f);

	cout << "KOSNA";

	parseModelData(data["modelData"]);
	parseMaterialData(data["materialData"]);
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

	const size_t matSize = sizeof(uint32_t) * triangleCount;
	CUdeviceptr dMaterialIndeces = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMaterialIndeces), matSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMaterialIndeces), materialIndices, matSize, cudaMemcpyHostToDevice));

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	uint32_t* triangleInputFlags = (uint32_t*)malloc(materialCount * sizeof(uint32_t));
	for (int i = 0; i < materialCount; i++) {
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
	triangleInput.triangleArray.numSbtRecords = materialCount;
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

uint32_t RawModel::getMaterialCount() {
	return materialCount;
}

vector<Material> RawModel::getMaterials() {
	return materials;
}
