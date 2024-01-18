#include "cuda_runtime.h"
#include "optix/optix.h"
#include "UtilityMathFunctions.h"

struct Params {
    float3* image;
    float3* normals;
    float3* albedo;
    unsigned int width;
    unsigned int height;
    int samplesPerPixel;
    int maxDepth;
    float3 camPosition;
    mat4 projectionMatrix;
    mat4 inverseProjection;
    mat4 viewMatrix;
    mat4 inverseView;
    OptixTraversableHandle handle;
};

struct RaygenData { };

struct MissData {
    float3 skyColorZenith;
    float3 skyColorHorizon;
    float3 groundColor;
    float3 sunDirection;
    float sunFocus;
    float sunIntensity;
};

struct HitgroupData {
    float3 color;
    float roughness;
    float metallic;
    float3 emissionColor;
    float emissionPower;

    float3* vertexNormals;
    uint3* vertexNormalIndices;
};