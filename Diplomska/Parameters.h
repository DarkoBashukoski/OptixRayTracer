#include "cuda_runtime.h"
#include "optix/optix.h"
#include "UtilityMathFunctions.h"

struct Params {
    uchar4* image;
    uint3* debug_buffer;
    unsigned int width;
    unsigned int height;
    float3 camPosition;
    mat4 projectionMatrix;
    mat4 inverseProjection;
    mat4 viewMatrix;
    mat4 inverseView;
    OptixTraversableHandle handle;
    unsigned int* rngState;
};

struct RayGenData { };

struct HitGroupData {
    float3 matColor;
};

struct MissData { 
    float3 topColor;
    float3 bottomColor;
};