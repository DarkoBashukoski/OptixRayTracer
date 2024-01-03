#pragma once
#ifndef ENTITY_H
#define ENTITY_H

#include <vector_types.h>
#include "RawModel.h"
#include "UtilityMathFunctions.h"

class Entity {
private:
	RawModel* rawModel;
	float3 position;
	float3 rotation;
	float3 scale;
	float transformtaion[12];
public:
	Entity(RawModel* _rawModel, float3 _position, float3 _rotation, float3 _scale);
	RawModel* getModel();
	float3 getPosition();
	float3 getRotation();
	float3 getScale();
	float* getTransformation();
};

#endif // !ENTITY_H
