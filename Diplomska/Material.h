#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector_types.h>

class Material {
private:
	
public:
	float3 color;
	float roughness;
	float metallic;
	float3 emissionColor;
	float emissionPower;

	Material(float3 _color, float _roughness, float _metallic, float3 _emissionColor, float _emissionPower);
	Material();
};

#endif // !MATERIAL_H
