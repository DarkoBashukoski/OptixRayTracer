#include "Material.h"

Material::Material(float3 _color, float _roughness, float _metallic, float3 _emissionColor, float _emissionPower) 
	: color(_color), roughness(_roughness), metallic(_metallic), emissionColor(_emissionColor), emissionPower(_emissionPower) {}

Material::Material() {
	color = { 0.0f, 0.0f, 0.0f };
	roughness = 0;
	metallic = 0;
	emissionColor = { 0.0f, 0.0f, 0.0 };
	emissionPower = 0;
}