#include "Entity.h"

Entity::Entity(RawModel* _rawModel, float3 _position, float3 _rotation, float3 _scale) : rawModel(_rawModel), position(_position), rotation(_rotation), scale(_scale) {
	mat4 transMatrix;
	transMatrix.identity();
	transMatrix = transMatrix * mat4::createTranslation(position);
	transMatrix = transMatrix * mat4::createRotationX(toRadians(rotation.x));
	transMatrix = transMatrix * mat4::createRotationY(toRadians(rotation.y));
	transMatrix = transMatrix * mat4::createRotationZ(toRadians(rotation.z));
	transMatrix = transMatrix * mat4::createScale(scale);

	transformtaion[0] = transMatrix.m11;
	transformtaion[1] = transMatrix.m12;
	transformtaion[2] = transMatrix.m13;
	transformtaion[3] = transMatrix.m14;
	transformtaion[4] = transMatrix.m21;
	transformtaion[5] = transMatrix.m22;
	transformtaion[6] = transMatrix.m23;
	transformtaion[7] = transMatrix.m24;
	transformtaion[8] = transMatrix.m31;
	transformtaion[9] = transMatrix.m32;
	transformtaion[10] = transMatrix.m33;
	transformtaion[11] = transMatrix.m34;
}

RawModel* Entity::getModel() {
	return rawModel;
}

float3 Entity::getPosition() {
	return position;
}

float3 Entity::getRotation() {
	return rotation;
}

float3 Entity::getScale() {
	return scale;
}

float* Entity::getTransformation() {
	return transformtaion;
}
