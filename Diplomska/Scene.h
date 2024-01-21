#pragma once
#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <fstream>
#include "Entity.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

class Scene {
private:
	std::vector<RawModel> models;
	std::vector<Entity> entities;
public:
	Scene(OptixDeviceContext context, std::string fileName);
	std::vector<Entity> getEntities();
};

#endif // !SCENE_H
