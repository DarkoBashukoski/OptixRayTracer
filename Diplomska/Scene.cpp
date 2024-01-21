#include "Scene.h"

Scene::Scene(OptixDeviceContext context, std::string fileName) {
	ifstream f(fileName);
	json data = json::parse(f);
	entities = vector<Entity>();
	
	for (json modelData : data["scene"]) {
		string modelName = modelData["modelName"].template get<string>();
		models.push_back(RawModel(context, modelName + ".json"));
		
		for (json instanceData : modelData["instances"]) {
			float3 translation{}, rotation{}, scale{};

			translation.x = instanceData["translation"]["x"];
			translation.y = instanceData["translation"]["y"];
			translation.z = instanceData["translation"]["z"];

			rotation.x = instanceData["rotation"]["x"];
			rotation.y = instanceData["rotation"]["y"];
			rotation.z = instanceData["rotation"]["z"];

			scale.x = instanceData["scale"]["x"];
			scale.y = instanceData["scale"]["y"];
			scale.z = instanceData["scale"]["z"];

			Entity e = Entity(&models.back(), translation, rotation, scale);
			entities.push_back(e);
		}
	}
}

std::vector<Entity> Scene::getEntities() {
	return entities;
}
