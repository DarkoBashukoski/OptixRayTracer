#include "Scene.h"

Scene::Scene(OptixDeviceContext context, std::string fileName) {
	ifstream f("Resources\\Scenes\\" + fileName + ".json");
	json data = json::parse(f);

	int modelCount = data["scene"].size();
	entities = vector<Entity>();
	models = vector<RawModel>();
	models.reserve(modelCount);
	
	for (json modelData : data["scene"]) {
		string modelName = modelData["modelName"].template get<string>();
		modelName = "Resources\\Models\\" + fileName + "\\" + modelName + ".json";
		models.push_back(RawModel(context, modelName));
		
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
