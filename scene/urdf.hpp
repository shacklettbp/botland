#pragma once

#include <scene/import.hpp>
#include <scene/scene.hpp>

namespace bot {

struct URDFLoader {
  struct Impl;

  URDFLoader();
  ~URDFLoader();

  std::unique_ptr<Impl> impl;

  struct BuiltinPrimitives {
    uint32_t cubeRenderIdx;
    uint32_t planeRenderIdx;
    uint32_t sphereRenderIdx;
    uint32_t capsuleRenderIdx;
  };

  struct URDFInfo {
    uint32_t idx;
    uint32_t numDofs;
    uint32_t numBodies;
  };

  // root_dir is the root of all the obj's that the URDF refers to.
  URDFInfo load(
      const std::string &path, 
      const std::string &root_dir,
      BuiltinPrimitives primitives,
      std::vector<std::string> &render_asset_paths,
      bool make_root_free_body = false,
      bool visualize_colliders = false);

  ModelData getModelData();
  ModelConfig * getModelConfigs(uint32_t &num_cfgs);



  uint32_t getBodyIndex(uint32_t urdf_index, const std::string &name);
};
    
}
