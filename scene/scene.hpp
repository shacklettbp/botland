#pragma once

#include <rt/err.hpp>
#include <rt/math.hpp>

namespace bot {

struct Body {
  enum class Type {
    FreeBody, // ndof = 6
    Hinge, // ndof = 1,
    Ball, // ndof = 3
    FixedBody, // ndof = 0
    Slider, // ndof = 1
    None
  };

  // Returns how many floats are required to store info for this body type
  static inline uint32_t getTypeDim(
      Type type, bool is_pos = false);
  static inline const char *getTypeStr(Type type);

  Type type;

  uint32_t qOffset;
  uint32_t qvOffset;

  // Index of parent in the `bodies` array in BodyGroup
  int32_t parentIndex;

  // This is relevant for anything that isn't FreeBody / FixedBody
  Vector3 axis;

  // Position of the joint axis relative to the parent's origin
  Vector3 relPositionParent;

  // Position of the current body's COM relative to the joint axis
  Vector3 relPositionLocal;

  // Rotation of the current body relative to parent
  Quat parentToChildRot;
};

// Either link or joint name hashes
struct BodyNameHash {
  uint32_t bodyIdx;
  uint32_t hash;
};

// This is what we use to attach colliders / visuals to the body
struct BodyObjAttach {
  uint32_t objID;
  Vector3 offset;
  Quat rot;
  Diag3x3 scale;

  // Index of the body this object is attached to
  uint16_t bodyIdx;

  // The index of the object within the list of objects attached 
  // to the body
  uint16_t subIdx;
};

struct BodyGroupInfo {
  // Number of degrees of freedom
  uint32_t numDofs;
  uint32_t qDim;

  // Number of bodies
  uint32_t numBodies;
  Body *bodies;

  uint32_t numVisuals;
  BodyObjAttach *visuals;

  uint32_t numNameHashes;
  BodyNameHash *nameHashes;
};

struct BodyGroupGeneralized {
  // Generalized position
  float *q;

  // Generalized velocity
  float *qv;
};

struct BodyPose {
  // Position / rotation of the center of mass
  Vector3 pos;
  Quat rot;
};

struct BodyObjTransform {
  Vector3 pos;
  Quat rot;
  Diag3x3 scale;
  uint32_t bodyIdx;
  uint32_t objID;
};

struct BodyGroupPose {
  // One pose per body
  uint32_t numPoses;
  BodyPose *poses;
};

struct BodyGroupRenderPose {
  uint32_t numTransforms;
  BodyObjTransform *transforms;
};

struct JointHinge {
  // In parent's basis
  Vector3 relPositionParent;

  // In child's basis
  Vector3 relPositionChild;

  // Rotation applied to child's vectors relative to
  // parent's coordinate system.
  Quat relParentRotation;

  // In child's basis
  Vector3 hingeAxis;

  // TODO: For now, we only support damping / friction loss for hinge
  float damping;
  float frictionLoss;
};

struct JointSlider {
  // In the parent's coordinate basis
  Vector3 relPositionParent;

  // In the child's coordinate basis
  Vector3 relPositionChild;

  // Rotation applied to child's vectors relative to
  // parent's coordinate system.
  Quat relParentRotation;

  // This is in the child's coordinate basis
  Vector3 slideVector;
};

struct JointBall {
  // In parent's basis
  Vector3 relPositionParent;

  // In child's basis
  Vector3 relPositionChild;

  // Rotation applied to child's vectors relative to
  // parent's coordinate system.
  Quat relParentRotation;
};

struct JointFixed {
  // In the parent's coordinate basis
  Vector3 relPositionParent;

  // In the child's coordinate basis
  Vector3 relPositionChild;

  // Rotation applied to child's vectors relative to
  // parent's coordinate system.
  Quat relParentRotation;
};

struct JointConnection {
  // These are body indices
  uint32_t parentIdx;
  uint32_t childIdx;

  // This determines which Joint struct to use
  Body::Type type;

  union {
    JointHinge hinge;
    JointBall ball;
    JointSlider slider;
    JointFixed fixed;
    // ...
  };
};

struct CollisionDisable {
  uint32_t aBody;
  uint32_t bBody;
};

enum class ResponseType : uint32_t {
  Dynamic,
  Kinematic,
  Static,
};

struct BodyDesc {
  Body::Type type;
  ResponseType responseType;
  uint32_t numVisualObjs;
  float mass;
  Diag3x3 inertia;
  float muS;
};

struct CollisionDesc {
  uint32_t objID;
  Vector3 offset;
  Quat rotation;
  Diag3x3 scale;

  // Required for URDF loading
  uint32_t linkIdx;
  // Index of the collider within the body
  uint32_t subIndex;

  // Optional to visualize the collision entities (-1 means we didn't
  // pass the collision objects to the renderer for visualization)
  int32_t renderObjID;
};

struct VisualDesc {
  uint32_t objID;
  Vector3 offset;
  Quat rotation;
  Diag3x3 scale;

  // Required for URDF loading
  uint32_t linkIdx;
  // Index of the collider within the body
  uint32_t subIndex;
};

// For loading pre-configured models
struct ModelConfig {
  // Assume that the first one is the root
  uint32_t numBodies;
  uint32_t bodiesOffset;

  uint32_t numConnections;
  uint32_t connectionsOffset;

  uint32_t numVisuals;
  uint32_t visualsOffset;

  uint32_t numHashes;
  uint32_t hashOffset;
};

// This is the data for all models in that could possibly be loaded.
struct ModelData {
  uint32_t numBodies;
  BodyDesc *bodies;

  uint32_t numConnections;
  JointConnection *connections;

  uint32_t numVisuals;
  VisualDesc *visuals;

  uint32_t numNameHashes;
  BodyNameHash *nameHashes;
};

BodyGroupInfo makeBodyGroup(
    uint32_t num_bodies,
    BodyDesc *body_descs,
    uint32_t num_visuals,
    VisualDesc *visuals,
    uint32_t num_connections,
    JointConnection *connections,
    uint32_t num_name_hashes,
    BodyNameHash *name_hashes,
    float global_scale = 1.f);

void destroyBodyGroup(BodyGroupInfo &body_grp);

// This returns the body group entity
BodyGroupInfo loadModel(ModelConfig cfg,
                        ModelData model_data,
                        float global_scale = 1.f);

// Fills in things like mass and inertia
BodyDesc makeCapsuleBodyDesc(
        Body::Type type,
        ResponseType response_type,
        float mu_s,
        float mass, // Total mass
        float radius,
        float cylinder_height);

// Because of the weird scaling properties of the capsule, we handle the
// capsule differently for rendering and physics.
// The visuals parameter needs to be a pointer with space for at least
// 3 visual desc's.
void writeCapsuleObjects(
      VisualDesc *visuals,
      float radius,
      float cylinder_height,
      uint32_t sphere_render_obj_idx,
      uint32_t cylinder_render_obj_idx);

BodyGroupGeneralized allocGeneralized(const BodyGroupInfo &bg);
BodyGroupPose allocBodyGroupPose(const BodyGroupInfo &bg);
BodyGroupRenderPose allocBodyGroupRenderPose(const BodyGroupInfo &bg);

void runFK(const BodyGroupInfo &bg_info,
           const BodyGroupGeneralized &bg_gen,
           BodyGroupPose &out);

void setRenderPose(const BodyGroupInfo &bg_info,
                   const BodyGroupPose &pose,
                   BodyGroupRenderPose &out);

void printBodyGroupInfo(const BodyGroupInfo &info);
void printBodyGroupPose(const BodyGroupPose &pose);

}

#include "scene.inl"
