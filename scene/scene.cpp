#include <stdio.h>
#include <cassert>
#include <rt/err.hpp>
#include <scene/scene.hpp>

namespace bot {

BodyGroupInfo makeBodyGroup(
    uint32_t num_bodies,
    BodyDesc *body_descs,
    uint32_t num_visuals,
    VisualDesc *visuals,
    uint32_t num_connections,
    JointConnection *connections,
    uint32_t num_name_hashes,
    BodyNameHash *name_hashes,
    float global_scale)
{
  (void)global_scale;

  // Actual number of DOFs (qv dim)
  uint32_t num_dofs = 0;
  uint32_t q_dim = 0;
  for (uint32_t i = 0; i < num_bodies; ++i) {
    num_dofs += Body::getTypeDim(body_descs[i].type);
    q_dim += Body::getTypeDim(body_descs[i].type, true);
  }

  BodyGroupInfo g = {
    .numDofs = num_dofs,
    .qDim = q_dim,
    .numBodies = num_bodies,
    .bodies = (Body *)malloc(num_bodies * sizeof(Body)),
    .numVisuals = num_visuals,
    .visuals = (BodyObjAttach *)malloc(num_visuals * sizeof(BodyObjAttach)),
    .numNameHashes = num_name_hashes,
    .nameHashes = name_hashes,
  };

  // Initialize all the bodies
  uint32_t qv_offset = 0;
  uint32_t q_offset = 0;
  for (uint32_t i = 0; i < num_bodies; ++i) {
    BodyDesc *bd_desc = &body_descs[i];
    Body *bd = &g.bodies[i];
    bd->type = bd_desc->type;
    bd->qOffset = q_offset;
    bd->qvOffset = qv_offset;

    // All to be initialized when processing joint connections
    bd->parentIndex = -1;
    bd->axis = Vector3::zero();
    bd->relPositionParent = Vector3::zero();
    bd->relPositionLocal = Vector3::zero();
    bd->parentToChildRot = Quat::id();

    qv_offset += Body::getTypeDim(bd->type);
    q_offset += Body::getTypeDim(bd->type, true);
  }

  for (uint32_t i = 0; i < num_connections; ++i) {
    JointConnection *conn = &connections[i];

    Body *c = &g.bodies[conn->childIdx];

    c->parentIndex = conn->parentIdx;

    switch (conn->type) {
    case Body::Type::Hinge: {
      c->axis = conn->hinge.hingeAxis;
      c->relPositionParent = conn->hinge.relPositionParent;
      c->relPositionLocal = conn->hinge.relPositionChild;
      c->parentToChildRot = conn->hinge.relParentRotation;
    } break;

    case Body::Type::Ball: {
      c->relPositionParent = conn->ball.relPositionParent;
      c->relPositionLocal = conn->ball.relPositionChild;
      c->parentToChildRot = conn->ball.relParentRotation;
    } break;

    case Body::Type::Slider: {
      c->axis = conn->slider.slideVector;
      c->relPositionParent = conn->slider.relPositionParent;
      c->relPositionLocal = conn->slider.relPositionChild;
      c->parentToChildRot = conn->slider.relParentRotation;
    } break;

    case Body::Type::FixedBody: {
      c->relPositionParent = conn->fixed.relPositionParent;
      c->relPositionLocal = conn->fixed.relPositionChild;
      c->parentToChildRot = conn->fixed.relParentRotation;
    } break;

    default: {
      FATAL("Received joint connection for invalid body type, %d", 
            (int)conn->type);
    } break;
    }
  }

  for (uint32_t i = 0; i < num_visuals; ++i) {
    VisualDesc *vd = &visuals[i];
    BodyObjAttach *oa = &g.visuals[i];
    oa->objID = vd->objID;
    oa->offset = vd->offset;
    oa->rot = vd->rotation;
    oa->scale = vd->scale;
    oa->bodyIdx = vd->linkIdx;
    oa->subIdx = vd->subIndex;
  }

  for (uint32_t i = 0; i < num_name_hashes; ++i) {
    g.nameHashes[i] = name_hashes[i];
  }
  
  return g;
}

BodyGroupGeneralized allocGeneralized(const BodyGroupInfo &bg)
{
  return BodyGroupGeneralized {
    .q = (float *)malloc(sizeof(float) * bg.qDim),
    .qv = (float *)malloc(sizeof(float) * bg.numDofs),
  };
}

BodyGroupPose allocBodyGroupPose(const BodyGroupInfo &bg)
{
  return BodyGroupPose {
    .numPoses = bg.numBodies,
    .poses = (BodyPose *)malloc(sizeof(BodyPose) * bg.numBodies),
  };
}

BodyGroupRenderPose allocBodyGroupRenderPose(const BodyGroupInfo &bg)
{
  return BodyGroupRenderPose {
    .numTransforms = bg.numVisuals,
    .transforms = (BodyObjTransform *)malloc(
        sizeof(BodyObjTransform) * bg.numVisuals),
  };
}

void destroyBodyGroup(BodyGroupInfo &grp_info)
{
  free(grp_info.bodies);
  free(grp_info.visuals);
  free(grp_info.nameHashes);
}

BodyGroupInfo loadModel(ModelConfig cfg,
                        ModelData model_data,
                        float global_scale)
{
  (void)global_scale;

  return makeBodyGroup(
    cfg.numBodies,
    model_data.bodies + cfg.bodiesOffset,
    cfg.numVisuals,
    model_data.visuals + cfg.visualsOffset,
    cfg.numConnections,
    model_data.connections + cfg.connectionsOffset,
    cfg.numHashes,
    model_data.nameHashes + cfg.hashOffset);
}

BodyDesc makeCapsuleBodyDesc(
    Body::Type type,
    ResponseType response_type,
    float mu_s,
    float mass,
    float radius,
    float cylinder_height)
{
  float r2 = radius * radius;
  float r3 = r2 * radius;
  float h2 = cylinder_height * cylinder_height;
  float rh = radius * cylinder_height;

  float cy_vol = PI * r2 * cylinder_height;
  float hs_vol = 2.f * PI * r3;
  float cp_vol = cy_vol + 2.f * hs_vol;

  float cy_mass = (cy_vol / cp_vol) * mass;
  float hs_mass = (hs_vol / cp_vol) * mass;

  Diag3x3 inertia_tensor = {
    cy_mass * (h2/12.f + r2*0.25f) +
      2.f * hs_mass * (r2*0.4f + h2*0.5f + rh*0.375f),
    cy_mass * (h2/12.f + r2*0.25f) +
      2.f * hs_mass * (r2*0.4f + h2*0.5f + rh*0.375f),
    cy_mass * (r2*0.5f) + 2.f * hs_mass * (r2*0.4f),
  };

  return BodyDesc {
    .type = type,
    .responseType = response_type,
    .numVisualObjs = 3,
    .mass = mass,
    .inertia = inertia_tensor,
    .muS = mu_s,
  };
}

void writeCapsuleObjects(
      VisualDesc *visuals,
      float radius,
      float cylinder_height,
      uint32_t sphere_render_obj_idx,
      uint32_t cylinder_render_obj_idx)
{
  visuals[0] = VisualDesc {
    .objID = cylinder_render_obj_idx,
    .offset = Vector3::all(0.f),
    .rotation = Quat::id(),
    .scale = { radius, radius, cylinder_height*0.5f },
    .linkIdx = 0xFFFF'FFFF,
    .subIndex = 0xFFFF'FFFF
  };

  visuals[1] = VisualDesc {
    .objID = sphere_render_obj_idx,
    .offset = { 0.f, 0.f, -cylinder_height*0.5f },
    .rotation = Quat::id(),
    .scale = { radius, radius, radius },
    .linkIdx = 0xFFFF'FFFF,
    .subIndex = 0xFFFF'FFFF
  };

  visuals[2] = VisualDesc {
    .objID = sphere_render_obj_idx,
    .offset = { 0.f, 0.f, cylinder_height*0.5f },
    .rotation = Quat::id(),
    .scale = { radius, radius, radius },
    .linkIdx = 0xFFFF'FFFF,
    .subIndex = 0xFFFF'FFFF
  };
}

void runFK(const BodyGroupInfo &bg_info,
           const BodyGroupGeneralized &bg_gen,
           BodyGroupPose &out)
{
  const float *all_q = bg_gen.q;

  BodyPose *all_poses = out.poses;
  Body *all_bodies = bg_info.bodies;

  { // Set the parent's state (we require that the root is fixed or free body
    uint32_t root_q_offset = all_bodies[0].qOffset;
    const float *q = all_q + root_q_offset;

    Vector3 com = { q[0], q[1], q[2] };

    all_poses[0] = {
      .pos = com,
      .rot = { q[3], q[4], q[5], q[6] },
    };
  }

  // Forward pass from parent to children
  for (int i = 1; i < (int)bg_info.numBodies; ++i) {
    const Body &body = all_bodies[i];

    const float *q = all_q + body.qOffset;

    BodyPose *curr_pose = all_poses + i;

    const BodyPose &parent_pose = all_poses[body.parentIndex];

    // We can calculate our stuff.
    switch (body.type) {
    case Body::Type::Hinge: {
      // Find the hinge axis orientation in world space
      Vector3 rotated_hinge_axis =
        parent_pose.rot.rotateVec(
            body.parentToChildRot.rotateVec(body.axis));
      rotated_hinge_axis = rotated_hinge_axis.normalize();

      // Calculate the composed rotation applied to the child entity.
      curr_pose->rot = parent_pose.rot *
        body.parentToChildRot *
        Quat::angleAxis(q[0], body.axis);

      // Calculate the composed COM position of the child
      //  (parent COM + R_{parent} * (rel_pos_parent + R_{hinge} * rel_pos_local))
      curr_pose->pos = parent_pose.pos +
        parent_pose.rot.rotateVec(
            body.relPositionParent +
            body.parentToChildRot.rotateVec(
              Quat::angleAxis(q[0], body.axis).
              rotateVec(body.relPositionLocal))
            );
    } break;

    case Body::Type::Slider: {
      // The composed rotation for this body is the same as the parent's
      curr_pose->rot = parent_pose.rot *
        body.parentToChildRot;

      curr_pose->pos = parent_pose.pos +
        parent_pose.rot.rotateVec(
            body.relPositionParent +
            body.parentToChildRot.rotateVec(
              body.relPositionLocal +
              q[0] * body.axis)
            );
    } break;

    case Body::Type::Ball: {
      Quat joint_rot = Quat{
        q[0], q[1], q[2], q[3]
      };

      // Calculate the composed rotation applied to the child entity.
      curr_pose->rot = parent_pose.rot *
        body.parentToChildRot *
        joint_rot;

      // Calculate the composed COM position of the child
      //  (parent COM + R_{parent} * (rel_pos_parent + R_{ball} * rel_pos_local))
      curr_pose->pos = parent_pose.pos +
        parent_pose.rot.rotateVec(
            body.relPositionParent +
            body.parentToChildRot.rotateVec(
              joint_rot.rotateVec(body.relPositionLocal))
            );
    } break;

    case Body::Type::FixedBody: {
      curr_pose->rot = parent_pose.rot;

      // This is the origin of the body
      curr_pose->pos =
        parent_pose.pos +
        parent_pose.rot.rotateVec(
            body.relPositionParent +
            body.parentToChildRot.rotateVec(
              body.relPositionLocal)
            );
    } break;

    default: {
      // Only hinges have parents
      assert(false);
    } break;
    }
  }
}

void setRenderPose(const BodyGroupInfo &bg_info,
                   const BodyGroupPose &pose,
                   BodyGroupRenderPose &out)
{
  for (uint32_t i = 0; i < bg_info.numVisuals; ++i) {
    const BodyObjAttach &attach = bg_info.visuals[i];
    const BodyPose &body_pose = pose.poses[attach.bodyIdx];
    BodyObjTransform &out_txfm = out.transforms[i];

    out_txfm.pos = body_pose.pos + 
                   body_pose.rot.rotateVec(attach.offset);
    out_txfm.rot = body_pose.rot * attach.rot;
    out_txfm.scale = attach.scale;
    out_txfm.bodyIdx = attach.bodyIdx;
    out_txfm.objID = attach.objID;
  }
}

void printBodyGroupInfo(const BodyGroupInfo &info)
{
  printf("Body group info:\n");
  printf("\t- numDofs = %u\n", info.numDofs);
  printf("\t- qDim = %u\n", info.qDim);
  printf("\t- numBodies = %u\n", info.numBodies);
  printf("\t- bodies:\n");
  for (uint32_t i = 0; i < info.numBodies; ++i) {
    Body *b = &info.bodies[i];
    printf("\t\t- type = %s\n", Body::getTypeStr(b->type));
    printf("\t\t- qOffset = %u\n", b->qOffset);
    printf("\t\t- qvOffset = %u\n", b->qvOffset);
    printf("\t\t- parentIndex = %d\n", b->parentIndex);
    printf("\t\t- axis = %f %f %f\n", 
        b->axis.x, b->axis.y, b->axis.z);
    printf("\t\t- relPositionParent = %f %f %f\n", 
        b->relPositionParent.x, b->relPositionParent.y, b->relPositionParent.z);
    printf("\t\t- relPositionLocal = %f %f %f\n", 
        b->relPositionLocal.x, b->relPositionLocal.y, b->relPositionLocal.z);
    printf("\t\t- parentToChildRot = %f %f %f %f\n", 
        b->parentToChildRot.w, 
        b->parentToChildRot.x, b->parentToChildRot.y, b->parentToChildRot.z);
  }

  printf("\t- numVisuals = %u\n", info.numVisuals);
  printf("\t- visuals:\n");
  for (uint32_t i = 0; i < info.numVisuals; ++i) {
    BodyObjAttach *v = &info.visuals[i];
    printf("\t\t- objID = %u\n", v->objID);
    printf("\t\t- offset = %f %f %f\n", 
        v->offset.x, v->offset.y, v->offset.z);
    printf("\t\t- rot = %f %f %f %f\n", 
        v->rot.w, v->rot.x, v->rot.y, v->rot.z);
    printf("\t\t- scale = %f %f %f\n", 
        v->scale.d0, v->scale.d1, v->scale.d2);
    printf("\t\t- bodyIdx = %u\n", 
        v->bodyIdx);
    printf("\t\t- subIdx = %u\n", 
        v->subIdx);
  }
}

void printBodyGroupPose(const BodyGroupPose &pose)
{
  printf("Body group pose:\n");
  printf("\t- numPoses = %u\n", pose.numPoses);
  for (uint32_t i = 0; i < pose.numPoses; ++i) {
    BodyPose *p = &pose.poses[i];
    printf("\t\t- pos = %f %f %f\n",
        p->pos.x, p->pos.y, p->pos.z);
    printf("\t\t- rot = %f %f %f %f\n",
        p->rot.w, p->rot.x, p->rot.y, p->rot.z);
  }
}
  
}
