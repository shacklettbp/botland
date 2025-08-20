#pragma once

#include "rt/math.hpp"

namespace bot {

struct HalfEdge {
  uint32_t next;
  uint32_t rootVertex;
  uint32_t face;
};

struct Plane {
  Vector3 normal; // Potentially unnormalized
  float d;
};

struct Segment {
  Vector3 p1;
  Vector3 p2;
};

struct HalfEdgeMesh {
  template <typename Fn>
    inline void iterateFaceIndices(uint32_t face, Fn &&fn) const;
  inline uint32_t twinIDX(uint32_t half_edge_id) const;
  inline uint32_t numEdges() const;
  inline uint32_t edgeToHalfEdge(uint32_t edge_id) const;

  HalfEdge *halfEdges;
  uint32_t *faceBaseHalfEdges;
  Plane *facePlanes;
  Vector3 *vertices;

  uint32_t numHalfEdges;
  uint32_t numFaces;
  uint32_t numVertices;
};

// Sphere at origin, ray_d must be normalized
inline float intersectRayOriginSphere(
    Vector3 ray_o,
    Vector3 ray_d,
    float r);

// Assumes (0, 0, 0) is at the base of the capsule line segment, and that
// ray_d is normalized.
// h is the length of the line segment (not overall height of the capsule).
inline float intersectRayZOriginCapsule(
    Vector3 ray_o,
    Vector3 ray_d,
    float r,
    float h);

// Returns non-unit normal
inline Vector3 computeTriangleGeoNormal(
    Vector3 ab,
    Vector3 ac,
    Vector3 bc);

inline Vector3 triangleClosestPointToOrigin(
    Vector3 a,
    Vector3 b,
    Vector3 c,
    Vector3 ab,
    Vector3 ac);

// Returns distance to closest point squared + closest point itself
// in *closest_point. If the hull is touching the origin returns 0 and
// *closest_point is invalid.
float hullClosestPointToOriginGJK(
    HalfEdgeMesh &hull,
    float err_tolerance2,
    Vector3 *closest_point);

// The segment has to be centered at the origin
float hullClosestPointToSegmentGJK(
    HalfEdgeMesh &hull,
    float err_tolerance2,
    Vector3 p1,
    Vector3 p2,
    Vector3 *closest_point);
  
}

#include "geo.inl"
