#ifdef BOT_SHADER_HOST_CPP_INCLUDE
#define PUB
#else
module shader_host;
#define PUB public
#endif

struct RenderVertex {
  float3 pos;
  float3 normal;
  float2 uv;
};

PUB struct ViewData {
  PUB float3x4 camTxfm; // 3x3 rotation matrix, last column is camera position
  PUB uint2 fbDims;
  PUB float2 screenScale;
  PUB float zNear;
  float pad0;
  float pad1;
  float pad2;
};

PUB struct NonUniformScaleObjectTransform {
  PUB float3x4 o2w;
  PUB float3x4 w2o;
};

PUB struct LightData {
  PUB float3 sunPosition;
  float pad1;
  PUB float3 sunColor;
  float pad2;
};

PUB struct GlobalPassData {
  PUB ViewData view;
  PUB LightData lights;
};

PUB struct ObjectPerDraw {
  PUB NonUniformScaleObjectTransform txfm;
};

#undef PUB
