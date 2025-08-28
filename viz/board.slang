import shader_host;
import utils;

struct BoardMaterial {
  Texture2D tex;
  SamplerState sampler;
};

ParameterBlock<GlobalPassData> global;

ParameterBlock<BoardMaterial> mat;

ParameterBlock<BoardDrawData> drawData;

struct V2F {
  float4 pos    : SV_Position;
  float3 normal : NORMAL0;
  float2 uv     : TEXCOORD0;
  float worldZ : TEXCOORD1;
};

[shader("vertex")]
V2F vertMain(uint idx: SV_VertexID)
{
  float3 pos_w, normal_w;
  float2 uv;
  cubeVert(idx, pos_w, normal_w, uv);

  uint2 grid_size = drawData.gridSize;

  pos_w.xy *= grid_size / 2;
  pos_w.xy += grid_size / 2 - 0.5f;
  pos_w.z *= 0.5f;
  pos_w.z -= 1.f;

  uv *= grid_size / 2;

  V2F out;
  out.pos = worldToClip(global.view, pos_w);
  out.normal = normal_w;
  out.uv = uv;

  out.worldZ = pos_w.z;

  return out;
}

[shader("fragment")]
float4 fragMain(V2F v2f) : SV_Target0
{
  float3 view_fwd = global.view.camTxfm[2].xyz;

  float3 n = normalize(v2f.normal);

  float dot_view = clamp(-dot(view_fwd, n), 0, 1);

  float brightness = lerp(0.25, 1, dot_view);

  float4 albedo = mat.tex.Sample(mat.sampler, v2f.uv);

  float4 color = brightness * albedo;
  return color;
}
