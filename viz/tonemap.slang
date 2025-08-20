import shader_host;
import utils;

struct TonemapInput {
  Texture2D hdr;
  SamplerState hdrSampler;
};

ParameterBlock<TonemapInput> tonemapInput;

[shader("vertex")]
FullscreenV2F vertMain(uint id : SV_VertexID)
{
  return fullscreenVS(id);
}

// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// MIT License, MJP, David Neubelt, Stephen Hill

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
static const float3x3 ACESInputMat =
{
  {0.59719, 0.35458, 0.04823},
  {0.07600, 0.90834, 0.01566},
  {0.02840, 0.13383, 0.83777},
};

// ODT_SAT => XYZ => D60_2_D65 => sRGB
static const float3x3 ACESOutputMat =
{
  { 1.60475, -0.53108, -0.07367},
  {-0.10208,  1.10813, -0.00605},
  {-0.00327, -0.07276,  1.07602},
};

float3 RRTAndODTFit(float3 v)
{
  float3 a = v * (v + 0.0245786f) - 0.000090537f;
  float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
  return a / b;
}

[shader("fragment")]
float4 fragMain(FullscreenV2F v2f) : SV_Target0
{
  float3 color = tonemapInput.hdr.Sample(tonemapInput.hdrSampler, v2f.uv).rgb;

  color = mul(ACESInputMat, color);
  color = RRTAndODTFit(color);
  color = mul(ACESOutputMat, color);

  color = saturate(color);
  return float4(color, 1);
}
