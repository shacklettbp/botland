import shader_host;
import utils;

struct NameMaterial {
  Texture2D fontTexture;
  SamplerState fontSampler;
};

ParameterBlock<GlobalPassData> global;

ParameterBlock<NameMaterial> nameMat;

ParameterBlock<TextPerDraw> perDraw;

struct V2F {
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
  float charIndex : TEXCOORD1;
};

[shader("vertex")]
V2F vertMain(uint idx: SV_VertexID)
{
  // We'll render text as a series of quads, one per character
  // Each character needs 6 vertices (2 triangles)
  uint charIdx = idx / 6;
  uint localIdx = idx % 6;
  
  // Bail out if we're past the number of characters
  if (charIdx >= perDraw.numChars) {
    V2F out;
    out.pos = float4(0, 0, 0, 0);
    out.uv = float2(0, 0);
    out.charIndex = 0;
    return out;
  }
  
  // Triangle strip pattern for a quad:
  float2 quadPattern[6] = {
    float2(0, 0), float2(1, 0), float2(1, 1),
    float2(0, 0), float2(1, 1), float2(0, 1)
  };
  
  float2 localVert = quadPattern[localIdx];
  
  // Get camera vectors for billboarding
  float3x4 camTxfm = global.view.camTxfm;
  float3 camRight = camTxfm[0].xyz;
  float3 camUp = camTxfm[1].xyz;
  float3 camPos = camTxfm[0].w * camTxfm[0].xyz + 
                  camTxfm[1].w * camTxfm[1].xyz + 
                  camTxfm[2].w * camTxfm[2].xyz;
  
  // Calculate character position offset
  float charWidth = perDraw.scale * 0.5;  // Approximate character width
  float totalWidth = charWidth * perDraw.numChars;
  float xOffset = -totalWidth * 0.5 + charIdx * charWidth;
  
  // Position in world space (above unit)
  float3 worldPos = perDraw.worldPos;
  worldPos.z += 0.7;  // Offset above unit
  
  // Apply billboard transformation
  float2 vertPos = (localVert - float2(0.5, 0.5)) * perDraw.scale;
  vertPos.x += xOffset;
  
  worldPos += camRight * vertPos.x;
  worldPos += camUp * vertPos.y;
  
  // Get UV coordinates for this character
  float4 charUV = perDraw.charUVs[charIdx];
  float2 uv = lerp(charUV.xy, charUV.zw, localVert);
  
  V2F out;
  out.pos = worldToClip(global.view, worldPos);
  out.uv = uv;
  out.charIndex = float(charIdx);
  
  return out;
}

[shader("fragment")]
float4 fragMain(V2F v2f) : SV_Target0
{
  // Sample font texture
  float4 texColor = nameMat.fontTexture.Sample(nameMat.fontSampler, v2f.uv);
  
  // Use alpha channel for text, multiply by color
  float alpha = texColor.a;
  
  // Apply text color
  float3 finalColor = perDraw.color.rgb;
  
  // Add slight darkening for better visibility
  float3 shadowColor = finalColor * 0.3;
  float shadowAlpha = smoothstep(0.3, 0.5, alpha);
  finalColor = lerp(shadowColor, finalColor, shadowAlpha);
  
  return float4(finalColor, alpha * perDraw.color.a);
}