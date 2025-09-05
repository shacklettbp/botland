import shader_host;
import utils;

ParameterBlock<GlobalPassData> global;
ParameterBlock<HealthBarPerDraw> perDraw;

struct V2F {
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
};

[shader("vertex")]
V2F vertMain(uint idx: SV_VertexID)
{
  // Generate multiple quads to form an arc
  // Each segment needs 6 vertices (2 triangles)
  const uint segments = 16; // Number of segments in the arc
  const uint vertsPerSegment = 6;
  
  // Determine which segment and which vertex within the segment
  uint segmentIdx = idx / vertsPerSegment;
  uint localIdx = idx % vertsPerSegment;
  
  // Triangle strip pattern for a quad:
  // 0: bottom-left, 1: bottom-right, 2: top-right
  // 3: bottom-left, 4: top-right, 5: top-left
  float2 quadPattern[6] = {
    float2(0, 0), float2(1, 0), float2(1, 1),
    float2(0, 0), float2(1, 1), float2(0, 1)
  };
  
  float2 localVert = quadPattern[localIdx];
  
  // Calculate position along the arc (0 to 1)
  // For each quad, we need the left edge at segmentIdx/segments
  // and the right edge at (segmentIdx+1)/segments
  float t0 = float(segmentIdx) / float(segments);
  float t1 = float(segmentIdx + 1) / float(segments);
  // localVert.x is 0 for left vertices, 1 for right vertices
  float t = (localVert.x == 1) ? t1 : t0;
  
  // Get camera vectors - we need to project to horizontal plane
  float3x4 camTxfm = global.view.camTxfm;
  float3 camForward = camTxfm[2].xyz;
  
  // Project camera forward to horizontal plane and normalize
  float2 camForwardHorizontal = normalize(camForward.xy);
  // Right vector is perpendicular to forward in horizontal plane
  float2 camRightHorizontal = float2(camForwardHorizontal.y, -camForwardHorizontal.x);
  
  // Arc parameters - 30% of circle = ~108 degrees = 1.88 radians
  // But we want just the bottom portion visible
  float arcAngle = 2.4;
  float startAngle = -arcAngle / 2;
  float angle = startAngle + t * arcAngle;
  
  // Inner and outer radius for thickness
  float innerRadius = 0.4;
  float outerRadius = 0.5;
  float radius = (localVert.y == 1) ? outerRadius : innerRadius;
  
  // Calculate position on the arc (in horizontal plane)
  // X is perpendicular to camera, Y is toward camera (both in horizontal plane)
  float2 arcPos = float2(sin(angle) * radius, -cos(angle) * radius);
  
  // Position relative to unit
  float3 worldPos = perDraw.worldPos;
  // Apply horizontal rotation to face camera
  worldPos.xy += camRightHorizontal * arcPos.x;
  worldPos.xy += camForwardHorizontal * arcPos.y;
  
  V2F out;
  out.pos = worldToClip(global.view, worldPos);
  out.uv = float2(t, localVert.y);
  
  return out;
}

[shader("fragment")]
float4 fragMain(V2F v2f) : SV_Target0
{
  float2 uv = v2f.uv;
  float healthPercent = perDraw.healthPercent;
  
  // Create background and foreground bar areas
  // For the arc, UV.x represents position along the arc
  float barFill = uv.x < healthPercent ? 1.0 : 0.0;
  
  // Color gradient based on health
  float3 healthColor;
  if (healthPercent > 0.6) {
    // Green to yellow
    float t = (healthPercent - 0.6) / 0.4;
    healthColor = lerp(float3(1, 1, 0), float3(0, 1, 0), t);
  } else if (healthPercent > 0.3) {
    // Yellow to orange
    float t = (healthPercent - 0.3) / 0.3;
    healthColor = lerp(float3(1, 0.5, 0), float3(1, 1, 0), t);
  } else {
    // Orange to red
    float t = healthPercent / 0.3;
    healthColor = lerp(float3(1, 0, 0), float3(1, 0.5, 0), t);
  }
  
  // Background color (dark gray)
  float3 bgColor = float3(0.2, 0.2, 0.2);
  
  // Mix foreground and background
  float3 finalColor = barFill > 0.5 ? healthColor : bgColor;
  
  // Add edge smoothing for anti-aliasing
  float edgeSmooth = fwidth(uv.x) * 2.0;
  float healthEdge = smoothstep(healthPercent - edgeSmooth, healthPercent + edgeSmooth, uv.x);
  finalColor = lerp(finalColor, bgColor, healthEdge * (1.0 - barFill));
  
  // Add a subtle outline
  float edgeAlpha = 1.0;
  float edgeDist = min(uv.y, 1.0 - uv.y);
  if (edgeDist < 0.25) {
    finalColor = lerp(finalColor, float3(0.4, 0.4, 0.4), 0.3);
  }
  
  // Apply team color tint (subtle)
  finalColor = lerp(finalColor, perDraw.teamColor, 0.15);
  
  return float4(finalColor, 0.85); // Slightly transparent
}