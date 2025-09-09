#include "viz.hpp"
#include <assert.h>
#include <filesystem>
#include <rt/math.hpp>
#include <cstring>

#include <gas/gas_imgui.hpp>

#include "bot-viz-global-shaders.hpp"
#include "bot-viz-material-shaders.hpp"
#include "shader_host.hpp"

#include "game/import.hpp"

namespace bot {

// Attack type string lookup table
constexpr const char* ATTACK_EFFECT_NAMES[] = {
  "None",
  "Poison Spread",
  "Healing Bloom",
  "Vampiric Bite",
  "Push",
};

constexpr const char* PASSIVE_ABILITY_NAMES[] = {
  "None",
  "Holy Aura",
};

static constexpr inline TextureFormat DEPTH_TEXTURE_FMT =
  TextureFormat::Depth32_Float;

static std::filesystem::path getShaderDir(ShaderByteCodeType type)
{
  std::filesystem::path shader_dir = BOT_SHADERS_OUT_DIR;
  switch (type) {
    case ShaderByteCodeType::SPIRV: shader_dir /= "spirv"; break;
    case ShaderByteCodeType::WGSL: shader_dir /= "wgsl"; break;
    case ShaderByteCodeType::MTLLib: shader_dir /= "mtl"; break;
    case ShaderByteCodeType::DXIL: shader_dir /= "dxil"; break;
    default: BOT_UNREACHABLE(); break;
  }
  
  return shader_dir;
}

static Scene loadModels(GPUDevice *gpu, GPUQueue tx_queue)
{
  namespace fs = std::filesystem;

  AssetImporter importer;

  {
    importer.importAsset(
        (fs::path(BOT_DATA_DIR) / "cube_render.obj").string());
    importer.importAsset(
        (fs::path(BOT_DATA_DIR) / "smooth_sphere_render.obj").string());
    importer.importAsset(
        (fs::path(BOT_DATA_DIR) / "location_effect.obj").string());
  }

  ImportedAssets &assets = importer.getImportedAssets();

  Buffer geo_buffer;
  std::vector<RenderMesh> render_meshes;
  std::vector<RenderObject> render_objects;
  {
    auto &src_objs = assets.objects;

    u32 total_num_geo_bytes = 0;
    for (const auto &src_obj : assets.objects) {
      for (const auto &src_mesh : src_obj.meshes) {
        total_num_geo_bytes = roundUp(total_num_geo_bytes, (u32)sizeof(RenderVertex));
        total_num_geo_bytes += sizeof(RenderVertex) * src_mesh.numVertices;
        total_num_geo_bytes += sizeof(u32) * src_mesh.numFaces * 3;
      }
    }

    using enum BufferUsage;
    
    Buffer staging = gpu->createStagingBuffer(total_num_geo_bytes);
    geo_buffer = gpu->createBuffer({
      .numBytes = total_num_geo_bytes,
      .usage = DrawVertex | DrawIndex | CopyDst,
    });

    u8 *staging_ptr;
    gpu->prepareStagingBuffers(1, &staging, (void **)&staging_ptr);

    u32 cur_buf_offset = 0;
    for (const auto &src_obj : src_objs) {
      render_objects.push_back({(u32)render_meshes.size(), (u32)src_obj.meshes.size()});

      for (const auto &src_mesh : src_obj.meshes) {
        cur_buf_offset = roundUp(cur_buf_offset, (u32)sizeof(RenderVertex));
        u32 vertex_offset = cur_buf_offset / sizeof(RenderVertex);

        RenderVertex *vertex_staging = (RenderVertex *)(staging_ptr + cur_buf_offset);

        for (i32 i = 0; i < (i32)src_mesh.numVertices; i++) {
          vertex_staging[i] = RenderVertex {
            .pos = src_mesh.positions[i],
            .normal = src_mesh.normals == nullptr ? WORLD_UP : src_mesh.normals[i],
            .uv = src_mesh.uvs == nullptr ? Vector2(0, 0) : src_mesh.uvs[i],
          };
        }

        cur_buf_offset += sizeof(RenderVertex) * src_mesh.numVertices;

        u32 index_offset = cur_buf_offset / sizeof(u32);
        u32 *indices_staging = (u32 *)(staging_ptr + cur_buf_offset);

        u32 num_index_bytes = sizeof(u32) * src_mesh.numFaces * 3;
        memcpy(indices_staging, src_mesh.indices, num_index_bytes);
        cur_buf_offset += num_index_bytes;

        render_meshes.push_back({
          .vertexOffset = vertex_offset,
          .indexOffset = index_offset, 
          .numTriangles = src_mesh.numFaces,
          .materialIndex = src_mesh.materialIdx == 0xFFFF'FFFF ? 0 :
            (u32)src_mesh.materialIdx,
        });
      }
    }
    chk(cur_buf_offset == total_num_geo_bytes);

    gpu->flushStagingBuffers(1, &staging);

    gpu->waitUntilReady(tx_queue);

    CommandEncoder upload_enc = gpu->createCommandEncoder(tx_queue);
    upload_enc.beginEncoding();

    CopyPassEncoder copy_enc = upload_enc.beginCopyPass();

    copy_enc.copyBufferToBuffer(staging, geo_buffer, 0, 0, total_num_geo_bytes);

    upload_enc.endCopyPass(copy_enc);
    upload_enc.endEncoding();

    gpu->submit(tx_queue, upload_enc);
    gpu->waitUntilWorkFinished(tx_queue);

    gpu->destroyCommandEncoder(upload_enc);
    gpu->destroyStagingBuffer(staging);
  }

  return {
    .geoBuffer = geo_buffer,
    .renderMeshes = std::move(render_meshes),
    .renderObjects = std::move(render_objects),
  };
}

void Viz::initSwapchain(Surface surface)
{
  SwapchainProperties swapchain_properties;
  swapchain = gpu->createSwapchain(
    surface, { SwapchainFormat::SDR_SRGB, SwapchainFormat::SDR_UNorm },
    &swapchain_properties);
  swapchainFormat = swapchain_properties.format;

  windowWidth = surface.width;
  windowHeight = surface.height;
}

void Viz::cleanupSwapchain()
{
  gpu->destroySwapchain(swapchain);
  swapchainFormat = TextureFormat::None;
  windowWidth = 0;
  windowHeight = 0;
}

void Viz::initSamplers()
{
  bilinearRepeatSampler = gpu->createSampler({
    .addressMode = SamplerAddressMode::Repeat,
  });

  nearestRepeatSampler = gpu->createSampler({
    .addressMode = SamplerAddressMode::Repeat,
    .mipmapFilterMode = SamplerFilterMode::Nearest,
    .magnificationFilterMode = SamplerFilterMode::Nearest,
    .minificationFilterMode = SamplerFilterMode::Nearest,
    .anisotropy = 1,
  });
}

void Viz::cleanupSamplers()
{
  gpu->destroySampler(nearestRepeatSampler);
  gpu->destroySampler(bilinearRepeatSampler);
}

void Viz::initParamBlockTypes()
{
  globalDataPBType = gpu->createParamBlockType({
    .uuid = "global_data_pb_type"_to_uuid,
    .buffers = {
      {
        .type = BufferBindingType::Uniform,
      },
    },
  });

  tonemapPBType = gpu->createParamBlockType({
    .uuid = "final_pass_input_pb_type"_to_uuid,
    .textures = {
      { .shaderUsage = ShaderStage::Fragment },
    },
    .samplers = {
      { .type = SamplerBindingType::Filtering },
    },
  });
}

void Viz::cleanupParamBlockTypes()
{
  gpu->destroyParamBlockType(tonemapPBType);
  gpu->destroyParamBlockType(globalDataPBType);
}

void Viz::initPassInterfaces()
{
  hdrPassInterface = gpu->createRasterPassInterface({
    .uuid = "hdr_raster_pass"_to_uuid,
    .depthAttachment = {
      .format = DEPTH_TEXTURE_FMT,
      .loadMode = AttachmentLoadMode::Clear,
    },
    .colorAttachments = {
      {
        .format = hdrRenderFormat,
        .loadMode = AttachmentLoadMode::Clear,
      },
    },
  });

  finalPassInterface = gpu->createRasterPassInterface({
    .uuid = "final_raster_pass"_to_uuid,
    .depthAttachment = {
      .format = DEPTH_TEXTURE_FMT,
      .loadMode = AttachmentLoadMode::Clear,
    },
    .colorAttachments = {
      {
        .format = swapchainFormat,
        .loadMode = AttachmentLoadMode::Clear,
      },
    },
  });
}

void Viz::cleanupPassInterfaces()
{
  gpu->destroyRasterPassInterface(finalPassInterface);
  gpu->destroyRasterPassInterface(hdrPassInterface);
}

void Viz::initMaterials(Runtime &rt)
{
  using enum VertexFormat;

  VizMaterialShaders shaders;
  StackAlloc alloc;

  std::filesystem::path shader_dir = 
    getShaderDir(gpuLib->backendShaderByteCodeType());

  chk(shaders.load(alloc, (shader_dir / "bot-viz-material-shaders.shader_blob").c_str()));

  ImageImporter importer;

  {
    auto src_tex = importer.importImage(
      (std::filesystem::path(BOT_DATA_DIR) / "checkerboard.png").c_str());

    if (!src_tex.has_value()) {
      FATAL("Failed to load checkerboard");
    }

    materials.boardTexture = gpu->createTexture({
      .format = TextureFormat::RGBA8_SRGB,
      .width = u16(src_tex->width),
      .height = u16(src_tex->height),
      .initData = { .ptr = src_tex->data },
    }, mainQueue);

    gpu->waitUntilWorkFinished(mainQueue);

    materials.boardPBType = gpu->createParamBlockType({
      .uuid = "board_materials_pb_type"_to_uuid,
      .textures = {
        { .shaderUsage = ShaderStage::Fragment },
      },
      .samplers = {
        { .type = SamplerBindingType::Filtering },
      },
    });

    materials.boardPB = gpu->createParamBlock({
      .typeID = materials.boardPBType,
      .textures = { materials.boardTexture },
      .samplers = { nearestRepeatSampler },
    });

    materials.boardShader = gpu->createRasterShader({
      .byteCode = shaders.getByteCode(MaterialShaderID::Board),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = hdrPassInterface,
      .paramBlockTypes = { globalDataPBType, materials.boardPBType },
      .numPerDrawBytes = sizeof(BoardDrawData),
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
      },
    });
  }
  
  {
    materials.genericLocationEffectShader = gpu->createRasterShader({
      .byteCode = shaders.getByteCode(MaterialShaderID::GenericLocationEffect),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = hdrPassInterface,
      .paramBlockTypes = { globalDataPBType },
      .numPerDrawBytes = sizeof(shader::GenericLocationEffectPerDraw),
      .vertexBuffers = {{
        .stride = sizeof(RenderVertex), .attributes = {
          { .offset = offsetof(RenderVertex, pos), .format = Vec3_F32 },
          { .offset = offsetof(RenderVertex, normal),  .format = Vec3_F32 },
          { .offset = offsetof(RenderVertex, uv), .format = Vec2_F32 },
        },
      }},
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
      },
    });
  }

  {
    materials.unitsShader = gpu->createRasterShader({
      .byteCode = shaders.getByteCode(MaterialShaderID::Units),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = hdrPassInterface,
      .paramBlockTypes = { globalDataPBType },
      .numPerDrawBytes = sizeof(UnitsPerDraw),
      .vertexBuffers = {{
        .stride = sizeof(RenderVertex), .attributes = {
          { .offset = offsetof(RenderVertex, pos), .format = Vec3_F32 },
          { .offset = offsetof(RenderVertex, normal),  .format = Vec3_F32 },
          { .offset = offsetof(RenderVertex, uv), .format = Vec2_F32 },
        },
      }},
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
      },
    });

    materials.healthBarShader = gpu->createRasterShader({
      .byteCode = shaders.getByteCode(MaterialShaderID::HealthBar),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = hdrPassInterface,
      .paramBlockTypes = { globalDataPBType },
      .numPerDrawBytes = sizeof(HealthBarPerDraw),
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
        .cullMode = CullMode::None,
        .blending = {
          {
            .colorOp = BlendOperation::Add,
            .srcColorFactor = BlendFactor::SrcAlpha,
            .dstColorFactor = BlendFactor::OneMinusSrcAlpha,
            .alphaOp = BlendOperation::Add,
            .srcAlphaFactor = BlendFactor::One,
            .dstAlphaFactor = BlendFactor::OneMinusSrcAlpha,
          }
        },
      },
    });
  }
  
  {
    // Initialize font atlas
    fontAtlas.init(rt, gpu, mainQueue, 24.0f); // 24px font size
    gpu->waitUntilWorkFinished(mainQueue);
    
    // Create text rendering resources
    materials.namePBType = gpu->createParamBlockType({
      .uuid = "text_materials_pb_type"_to_uuid,
      .textures = {
        { .shaderUsage = ShaderStage::Fragment },
      },
      .samplers = {
        { .type = SamplerBindingType::Filtering },
      },
    });
    
    materials.namePB = gpu->createParamBlock({
      .typeID = materials.namePBType,
      .textures = { fontAtlas.texture },
      .samplers = { bilinearRepeatSampler },
    });
    
    materials.nameShader = gpu->createRasterShader({
      .byteCode = shaders.getByteCode(MaterialShaderID::Name),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = hdrPassInterface,
      .paramBlockTypes = { globalDataPBType, materials.namePBType },
      .numPerDrawBytes = sizeof(shader::NamePerDraw),
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
        .writeDepth = false,
        .cullMode = CullMode::None,
        .blending = {
          {
            .colorOp = BlendOperation::Add,
            .srcColorFactor = BlendFactor::SrcAlpha,
            .dstColorFactor = BlendFactor::OneMinusSrcAlpha,
            .alphaOp = BlendOperation::Add,
            .srcAlphaFactor = BlendFactor::One,
            .dstAlphaFactor = BlendFactor::OneMinusSrcAlpha,
          }
        },
      },
    });
  }
}

void Viz::cleanupMaterials()
{
  {
    gpu->destroyRasterShader(materials.nameShader);
    gpu->destroyParamBlock(materials.namePB);
    gpu->destroyParamBlockType(materials.namePBType);
    fontAtlas.destroy(gpu);
    
    gpu->destroyRasterShader(materials.healthBarShader);
    gpu->destroyRasterShader(materials.genericLocationEffectShader);
    gpu->destroyRasterShader(materials.unitsShader);
  }

  {
    gpu->destroyRasterShader(materials.boardShader);
    gpu->destroyParamBlock(materials.boardPB);
    gpu->destroyParamBlockType(materials.boardPBType);
    gpu->destroyTexture(materials.boardTexture);
  }
}

void Viz::initFrameInputs()
{
  for (i32 i = 0; i < Viz::NUM_FRAMES_IN_FLIGHT; i++) {
    FrameInput &input = frames[i].input;
    input.globalDataBuffer = gpu->createBuffer({
      .numBytes = sizeof(GlobalPassData),
      .usage = BufferUsage::CopyDst | BufferUsage::ShaderUniform,
    });

    input.globalDataPB = gpu->createParamBlock({
      .typeID = globalDataPBType,
      .buffers = {
        {
          .buffer = input.globalDataBuffer
        },
      },
    });
  }
}

void Viz::cleanupFrameInputs()
{
  for (i32 i = 0; i < Viz::NUM_FRAMES_IN_FLIGHT; i++) {
    FrameInput &input = frames[i].input;
    gpu->destroyParamBlock(input.globalDataPB);
    gpu->destroyBuffer(input.globalDataBuffer);
  }
}

void Viz::initRenderFrames()
{
  for (i32 i = 0; i < Viz::NUM_FRAMES_IN_FLIGHT; i++) {
    RenderFrame &frame = frames[i].render;

    frame.depth = gpu->createTexture({
      .format = DEPTH_TEXTURE_FMT,
      .width = (u16)windowWidth,
      .height = (u16)windowHeight,
      .usage = TextureUsage::DepthAttachment,
    });

    frame.hdr = gpu->createTexture({
      .format = hdrRenderFormat,
      .width = (u16)windowWidth,
      .height = (u16)windowHeight,
      .usage = TextureUsage::ColorAttachment | TextureUsage::ShaderSampled,
    });

    frame.hdrPass = gpu->createRasterPass({
      .interface = hdrPassInterface,
      .depthAttachment = frame.depth,
      .colorAttachments = { frame.hdr },
    });

    frame.finalPass = gpu->createRasterPass({
      .interface = finalPassInterface,
      .depthAttachment = frame.depth,
      .colorAttachments = { swapchain.proxyAttachment() },
    });

    frame.tonemapPB = gpu->createParamBlock({
      .typeID = tonemapPBType,
      .textures = {
        frame.hdr,
      },
      .samplers = {
        bilinearRepeatSampler,
      },
    });
  }
}

void Viz::cleanupRenderFrames()
{
  for (i32 i = 0; i < Viz::NUM_FRAMES_IN_FLIGHT; i++) {
    RenderFrame &frame = frames[i].render;

    gpu->destroyParamBlock(frame.tonemapPB);

    gpu->destroyRasterPass(frame.finalPass);
    gpu->destroyRasterPass(frame.hdrPass);
    gpu->destroyTexture(frame.hdr);
    gpu->destroyTexture(frame.depth);
  }
}

void Viz::loadGlobalShaders()
{
  using enum VertexFormat;

  VizGlobalShaders shaders;
  StackAlloc alloc;

  std::filesystem::path shader_dir = 
    getShaderDir(gpuLib->backendShaderByteCodeType());

  chk(shaders.load(alloc, (shader_dir / "bot-viz-global-shaders.shader_blob").c_str()));

  tonemapShader = gpu->createRasterShader({
    .byteCode = shaders.getByteCode(GlobalShaderID::Tonemap),
    .vertexEntry = "vertMain",
    .fragmentEntry = "fragMain",
    .rasterPass = finalPassInterface,
    .paramBlockTypes = { tonemapPBType },
    .numPerDrawBytes = 0,
    .vertexBuffers = {},
    .rasterConfig = {
      .depthCompare = DepthCompare::Disabled,
      .writeDepth = false,
      .cullMode = CullMode::None,
    },
  });
}

void Viz::cleanupGlobalShaders()
{
  gpu->destroyRasterShader(tonemapShader);
}

void Viz::init(RTStateHandle rt_state_hdl, Sim *sim_state,
               GPULib *gpu_lib, GPUDevice *gpu_in, Surface surface)
{
  Runtime rt(rt_state_hdl, 0);

  rtStateHdl = rt_state_hdl;
  sim = sim_state;

  gpuLib = gpu_lib;
  gpu = gpu_in;
  mainQueue = gpu->getMainQueue();

  initSwapchain(surface);

  if ((gpu->getSupportedFeatures() & GPUFeatures::RenderableRG11B10_Float) !=
      GPUFeatures::None) {
    hdrRenderFormat = TextureFormat::RG11B10_Float;
  } else {
    hdrRenderFormat = TextureFormat::RGBA16_Float;
  }

  initSamplers();

  initParamBlockTypes();
  initPassInterfaces();

  loadGlobalShaders();

  initMaterials(rt);

  initFrameInputs();
  initRenderFrames();

  ImGuiSystem::init(gpu, mainQueue, finalPassInterface,
    getShaderDir(gpuLib->backendShaderByteCodeType()).c_str(),
    (std::filesystem::path(BOT_DATA_DIR) / "imgui_font.ttf").c_str(),
    12.f);

  frameEnc = gpu->createCommandEncoder(mainQueue);

  scene = loadModels(gpu, mainQueue);
}

void Viz::shutdown()
{
  gpu->waitUntilWorkFinished(mainQueue);

  gpu->destroyCommandEncoder(frameEnc);

  ImGuiSystem::shutdown(gpu);

  cleanupRenderFrames();
  cleanupFrameInputs();

  cleanupMaterials();

  cleanupGlobalShaders();

  cleanupPassInterfaces();
  cleanupParamBlockTypes();

  cleanupSamplers();

  cleanupSwapchain();

  gpu = nullptr;
  gpuLib = nullptr;
}

void Viz::resize(SimRT &rt, Surface new_surface)
{
  // FIXME
  gpu->waitUntilWorkFinished(mainQueue);

  cleanupRenderFrames();
  cleanupSwapchain();

  initSwapchain(new_surface);
  initRenderFrames();

  render(rt);
}

GridPos Viz::screenToGridPos(Vector2 screenPos)
{
  // Convert screen coordinates (0,0 top-left) to NDC (-1,-1 to 1,1)
  float ndcX = (2.0f * screenPos.x / windowWidth) - 1.0f;
  float ndcY = 1.0f - (2.0f * screenPos.y / windowHeight);
  
  // Calculate aspect ratio and FOV scale
  float aspect_ratio = (float)windowWidth / (float)windowHeight;
  float fov_scale = 1.f / tanf(toRadians(cam.fov * 0.5f));
  
  // Unproject from NDC to view space direction
  Vector3 rayDirView = {
    ndcX * aspect_ratio / fov_scale,
    ndcY / fov_scale,
    1.0f
  };
  
  // Transform ray direction to world space
  Vector3 rayDirWorld = normalize(
    cam.right * rayDirView.x +
    cam.up * rayDirView.y +
    cam.fwd
  );
  
  // Ray-plane intersection with z=0 grid plane
  if (fabsf(rayDirWorld.z) < 0.001f) {
    return { -1, -1 }; // Ray is parallel to the grid plane
  }
  
  float t = -cam.position.z / rayDirWorld.z;
  
  if (t < 0) {
    return { -1, -1 }; // Ray doesn't intersect the grid plane
  }
  
  Vector3 intersection = cam.position + rayDirWorld * t;
  
  // Convert world coordinates to grid coordinates
  i32 gridX = (i32)floorf(intersection.x + 0.5f);
  i32 gridY = (i32)floorf(intersection.y + 0.5f);
  
  // Check bounds
  if (gridX < 0 || gridX >= GRID_SIZE || gridY < 0 || gridY >= GRID_SIZE) {
    return { -1, -1 };
  }
  
  return { gridX, gridY };
}

static UIControl::Flag updateCamera(OrbitCam &cam, UserInput &input,
                                    UserInputEvents &events, float delta_t)
{
  constexpr float MOUSE_SPEED = 1e-1f;
  constexpr float SCROLL_SPEED = 5.f;
  constexpr float CAM_MOVE_SPEED = 5.f;

  UIControl::Flag result = UIControl::None;
  Vector2 mouse_delta = events.mouseDelta();
  Vector2 mouse_scroll = events.mouseScroll();
  Vector3 translate = Vector3::zero();
  
  bool control_camera = input.isDown(InputID::MouseRight) || input.isDown(InputID::Shift);

  if (control_camera) {
    result |= UIControl::RawMouseMode;

    cam.azimuth -= mouse_delta.y * MOUSE_SPEED * delta_t;
    cam.heading += mouse_delta.x * MOUSE_SPEED * delta_t;
    cam.azimuth = std::clamp(cam.azimuth, -0.49f * PI, 0.49f * PI);

    while (cam.heading > PI) {
      cam.heading -= 2.f * PI;
    }
    while (cam.heading < -PI) {
      cam.heading += 2.f * PI;
    }

    if (mouse_scroll.y != 0.f) {
      float zoom_change = -mouse_scroll.y * SCROLL_SPEED * delta_t;
      if (zoom_change < 0.f) {
        cam.zoom /= 1.f - zoom_change;
      }
      if (zoom_change > 0.f) {
        cam.zoom *= 1.f + zoom_change;
      }
    }
  }

  cam.fwd = {
    .x = cosf(cam.heading) * cos(cam.azimuth),
    .y = sinf(cam.heading) * cosf(cam.azimuth),
    .z = -sinf(cam.azimuth),
  };

  cam.right = normalize(cross(cam.fwd, WORLD_UP));
  cam.up = normalize(cross(cam.right, cam.fwd));

  if (control_camera) {
    // Move the focus point.
    if (input.isDown(InputID::W)) {
      translate += cam.up;
    }
    if (input.isDown(InputID::A)) {
      translate -= cam.right;
    }
    if (input.isDown(InputID::S)) {
      translate -= cam.up;
    }
    if (input.isDown(InputID::D)) {
      translate += cam.right;
    }
  }

  cam.target += translate * CAM_MOVE_SPEED * delta_t;
  cam.position = cam.target - cam.fwd * cam.zoom;

  return result;
}

void Viz::buildImguiWidgets()
{
  World *world = sim->activeWorlds[curVizActiveWorld];

  // Fixed Turn Order window (upper-right)
  {
    const float panelWidth = 280.0f;
    const float panelHeight = 220.0f;
    
    ImGui::SetNextWindowPos(ImVec2(float(windowWidth) / 2 - panelWidth, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoBringToFrontOnFocus;
    if (ImGui::Begin("Turn Order", nullptr, flags)) {
      if (world->turnHead) {
        ImGui::Text("Units in turn order:");
        ImGui::Separator();

        UnitID id = world->turnHead;
        for (i32 i = 0; i < world->numAliveUnits; i++) {
          UnitPtr u = world->units.get(id);
          if (!u) break;

          bool isCurrent = (id == world->turnCur);
          ImVec4 teamColor = (u->team == 0) ? ImVec4(1.0f, 0.2f, 0.2f, 1.0f)
                                            : ImVec4(0.2f, 0.2f, 1.0f, 1.0f);
          if (isCurrent) {
            // Slightly brighter for the current unit
            teamColor.x = fminf(teamColor.x + 0.2f, 1.0f);
            teamColor.y = fminf(teamColor.y + 0.2f, 1.0f);
            teamColor.z = fminf(teamColor.z + 0.2f, 1.0f);
          }

          ImGui::TextColored(teamColor, "%s%s: (%d %d)  HP:%d  Speed:%d",
                             isCurrent ? "> " : "  ", u->name.data, u->pos.x, u->pos.y,
                             u->hp, u->speed);

          // Advance in the circular list (guard with count to avoid infinite loop)
          id = u->turnListItem.next ? u->turnListItem.next : UnitID::none();
          if (!id) break;
        }
      } else {
        ImGui::Text("No turn data available.");
      }
    }
    ImGui::End();
  }
  
  // Show unit inspector window if a unit is selected
  if (selectedUnit) {
    UnitPtr unit = world->units.get(selectedUnit);
    if (unit) {
      ImGui::Begin("Unit Inspector");
      
      ImGui::Text("Grid Position: (%d, %d)", unit->pos.x, unit->pos.y);
      ImGui::Separator();
      
      // Team color indicator
      const char* teamName = (unit->team == 0) ? "Red Team" : "Blue Team";
      if (unit->team == 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "%s", teamName);
      } else {
        ImGui::TextColored(ImVec4(0.2f, 0.2f, 1.0f, 1.0f), "%s", teamName);
      }
      
      ImGui::Separator();
      
      // Health bar
      ImGui::Text("Health: %d / %d", unit->hp, DEFAULT_HP);
      float healthPercent = (float)unit->hp / (float)DEFAULT_HP;
      ImGui::ProgressBar(healthPercent, ImVec2(0.0f, 0.0f));
      
      // Speed
      ImGui::Text("Speed: %d", unit->speed);

      // Attack
      ImGui::Text("Attack Range: %d", unit->attackProp.range);
      ImGui::Text("Attack Damage: %d", unit->attackProp.damage);
      ImGui::Text("Attack Effect: %s",
        ATTACK_EFFECT_NAMES[static_cast<u32>(unit->attackProp.effect)]);
      
      ImGui::Text("Passive Ability: %s",
        PASSIVE_ABILITY_NAMES[static_cast<u32>(unit->passiveAbility)]);
      
      ImGui::End();
    } else {
      // Unit no longer exists or is dead, close the inspector
      selectedUnit = UnitID::none();
    }
  }
  
  // Debug window
  ImGui::Begin("Debug");
  ImGui::Text("Click on a unit to inspect its stats");
  if (selectedGridPos.x >= 0 && selectedGridPos.y >= 0) {
    ImGui::Text("Selected cell: (%d, %d)", selectedGridPos.x, selectedGridPos.y);
  }
  ImGui::End();
}

UIControl Viz::runUI(SimRT &rt, UserInput &input, UserInputEvents &events,
                     const char *text_input, float ui_scale, float delta_t)
{
  if (events.downEvent(InputID::K1)) {
    World *world = sim->activeWorlds[curVizActiveWorld];
    destroyWorld(rt, world);
    sim->activeWorlds[curVizActiveWorld] =
      createWorld(rt, (u64)curVizActiveWorld | (u64(numWorldResets++) << 32));
  }

  UIControl ui_ctrl {};
  ui_ctrl.flags |= updateCamera(cam, input, events, delta_t);
  // Handle unit selection on left click (when not camera dragging)
  if (!input.isDown(InputID::MouseRight) && 
      !input.isDown(InputID::Shift)) {
    if (events.downEvent(InputID::MouseLeft)) {
      Vector2 mousePos = input.mousePosition();
      GridPos clickedPos = screenToGridPos(mousePos);
      
      selectedGridPos = clickedPos;
      
      if (clickedPos.x == -1 && clickedPos.y == -1) {
        selectedUnit = UnitID::none();
      } else {
        World *world = sim->activeWorlds[curVizActiveWorld];
        selectedGridPos = clickedPos;
        
        // Check if there's a unit at this position
        Cell& cell = world->grid[clickedPos.y][clickedPos.x];
        if (cell.actorID != GenericID::none() && cell.actorID.type == (i32)ActorType::Unit) {
          // Try to get the unit
          selectedUnit = UnitID::fromGeneric(cell.actorID);
          UnitPtr unit = world->units.get(selectedUnit);
          if (!unit) {
            selectedUnit = UnitID::none();
          }
        } else {
          // No unit at this position
          selectedUnit = UnitID::none();
        }
      }
    }
    
    World *world = sim->activeWorlds[curVizActiveWorld];
    bool player_moved = false;
    if (selectedUnit == world->turnCur) {
      UnitAction playerAction = {};
      if (events.downEvent(InputID::W)) {
        player_moved = true;
        playerAction.move = MoveAction::Up;
      } else if (events.downEvent(InputID::A)) {
        player_moved = true;
        playerAction.move = MoveAction::Left;
      } else if (events.downEvent(InputID::S)) {
        player_moved = true;
        playerAction.move = MoveAction::Down;
      } else if (events.downEvent(InputID::D)) {
        player_moved = true;
        playerAction.move = MoveAction::Right;
      } else if (events.downEvent(InputID::Space)) {
        player_moved = true;
        playerAction.move = MoveAction::Wait;
      }
      
      if (player_moved) {
        stepWorld(rt, world, playerAction);
      }
    }
  }

  ImGuiSystem::UIControl imgui_ctrl = {};
  ImGuiSystem::newFrame(input, events, windowWidth, windowHeight,
                        ui_scale, delta_t, text_input, &imgui_ctrl);

  buildImguiWidgets();

  if ((imgui_ctrl.type & ImGuiSystem::UIControl::EnableIME)) {
    ui_ctrl.flags |= UIControl::EnableIME;
    ui_ctrl.imePos = imgui_ctrl.pos;
    ui_ctrl.imeLineHeight = imgui_ctrl.lineHeight;
  }

  if ((imgui_ctrl.type & ImGuiSystem::UIControl::DisableIME)) {
    ui_ctrl.flags |= UIControl::DisableIME;
  }
  
  #if 0
  World *world = sim->activeWorlds[curVizActiveWorld];
  static float elapsed_time = 0.0f;
  elapsed_time += delta_t;
  if (elapsed_time >= 0.1f) {
    playerAction.move = MoveAction(rand() % (int)MoveAction::NUM_MOVE_ACTIONS);

    stepWorld(rt, world, playerAction);
    
    elapsed_time = 0.0f;
  }
  #endif

  return ui_ctrl;
}

static void stageViewData(OrbitCam &cam,
                          u32 render_width, u32 render_height,
                          GlobalPassData *out)
{
  float aspect_ratio = (float)render_width / (float)render_height;

  float fov_scale = 1.f / tanf(toRadians(cam.fov * 0.5f));

  float screen_x_scale = fov_scale / aspect_ratio;
  float screen_y_scale = fov_scale;

  out->view.camTxfm.rows[0] = Vector4::fromVec3W(cam.right, cam.position.x);
  out->view.camTxfm.rows[1] = Vector4::fromVec3W(cam.up, cam.position.y);
  out->view.camTxfm.rows[2] = Vector4::fromVec3W(cam.fwd, cam.position.z);

  out->view.fbDims = { render_width, render_height };
  out->view.screenScale = Vector2(screen_x_scale, screen_y_scale);
  out->view.zNear = 1.f;
}

static NonUniformScaleObjectTransform computeNonUniformScaleTxfm(
    Vector3 t, Quat r, Diag3x3 s)
{
  float x2 = r.x * r.x;
  float y2 = r.y * r.y;
  float z2 = r.z * r.z;
  float xz = r.x * r.z;
  float xy = r.x * r.y;
  float yz = r.y * r.z;
  float wx = r.w * r.x;
  float wy = r.w * r.y;
  float wz = r.w * r.z;

  float y2z2 = y2 + z2;
  float x2z2 = x2 + z2;
  float x2y2 = x2 + y2;

  Diag3x3 ds = 2.f * s;
  Diag3x3 i_s = 1.f / s;
  Diag3x3 i_ds = 2.f * i_s;

  NonUniformScaleObjectTransform out;
  out.o2w = {{
    { s.d0 - ds.d0 * y2z2, ds.d1 * (xy - wz), ds.d2 * (xz + wy), t.x },
    { ds.d0 * (xy + wz), s.d1 - ds.d1 * x2z2, ds.d2 * (yz - wx), t.y },
    { ds.d0 * (xz - wy), ds.d1 * (yz + wx), s.d2 - ds.d2 * x2y2, t.z },
  }};

  Vector3 w2o_r0 = 
      { i_s.d0 - i_ds.d0 * y2z2, i_ds.d1 * (xy + wz), ds.d2 * (xz - wy) };
  Vector3 w2o_r1 =
      { i_ds.d0 * (xy - wz), i_s.d1 - i_ds.d1 * x2z2, i_ds.d2 * (yz + wx) };
  Vector3 w2o_r2 =
      { i_ds.d0 * (xz + wy), i_ds.d1 * (yz - wx), i_s.d2 - i_ds.d2 * x2y2 };

  out.w2o = {{
    Vector4::fromVec3W(w2o_r0, -dot(w2o_r0, t)),
    Vector4::fromVec3W(w2o_r1, -dot(w2o_r1, t)),
    Vector4::fromVec3W(w2o_r2, -dot(w2o_r2, t)),
  }};

  return out;
}

void Viz::renderGenericLocationEffects(SimRT &rt, FrameState &frame,
                                       RasterPassEncoder &enc)
{ 
  (void)rt;
  
  World *world = sim->activeWorlds[curVizActiveWorld];

  enc.setParamBlock(0, frame.input.globalDataPB);
  enc.setShader(materials.genericLocationEffectShader);
  enc.setVertexBuffer(0, scene.geoBuffer);
  enc.setIndexBufferU32(scene.geoBuffer);
  
  RenderObject obj = scene.renderObjects[2];
  
  for (auto effect : world->locationEffects) {
    Vector4 color = {};
    switch (effect->type) {
      case LocationEffectType::Poison:
        color = Vector4(0.2f, 1.0f, 0.2f, 1.0f);
        break;
      case LocationEffectType::Healing:
        color = Vector4(0.2f, 0.2f, 1.0f, 1.0f);
        break;
      default:
        color = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
        break;
    }

    enc.drawData(shader::GenericLocationEffectPerDraw {
      .txfm = computeNonUniformScaleTxfm(
        { float(effect->pos.x), float(effect->pos.y), 0.15f },
        Quat::id(), { 1.f, 1.f, 0.5f }),
      .color = color,
    });

    for (u32 mesh_idx = 0; mesh_idx < obj.numMeshes; mesh_idx++) {
      RenderMesh mesh = scene.renderMeshes[obj.meshOffset + mesh_idx];
      enc.drawIndexed(mesh.vertexOffset, mesh.indexOffset,
                      mesh.numTriangles);
    }
  }
}

void Viz::renderUnits(SimRT &rt, FrameState &frame, RasterPassEncoder &enc)
{
  (void)rt;

  World *world = sim->activeWorlds[curVizActiveWorld];

  // First pass: render all unit meshes
  enc.setParamBlock(0, frame.input.globalDataPB);
  enc.setShader(materials.unitsShader);
  enc.setVertexBuffer(0, scene.geoBuffer);
  enc.setIndexBufferU32(scene.geoBuffer);
  
  for (auto unit : world->units) {
    assert(unit->hp > 0);
    
    enc.drawData(UnitsPerDraw {
      .txfm = computeNonUniformScaleTxfm(
        { float(unit->pos.x), float(unit->pos.y), 0.5f },
        Quat::id(), { 0.30f, 0.30f, 0.5f }),
      .color = (unit->team == 0) ?
        Vector4(1, 0, 0, 1) :
        Vector4(0, 0, 1, 1),
    });

    RenderObject obj = scene.renderObjects[1];

    for (u32 mesh_idx = 0; mesh_idx < obj.numMeshes; mesh_idx++) {
      RenderMesh mesh = scene.renderMeshes[obj.meshOffset + mesh_idx];
      enc.drawIndexed(mesh.vertexOffset, mesh.indexOffset,
                      mesh.numTriangles);
    }
  }

  // Second pass: render health bars
  enc.setShader(materials.healthBarShader);
  enc.setParamBlock(0, frame.input.globalDataPB);
  
  for (auto unit : world->units) {
    float healthPercent = float(unit->hp) / float(DEFAULT_HP);
    
    Vector3 teamColor = (unit->team == 0) ?
      Vector3(1.0f, 0.2f, 0.2f) :  // Red team
      Vector3(0.2f, 0.2f, 1.0f);   // Blue team
    
    enc.drawData(HealthBarPerDraw {
      .worldPos = { float(unit->pos.x), float(unit->pos.y), 0.01f },
      .healthPercent = healthPercent,
      .teamColor = teamColor,
    });
    
    // Draw triangles for the arc (16 segments * 2 triangles = 32 triangles)
    enc.draw(0, 32);
  }
  
  // Third pass: render unit names
  enc.setShader(materials.nameShader);
  enc.setParamBlock(0, frame.input.globalDataPB);
  enc.setParamBlock(1, materials.namePB);
  
  for (auto unit : world->units) {
    // Prepare text data
    shader::NamePerDraw nameData = {};
    nameData.worldPos = { float(unit->pos.x), float(unit->pos.y), 1.2f };
    nameData.scale = 0.3f; // Adjust scale as needed
    nameData.color = (unit->team == 0) ?
      Vector4(1.0f, 0.2f, 0.2f, 1.0f) :  // Red team
      Vector4(0.2f, 0.2f, 1.0f, 1.0f);   // Blue team
    
    // Convert unit name to UV coordinates
    const char* name = unit->name.data;
    u32 nameLen = strlen(name);
    nameData.numChars = std::min(nameLen, 16_u32);
    
    for (u32 i = 0; i < nameData.numChars; i++) {
      CharInfo charInfo = fontAtlas.getCharInfo(name[i]);
      nameData.charUVs[i] = Vector4(charInfo.x0, charInfo.y0, charInfo.x1, charInfo.y1);
    }
    
    enc.drawData(nameData);
    
    // Draw quads for text (6 vertices per character)
    enc.draw(0, nameData.numChars * 2);
  }
}

void Viz::renderBoard(SimRT &rt, FrameState &frame, RasterPassEncoder &enc)
{
  (void)rt;
  enc.setParamBlock(0, frame.input.globalDataPB);

  enc.setShader(materials.boardShader);
  enc.setParamBlock(1, materials.boardPB);
  
  World *world = sim->activeWorlds[curVizActiveWorld];
  
  std::array<int, 2> curTurnPos = { -1, -1 };

  {
    UnitPtr unit = world->units.get(world->turnCur);
    if (unit) {
      curTurnPos = { unit->pos.x, unit->pos.y };
    }
  }

  enc.drawData(BoardDrawData {
    .gridSize = { GRID_SIZE, GRID_SIZE },
    .curTurnPos = curTurnPos,
    .curSelectPos = { selectedGridPos.x, selectedGridPos.y },
  });

  enc.draw(0, 12);
}

void Viz::render(SimRT &rt)
{
  gpu->waitUntilReady(mainQueue);

  FrameState &frame = frames[curFrameIdx];
  curFrameIdx = (curFrameIdx + 1) % NUM_FRAMES_IN_FLIGHT;
  
  frameEnc.beginEncoding();

  {
    CopyPassEncoder enc = frameEnc.beginCopyPass();

    MappedTmpBuffer global_pass_data_staging = enc.tmpBuffer(
        sizeof(GlobalPassData));

    GlobalPassData *global_pass_data_staging_ptr =
      (GlobalPassData *)global_pass_data_staging.ptr;

    stageViewData(cam, windowWidth, windowHeight, global_pass_data_staging_ptr);

    enc.copyBufferToBuffer(
      global_pass_data_staging.buffer, frame.input.globalDataBuffer,
      global_pass_data_staging.offset, 0, sizeof(GlobalPassData));

    frameEnc.endCopyPass(enc);
  }

  {
    RasterPassEncoder enc = frameEnc.beginRasterPass(frame.render.hdrPass);
    renderBoard(rt, frame, enc);
    renderGenericLocationEffects(rt, frame, enc);
    renderUnits(rt, frame, enc);
    frameEnc.endRasterPass(enc);
  }

  auto [swapchain_tex, swapchain_status] = gpu->acquireSwapchainImage(swapchain);
  assert(swapchain_status == SwapchainStatus::Valid);

  {
    RasterPassEncoder enc = frameEnc.beginRasterPass(frame.render.finalPass);

    enc.setShader(tonemapShader);
    enc.setParamBlock(0, frame.render.tonemapPB);
    enc.draw(0, 1);

    ImGuiSystem::render(enc);

    frameEnc.endRasterPass(enc);
  }

  frameEnc.endEncoding();
  gpu->submit(mainQueue, frameEnc);
  gpu->presentSwapchainImage(swapchain);
}

}
