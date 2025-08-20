#include "wgpu.hpp"
#include "wgpu_init.hpp"

#include <rt/macros.hpp>

#ifndef EMSCRIPTEN
#include <dawn/native/DawnNative.h>
#endif


#ifdef BOT_OS_LINUX
#include "linux.hpp"
#endif

#ifdef BOT_OS_WINDOWS
#include "windows.hpp"
#endif

//#define GAS_WGPU_DEBUG_PRINT (1)

namespace bot::gas::webgpu {

namespace {

class InitDeviceRequest {
public:
  InitDeviceRequest(WebGPULib *lib, i32 device_idx,
                    Span<const Surface> surfaces,
                    void (*cb)(GPUDevice *, void *),
                    void *cb_data);

  void busyWait();
  void release();

private:
  WebGPULib *lib_;
  i32 device_idx_;
  void (*cb_)(GPUDevice *, void *);
  void *cb_data_;
  int ref_count_;

  WGPUAdapter adapter_;
  WGPULimits limits_;
  GPUFeatures supported_features_;

  WGPUFuture adapter_future_;
  WGPUFuture device_future_;

  void requestAdapterCB(WGPURequestAdapterStatus status,
                        WGPUAdapter requested_adapter,
                        WGPUStringView err_msg);
  void requestDeviceCB(WGPURequestDeviceStatus status,
                       WGPUDevice requested_device,
                       WGPUStringView err_msg);
};

#ifndef EMSCRIPTEN
void instanceLoggingCB(wgpu::LoggingType type,
                       wgpu::StringView message,
                       void *userdata)
{
  (void)type;
  (void)userdata;
  fprintf(stderr, " Instance Logging: %s\n", message.data);
}

void deviceLoggingCB(WGPULoggingType type,
                     WGPUStringView message,
                     void *userdata1,
                     void *userdata2)
{
  (void)type;
  (void)userdata1;
  (void)userdata2;
  fprintf(stderr, " Device Logging: %.*s\n", (int)message.length, message.data);
}
#endif

void deviceLostCB(const WGPUDevice *wgpu_dev,
                  WGPUDeviceLostReason lost_reason,
                  WGPUStringView message,
                  void *userdata1,
                  void *userdata2)
{
  (void)lost_reason;
  (void)userdata2;

  WGPUDevice *destroying_device = (WGPUDevice *)userdata1;

  if (*wgpu_dev == *destroying_device) {
    return;
  }

  FATAL("WGPU device lost: %.*s", (int)message.length, message.data);
}

void uncapturedErrorCB(const WGPUDevice *wgpu_dev,
                       WGPUErrorType error_type,
                       WGPUStringView message,
                       void *userdata1,
                       void *userdata2)
{
  (void)wgpu_dev;
  (void)error_type;
  (void)userdata1;
  (void)userdata2;

  fprintf(stderr, " uncaptured error: %.*s\n", (int)message.length, message.data);
  debuggerBreakPoint();
}

inline wgpu::TextureFormat convertTextureFormat(TextureFormat in)
{
  using O = wgpu::TextureFormat;
  using enum TextureFormat;

  switch (in) {
    case None: return O::Undefined;
    case RGBA8_UNorm: return O::RGBA8Unorm;
    case RGBA8_SRGB: return O::RGBA8UnormSrgb;
    case BGRA8_UNorm: return O::BGRA8Unorm;
    case BGRA8_SRGB: return O::BGRA8UnormSrgb;
    case Depth32_Float: return O::Depth32Float;
    case RG11B10_Float: return O::RG11B10Ufloat;
    case RGBA16_Float: return O::RGBA16Float;
    default: BOT_UNREACHABLE();
  }
}

inline TextureFormat convertWebGPUTextureFormat(wgpu::TextureFormat in)
{
  using In = wgpu::TextureFormat;
  using enum TextureFormat;

  switch (in) {
    case In::RGBA8Unorm: return RGBA8_UNorm;
    case In::RGBA8UnormSrgb: return RGBA8_SRGB;
    case In::BGRA8Unorm: return BGRA8_UNorm;
    case In::BGRA8UnormSrgb: return BGRA8_SRGB;
    default: return None;
  }
};

inline wgpu::TextureUsage convertTextureUsage(TextureUsage in)
{
  using O = wgpu::TextureUsage;
  using enum TextureUsage;

  u64 out = (u64)O::None;
  static_assert(sizeof(u64) == sizeof(O));

  if ((in & CopySrc) == CopySrc) {
    out |= (u64)O::CopySrc;
  }

  if ((in & CopyDst) == CopyDst) {
    out |= (u64)O::CopyDst;
  }

  if ((in & ShaderSampled) == ShaderSampled) {
    out |= (u64)O::TextureBinding;
  }

  if ((in & ShaderStorage) == ShaderSampled) {
    out |= (u64)O::StorageBinding;
  }

  if ((in & ColorAttachment) == ColorAttachment) {
    out |= (u64)O::RenderAttachment;
  }

  if ((in & DepthAttachment) == DepthAttachment) {
    out |= (u64)O::RenderAttachment;
  }

  return (O)out;
}

inline wgpu::AddressMode convertSamplerAddressMode(SamplerAddressMode in)
{
  using O = wgpu::AddressMode;
  using enum SamplerAddressMode;

  switch (in) {
    case Clamp: return O::ClampToEdge;
    case Repeat: return O::Repeat;
    case MirrorRepeat: return O::MirrorRepeat;
    case InheritUMode: return O::Undefined;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::FilterMode convertSamplerFilterMode(SamplerFilterMode in)
{
  using O = wgpu::FilterMode;
  using enum SamplerFilterMode;

  switch (in) {
    case Nearest: return O::Nearest;
    case Linear:  return O::Linear;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::MipmapFilterMode
    convertSamplerFilterModeToMipFilterMode(SamplerFilterMode in)
{
  using O = wgpu::MipmapFilterMode;
  using enum SamplerFilterMode;

  switch (in) {
    case Nearest: return O::Nearest;
    case Linear:  return O::Linear;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::BufferUsage convertBufferUsage(BufferUsage in)
{
  using O = wgpu::BufferUsage;
  using enum BufferUsage;

  u64 out = (u64)O::None;
  static_assert(sizeof(u64) == sizeof(O));

  if ((in & CopySrc) == CopySrc) {
    out |= (u64)O::CopySrc;
  }

  if ((in & CopyDst) == CopyDst) {
    out |= (u64)O::CopyDst;
  }

  if ((in & DrawIndex) == DrawIndex) {
    out |= (u64)O::Index;
  }

  if ((in & DrawVertex) == DrawVertex) {
    out |= (u64)O::Vertex;
  }

  if ((in & ShaderUniform) == ShaderUniform) {
    out |= (u64)O::Uniform;
  }

  if ((in & ShaderStorage) == ShaderStorage) {
    out |= (u64)O::Storage;
  }

  return (O)out;
}

inline wgpu::CompareFunction convertDepthCompare(DepthCompare in)
{
  using O = wgpu::CompareFunction;
  using enum DepthCompare;

  switch (in) {
    case GreaterOrEqual: return O::GreaterEqual;
    case LessOrEqual: return O::LessEqual;
    case Disabled: return O::Always;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::CullMode convertCullMode(CullMode in)
{
  using O = wgpu::CullMode;
  using enum CullMode;

  switch (in) {
    case None: return O::None;
    case FrontFace: return O::Front;
    case BackFace: return O::Back;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::LoadOp convertAttachmentLoadMode(AttachmentLoadMode in)
{
  using O = wgpu::LoadOp;
  using enum AttachmentLoadMode;

  switch (in) {
    case Load: return O::Load;
    case Clear: return O::Clear;
    case Undefined: return O::Undefined;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::StoreOp convertAttachmentStoreMode(
    AttachmentStoreMode in)
{
  using O = wgpu::StoreOp;
  using enum AttachmentStoreMode;

  switch (in) {
    case Store: return O::Store;
    case Undefined: return O::Undefined;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::ShaderStage convertShaderStage(ShaderStage in)
{
  using O = wgpu::ShaderStage;
  using enum ShaderStage;

  u64 out = (u64)O::None;
  static_assert(sizeof(u64) == sizeof(O));

  if ((in & Vertex) == Vertex) {
    out |= (u64)O::Vertex;
  }

  if ((in & Fragment) == Fragment) {
    out |= (u64)O::Fragment;
  }

  if ((in & Compute) == Compute) {
    out |= (u64)O::Compute;
  }

  return (O)out;
}

inline wgpu::BufferBindingType convertBufferBindingType(BufferBindingType in)
{
  using O = wgpu::BufferBindingType;
  using enum BufferBindingType;

  switch (in) {
    case Uniform: case DynamicUniform: return O::Uniform;
    case Storage: return O::ReadOnlyStorage;
    case StorageRW: return O::Storage;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::VertexFormat convertVertexFormat(VertexFormat in)
{
  using O = wgpu::VertexFormat;
  using enum VertexFormat;
  switch (in) {
    case Scalar_F32: return O::Float32;
    case Vec2_F32: return O::Float32x2;
    case Vec3_F32: return O::Float32x3;
    case Vec4_UNorm8: return O::Unorm8x4;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::BlendOperation convertBlendOp(BlendOperation in)
{
  using O = wgpu::BlendOperation;
  using enum BlendOperation;

  switch (in) {
    case None: return O::Undefined;
    case Add: return O::Add;
    case Subtract: return O::Subtract;
    default: BOT_UNREACHABLE();
  }
}

inline wgpu::BlendFactor convertBlendFactor(BlendFactor in)
{
  using O = wgpu::BlendFactor;
  using enum BlendFactor;

  switch (in) {
    case Zero: return O::Zero;
    case One: return O::One;
    case Src: return O::Src;
    case OneMinusSrc: return O::OneMinusSrc;
    case SrcAlpha: return O::SrcAlpha;
    case OneMinusSrcAlpha: return O::OneMinusSrcAlpha;
    case Dst: return O::Dst;
    case OneMinusDst: return O::OneMinusDst;
    case DstAlpha: return O::DstAlpha;
    case OneMinusDstAlpha: return O::OneMinusDstAlpha;
    default: BOT_UNREACHABLE();
  }
}

}

GPULib * WebGPULib::init(const APIConfig &cfg)
{
  wgpu::InstanceDescriptor inst_desc {
    .capabilities = {
      .timedWaitAnyEnable = false,
    },
  };

#ifndef EMSCRIPTEN
  dawn::native::DawnInstanceDescriptor dawn_inst_desc;
  if (cfg.enableValidation) {
    dawn_inst_desc.nextInChain = inst_desc.nextInChain;
    inst_desc.nextInChain = &dawn_inst_desc;

    dawn_inst_desc.backendValidationLevel =
      dawn::native::BackendValidationLevel::Full;
    dawn_inst_desc.SetLoggingCallback(instanceLoggingCB, (void *)nullptr);
  }
#endif

  wgpu::Instance instance = wgpu::CreateInstance(&inst_desc);
  auto *api = new WebGPULib();
  api->inst = std::move(instance);
  api->destroyingDevice = nullptr;
  api->debugPipelineCompilation = cfg.debugPipelineCompilation;
  api->errorsAreFatal = cfg.errorsAreFatal;
  return api;
}

void WebGPULib::shutdown()
{
  delete this;
}

Surface WebGPULib::createSurface(void *os_data, i32 width, i32 height)
{
#if defined(BOT_OS_LINUX)
  LinuxWindowHandle &linux_hdl = *(LinuxWindowHandle *)os_data;

  wgpu::SurfaceSourceXlibWindow from_xlib;
  wgpu::SurfaceSourceWaylandSurface from_wayland;
  wgpu::SurfaceDescriptor surface_desc;

  switch (linux_hdl.backend) {
    case LinuxWindowHandle::Backend::X11: {
      from_xlib.display = linux_hdl.x11.display;
      from_xlib.window = linux_hdl.x11.window;

      surface_desc = wgpu::SurfaceDescriptor {
        .nextInChain = &from_xlib,
        .label = nullptr,
      };
    } break;
    case LinuxWindowHandle::Backend::Wayland: {
      from_wayland.display = linux_hdl.wayland.display;
      from_wayland.surface = linux_hdl.wayland.surface;

      surface_desc = wgpu::SurfaceDescriptor {
        .nextInChain = &from_wayland,
        .label = nullptr,
      };
    } break;
    default: BOT_UNREACHABLE();
  }
#elif defined(BOT_OS_MACOS)
  wgpu::SurfaceSourceMetalLayer from_metal({
    .nextInChain = nullptr,
    .layer = os_data,
  });
  
  wgpu::SurfaceDescriptor surface_desc {
    .nextInChain = &from_metal,
    .label = nullptr,
  };
#elif defined(BOT_OS_WINDOWS)
  Win32WindowHandle &win32_hdl = *(Win32WindowHandle *)os_data;

  wgpu::SurfaceSourceWindowsHWND from_windows_hwnd({
    .nextInChain = nullptr,
    .hinstance = win32_hdl.hinstance,
    .hwnd = win32_hdl.hwnd,
  });
  
  wgpu::SurfaceDescriptor surface_desc {
    .nextInChain = &from_windows_hwnd,
    .label = nullptr,
  };
#elif defined(EMSCRIPTEN)
  const char *selector = (const char *)os_data;

  wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas_desc {};
  canvas_desc.selector = selector;

  wgpu::SurfaceDescriptor surface_desc {
    .nextInChain = &canvas_desc,
    .label = nullptr,
  };
#else
  static_assert(false, "Unimplemented");
#endif

  wgpu::Surface surface = inst.CreateSurface(&surface_desc);
  return Surface {
    .hdl = {
      .ptr = surface.MoveToCHandle(),
    },
    .width = width,
    .height = height,
  };
}

void WebGPULib::destroySurface(Surface surface)
{
  wgpuSurfaceRelease((WGPUSurface)surface.hdl.ptr);
} 

static WGPUWaitStatus busyWaitForFuture(WGPUInstance inst, WGPUFuture future)
{
#ifdef EMSCRIPTEN
  (void)inst;
  (void)future;
  FATAL("Cannot busy wait on web");
#else
  WGPUFutureWaitInfo wait_info {
    .future = future,
    .completed = false,
  };

  WGPUWaitStatus wait_status;
  while ((wait_status = wgpuInstanceWaitAny(inst, 1, &wait_info, 0)) ==
          WGPUWaitStatus_TimedOut)
  {}

  return wait_status;
#endif
}

InitDeviceRequest::InitDeviceRequest(WebGPULib *lib, i32 device_idx,
                                     Span<const Surface> surfaces,
                                     void (*cb)(GPUDevice *, void *),
                                     void *cb_data)
{
  lib_ = lib;
  device_idx_ = device_idx;
  cb_ = cb;
  cb_data_ = cb_data;
  ref_count_ = 1;

  // Cannot select specific GPU in webgpu
  chk(device_idx_ == 0);
  chk(surfaces.size() <= 1); // Can only have one compatible surface

  WGPURequestAdapterOptions request_options {};
  request_options.powerPreference = WGPUPowerPreference_HighPerformance;

  if (surfaces.size() == 1) {
    request_options.compatibleSurface = (WGPUSurface)surfaces[0].hdl.ptr;
  }

  auto request_adapter_wrapper = [](
    WGPURequestAdapterStatus status, WGPUAdapter adapter,
    WGPUStringView err_msg, void *request_raw_ptr, void *)
  {
    InitDeviceRequest *request = (InitDeviceRequest *)request_raw_ptr;
    request->requestAdapterCB(status, adapter, err_msg);
  };

  ref_count_++;
  adapter_future_ = wgpuInstanceRequestAdapter(
    lib->inst.Get(), &request_options, WGPURequestAdapterCallbackInfo {
      .nextInChain = nullptr,
      .mode = WGPUCallbackMode_AllowSpontaneous,
      .callback = request_adapter_wrapper,
      .userdata1 = this,
      .userdata2 = nullptr,
    });
}

void InitDeviceRequest::busyWait()
{
  WGPUWaitStatus wait_status = busyWaitForFuture(lib_->inst.Get(), adapter_future_);
  if (wait_status != WGPUWaitStatus_Success) {
    FATAL("Requesting adapter failed during wait");
  }

  wait_status = busyWaitForFuture(lib_->inst.Get(), device_future_);
  if (wait_status != WGPUWaitStatus_Success) {
    FATAL("Requesting device failed during wait");
  }
}

void InitDeviceRequest::release()
{
  ref_count_--;
  if (ref_count_ == 0) {
    delete this;
  }
}

void InitDeviceRequest::requestAdapterCB(
    WGPURequestAdapterStatus status, WGPUAdapter requested_adapter,
    WGPUStringView err_msg)
{
  if (status != WGPURequestAdapterStatus_Success) {
    FATAL("Requesting adapter failed: %.*s", err_msg.length, err_msg.data);
  }

  adapter_ = requested_adapter;

  limits_.nextInChain = nullptr;
  WGPUStatus limits_status = wgpuAdapterGetLimits(adapter_, &limits_);
  if (limits_status != WGPUStatus_Success) {
    FATAL("Failed to get supported limits from adapter");
  }
  if (limits_.maxUniformBufferBindingSize > 65536) {
    limits_.maxUniformBufferBindingSize = 65536;
  }

  supported_features_ = GPUFeatures::None;
  WGPUSupportedFeatures features;
  wgpuAdapterGetFeatures(adapter_, &features);
  for (size_t i = 0; i < features.featureCount; i++) {
    WGPUFeatureName feature = features.features[i];
    switch (feature) {
      case WGPUFeatureName_RG11B10UfloatRenderable: {
        supported_features_ |= GPUFeatures::RenderableRG11B10_Float;
      } break;
      default: break;
    }
  }
  wgpuSupportedFeaturesFreeMembers(features);

  WGPUDeviceDescriptor dev_desc {};
  dev_desc.requiredLimits = &limits_;

  auto required_features = std::to_array<WGPUFeatureName>({
    WGPUFeatureName_RG11B10UfloatRenderable,
  });
  if ((supported_features_ & GPUFeatures::RenderableRG11B10_Float) !=
      GPUFeatures::None) {
    dev_desc.requiredFeatureCount = required_features.size();
    dev_desc.requiredFeatures = required_features.data();
  }

#ifndef EMSCRIPTEN
  WGPUDawnTogglesDescriptor dev_toggles_desc {};
  auto toggles = std::to_array({"dump_shaders"});
  dev_toggles_desc.enabledToggleCount = 1;
  dev_toggles_desc.enabledToggles = toggles.data();

  if (lib_->debugPipelineCompilation) {
    dev_desc.nextInChain = &dev_toggles_desc.chain;
  }
#endif

  dev_desc.deviceLostCallbackInfo = WGPUDeviceLostCallbackInfo {
    .nextInChain = nullptr,
    .mode = WGPUCallbackMode_AllowSpontaneous,
    .callback = deviceLostCB,
    .userdata1 = &lib_->destroyingDevice,
    .userdata2 = nullptr,
  };

  dev_desc.uncapturedErrorCallbackInfo = WGPUUncapturedErrorCallbackInfo {
    .nextInChain = nullptr,
    .callback = uncapturedErrorCB,
    .userdata1 = nullptr,
    .userdata2 = nullptr,
  };

  auto request_device_wrapper =  [](WGPURequestDeviceStatus status,
                                    WGPUDevice requested_device,
                                    WGPUStringView message,
                                    void *request_raw_ptr, void *)
  {
    InitDeviceRequest *request = (InitDeviceRequest *)request_raw_ptr;
    request->requestDeviceCB(status, std::move(requested_device), message);
  };

  device_future_ = wgpuAdapterRequestDevice(
    adapter_, &dev_desc, WGPURequestDeviceCallbackInfo {
      .nextInChain = nullptr,
      .mode = WGPUCallbackMode_AllowSpontaneous,
      .callback = request_device_wrapper,
      .userdata1 = this,
      .userdata2 = nullptr,
    });
}
  
void InitDeviceRequest::requestDeviceCB(WGPURequestDeviceStatus status,
                                        WGPUDevice device,
                                        WGPUStringView message)
{
  if (status != WGPURequestDeviceStatus_Success) {
    FATAL("Requesting device failed: %.*s", message.length, message.data);
  }

#ifndef EMSCRIPTEN
  wgpuDeviceSetLoggingCallback(device, WGPULoggingCallbackInfo {
    .nextInChain = nullptr,
    .callback = deviceLoggingCB,
    .userdata1 = nullptr,
    .userdata2 = nullptr,
  });
#endif

  WGPUQueue queue = wgpuDeviceGetQueue(device);

  BackendLimits backend_limits {
    .maxNumUniformBytes = (u32)limits_.maxUniformBufferBindingSize ,
  };

  Backend *backend = new Backend(adapter_, device, queue,
    lib_->inst, backend_limits, supported_features_, lib_->errorsAreFatal);

  auto cb = cb_;
  void *cb_data = cb_data_;
  release();

  cb(backend, cb_data);
}

GPUDevice * WebGPULib::createDevice(
    i32 gpu_idx, Span<const Surface> surfaces)
{
  GPUDevice *out;

  auto cb = [](GPUDevice *dev, void *out_ptr) {
    *(GPUDevice **)out_ptr = dev;
  };

  InitDeviceRequest request(this, gpu_idx, surfaces, cb, (void *)&out);
  request.busyWait();
  // Don't call release here because the request is stack allocated

  return out;
}

void WebGPULib::createDeviceAsync(
    i32 gpu_idx, Span<const Surface> surfaces,
    void (*cb)(GPUDevice *, void *), void *cb_data)
{
  auto *request = new InitDeviceRequest(this, gpu_idx, surfaces, cb, cb_data);
  // Unfortunately InitDeviceRequest has to be reference counted, because on
  // native all the callbacks actually happen directly in the InitDeviceRequest constructor,
  // while on web they're called later.
  request->release();
}

void WebGPULib::destroyDevice(GPUDevice *gpu)
{
  auto wgpu_backend = static_cast<Backend *>(gpu);
  wgpu_backend->destroy();

  // Hacky, the device lost callback has a pointer
  // to the destroyingDevice member in the WebGPULib class,
  // which it checks to see if we're manually destroying this device.
  destroyingDevice = wgpu_backend->dev.Get();

  delete wgpu_backend;

  destroyingDevice = nullptr;
}

ShaderByteCodeType WebGPULib::backendShaderByteCodeType()
{
  return ShaderByteCodeType::WGSL;
}

Backend::Backend(wgpu::Adapter &&adapter_in,
                 wgpu::Device &&dev_in,
                 wgpu::Queue &&queue_in,
                 wgpu::Instance &inst_in,
                 BackendLimits &limits_in,
                 GPUFeatures supported_features,
                 bool errors_are_fatal)
  : BackendCommon(supported_features, errors_are_fatal),
    adapter(std::move(adapter_in)),
    dev(std::move(dev_in)),
    queue(std::move(queue_in)),
    inst(inst_in),
    limits(limits_in)
{
  {
    for (i32 i = 0; i < MAX_TMP_STAGING_BUFFERS; i++) {
      stagingBelt.cbStates[i] = { this, i };
    }
    stagingBelt.numFree = 0;
    stagingBelt.numAllocated = 0;
    stagingBelt.stagingBufferHandlesBase =
        buffers.reserveRows(MAX_TMP_STAGING_BUFFERS);

  }

  {
    wgpu::BindGroupLayoutEntry layout_entry {
      .binding = 0,
      .visibility = wgpu::ShaderStage((u64)wgpu::ShaderStage::Vertex |
                                      (u64)wgpu::ShaderStage::Fragment |
                                      (u64)wgpu::ShaderStage::Compute),
      .buffer = wgpu::BufferBindingLayout {
        .type = wgpu::BufferBindingType::Uniform,
        .hasDynamicOffset = true,
        .minBindingSize = limits.maxNumUniformBytes,
      },
    };

    wgpu::BindGroupLayoutDescriptor layout_desc {
      .entryCount = 1,
      .entries = &layout_entry,
    };

    tmpDynamicUniformLayout = dev.CreateBindGroupLayout(&layout_desc);
  }

  for (BackendQueueData &queue_data : queueDatas) {
    GPUTmpInputState &gpu_tmp_input = queue_data.gpuTmpInput;
    gpu_tmp_input.curTmpStagingRange = 0;
    gpu_tmp_input.curTmpInputRange = 0;
    gpu_tmp_input.maxNumUsedTmpGPUBuffers = 0;

    gpu_tmp_input.tmpBufferHandlesBase =
        buffers.reserveRows(MAX_TMP_BUFFERS_PER_QUEUE);

    TmpParamBlockState &tmp_param_block_state = queue_data.tmpParamBlockState;
    tmp_param_block_state.numLive = 0;
    tmp_param_block_state.baseHandleOffset =
        paramBlocks.reserveRows(MAX_TMP_PARAM_BLOCKS_PER_QUEUE);
  }

  // Pre allocate a tmp data block for the main queue
  {
    GPUTmpInputState &gpu_tmp_input = queueDatas[0].gpuTmpInput;
    allocGPUTmpBuffer(gpu_tmp_input, 0);
  }

}

void Backend::destroy()
{
  for (BackendQueueData &queue_data : queueDatas) {
    GPUTmpInputState &gpu_tmp_input = queue_data.gpuTmpInput;

    for (i32 i = 0; i < (i32)gpu_tmp_input.maxNumUsedTmpGPUBuffers; i++) {
      auto [to_buffer, _, id] = buffers.get(
          gpu_tmp_input.tmpBufferHandlesBase, i);
      to_buffer->Destroy();
      to_buffer->~Buffer();
    }

    buffers.releaseRows(
      gpu_tmp_input.tmpBufferHandlesBase, MAX_TMP_BUFFERS_PER_QUEUE);

    TmpParamBlockState &tmp_param_block_state = queue_data.tmpParamBlockState;
    chk(tmp_param_block_state.numLive == 0);
    paramBlocks.releaseRows(tmp_param_block_state.baseHandleOffset,
                            MAX_TMP_PARAM_BLOCKS_PER_QUEUE);
  }

  chk(stagingBelt.numFree == stagingBelt.numAllocated);
  for (i32 i = 0; i < stagingBelt.numAllocated; i++) {
    stagingBelt.buffers[i].Unmap();
    stagingBelt.buffers[i].Destroy();

    auto [to_buffer, _, id] = buffers.get(
        stagingBelt.stagingBufferHandlesBase, i);

    to_buffer->~Buffer();
  }

  buffers.releaseRows(stagingBelt.stagingBufferHandlesBase,
      MAX_TMP_STAGING_BUFFERS);
}

void Backend::createGPUResources(i32 num_buffers,
                                 const BufferInit *buffer_inits,
                                 Buffer *buffer_handles_out,
                                 i32 num_textures,
                                 const TextureInit *texture_inits,
                                 Texture *texture_handles_out,
                                 GPUQueue tx_queue)
{
  u32 buffer_tbl_offset;
  if (num_buffers > 0) {
    buffer_tbl_offset = buffers.reserveRows(num_buffers);
    if (buffer_tbl_offset == AllocOOM) [[unlikely]] {
      reportError(ErrorStatus::TableFull);
      return;
    }
  }

  u32 texture_tbl_offset;
  if (num_textures > 0) {
    texture_tbl_offset = textures.reserveRows(num_textures);
    if (texture_tbl_offset == AllocOOM) [[unlikely]] {
      if (num_buffers > 0) {
        buffers.releaseRows(buffer_tbl_offset, num_buffers);
      }
      reportError(ErrorStatus::TableFull);
      return;
    }
  }

  wgpu::CommandEncoder upload_enc;
  GPUTmpMemBlock staging_block {};
  if (tx_queue.id != -1) {
    upload_enc = dev.CreateCommandEncoder();
    waitUntilReady(tx_queue);
  }

  auto stagingAlloc = [this, tx_queue, &staging_block]
    (u32 num_bytes) -> StagingHandle 
  {
    if (num_bytes > GPUTmpMemBlock::BLOCK_SIZE) {
      return {
        .buffer = {},
        .offset = 0,
        .ptr = nullptr,
      }; 
    }
    
    // FIXME move to validation
    chk(tx_queue.id != -1);

    u32 offset = staging_block.alloc(num_bytes, 4);
    if (staging_block.blockFull()) {
      staging_block = allocGPUTmpStagingBlock(tx_queue);
      staging_block.offset += num_bytes;
      offset = 0;
    }

    return {
      staging_block.buffer,
      offset,
      staging_block.ptr + offset,
    };
  };

  for (i32 buf_idx = 0; buf_idx < num_buffers; buf_idx++) {
    const BufferInit &buf_init = buffer_inits[buf_idx];

    auto [to_hot, to_cold, id] = buffers.get(buffer_tbl_offset, buf_idx);

    wgpu::BufferUsage wgpu_usage = convertBufferUsage(buf_init.usage);

    StagingHandle staging = buf_init.initData;
    if (staging.ptr) {
      wgpu_usage |= wgpu::BufferUsage::CopyDst;
    }

    wgpu::BufferDescriptor buf_desc {
      .usage = wgpu_usage,
      .size = buf_init.numBytes,
      .mappedAtCreation = false,
    };

    new (to_hot) wgpu::Buffer(dev.CreateBuffer(&buf_desc));
    *to_cold = buf_init;
    buffer_handles_out[buf_idx] = id;

    if (staging.ptr) {
      if (staging.buffer.null()) {
        staging = stagingAlloc(buf_init.numBytes);
        memcpy(staging.ptr, buf_init.initData.ptr, buf_init.numBytes);
      }

      upload_enc.CopyBufferToBuffer(*buffers.hot(staging.buffer), 0,
                                    *to_hot, 0, buf_init.numBytes);
    }
  }

  for (i32 tex_idx = 0; tex_idx < num_textures; tex_idx++) {
    const TextureInit &tex_init = texture_inits[tex_idx];

    wgpu::TextureUsage wgpu_usage = convertTextureUsage(tex_init.usage);

    wgpu::TextureDimension dim;
    if (tex_init.depth == 0) {
      if (tex_init.height == 0) {
        dim = wgpu::TextureDimension::e1D;
      } else {
        dim = wgpu::TextureDimension::e2D;
      }
    } else {
      dim = wgpu::TextureDimension::e3D;
    }

    u32 width = (u32)tex_init.width;
    u32 height = tex_init.height != 0 ? (u32)tex_init.height : 1;
    u32 depth = tex_init.depth != 0 ? (u32)tex_init.depth : 1;

    StagingHandle staging = tex_init.initData;
    if (staging.ptr) {
      wgpu_usage |= wgpu::TextureUsage::CopyDst;
    }

    wgpu::TextureDescriptor tex_desc {
      .usage = wgpu_usage,
      .dimension = dim,
      .size = {
        .width = width,
        .height = height,
        .depthOrArrayLayers = depth,
      },
      .format = convertTextureFormat(tex_init.format),
      .mipLevelCount = (u32)tex_init.numMipLevels,
      .sampleCount = 1,
      .viewFormatCount = 0,
      .viewFormats = nullptr,
    };

    auto [to_hot, to_cold, id] = textures.get(texture_tbl_offset, tex_idx);

    wgpu::Texture wgpu_tex = dev.CreateTexture(&tex_desc);

    wgpu::TextureViewDescriptor view_desc {
      .format = tex_desc.format,
    };

    wgpu::TextureView wgpu_tex_view = wgpu_tex.CreateView(&view_desc);

    u32 bytes_per_texel = bytesPerTexelForFormat(tex_init.format);
    new (to_cold) BackendTextureCold {
      .texture = std::move(wgpu_tex),
      .baseWidth = width,
      .baseHeight = height,
      .baseDepth = depth,
      .numBytesPerTexel = bytes_per_texel,
    };

    new (to_hot) BackendTexture {
      .view = std::move(wgpu_tex_view),
    };

    texture_handles_out[tex_idx] = id;

    chk(tex_init.numMipLevels == 1);
    u32 num_bytes = width * height * depth * bytes_per_texel;

    if (staging.ptr) {
      if (staging.buffer.null()) {
        staging = stagingAlloc(num_bytes);
        memcpy(staging.ptr, tex_init.initData.ptr, num_bytes);
      }

      u32 cur_offset = staging.offset;
      u32 mip_idx = 0;

      wgpu::TexelCopyBufferInfo src {
        .layout = {
          .offset = cur_offset,
          .bytesPerRow = width * bytes_per_texel,
        },
        .buffer = *buffers.hot(staging.buffer),
      };

      if (depth > 1) {
        src.layout.rowsPerImage = height;
      }

      wgpu::TexelCopyTextureInfo dst {
        .texture = to_cold->texture,
        .mipLevel = mip_idx,
      };

      wgpu::Extent3D copy_size {
        .width = width,
        .height = height,
        .depthOrArrayLayers = depth,
      };

      upload_enc.CopyBufferToTexture(&src, &dst, &copy_size);
    }
  }

  if (tx_queue.id != -1) {
    GPUTmpInputState &gpu_tmp_input = queueDatas[tx_queue.id].gpuTmpInput;
    unmapActiveStagingBuffers(gpu_tmp_input);

    wgpu::CommandBuffer cmd_buf = upload_enc.Finish();
    queue.Submit(1, &cmd_buf);

    mapActiveStagingBuffers(gpu_tmp_input);

    gpu_tmp_input.curTmpStagingRange = 0;
  }
}

void Backend::destroyGPUResources(i32 num_buffers,
                                  const Buffer *buffer_hdls,
                                  i32 num_textures,
                                  const Texture *texture_hdls)
{
  buffers.releaseResources(num_buffers, buffer_hdls,
    [](wgpu::Buffer *to_buf, BufferInit *)
  {
    to_buf->Destroy();
    to_buf->~Buffer();
  });

  textures.releaseResources(num_textures, texture_hdls,
    [](BackendTexture *to_hot, BackendTextureCold *to_cold)
  {
    to_hot->~BackendTexture();

    to_cold->texture.Destroy();
    to_cold->~BackendTextureCold();
  });
}

Buffer Backend::createStagingBuffer(u32 num_bytes)
{
  u32 tbl_offset = buffers.reserveRows(1);
  if (tbl_offset == AllocOOM) [[unlikely]] {
    reportError(ErrorStatus::TableFull);
    return {};
  }

  wgpu::BufferDescriptor buf_desc {
    .usage = (wgpu::BufferUsage)(
        (u64)wgpu::BufferUsage::MapWrite |
        (u64)wgpu::BufferUsage::CopySrc),
    .size = num_bytes,
    .mappedAtCreation = true,
  };

  auto [to_hot, to_cold, id] = buffers.get(tbl_offset, 0);

  new (to_hot) wgpu::Buffer(dev.CreateBuffer(&buf_desc));
  *to_cold = {};

  return id;
}

void Backend::destroyStagingBuffer(Buffer staging)
{
  buffers.releaseResources(1, &staging,
    [](wgpu::Buffer *to_buf, auto)
  {
    to_buf->Destroy();
    to_buf->~Buffer();
  });
}

void Backend::prepareStagingBuffers(i32 num_buffers,
                                    Buffer *buffer_hdls,
                                    void **mapped_out)
{
  for (i32 buf_idx = 0; buf_idx < num_buffers; buf_idx++) {
    Buffer hdl = buffer_hdls[buf_idx];
    wgpu::Buffer *to_buffer = buffers.hot(hdl);
    if (!to_buffer) [[unlikely]] {
      reportError(ErrorStatus::NullBuffer);
      continue;
    }

    if (to_buffer->GetMapState() == wgpu::BufferMapState::Unmapped) {
      // FIXME
      auto map_cb = [](wgpu::MapAsyncStatus, char const *, void *) {};

      wgpu::Future map_future = to_buffer->MapAsync(
          wgpu::MapMode::Write, 0, WGPU_WHOLE_MAP_SIZE,
          wgpu::CallbackMode::WaitAnyOnly, map_cb, (void *)nullptr);

      WGPUWaitStatus map_wait_status = busyWaitForFuture(inst.Get(), map_future);
      chk(map_wait_status == WGPUWaitStatus_Success);
    }

    mapped_out[buf_idx] = to_buffer->GetMappedRange();
  }
}

void Backend::flushStagingBuffers(i32 num_buffers, Buffer *buffer_hdls)
{
  for (i32 buf_idx = 0; buf_idx < num_buffers; buf_idx++) {
    Buffer hdl = buffer_hdls[buf_idx];
    wgpu::Buffer *to_buffer = buffers.hot(hdl);
    if (!to_buffer) [[unlikely]] {
      reportError(ErrorStatus::NullBuffer);
      continue;
    }

    to_buffer->Unmap();
  }
}

Buffer Backend::createReadbackBuffer(u32 num_bytes)
{
  u32 tbl_offset = buffers.reserveRows(1);
  if (tbl_offset == AllocOOM) [[unlikely]] {
    reportError(ErrorStatus::TableFull);
    return {};
  }

  wgpu::BufferDescriptor buf_desc {
    .usage = (wgpu::BufferUsage)(
        (u64)wgpu::BufferUsage::MapRead |
        (u64)wgpu::BufferUsage::CopyDst),
    .size = num_bytes,
  };

  auto [to_hot, to_cold, id] = buffers.get(tbl_offset, 0);

  new (to_hot) wgpu::Buffer(dev.CreateBuffer(&buf_desc));
  *to_cold = {};

  return id;
}

void Backend::destroyReadbackBuffer(Buffer buffer)
{
  buffers.releaseResources(1, &buffer,
    [](wgpu::Buffer *to_buf, auto)
  {
    to_buf->Destroy();
    to_buf->~Buffer();
  });
}

void * Backend::beginReadback(Buffer buffer)
{
  wgpu::Buffer *to_buffer = buffers.hot(buffer);

  wgpu::MapAsyncStatus map_status;
  auto map_cb = [](wgpu::MapAsyncStatus status, char const *,
                   wgpu::MapAsyncStatus *out)
  {
    *out = status;
  };

  wgpu::Future map_future = to_buffer->MapAsync(
      wgpu::MapMode::Read, 0, WGPU_WHOLE_MAP_SIZE,
      wgpu::CallbackMode::WaitAnyOnly, map_cb, &map_status);

  WGPUWaitStatus map_wait_status = busyWaitForFuture(inst.Get(), map_future);
  if (map_wait_status != WGPUWaitStatus_Success) {
    FATAL("Failed to wait while mapping readback buffer: %lu",
          (u64)map_wait_status);
  }

  if (map_status != wgpu::MapAsyncStatus::Success) {
    FATAL("Failed to map readback buffer: %lu",
          (u64)map_wait_status);
  }

  chk(to_buffer->GetMapState() == wgpu::BufferMapState::Mapped);

  return (void *)to_buffer->GetConstMappedRange();
}

void Backend::endReadback(Buffer buffer)
{
  buffers.hot(buffer)->Unmap();
}
 
Buffer Backend::createStandaloneBuffer(BufferInit init, bool external_export)
{
  (void)init;
  (void)external_export;
  FATAL("Unimplemented");
}

void Backend::destroyStandaloneBuffer(Buffer buffer)
{
  (void)buffer;
  FATAL("Unimplemented");
}

Texture Backend::createStandaloneTexture(TextureInit init)
{
  (void)init;
  FATAL("Unimplemented");
}

void Backend::destroyStandaloneTexture(Texture texture)
{
  (void)texture;
  FATAL("Unimplemented");
}


void Backend::createSamplers(i32 num_samplers,
                             const SamplerInit *sampler_inits,
                             Sampler *handles_out)
{
  u32 tbl_offset = samplers.reserveRows(num_samplers);
  if (tbl_offset == AllocOOM) [[unlikely]] {
    reportError(ErrorStatus::TableFull);
    return;
  }

  for (i32 sampler_idx = 0; sampler_idx < num_samplers; sampler_idx++) {
    const SamplerInit &sampler_init = sampler_inits[sampler_idx];

    auto [to_out, _, id] = samplers.get(tbl_offset, sampler_idx);

    wgpu::SamplerDescriptor sampler_descriptor;

    sampler_descriptor.addressModeU =
      convertSamplerAddressMode(sampler_init.addressModeU);

    if (sampler_init.addressModeV == SamplerAddressMode::InheritUMode) {
      sampler_descriptor.addressModeV = sampler_descriptor.addressModeU;
    } else {
      sampler_descriptor.addressModeV =
          convertSamplerAddressMode(sampler_init.addressModeV);
    }

    if (sampler_init.addressModeW == SamplerAddressMode::InheritUMode) {
      sampler_descriptor.addressModeW = sampler_descriptor.addressModeU;
    } else {
      sampler_descriptor.addressModeW =
          convertSamplerAddressMode(sampler_init.addressModeW);
    }

    sampler_descriptor.magFilter =
        convertSamplerFilterMode(sampler_init.magnificationFilterMode);
    sampler_descriptor.minFilter =
        convertSamplerFilterMode(sampler_init.minificationFilterMode);
    sampler_descriptor.mipmapFilter =
        convertSamplerFilterModeToMipFilterMode(sampler_init.mipmapFilterMode);
    sampler_descriptor.lodMinClamp = sampler_init.mipClamp[0];
    sampler_descriptor.lodMaxClamp = sampler_init.mipClamp[1];
    sampler_descriptor.maxAnisotropy = sampler_init.anisotropy;

    new (to_out) wgpu::Sampler(dev.CreateSampler(&sampler_descriptor));

    handles_out[sampler_idx] = id;
  }
}

void Backend::destroySamplers(i32 num_samplers, Sampler *handles)
{
  samplers.releaseResources(num_samplers, handles,
    [](wgpu::Sampler *to_sampler, auto)
  {
    to_sampler->~Sampler();
  });
}

void Backend::createParamBlockTypes(
    i32 num_types,
    const ParamBlockTypeInit *blk_types,
    ParamBlockType *handles_out)
{
  u32 tbl_offset = paramBlockTypes.reserveRows(num_types);
  if (tbl_offset == AllocOOM) [[unlikely]] {
    reportError(ErrorStatus::TableFull);
    return;
  }

  for (i32 type_idx = 0; type_idx < num_types; type_idx++) {
    const ParamBlockTypeInit &type_init = blk_types[type_idx];

    auto [to_block_type, to_uuid, id] =
      paramBlockTypes.get(tbl_offset, type_idx);

    i32 auto_binding_idx = 0;
    wgpu::BindGroupLayoutEntry layout_entries[MAX_BINDINGS_PER_GROUP];

    const i32 num_buffer_bindings = (i32)type_init.buffers.size();
    const i32 num_texture_bindings = (i32)type_init.textures.size();
    const i32 num_sampler_bindings = (i32)type_init.samplers.size();

    i32 out_binding_idx = 0;
    for (i32 buffer_binding_idx = 0;
         buffer_binding_idx < num_buffer_bindings;
         buffer_binding_idx++) {
      const BufferBindingConfig buffer_cfg =
        type_init.buffers[buffer_binding_idx];
      chk(buffer_cfg.numBuffers == 1);

      i32 binding = buffer_cfg.bindLocation;
      if (binding == -1) {
        binding = auto_binding_idx;
      }
      auto_binding_idx = binding + 1;

      layout_entries[out_binding_idx++] = wgpu::BindGroupLayoutEntry {
        .binding = (u32)binding,
        .visibility = convertShaderStage(buffer_cfg.shaderUsage),
        .buffer = wgpu::BufferBindingLayout {
          .type = convertBufferBindingType(buffer_cfg.type),
          .hasDynamicOffset =
              buffer_cfg.type == BufferBindingType::DynamicUniform,
          .minBindingSize = 0,
        },
      };
    }

    for (i32 texture_binding_idx = 0;
         texture_binding_idx < num_texture_bindings;
         texture_binding_idx++) {
      const TextureBindingConfig texture_cfg =
        type_init.textures[texture_binding_idx];
      chk(texture_cfg.numTextures == 1);

      i32 binding = texture_cfg.bindLocation;
      if (binding == -1) {
        binding = auto_binding_idx;
      }
      auto_binding_idx = binding + 1;

      using enum TextureBindingType;

      wgpu::TextureSampleType sample_type;
      wgpu::TextureViewDimension tex_dim;
      switch (texture_cfg.type) {
        case Texture1D: {
          sample_type = wgpu::TextureSampleType::Float;
          tex_dim = wgpu::TextureViewDimension::e1D;
        } break;
        case Texture2D: {
          sample_type = wgpu::TextureSampleType::Float;
          tex_dim = wgpu::TextureViewDimension::e2D;
        } break;
        case Texture3D: {
          sample_type = wgpu::TextureSampleType::Float;
          tex_dim = wgpu::TextureViewDimension::e3D;
        } break;
        case DepthTexture2D: {
          sample_type = wgpu::TextureSampleType::Depth;
          tex_dim = wgpu::TextureViewDimension::e2D;
        } break;
        case UnfilterableTexture2D: {
          sample_type = wgpu::TextureSampleType::UnfilterableFloat;
          tex_dim = wgpu::TextureViewDimension::e2D;
        } break;
        default: BOT_UNREACHABLE();
      }

      layout_entries[out_binding_idx++] = wgpu::BindGroupLayoutEntry {
        .binding = (u32)binding,
        .visibility = convertShaderStage(texture_cfg.shaderUsage),
        .texture = wgpu::TextureBindingLayout {
          .sampleType = sample_type,
          .viewDimension = tex_dim,
        },
      };
    }

    for (i32 sampler_binding_idx = 0;
         sampler_binding_idx < num_sampler_bindings;
         sampler_binding_idx++) {
      const SamplerBindingConfig sampler_cfg =
        type_init.samplers[sampler_binding_idx];
      chk(sampler_cfg.numSamplers == 1);

      i32 binding = sampler_cfg.bindLocation;
      if (binding == -1) {
        binding = auto_binding_idx;
      }
      auto_binding_idx = binding + 1;

      wgpu::SamplerBindingType wgpu_sampler_binding_type;
      using enum SamplerBindingType;
      switch (sampler_cfg.type) {
        case Filtering: {
          wgpu_sampler_binding_type = wgpu::SamplerBindingType::Filtering;
        } break;
        case Comparison: {
          wgpu_sampler_binding_type = wgpu::SamplerBindingType::Comparison;
        } break;
        case NonFiltering: {
          wgpu_sampler_binding_type = wgpu::SamplerBindingType::NonFiltering;
        } break;
        default: BOT_UNREACHABLE();
      }

      layout_entries[out_binding_idx++] = wgpu::BindGroupLayoutEntry {
        .binding = (u32)binding,
        .visibility = convertShaderStage(sampler_cfg.shaderUsage),
        .sampler = wgpu::SamplerBindingLayout {
          .type = wgpu_sampler_binding_type,
        },
      };
    }

    wgpu::BindGroupLayoutDescriptor layout_descriptor;
    layout_descriptor.entryCount = size_t(out_binding_idx);
    layout_descriptor.entries = layout_entries;

    new (to_block_type) BackendParamBlockType {
      .layout = dev.CreateBindGroupLayout(&layout_descriptor),
    };

    *to_uuid = ParamBlockTypeID(type_init.uuid);

    paramBlockTypeIDs.insert(type_init.uuid, id.id);

    handles_out[type_idx] = id;
  }
}

void Backend::destroyParamBlockTypes(i32 num_types,
                                     ParamBlockType *handles)
{
  paramBlockTypes.releaseResources(num_types, handles,
    [this](BackendParamBlockType *to_param_block_type,
           ParamBlockTypeID *to_uuid)
  {
    to_param_block_type->~BackendParamBlockType();
    paramBlockTypeIDs.remove(to_uuid->asUUID());
  });
}

void Backend::createParamBlocks(i32 num_blks,
                               const ParamBlockInit *blk_inits,
                               ParamBlock *handles_out)
{
  u32 tbl_offset = paramBlocks.reserveRows(num_blks);
  if (tbl_offset == AllocOOM) [[unlikely]] {
    reportError(ErrorStatus::TableFull);
    return;
  }

  for (i32 blk_idx = 0; blk_idx < num_blks; blk_idx++) {
    const ParamBlockInit &init = blk_inits[blk_idx];

    auto [to_group, _, id] = paramBlocks.get(tbl_offset, blk_idx);

    new (to_group) wgpu::BindGroup(createBindGroup(init));

    handles_out[blk_idx] = id;
  }
}

void Backend::destroyParamBlocks(i32 num_blks, ParamBlock *blks)
{
  paramBlocks.releaseResources(num_blks, blks,
    [](wgpu::BindGroup *to_group, auto)
  {
    to_group->~BindGroup();
  });
}

ParamBlock Backend::createTemporaryParamBlock(
    GPUQueue queue_hdl,
    ParamBlockInit init)
{
  TmpParamBlockState &tmp_state = queueDatas[queue_hdl.id].tmpParamBlockState;

  i32 tmp_idx = AtomicU32Ref(tmp_state.numLive).fetch_add_relaxed(1);
  chk(tmp_idx < MAX_TMP_PARAM_BLOCKS_PER_QUEUE);
  auto [to_group, _, id] = paramBlocks.get(tmp_state.baseHandleOffset, tmp_idx);

  new (to_group) wgpu::BindGroup(createBindGroup(init));

  return id;
}

wgpu::BindGroup Backend::createBindGroup(ParamBlockInit init)
{
  wgpu::BindGroupLayout layout =
    getBindGroupLayoutByParamBlockTypeID(init.typeID);

  wgpu::BindGroupEntry entries[MAX_BINDINGS_PER_GROUP];

  i32 entry_idx = 0;
  for (BufferBinding binding : init.buffers) {
    wgpu::Buffer *to_buf = buffers.hot(binding.buffer);
    entries[entry_idx] = wgpu::BindGroupEntry {
      .binding = (u32)entry_idx,
      .buffer = *to_buf,
      .offset = (u64)binding.offset,
      .size = binding.numBytes == 0xFFFF'FFFF ?
        WGPU_WHOLE_SIZE : (u64)binding.numBytes,
    };

    entry_idx += 1;
  }

  for (Texture texture : init.textures) {
    BackendTexture *to_tex = textures.hot(texture);
    entries[entry_idx] = wgpu::BindGroupEntry {
      .binding = (u32)entry_idx,
      .textureView = to_tex->view,
    };
    entry_idx += 1;
  }

  for (Sampler sampler : init.samplers) {
    wgpu::Sampler *to_sampler = samplers.hot(sampler);
    entries[entry_idx] = wgpu::BindGroupEntry {
      .binding = (u32)entry_idx,
      .sampler = *to_sampler,
    };
    entry_idx += 1;
  }

  wgpu::BindGroupDescriptor descriptor {
    .layout = layout,
    .entryCount = size_t(
      init.buffers.size() +
      init.textures.size() +
      init.samplers.size()),
    .entries = entries,
  }; 

  return wgpu::BindGroup(dev.CreateBindGroup(&descriptor));
}

void Backend::createRasterPassInterfaces(
    i32 num_interfaces,
    const RasterPassInterfaceInit *interface_inits,
    RasterPassInterface *handles_out)
{
  u32 tbl_offset = rasterPassInterfaces.reserveRows(num_interfaces);
  if (tbl_offset == AllocOOM) [[unlikely]] {
    reportError(ErrorStatus::TableFull);
    return;
  }

  for (i32 interface_idx = 0; interface_idx < num_interfaces;
       interface_idx++) {
    const RasterPassInterfaceInit &interface_init =
      interface_inits[interface_idx];

    auto [to_cfg, to_uuid, id] =
      rasterPassInterfaces.get(tbl_offset, interface_idx);

    const DepthAttachmentConfig &depth_cfg = interface_init.depthAttachment;
    to_cfg->depthAttachment = {
      .format = convertTextureFormat(depth_cfg.format),
      .loadOp = convertAttachmentLoadMode(depth_cfg.loadMode),
      .storeOp = convertAttachmentStoreMode(depth_cfg.storeMode),
      .clearValue = depth_cfg.clearValue,
    };

    i32 num_color_attachments = (i32)interface_init.colorAttachments.size();
    for (i32 i = 0; i < num_color_attachments; i++) {
      const ColorAttachmentConfig &color_cfg =
          interface_init.colorAttachments[i];
      to_cfg->colorAttachments[i] = {
        .format = convertTextureFormat(color_cfg.format),
        .loadOp = convertAttachmentLoadMode(color_cfg.loadMode),
        .storeOp = convertAttachmentStoreMode(color_cfg.storeMode),
        .clearValue = color_cfg.clearValue,
      };
    }

    to_cfg->numColorAttachments = num_color_attachments;

    *to_uuid = RasterPassInterfaceID(interface_init.uuid);

    rasterPassInterfaceIDs.insert(interface_init.uuid, id.id);

    handles_out[interface_idx] = id;
  }
}

void Backend::destroyRasterPassInterfaces(
    i32 num_interfaces, RasterPassInterface *handles)
{
  rasterPassInterfaces.releaseResources(num_interfaces, handles,
    [this](auto, RasterPassInterfaceID *to_uuid)
  {
    rasterPassInterfaceIDs.remove(to_uuid->asUUID());
  });
}

void Backend::createRasterPasses(
    i32 num_passes,
    const RasterPassInit *pass_inits,
    RasterPass *handles_out)
{
  u32 tbl_offset = rasterPasses.reserveRows(num_passes);
  if (tbl_offset == AllocOOM) [[unlikely]] {
    reportError(ErrorStatus::TableFull);
    return;
  }

  for (i32 pass_idx = 0; pass_idx < num_passes; pass_idx++) {
    const RasterPassInit &pass_init = pass_inits[pass_idx];

    auto [out, _, id] = rasterPasses.get(tbl_offset, pass_idx);

    new (out) BackendRasterPass {};

    const BackendRasterPassConfig *cfg =
      rasterPassInterfaces.hot(pass_init.interface);

    if (!pass_init.depthAttachment.null()) {
      out->depthAttachment = {
        .view = textures.hot(pass_init.depthAttachment)->view,
        .loadOp = cfg->depthAttachment.loadOp,
        .storeOp = cfg->depthAttachment.storeOp,
        .clearValue = cfg->depthAttachment.clearValue,
      };
    } else {
      chk(cfg->depthAttachment.format == wgpu::TextureFormat::Undefined);
    }

    out->numColorAttachments = (i32)pass_init.colorAttachments.size();
    out->swapchainAttachmentIndex = -1;
    for (i32 i = 0; i < (i32)pass_init.colorAttachments.size(); i++) {
      const BackendColorAttachmentConfig &attach_cfg =
          cfg->colorAttachments[i];
      BackendColorAttachment &out_attach = out->colorAttachments[i];

      Texture tex_hdl = pass_init.colorAttachments[i];
      if (tex_hdl.gen == 0) {
        chk(out->swapchainAttachmentIndex == -1);
        out->swapchainAttachmentIndex = i;
        out->swapchain = { tex_hdl.id };
      } else {
        out_attach.view = textures.hot(tex_hdl)->view;
      }
      out_attach.loadOp = attach_cfg.loadOp;
      out_attach.storeOp = attach_cfg.storeOp;
      out_attach.clearValue = attach_cfg.clearValue;
    }

    handles_out[pass_idx] = id;
  }
}

void Backend::destroyRasterPasses(
    i32 num_passes, RasterPass *handles)
{
  rasterPasses.releaseResources(num_passes, handles,
    [](BackendRasterPass *to_pass, auto)
  {
    to_pass->~BackendRasterPass();
  });
}

void Backend::createRasterShaders(i32 num_shaders,
                                  const RasterShaderInit *shader_inits,
                                  RasterShader *handles_out)
{
  u32 tbl_offset = rasterShaders.reserveRows(num_shaders);
  if (tbl_offset == AllocOOM) [[unlikely]] {
    reportError(ErrorStatus::TableFull);
    return;
  }

  for (i32 shader_idx = 0; shader_idx < num_shaders; shader_idx++) {
    const RasterShaderInit &shader_init = shader_inits[shader_idx];
    const RasterHWConfig &raster_cfg = shader_init.rasterConfig;

    wgpu::ShaderSourceWGSL wgsl_desc {{
      .nextInChain = nullptr,
      .code = wgpu::StringView((const char *)shader_init.byteCode.data,
        (size_t)shader_init.byteCode.numBytes),
    }};
    wgpu::ShaderModuleDescriptor shader_mod_desc {
      .nextInChain = &wgsl_desc,
    };

    wgpu::ShaderModule shader_mod = dev.CreateShaderModule(&shader_mod_desc);

    wgpu::PrimitiveState primitive_state {
      .cullMode = convertCullMode(raster_cfg.cullMode),
    };

    const BackendRasterPassConfig *pass_cfg = 
        getRasterPassConfigByID(shader_init.rasterPass);

    std::array<
        std::array<wgpu::VertexAttribute,
                   MAX_VERTEX_ATTRIBUTES>,
        MAX_VERTEX_BUFFERS_PER_SHADER> vertex_attributes;

    std::array<wgpu::VertexBufferLayout, MAX_VERTEX_BUFFERS_PER_SHADER>
        vertex_buffers;

    const i32 num_vertex_buffers = (i32)shader_init.vertexBuffers.size();
    chk(num_vertex_buffers < MAX_VERTEX_BUFFERS_PER_SHADER);
    for (i32 vbuf_idx = 0; vbuf_idx < num_vertex_buffers; vbuf_idx++) {
      const VertexBufferConfig &vbuf_cfg = shader_init.vertexBuffers[vbuf_idx];
      wgpu::VertexBufferLayout &out_layout = vertex_buffers[vbuf_idx];
      out_layout.arrayStride = vbuf_cfg.stride;
      out_layout.stepMode = wgpu::VertexStepMode::Vertex;
      
      wgpu::VertexAttribute *out_attrs = vertex_attributes[vbuf_idx].data();
      const i32 num_attrs = vbuf_cfg.attributes.size();
      chk(num_attrs < MAX_VERTEX_ATTRIBUTES);
      for (i32 i = 0; i < num_attrs; i++) {
        VertexAttributeConfig attr_cfg = vbuf_cfg.attributes[i];
        out_attrs[i] = {
          .format = convertVertexFormat(attr_cfg.format),
          .offset = attr_cfg.offset,
          .shaderLocation = (u32)i,
        };
      }

      out_layout.attributeCount = (size_t)num_attrs;
      out_layout.attributes = out_attrs;
    }

    wgpu::VertexState vert_state {
      .module = shader_mod,
      .entryPoint = shader_init.vertexEntry,
      .constantCount = 0,
      .constants = nullptr,
      .bufferCount = (size_t)num_vertex_buffers,
      .buffers = vertex_buffers.data(),
    };

    wgpu::DepthStencilState depth_state {
      .format = pass_cfg->depthAttachment.format,
      .depthWriteEnabled = raster_cfg.writeDepth,
      .depthCompare = convertDepthCompare(raster_cfg.depthCompare),
      .depthBias = raster_cfg.depthBias,
      .depthBiasSlopeScale = raster_cfg.depthBiasSlope,
      .depthBiasClamp = raster_cfg.depthBiasClamp,
    };

    wgpu::ColorTargetState color_tgt_states[MAX_COLOR_ATTACHMENTS];
    
    for (i32 i = 0; i < pass_cfg->numColorAttachments; i++) {
      const BackendColorAttachmentConfig &attach_cfg  =
          pass_cfg->colorAttachments[i];
      color_tgt_states[i] = {
        .format = attach_cfg.format,
      };
    }

    chk(raster_cfg.blending.size() == 0 ||
      raster_cfg.blending.size() == pass_cfg->numColorAttachments);

    wgpu::BlendState blend_states[MAX_COLOR_ATTACHMENTS];

    for (i32 i = 0; i < raster_cfg.blending.size(); i++) {
      BlendingConfig blend_cfg = raster_cfg.blending[i];
      wgpu::BlendState &blend_out = blend_states[i];
      blend_out.color.operation = convertBlendOp(blend_cfg.colorOp);
      blend_out.color.srcFactor = convertBlendFactor(blend_cfg.srcColorFactor);
      blend_out.color.dstFactor = convertBlendFactor(blend_cfg.dstColorFactor);
      blend_out.alpha.operation = convertBlendOp(blend_cfg.alphaOp);
      blend_out.alpha.srcFactor = convertBlendFactor(blend_cfg.srcAlphaFactor);
      blend_out.alpha.dstFactor = convertBlendFactor(blend_cfg.dstAlphaFactor);

      if (blend_out.color.operation != wgpu::BlendOperation::Undefined &&
          blend_out.alpha.operation != wgpu::BlendOperation::Undefined) {
        color_tgt_states[i].blend = &blend_out;
      }
    }

    wgpu::FragmentState frag_state {
      .module = shader_mod,
      .entryPoint = shader_init.fragmentEntry,
      .constantCount = 0,
      .constants = nullptr,
      .targetCount = (size_t)pass_cfg->numColorAttachments,
      .targets = color_tgt_states,
    };

    wgpu::BindGroupLayout bind_group_layouts[MAX_BIND_GROUPS_PER_SHADER];

    i32 num_bind_groups = shader_init.paramBlockTypes.size();
    for (i32 i = 0; i < num_bind_groups; i++) {
      bind_group_layouts[i] =
        getBindGroupLayoutByParamBlockTypeID(shader_init.paramBlockTypes[i]);
    }

    i32 per_draw_bind_group_slot = -1;
    if (shader_init.numPerDrawBytes > 0) {
      per_draw_bind_group_slot = num_bind_groups++;
      bind_group_layouts[per_draw_bind_group_slot] = tmpDynamicUniformLayout;
    }

    wgpu::PipelineLayoutDescriptor layout_descriptor {
      .bindGroupLayoutCount = (size_t)num_bind_groups,
      .bindGroupLayouts = bind_group_layouts,
    };

    wgpu::PipelineLayout pipeline_layout =
      dev.CreatePipelineLayout(&layout_descriptor);

    wgpu::RenderPipelineDescriptor pipeline_desc {
      .layout = std::move(pipeline_layout),
      .vertex = vert_state,
      .primitive = primitive_state,
      .depthStencil =
          pass_cfg->depthAttachment.format == wgpu::TextureFormat::Undefined ?
              nullptr : &depth_state,
      .fragment = &frag_state,
    };

    wgpu::RenderPipeline pipeline = dev.CreateRenderPipeline(&pipeline_desc);

    auto [out, _, id] = rasterShaders.get(tbl_offset, shader_idx);
    new (out) BackendRasterShader {
      .pipeline = std::move(pipeline),
      .perDrawBindGroupSlot = per_draw_bind_group_slot,
    };
    handles_out[shader_idx] = id;
  }
}

void Backend::destroyRasterShaders(i32 num_shaders, RasterShader *handles)
{
  rasterShaders.releaseResources(num_shaders, handles,
    [](BackendRasterShader *to_shader, auto)
  {
    to_shader->~BackendRasterShader();
  });
}

Swapchain Backend::createSwapchain(Surface surface,
                                   Span<const SwapchainFormat> format_preferences,
                                   SwapchainProperties *properties)
{
  wgpu::Surface wgpu_surface((WGPUSurface)surface.hdl.ptr);

  wgpu::SurfaceCapabilities capabilities;
  wgpu_surface.GetCapabilities(adapter, &capabilities);
  bool supports_copy_dst =
    ((u64)capabilities.usages & (u64)wgpu::TextureUsage::CopyDst) != 0;

  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;

  for (i64 pref_idx = 0; pref_idx < format_preferences.size(); pref_idx++) {
    SwapchainFormat desired_fmt = format_preferences[pref_idx];

    for (i64 format_capability_idx = 0;
         format_capability_idx < (i64)capabilities.formatCount;
         format_capability_idx++) {
      wgpu::TextureFormat capable_format = capabilities.formats[format_capability_idx];
      bool valid = false;

      switch (desired_fmt) {
        case SwapchainFormat::SDR_UNorm: {
          switch (capable_format) {
            case wgpu::TextureFormat::RGBA8Unorm:
            case wgpu::TextureFormat::BGRA8Unorm:
              valid = true;
            default: break;
          }
        } break;
        case SwapchainFormat::SDR_SRGB: {
          switch (capable_format) {
            case wgpu::TextureFormat::RGBA8UnormSrgb:
            case wgpu::TextureFormat::BGRA8UnormSrgb:
              valid = true;
            default: break;
          }
        } break;
        case SwapchainFormat::HDR_10B: {
          switch (capable_format) {
            case wgpu::TextureFormat::RGB10A2Uint:
            case wgpu::TextureFormat::RGB10A2Unorm:
              valid = true;
            default: break;
          }
        } break;
        case SwapchainFormat::HDR_16B: {
          switch (capable_format) {
            case wgpu::TextureFormat::RGBA16Float:
              valid = true;
            default: break;
          }
        } break;
        default: BOT_UNREACHABLE();
      }

      if (valid) {
        format = capable_format;
        break;
      }
    }

    if (format != wgpu::TextureFormat::Undefined) {
      break;
    }
  }

  if (format == wgpu::TextureFormat::Undefined) {
    FATAL("Swapchain cannot support any requested format type");
  }

  wgpu::TextureUsage swapchain_usage = wgpu::TextureUsage::RenderAttachment;
  if (supports_copy_dst) {
    swapchain_usage |= wgpu::TextureUsage::CopyDst;
  }

  wgpu::SurfaceConfiguration surface_cfg {
    .device = dev,
    .format = format,
    .usage = swapchain_usage,
    .width = (u32)surface.width,
    .height = (u32)surface.height,
    .viewFormatCount = 0,
    .viewFormats = nullptr,
    .alphaMode = wgpu::CompositeAlphaMode::Opaque,
    .presentMode = wgpu::PresentMode::Fifo,
  };
  wgpu_surface.Configure(&surface_cfg);

  i32 swapchain_idx = swapchains.popFromFreeList();

  Texture reserved_hdl;
  {
    u32 tbl_row = textures.reserveRows(1);
    if (tbl_row == AllocOOM) {
      FATAL("Out of texture handles while creating swapchain");
    }

    auto [to_hot, to_cold, id] = textures.get(tbl_row, 0);

    new (to_cold) BackendTextureCold {};
    new (to_hot) BackendTexture {};
    reserved_hdl = id;
  }

  TextureFormat mapped_fmt = convertWebGPUTextureFormat(format);

  if (mapped_fmt == TextureFormat::None) {
    FATAL("Unknown / unsupported swapchain format: 0x%X", (u32)format);
  }

  *properties = {
    .format = mapped_fmt,
    .supportsCopyDst = supports_copy_dst,
  };

  new (&swapchains[swapchain_idx]) BackendSwapchain {
    .surface = std::move(wgpu_surface),
    .view = {},
    .reservedHandle = reserved_hdl,
  };

  return Swapchain { swapchain_idx };
}

void Backend::destroySwapchain(Swapchain swapchain)
{
  i32 swapchain_idx = swapchain.id;

  BackendSwapchain &wgpu_swapchain = swapchains[swapchain.id];
  chk(wgpu_swapchain.view == nullptr);

  {
    textures.releaseResources(1, &wgpu_swapchain.reservedHandle,
      [](BackendTexture *to_hot, BackendTextureCold *to_cold)
    {
      chk(to_hot->view == nullptr &&
             to_cold->texture == nullptr);

      to_hot->~BackendTexture();
      to_cold->~BackendTextureCold();
    });
  }

  wgpu_swapchain.surface.Unconfigure();
  wgpu_swapchain.~BackendSwapchain();

  swapchains.pushToFreeList(swapchain_idx);
}

AcquireSwapchainResult Backend::acquireSwapchainImage(Swapchain swapchain)
{
  BackendSwapchain &wgpu_swapchain = swapchains[swapchain.id];

  wgpu::SurfaceTexture surface_tex;
  wgpu_swapchain.surface.GetCurrentTexture(&surface_tex);

  bool optimal = surface_tex.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal;
  bool suboptimal = surface_tex.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal;

  if (!optimal && !suboptimal) [[unlikely]] {
    return {
      .texture = {},
      .status = SwapchainStatus::Invalid,
    };
  }

  wgpu::Texture tex = surface_tex.texture;
  wgpu::TextureView view = tex.CreateView();

  wgpu_swapchain.view = view;

  auto [to_hot, to_cold, id] =
    textures.get(wgpu_swapchain.reservedHandle.id, 0);
  to_cold->texture = std::move(tex);
  to_hot->view = std::move(view);

  return {
    .texture = wgpu_swapchain.reservedHandle,
    .status = suboptimal ?
      SwapchainStatus::Suboptimal : SwapchainStatus::Valid,
  };
}

void Backend::presentSwapchainImage(Swapchain swapchain)
{
  BackendSwapchain &wgpu_swapchain = swapchains[swapchain.id];

#ifndef EMSCRIPTEN
  wgpu_swapchain.surface.Present();
#endif

  wgpu_swapchain.view = nullptr;

  auto [to_hot, to_cold, id] =
    textures.get(wgpu_swapchain.reservedHandle.id, 0);
  to_hot->view = nullptr;
  to_cold->texture = nullptr;
}

void Backend::waitUntilReady(GPUQueue)
{
  inst.ProcessEvents();
}

void Backend::waitUntilWorkFinished(GPUQueue)
{
  inst.ProcessEvents();
  // Essentially a no-op on webgpu
}

void Backend::waitUntilIdle()
{
  inst.ProcessEvents();

  wgpu::QueueWorkDoneStatus queue_status;
  auto workDoneCB = [&queue_status](wgpu::QueueWorkDoneStatus status_in)
  {
    queue_status = status_in;
  };

  wgpu::Future future = queue.OnSubmittedWorkDone(
      wgpu::CallbackMode::WaitAnyOnly, workDoneCB);

  WGPUWaitStatus wait_status = busyWaitForFuture(inst.Get(), future);
  if (wait_status != WGPUWaitStatus_Success) {
    FATAL("WebGPU backend waitUntilIdle: error while waiting for work done callback: %lu", (u64)wait_status);
  }

  if (queue_status != wgpu::QueueWorkDoneStatus::Success) {
    FATAL("WebGPU backend waitUntilIdle: work done callback failure: %lu",
         (u64)queue_status);
  }

  inst.ProcessEvents();
}

ShaderByteCodeType Backend::backendShaderByteCodeType()
{
  return ShaderByteCodeType::WGSL;
}

// GPUTmpInputState::MAX_BUFFERS frontend buffer handles are reserved starting
// at GPUTmpInputState::bufferHandlesBase  -1 for these temporary buffers.
// This lets us abuse the return value of this function to both refer to the
// actual buffer handle and the pre created bind group that refers to these
// full buffers.
GPUTmpMemBlock Backend::allocGPUTmpStagingBlock(GPUQueue queue_hdl)
{
  BackendQueueData &queue_data = queueDatas[queue_hdl.id];
  GPUTmpInputState &state = queue_data.gpuTmpInput;

  AtomicU64Ref staging_range_atomic(state.curTmpStagingRange);

  while (true) {
    u64 offset_range = staging_range_atomic.fetch_add<sync::acq_rel>(1);
    u32 global_offset = (u32)offset_range;
    u32 range_end = u32(offset_range >> 32);

    if (global_offset < range_end) [[likely]] {
      u32 buf_idx = global_offset / NUM_BLOCKS_PER_TMP_BUFFER;
      u32 buf_offset = (global_offset % NUM_BLOCKS_PER_TMP_BUFFER) *
          GPUTmpMemBlock::BLOCK_SIZE;

      i32 staging_belt_idx = state.tmpStagingBuffers[buf_idx];
      u8 *ptr = stagingBelt.ptrs[staging_belt_idx];

      Buffer buffer_hdl {
        .gen = 1,
        .id = u16(stagingBelt.stagingBufferHandlesBase + staging_belt_idx),
      };

      return {
        .ptr = ptr,
        .buffer = buffer_hdl,
        .offset = buf_offset,
      };
    }

    spinLock(&state.lock);

    offset_range = staging_range_atomic.load<sync::relaxed>();
    global_offset = (u32)offset_range;
    range_end = u32(offset_range >> 32);

    if (global_offset < range_end) {
      spinUnlock(&state.lock);
      continue;
    }

    global_offset = range_end;
    u32 buf_idx = global_offset / NUM_BLOCKS_PER_TMP_BUFFER;
    chk(buf_idx < MAX_TMP_BUFFERS_PER_QUEUE);

    i32 staging_belt_idx = allocStagingBufferFromBelt();

    state.tmpStagingBuffers[buf_idx] = staging_belt_idx;

    staging_range_atomic.store<sync::release>(
      (u64(global_offset + NUM_BLOCKS_PER_TMP_BUFFER) << 32) |
       u64(global_offset + 1));

    spinUnlock(&state.lock);

    Buffer buffer_hdl {
      .gen = 1,
      .id = u16(stagingBelt.stagingBufferHandlesBase + staging_belt_idx),
    };

    return {
      .ptr = stagingBelt.ptrs[staging_belt_idx],
      .buffer = buffer_hdl,
      .offset = 0,
    };
  }
}

GPUTmpMemBlock Backend::allocGPUTmpInputBlock(GPUQueue queue_hdl)
{
  BackendQueueData &queue_data = queueDatas[queue_hdl.id];
  GPUTmpInputState &state = queue_data.gpuTmpInput;

  AtomicU64Ref tmp_input_range_atomic(state.curTmpInputRange);

  while (true) {
    u64 offset_range = tmp_input_range_atomic.fetch_add<sync::acq_rel>(1);
    u32 global_offset = (u32)offset_range;
    u32 range_end = u32(offset_range >> 32);

    if (global_offset < range_end) [[likely]] {
      u32 buf_idx = global_offset / NUM_BLOCKS_PER_TMP_BUFFER;
      u32 buf_offset = (global_offset % NUM_BLOCKS_PER_TMP_BUFFER) *
          GPUTmpMemBlock::BLOCK_SIZE;

      i32 staging_belt_idx = state.gpuTmpInputStagingBuffers[buf_idx];
      u8 *ptr = stagingBelt.ptrs[staging_belt_idx];

      Buffer buffer_hdl {
        .gen = 1,
        .id = u16(state.tmpBufferHandlesBase + buf_idx),
      };

      return {
        .ptr = ptr,
        .buffer = buffer_hdl,
        .offset = buf_offset,
      };
    }

    spinLock(&state.lock);

    offset_range = tmp_input_range_atomic.load<sync::relaxed>();
    global_offset = (u32)offset_range;
    range_end = u32(offset_range >> 32);

    if (global_offset < range_end) {
      spinUnlock(&state.lock);
      continue;
    }

    global_offset = range_end;
    u32 buf_idx = global_offset / NUM_BLOCKS_PER_TMP_BUFFER;
    chk(buf_idx < MAX_TMP_BUFFERS_PER_QUEUE);

    i32 staging_belt_idx = allocStagingBufferFromBelt();
    state.gpuTmpInputStagingBuffers[buf_idx] = staging_belt_idx;
    allocGPUTmpBuffer(state, buf_idx);

    tmp_input_range_atomic.store<sync::release>(
      (u64(global_offset + NUM_BLOCKS_PER_TMP_BUFFER) << 32) |
       u64(global_offset + 1));

    spinUnlock(&state.lock);

    Buffer buffer_hdl {
      .gen = 1,
      .id = u16(state.tmpBufferHandlesBase + buf_idx),
    };

    return {
      .ptr = stagingBelt.ptrs[staging_belt_idx],
      .buffer = buffer_hdl,
      .offset = 0,
    };
  }
}

void Backend::unmapActiveStagingBuffers(GPUTmpInputState &gpu_tmp_input)
{
  u32 end_staging_offset = u32(gpu_tmp_input.curTmpStagingRange >> 32);
  i32 num_active_staging_buffers =
      (i32)end_staging_offset / NUM_BLOCKS_PER_TMP_BUFFER;

  for (i32 i = 0; i < num_active_staging_buffers; i++) {
    i32 staging_belt_idx = gpu_tmp_input.tmpStagingBuffers[i];
    stagingBelt.buffers[staging_belt_idx].Unmap();
  }

  u32 end_tmp_input_offset = u32(gpu_tmp_input.curTmpInputRange >> 32);
  i32 num_active_tmp_input_buffers =
      (i32)end_tmp_input_offset / NUM_BLOCKS_PER_TMP_BUFFER;

  for (i32 i = 0; i < num_active_tmp_input_buffers; i++) {
    i32 staging_belt_idx = gpu_tmp_input.gpuTmpInputStagingBuffers[i];
    stagingBelt.buffers[staging_belt_idx].Unmap();
  }
}

void Backend::mapActiveStagingBuffers(GPUTmpInputState &gpu_tmp_input)
{
  auto mapStagingBuffer =
    [this]
  (i32 idx)
  {
    stagingBelt.buffers[idx].MapAsync(wgpu::MapMode::Write, 0,
        TMP_BUFFER_SIZE,
        wgpu::CallbackMode::AllowSpontaneous,
        returnBufferToStagingBeltCallback,
        (void *)&stagingBelt.cbStates[idx]);
  };

  u32 end_staging_offset = u32(gpu_tmp_input.curTmpStagingRange >> 32);
  i32 num_active_staging_buffers =
      (i32)end_staging_offset / NUM_BLOCKS_PER_TMP_BUFFER;

  for (i32 i = 0; i < num_active_staging_buffers; i++) {
    mapStagingBuffer(gpu_tmp_input.tmpStagingBuffers[i]);
  }

  u32 end_tmp_input_offset = u32(gpu_tmp_input.curTmpInputRange >> 32);
  i32 num_active_tmp_input_buffers =
      (i32)end_tmp_input_offset / NUM_BLOCKS_PER_TMP_BUFFER;

  for (i32 i = 0; i < num_active_tmp_input_buffers; i++) {
    mapStagingBuffer(gpu_tmp_input.gpuTmpInputStagingBuffers[i]);
  }
}

void Backend::submit(GPUQueue queue_hdl, FrontendCommands *cmds)
{
#ifdef GAS_WGPU_DEBUG_PRINT
  printf("WGPU: begin submit\n");
#endif

  BackendQueueData &queue_data = queueDatas[queue_hdl.id];

  wgpu::CommandEncoder wgpu_enc = dev.CreateCommandEncoder();

  GPUTmpInputState &gpu_tmp_input = queue_data.gpuTmpInput;

  // Any tmp buffers used in raster / compute passes must be
  // copied to GPU-visible buffers
  {
    unmapActiveStagingBuffers(gpu_tmp_input);
    u32 end_tmp_input_offset = u32(gpu_tmp_input.curTmpInputRange >> 32);
    i32 num_active_tmp_input_buffers =
        (i32)end_tmp_input_offset / NUM_BLOCKS_PER_TMP_BUFFER;

    for (i32 i = 0; i < (i32)num_active_tmp_input_buffers - 1; i++) {
      i32 staging_belt_idx = gpu_tmp_input.gpuTmpInputStagingBuffers[i];

      wgpu::Buffer &staging_buf =
          stagingBelt.buffers[staging_belt_idx];
      
      auto [to_gpu_buf, _1, _2] = buffers.get(
          gpu_tmp_input.tmpBufferHandlesBase, i);

      wgpu_enc.CopyBufferToBuffer(staging_buf, 0,
          *to_gpu_buf, 0, TMP_BUFFER_SIZE);
    }

    if (num_active_tmp_input_buffers > 0) {
      i32 i = num_active_tmp_input_buffers - 1;

      u32 cur_tmp_input_offset = u32(gpu_tmp_input.curTmpInputRange);

      i32 num_end_blocks = cur_tmp_input_offset % NUM_BLOCKS_PER_TMP_BUFFER;

      i32 staging_belt_idx = gpu_tmp_input.gpuTmpInputStagingBuffers[i];

      wgpu::Buffer &staging_buf =
          stagingBelt.buffers[staging_belt_idx];
      
      auto [to_gpu_buf, _1, _2] = buffers.get(
          gpu_tmp_input.tmpBufferHandlesBase, i);

      wgpu_enc.CopyBufferToBuffer(staging_buf, 0,
          *to_gpu_buf, 0, num_end_blocks * GPUTmpMemBlock::BLOCK_SIZE);
    }
  }

  CommandDecoder decoder(cmds);

  auto encodeRasterPass = [&]()
  {
    decoder.resetDrawParams();

    auto raster_pass = decoder.id<RasterPass>();
    const BackendRasterPass &backend_pass = *rasterPasses.hot(raster_pass);

    wgpu::RenderPassColorAttachment color_attachments[MAX_COLOR_ATTACHMENTS];
    wgpu::RenderPassDepthStencilAttachment depth_attachment;

    for (i32 i = 0; i < backend_pass.numColorAttachments; i++) {
      const BackendColorAttachment &in =
        backend_pass.colorAttachments[i];
      wgpu::RenderPassColorAttachment &out =
        color_attachments[i];

      if (i == backend_pass.swapchainAttachmentIndex) {
        BackendSwapchain &backend_swapchain =
          swapchains[backend_pass.swapchain.id];
        out.view = backend_swapchain.view;
      } else {
        out.view = in.view;
      }
      out.loadOp = in.loadOp;
      out.storeOp = in.storeOp;
      out.clearValue = {
        in.clearValue.x, 
        in.clearValue.y, 
        in.clearValue.z, 
        in.clearValue.w, 
      };
    }

    wgpu::RenderPassDescriptor pass_descriptor;
    pass_descriptor.colorAttachmentCount =
      (size_t)backend_pass.numColorAttachments;
    pass_descriptor.colorAttachments = color_attachments;

    if (backend_pass.depthAttachment.view == nullptr) {
      pass_descriptor.depthStencilAttachment = nullptr;
    } else {
      depth_attachment.view = backend_pass.depthAttachment.view;
      depth_attachment.depthLoadOp =
        backend_pass.depthAttachment.loadOp;
      depth_attachment.depthStoreOp =
        backend_pass.depthAttachment.storeOp;
      depth_attachment.depthClearValue =
        backend_pass.depthAttachment.clearValue;

      pass_descriptor.depthStencilAttachment = &depth_attachment;
    }

    wgpu::RenderPassEncoder pass_enc =
        wgpu_enc.BeginRenderPass(&pass_descriptor);

    wgpu::BindGroup dynamic_tmp_input_bind_group;
    i32 dynamic_bind_group_idx = -1;
    while (true) {
      auto updateDrawState =
        [&]
      (CommandCtrl ctrl)
      {
        if (RasterShader shader = decoder.drawShader(ctrl); !shader.null()) {
          BackendRasterShader *to_raster_shader = rasterShaders.hot(shader);
          pass_enc.SetPipeline(to_raster_shader->pipeline);
          dynamic_bind_group_idx = to_raster_shader->perDrawBindGroupSlot;
        }

        if (ParamBlock pb0 = decoder.drawParamBlock0(ctrl); !pb0.null()) {
          pass_enc.SetBindGroup(0, *paramBlocks.hot(pb0));
        }

        if (ParamBlock pb1 = decoder.drawParamBlock1(ctrl); !pb1.null()) {
          pass_enc.SetBindGroup(1, *paramBlocks.hot(pb1));
        }

        if (ParamBlock pb2 = decoder.drawParamBlock2(ctrl); !pb2.null()) {
          pass_enc.SetBindGroup(2, *paramBlocks.hot(pb2));
        }

        if (Buffer data_buf = decoder.drawDataBuffer(ctrl); !data_buf.null()) {
          i32 tmp_buf_idx =
              (i32)data_buf.id - gpu_tmp_input.tmpBufferHandlesBase;

          dynamic_tmp_input_bind_group =
              gpu_tmp_input.tmpGPUBufferBindGroups[tmp_buf_idx];
        }

        if (u32 data_offset = decoder.drawDataOffset(ctrl);
            data_offset != 0xFFFF'FFFF) {
          pass_enc.SetBindGroup(dynamic_bind_group_idx,
                                dynamic_tmp_input_bind_group, 1, &data_offset);
        }

        if (Buffer vb0 = decoder.drawVertexBuffer0(ctrl); !vb0.null()) {
          pass_enc.SetVertexBuffer(0, *buffers.hot(vb0));
        }

        if (Buffer vb1 = decoder.drawVertexBuffer1(ctrl); !vb1.null()) {
          pass_enc.SetVertexBuffer(1, *buffers.hot(vb1));
        }

        if (Buffer ib = decoder.drawIndexBuffer32(ctrl); !ib.null()) {
          pass_enc.SetIndexBuffer(*buffers.hot(ib),
                                  wgpu::IndexFormat::Uint32);
        }
        if (Buffer ib = decoder.drawIndexBuffer16(ctrl); !ib.null()) {
          pass_enc.SetIndexBuffer(*buffers.hot(ib),
                                  wgpu::IndexFormat::Uint16);
        }

        DrawParams draw_params = decoder.drawParams(ctrl);

#ifdef GAS_WGPU_DEBUG_PRINT
        printf(R"(DrawParams:
  NumTriangles: %u
  NumInstances: %u
  VertexOffset: %u
  InstanceOffset: %u
)", draw_params.numTriangles, draw_params.numInstances,
    draw_params.vertexOffset, draw_params.instanceOffset);
#endif

        return draw_params;
      };

      CommandCtrl ctrl = decoder.ctrl();

#ifdef GAS_WGPU_DEBUG_PRINT
      debugPrintDrawCommandCtrl(ctrl);
#endif

      CommandCtrl ctrl_masked = ctrl & 
          (CommandCtrl::RasterDraw |
           CommandCtrl::RasterDrawIndexed |
           CommandCtrl::RasterScissors);

      switch (ctrl_masked) {
        case CommandCtrl::None: {
          pass_enc.End();
          return;
        } break;
        case CommandCtrl::RasterDraw: {
          DrawParams draw_params = updateDrawState(ctrl);

          pass_enc.Draw(draw_params.numTriangles * 3,
                        draw_params.numInstances,
                        draw_params.vertexOffset,
                        draw_params.instanceOffset);
        } break;
        case CommandCtrl::RasterDrawIndexed: {
          DrawParams draw_params = updateDrawState(ctrl);

          pass_enc.DrawIndexed(draw_params.numTriangles * 3,
                               draw_params.numInstances,
                               draw_params.indexOffset,
                               draw_params.vertexOffset,
                               draw_params.instanceOffset);
        } break;
        case CommandCtrl::RasterScissors: {
          ScissorParams scissors = decoder.scissorParams();

          pass_enc.SetScissorRect(scissors.offsetX, scissors.offsetY,
                                  scissors.width, scissors.height);
        } break;
        default: BOT_UNREACHABLE();
      }
    }

    pass_enc.End();
  };
   
  auto encodeCopyPass = [&]()
  {
    decoder.resetCopyCommand();

    while (true) {
      CommandCtrl ctrl = decoder.ctrl();

      CommandCtrl ctrl_masked = ctrl & 
          (CommandCtrl::CopyCmdBufferToBuffer |
           CommandCtrl::CopyCmdBufferToTexture |
           CommandCtrl::CopyCmdTextureToBuffer |
           CommandCtrl::CopyCmdBufferClear);

      switch (ctrl_masked) {
        case CommandCtrl::None: {
          return;
        } break;
        case CommandCtrl::CopyCmdBufferToBuffer: {
          CopyBufferToBufferCmd b2b =
              decoder.copyBufferToBuffer(ctrl);

          wgpu_enc.CopyBufferToBuffer(*buffers.hot(b2b.src), b2b.srcOffset,
                                      *buffers.hot(b2b.dst), b2b.dstOffset,
                                      b2b.numBytes);
        } break;
        case CommandCtrl::CopyCmdBufferToTexture: {
          CopyBufferToTextureCmd b2t =
              decoder.copyBufferToTexture(ctrl);

          BackendTextureCold *to_tex_data;
          {
            if (!textures.hot(b2t.dst)) [[unlikely]] {
              to_tex_data = nullptr;
            }
            to_tex_data = textures.cold(b2t.dst);
          }

          u32 mip0_width = to_tex_data->baseWidth;
          u32 mip0_height = to_tex_data->baseHeight;
          u32 mip0_depth = to_tex_data->baseDepth;
          u32 bytes_per_texel = to_tex_data->numBytesPerTexel;
              
          u32 width = std::max(mip0_width >> b2t.dstMipLevel, 1_u32);
          u32 height = std::max(mip0_height >> b2t.dstMipLevel, 1_u32);
          u32 depth = std::max(mip0_depth >> b2t.dstMipLevel, 1_u32);

          wgpu::TexelCopyBufferInfo src {
            .layout = {
              .offset = b2t.srcOffset,
              .bytesPerRow = width * bytes_per_texel,
            },
            .buffer = *buffers.hot(b2t.src),
          };

          wgpu::TexelCopyTextureInfo dst {
            .texture = to_tex_data->texture,
            .mipLevel = b2t.dstMipLevel,
          };

          wgpu::Extent3D copy_size {
            .width = width,
            .height = height,
            .depthOrArrayLayers = depth,
          };

          wgpu_enc.CopyBufferToTexture(&src, &dst, &copy_size);
        } break;
        case CommandCtrl::CopyCmdTextureToBuffer: {
          CopyTextureToBufferCmd t2b =
              decoder.copyTextureToBuffer(ctrl);

          BackendTextureCold *to_tex_data;
          {
            if (!textures.hot(t2b.src)) [[unlikely]] {
              to_tex_data = nullptr;
            }
            to_tex_data = textures.cold(t2b.src);
          }

          u32 mip0_width = to_tex_data->baseWidth;
          u32 mip0_height = to_tex_data->baseHeight;
          u32 mip0_depth = to_tex_data->baseDepth;
          u32 bytes_per_texel = to_tex_data->numBytesPerTexel;
              
          u32 width = std::max(mip0_width >> t2b.srcMipLevel, 1_u32);
          u32 height = std::max(mip0_height >> t2b.srcMipLevel, 1_u32);
          u32 depth = std::max(mip0_depth >> t2b.srcMipLevel, 1_u32);

          wgpu::TexelCopyTextureInfo src {
            .texture = to_tex_data->texture,
            .mipLevel = t2b.srcMipLevel,
          };

          wgpu::TexelCopyBufferInfo dst {
            .layout = {
              .offset = t2b.dstOffset,
              .bytesPerRow = width * bytes_per_texel,
            },
            .buffer = *buffers.hot(t2b.dst),
          };

          wgpu::Extent3D copy_size {
            .width = width,
            .height = height,
            .depthOrArrayLayers = depth,
          };

          wgpu_enc.CopyTextureToBuffer(&src, &dst, &copy_size);
        } break;
        case CommandCtrl::CopyCmdBufferClear: {
          CopyClearBufferCmd clear = decoder.copyClear(ctrl);

          wgpu_enc.ClearBuffer(*buffers.hot(clear.buffer), clear.offset,
                               clear.numBytes);
        } break;
        default: BOT_UNREACHABLE();
      }
    }
  };

  for (CommandCtrl ctrl; (ctrl = decoder.ctrl()) != CommandCtrl::None;) {
    switch (ctrl) {
      case CommandCtrl::RasterPass: {
        encodeRasterPass();
      } break;
      case CommandCtrl::ComputePass: {
      } break;
      case CommandCtrl::CopyPass: {
        encodeCopyPass();
      } break;
      default: BOT_UNREACHABLE();
    }
  }

  wgpu::CommandBuffer cmd_buf = wgpu_enc.Finish();
  queue.Submit(1, &cmd_buf);

  mapActiveStagingBuffers(gpu_tmp_input);


  // Clear staging buffer tracking for next submission
  {
    gpu_tmp_input.curTmpStagingRange = 0;
    gpu_tmp_input.curTmpInputRange = 0;
  }

  // Destroy temporary parameter blocks (webgpu manages keeping these alive
  // until the submission is done
  {
    TmpParamBlockState &tmp_param_block_state = queue_data.tmpParamBlockState;

    for (i32 i = 0; i < (i32)tmp_param_block_state.numLive; i++) {
      auto [to_param_block, _1, _2] = paramBlocks.get(
          tmp_param_block_state.baseHandleOffset, i);

      to_param_block->~BindGroup();
    }

    tmp_param_block_state.numLive = 0;
  }
}

BackendRasterPassConfig * Backend::getRasterPassConfigByID(
    RasterPassInterfaceID id)
{
  if (!id.isUUID()) {
    return rasterPassInterfaces.hot(id.asHandle());
  } else {
    i32 tbl_row = rasterPassInterfaceIDs.lookup(id.asUUID());
    if (tbl_row == ResourceUUIDMap::NOT_FOUND) [[unlikely]] {
      return nullptr;
    } else {
      auto [to_cfg, _1, _2] = rasterPassInterfaces.get(tbl_row, 0);
      return to_cfg;
    }
  }
}

wgpu::BindGroupLayout Backend::getBindGroupLayoutByParamBlockTypeID(
    ParamBlockTypeID id)
{
  if (!id.isUUID()) {
    return paramBlockTypes.hot(id.asHandle())->layout;
  } else {
    i32 tbl_row = paramBlockTypeIDs.lookup(id.asUUID());
    if (tbl_row == ResourceUUIDMap::NOT_FOUND) [[unlikely]] {
      return nullptr;
    } else {
      auto [to_block_type, _1, _2] = paramBlockTypes.get(tbl_row, 0);
      return to_block_type->layout;
    }
  }
}

i32 Backend::allocStagingBufferFromBelt()
{
  spinLock(&stagingBelt.lock);
  BOT_DEFER(spinUnlock(&stagingBelt.lock));

  if (stagingBelt.numFree > 0) {
    i32 idx = stagingBelt.freeList[--stagingBelt.numFree];
    return idx;
  }

  u32 idx = stagingBelt.numAllocated++;
  chk(idx < MAX_TMP_STAGING_BUFFERS);

  wgpu::BufferDescriptor buffer_desc {
    .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
    .size = TMP_BUFFER_SIZE,
    .mappedAtCreation = true,
  };

  stagingBelt.buffers[idx] = dev.CreateBuffer(&buffer_desc);
  stagingBelt.ptrs[idx] =
     (u8 *)stagingBelt.buffers[idx].GetMappedRange();

  { // Save buffer to handle slot
    auto [to_buffer, to_buffer_metadata, _2] = buffers.get(
        stagingBelt.stagingBufferHandlesBase, idx);
    new (to_buffer) wgpu::Buffer(stagingBelt.buffers[idx]);
    *to_buffer_metadata = {
      .numBytes = TMP_BUFFER_SIZE,
      .usage = BufferUsage::CopySrc,
    };
  }

  return idx;
}

void Backend::returnBufferToStagingBeltCallback(
    wgpu::MapAsyncStatus async_status,
    const char *msg,
    void *user_data)
{
  if (async_status != wgpu::MapAsyncStatus::Success) {
    FATAL("Error while remapping buffer in staging belt: %lu, %s",
          (u64)async_status, msg);
  }

  auto [to_backend, buf_idx] = *(StagingBelt::CallbackState *)user_data;

  StagingBelt &belt = to_backend->stagingBelt;

  spinLock(&belt.lock);

  belt.freeList[belt.numFree++] = buf_idx;

  spinUnlock(&belt.lock);
}

#if 0
GPUTmpMemBlock Backend::allocTmpDynUniformBlock(BackendQueueData &queue_data)
{
  GPUTmpInputState &state = queue_data.gpuTmpInput;

  AtomicU64Ref atomic_free_range(state.curDynUniformRange);

  while (true) {
    u64 offset_range = atomic_free_range.fetch_add<sync::acq_rel>(1);

    u32 global_offset = (u32)offset_range;
    u32 range_end = u32(offset_range >> 32);

    if (global_offset < range_end) [[likely]] {
      u32 buf_idx = global_offset / NUM_BLOCKS_PER_TMP_BUFFER;
      u32 buf_offset = (global_offset % TmpGPUBuffer::NUM_BLOCKS) *
          GPUTmpMemBlock::BLOCK_SIZE;

      TmpGPUBuffer &cur_tmp = state.buffers[buf_idx];

      Buffer buffer_hdl {
        .gen = 1,
        .id = u16(state.bufferHandlesBase + buf_idx),
      };

      return {
        .buffer = buffer_hdl,
        .offset = buf_offset,
      };
    }

    queue_data.allocLock.lock();

    offset_range = atomic_free_range.load<sync::relaxed>();
    global_offset = (u32)offset_range;
    range_end = u32(offset_range >> 32);

    if (global_offset < range_end) {
      queue_data.allocLock.unlock();
      continue;
    }

    u32 new_global_offset = range_end;
    u32 buf_idx = new_global_offset / TmpGPUBuffer::NUM_BLOCKS;
    chk(buf_idx < GPUTmpInputState::MAX_BUFFERS);

    u32 new_buf_handle_idx = state.bufferHandlesBase + buf_idx;
    state.buffers[buf_idx] = allocTmpDynamicUniformBuffer(new_buf_handle_idx);

    TmpGPUBuffer &new_tmp = state.buffers[buf_idx];
    u8 *ptr = stagingBelt.ptrs[new_tmp.stagingBeltIndex];

    atomic_free_range.store<sync::release>(
      (u64(new_global_offset + TmpGPUBuffer::NUM_BLOCKS) << 32) |
       u64(new_global_offset + 1));

    queue_data.allocLock.unlock();

    Buffer buffer_hdl {
      .gen = 1,
      .id = u16(new_buf_handle_idx),
    };

    return {
      .ptr = ptr,
      .buffer = buffer_hdl,
      .offset = 0,
    };
  }
}
#endif

void Backend::allocGPUTmpBuffer(GPUTmpInputState &state, i32 buf_idx)
{
  if (buf_idx < (i32)state.maxNumUsedTmpGPUBuffers) {
    return;
  }

  wgpu::BufferDescriptor buffer_desc {
    .usage = wgpu::BufferUsage::Uniform | 
             wgpu::BufferUsage::Storage | 
             wgpu::BufferUsage::Vertex | 
             wgpu::BufferUsage::Index | 
             wgpu::BufferUsage::CopySrc |
             wgpu::BufferUsage::CopyDst,
    .size = TMP_BUFFER_SIZE,
  };

  auto [to_buffer, _, id] = buffers.get(
      state.tmpBufferHandlesBase, buf_idx);

  new (to_buffer) wgpu::Buffer(dev.CreateBuffer(&buffer_desc));

  wgpu::BindGroupEntry bind_group_entry {
    .binding = 0,
    .buffer = *to_buffer,
    .offset = 0,
    .size = limits.maxNumUniformBytes,
  };

  wgpu::BindGroupDescriptor bind_group_desc {
    .layout = tmpDynamicUniformLayout,
    .entryCount = 1,
    .entries = &bind_group_entry,
  };

  wgpu::BindGroup bind_group = dev.CreateBindGroup(&bind_group_desc);

  state.tmpGPUBufferBindGroups[buf_idx] = bind_group;

  state.maxNumUsedTmpGPUBuffers += 1;
}

GPULib * initWebGPU(const APIConfig &cfg)
{
  return WebGPULib::init(cfg);
}

}
