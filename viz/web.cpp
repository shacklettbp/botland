#include "viz.hpp"

namespace bot {

namespace {
namespace web {

GPULib *gpu_lib = nullptr;
Surface canvas_surface;
Viz viz = {};

void register_callbacks()
{
}

void init()
{
  gpu_lib = GPULib::init(GPUAPISelect::WebGPU, {
    .errorsAreFatal = true,
    .enablePresent = true,
  });

  const char *canvas_selector = "#main-render-canvas";
  canvas_surface = gpu_lib->createSurface((void *)canvas_selector, 1920, 1080);

  auto device_cb = [](GPUDevice *dev, void *) {
    viz.init(gpu_lib, dev, canvas_surface);
    register_callbacks();
  };

  gpu_lib->createDeviceAsync(0, {canvas_surface}, device_cb, nullptr);
}

extern "C" void rdb_web_shutdown()
{
  GPUDevice *gpu = viz.gpu;

  viz.shutdown();
  gpu_lib->destroySurface(canvas_surface);
  gpu_lib->destroyDevice(gpu);
  gpu_lib->shutdown();
}

extern "C" void rdb_web_render()
{
  viz.render();
}

}
}

}

int main()
{
  rdb::web::init();
  return 0;
}
