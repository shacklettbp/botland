#include <rt/rt.hpp>
#include <filesystem>
#include <gas/gas.hpp>
#include <rt/types.hpp>
#include <gas/gas_ui.hpp>

#include "game/backend.hpp"
#include "viz.hpp"

namespace bot {

#if 0
// This is testing code for backend
struct Backend {
  enum class ExecMode {
    CPU, GPU, None
  };

  ExecMode execMode;
  RuntimeState *rtState;
  SimRT *sim;

  Backend(ExecMode exec_mode, u32 num_worlds);

  void step();
};

Backend::Backend()
  : execMode(exec_mode),
    rtState(nullptr)
{
  switch (execMode) {
  case ExecMode::CPU: {
    rtState = createRuntimeState({});
  } break;

  case ExecMode::GPU: {

  } break;

  default: {
    FATAL("Passed incorrect execution mode");
  } break;
  }
}
#endif


struct Frontend {
  UISystem * ui_sys;
  Window * window;

  Viz viz;

  Swapchain swapchain;

  Backend *backend;

  inline void init(Backend *backend);
  inline void shutdown();

  inline void handleUIControl(UIControl ui_ctrl);
  inline void loop();
};

void Frontend::init(Backend *be)
{
  backend = be;
  ui_sys = UISystem::init({
    .enableValidation = true,
    .errorsAreFatal = true,
  });
  window = ui_sys->createMainWindow(
    "Botland", 1920*2, 1080*2, WindowInitFlags::Resizable);

  GPULib *gpu_lib = ui_sys->gpuLib();
  viz.init(backendRTStateHandle(be), backendSimState(be),
    gpu_lib, gpu_lib->createDevice(0, {window->surface}), window->surface);
}

void Frontend::shutdown()
{
  GPUDevice *gpu = viz.gpu;

  gpu->waitUntilIdle();

  viz.shutdown();
  ui_sys->gpuLib()->destroyDevice(gpu);

  ui_sys->processEvents();
  ui_sys->destroyMainWindow();
  ui_sys->shutdown();
}

void Frontend::handleUIControl(UIControl ui_ctrl)
{
  if ((ui_ctrl.flags & UIControl::RawMouseMode) != UIControl::None) {
    ui_sys->enableRawMouseInput(window);
  } else {
    ui_sys->disableRawMouseInput(window);
  }

  if ((ui_ctrl.flags & UIControl::EnableIME) != UIControl::None) {
    ui_sys->beginTextEntry(window, ui_ctrl.imePos, ui_ctrl.imeLineHeight);
  }

  if ((ui_ctrl.flags & UIControl::DisableIME) != UIControl::None) {
    ui_sys->endTextEntry(window);
  }
}

void Frontend::loop()
{
  SimRT rt(backendRTStateHandle(backend), 0, backendSimState(backend));

  bool should_loop = true;

  auto prev_frame_start_time = std::chrono::steady_clock::now();
  while (should_loop) {
    {
      bool should_exit = ui_sys->processEvents();
      if (should_exit ||
          (window->state & WindowState::ShouldClose) != WindowState::None) {
        should_loop = false;
      }
    }

    auto cur_frame_start_time = std::chrono::steady_clock::now();
    float delta_t;
    {
      std::chrono::duration<float> duration =
        cur_frame_start_time - prev_frame_start_time;
      delta_t = duration.count();
    }
    prev_frame_start_time = cur_frame_start_time;

    UIControl ui_ctrl = viz.runUI(rt, ui_sys->inputState(), 
      ui_sys->inputEvents(), ui_sys->inputText(), window->systemUIScale, delta_t);
    handleUIControl(ui_ctrl);

    viz.render(rt);
  }
}

}

int main(int argc, const char *argv[])
{
  using namespace bot;

  int gpu_id = -1;
  int num_worlds = 1;

  if (argc != 3) {
    printf("./native_frontend [cpu|gpu] [# worlds]\n");
    return -1;
  } else {
    const char *backend_type = argv[1];
    const char *num_worlds_str = argv[2];

    if (!strcmp(backend_type, "cpu")) {
      gpu_id = -1;
    } else if (!strcmp(backend_type, "gpu")) {
      gpu_id = 0;
    } else {
      printf("./native_frontend [cpu|gpu] [# worlds]\n");
      return -1;
    }

    num_worlds = std::stoi(num_worlds_str);
  }

  (void)argc;
  (void)argv;

  Backend *backend = backendInit({
    .rt = {},
    .cuda = CUDAConfig {
      .gpuID = gpu_id,
      .memPoolSizeMB = 128,
    },
    .sim = {
      .numActiveWorlds = num_worlds,
    },
  });

  Runtime rt = Runtime(backendRTStateHandle(backend), 0);

  Frontend frontend;
  frontend.init(backend);
  frontend.loop();
  frontend.shutdown();

  backendShutdown(backend);


  return 0;
}
