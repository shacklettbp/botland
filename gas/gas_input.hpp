#pragma once

#include <rt/types.hpp>
#include <rt/math.hpp>
#include <rt/utils.hpp>

#include <array>

namespace bot::gas {

enum class InputID : u32 {
  MouseLeft, MouseRight, MouseMiddle, Mouse4, Mouse5,
  A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
  K1, K2, K3, K4, K5, K6, K7, K8, K9, K0,
  Shift, Space, BackSpace, Esc, Enter,
  NUM_IDS,
};

class UserInput {
public:
  inline Vector2 mousePosition() const;

  inline bool isDown(InputID id) const;
  inline bool isUp(InputID id) const;

  inline void setMousePosition(Vector2 p);
  inline void setDown(InputID id);
  inline void setUp(InputID id);

private:
  static constexpr inline u32 NUM_BITFIELDS =
      roundToAlignment((u32)InputID::NUM_IDS, (u32)32);

  Vector2 mouse_pos_;

  std::array<u32, NUM_BITFIELDS> states_;
};

class UserInputEvents {
public:
  inline bool downEvent(InputID id) const;
  inline bool upEvent(InputID id) const;

  inline Vector2 mouseDelta() const;
  inline Vector2 mouseScroll() const;

  inline void recordDownEvent(InputID id);
  inline void recordUpEvent(InputID id);

  inline void updateMouseDelta(Vector2 d);
  inline void updateMouseScroll(Vector2 d);

  void merge(const UserInputEvents &o);
  void clear();

private:
  static constexpr inline u32 NUM_BITFIELDS =
      2 * roundToAlignment((u32)InputID::NUM_IDS, (u32)32);

  std::array<u32, NUM_BITFIELDS> events_;
  Vector2 mouse_delta_;
  Vector2 mouse_scroll_;
};

}

#include "gas_input.inl"
