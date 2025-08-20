namespace bot::gas {

Vector2 UserInput::mousePosition() const
{
  return mouse_pos_;
}

bool UserInput::isDown(InputID id) const
{
  i32 id_idx = (i32)id / 32;
  i32 id_bit = (i32)id % 32;
  return (states_[id_idx] & (1 << id_bit)) != 0;
}

bool UserInput::isUp(InputID id) const
{
  i32 id_idx = (i32)id / 32;
  i32 id_bit = (i32)id % 32;
  return (states_[id_idx] & (1 << id_bit)) == 0;
}

void UserInput::setMousePosition(Vector2 p)
{
  mouse_pos_ = p;
}

void UserInput::setDown(InputID id)
{
  i32 state_idx = (i32)id / 32;
  i32 state_bit = (i32)id % 32;

  states_[state_idx] |= (1 << state_bit);
}

void UserInput::setUp(InputID id)
{
  i32 state_idx = (i32)id / 32;
  i32 state_bit = (i32)id % 32;

  states_[state_idx] &= ~(1 << state_bit);
}

bool UserInputEvents::downEvent(InputID id) const
{
  i32 id_idx = (i32)id / 16;
  i32 id_bit = (i32)id % 16;
  return (events_[id_idx] & (1 << (2 * id_bit))) != 0;
}

bool UserInputEvents::upEvent(InputID id) const
{
  i32 id_idx = (i32)id / 16;
  i32 id_bit = (i32)id % 16;
  return (events_[id_idx] & (1 << (2 * id_bit + 1))) != 0;
}

Vector2 UserInputEvents::mouseDelta() const
{
  return mouse_delta_;
}

Vector2 UserInputEvents::mouseScroll() const
{
  return mouse_scroll_;
}

void UserInputEvents::recordDownEvent(InputID id)
{
  i32 event_idx = (i32)id / 16;
  i32 event_bit = (i32)id % 16;
  events_[event_idx] |= (1 << (2 * event_bit));
}

void UserInputEvents::recordUpEvent(InputID id)
{
  i32 event_idx = (i32)id / 16;
  i32 event_bit = (i32)id % 16;
  events_[event_idx] |= (1 << (2 * event_bit + 1));
}

void UserInputEvents::updateMouseDelta(Vector2 d)
{
  mouse_delta_ += d;
}

void UserInputEvents::updateMouseScroll(Vector2 d)
{
  mouse_scroll_ += d;
}

}
