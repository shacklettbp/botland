namespace bot {

uint32_t Body::getTypeDim(Type type, bool is_pos)
{
  switch (type) {
  case Type::FreeBody: {
    return is_pos ? 7 : 6;
  }

  case Type::Hinge: {
    return 1;
  } break;

  case Type::Ball: {
    return is_pos ? 4 : 3;
  }

  case Type::FixedBody: {
    return is_pos ? 7 : 0;
  }

  case Type::Slider: {
    return 1;
  }

  default: {
    // Something is wrong if we're here
    FATAL("getTypeDim received invalid body type: %d", (int)type);
    return 0;
  }
  }
}

const char *Body::getTypeStr(Type type)
{
  switch (type) {
  case Type::FreeBody: {
    return "FreeBody";
  }

  case Type::Hinge: {
    return "Hinge";
  } break;

  case Type::Ball: {
    return "Ball";
  }

  case Type::FixedBody: {
    return "FixedBody";
  }

  case Type::Slider: {
    return "Slider";
  }

  default: {
    // Something is wrong if we're here
    return "<Invalid>";
  }
  }
}

}
