namespace bot {

inline UIControl::Flag & operator|=(UIControl::Flag &a, UIControl::Flag b)
{
    a = UIControl::Flag(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline UIControl::Flag operator|(UIControl::Flag a, UIControl::Flag b)
{
    a |= b;

    return a;
}

inline UIControl::Flag & operator&=(UIControl::Flag &a, UIControl::Flag b)
{
    a = UIControl::Flag(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    return a;
}

inline UIControl::Flag operator&(UIControl::Flag a, UIControl::Flag b)
{
    a &= b;

    return a;
}

}
