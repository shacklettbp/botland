#include "macros.hpp"
#include "utils.hpp"

namespace bot {
namespace rand {

constexpr RandKey initKey(u32 seed, u32 seed_upper)
{
    return split_i(RandKey { seed, seed_upper }, 0);
}

constexpr u32 bits32(RandKey k)
{
    return k.a ^ k.b;
}

constexpr u64 bits64(RandKey k)
{
    return ((u64)k.b << 32_u64) | (u64)k.a;
}

constexpr i32 sampleI32Biased(RandKey k, i32 a, i32 b)
{
    u32 s = (u32)(b - a);
    u32 x = bits32(k);

    return u32mulhi(x, s);
}

constexpr float sampleUniform(RandKey k)
{
    return bitsToFloat01(bits32(k));
}

constexpr bool sampleBool(RandKey k)
{
    u32 bits = bits32(k);

    u32 num_set = 
#ifdef BOT_GPU
        (u32)__popc(bits);
#else
        std::popcount(bits);
#endif

    return (num_set & 1) == 0;
}

constexpr Vector2 sample2xUniform(RandKey k)
{
    return Vector2 {
        .x = bitsToFloat01(k.a),
        .y = bitsToFloat01(k.b),
    };
}

constexpr float bitsToFloat01(u32 rand_bits)
{
    // This implementation (and the one commented out below), generate random
    // numbers in the interval [0, 1). This is done by randomizing the mantissa
    // while leaving the exponent at 0. This means some small random numbers
    // near 0 won't be output (for example 2^-32 won't be output). The plus
    // side is that 1 - sampleUniform(k) will at most be equal to 1 - float
    // epsilon (not 1).
    //
    // random123 contains an implementation of this idea as well:
    // https://github.com/DEShawResearch/random123/blob/main/include/Random123/u01fixedpt.h
    // That library seems to advocate a version that will randomly generate
    // smaller floats as well (such as dividing by 2^-32) but only provides
    // implementations that generate in the range (0, 1]
    return (rand_bits >> 8_u32) * 0x1p-24f;

#if 0
    constexpr u32 exponent = 0x3f800000;
    u32 raw = (exponent | (rand_bits >> 9)) - 1;

#ifdef BOT_GPU
    return __uint_as_float(raw);
#else
    return std::bit_cast<float>(raw);
#endif
#endif
}

}

RNG::RNG()
    : k_(RandKey { 0, 0 }),
      count_(0)
{}

RNG::RNG(RandKey k)
    : k_(k),
      count_(0)
{}

RNG::RNG(u32 seed)
    : RNG(rand::initKey(seed))
{}

i32 RNG::sampleI32(i32 a, i32 b)
{
    RandKey sample_k = advance();
    return rand::sampleI32(sample_k, a, b);
}

i32 RNG::sampleI32Biased(i32 a, i32 b)
{
    RandKey sample_k = advance();
    return rand::sampleI32Biased(sample_k, a, b);
}

float RNG::sampleUniform()
{
    RandKey sample_k = advance();
    return rand::sampleUniform(sample_k);
}

bool RNG::sampleBool()
{
    RandKey sample_k = advance();
    return rand::sampleBool(sample_k);
}

RandKey RNG::randKey()
{
    return advance();
}

RandKey RNG::advance()
{
    RandKey sample_k = rand::split_i(k_, count_);
    count_ += 1;

    return sample_k;
}

}
