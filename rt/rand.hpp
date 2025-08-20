#pragma once

#include "types.hpp"
#include "math.hpp"

namespace bot {

struct RandKey {
    u32 a;
    u32 b;
};

namespace rand {

constexpr inline RandKey initKey(u32 seed, u32 seed_upper = 0);
RandKey split_i(
  RandKey src, u32 idx, u32 idx_upper = 0);

constexpr inline u32 bits32(RandKey k);
constexpr inline u64 bits64(RandKey k);

i32 sampleI32(RandKey k, i32 a, i32 b);
constexpr inline i32 sampleI32Biased(RandKey k, i32 a, i32 b);
constexpr inline float sampleUniform(RandKey k);
constexpr inline bool sampleBool(RandKey k);
constexpr inline Vector2 sample2xUniform(RandKey k);
constexpr inline float bitsToFloat01(u32 rand_bits);


}

class RNG {
public:
    inline RNG();
    inline RNG(RandKey k);
    inline RNG(u32 seed);

    inline i32 sampleI32(i32 a, i32 b);
    inline i32 sampleI32Biased(i32 a, i32 b);
    inline float sampleUniform();
    inline bool sampleBool();

    inline RandKey randKey();

    RNG(const RNG &) = default;
    RNG(RNG &&) = default;
    RNG & operator=(const RNG &) = default;
    RNG & operator=(RNG &&) = default;

private:
    inline RandKey advance();

    RandKey k_;
    u32 count_;
};

}

#include "rand.inl"
