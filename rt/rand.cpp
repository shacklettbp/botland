#include "rand.hpp"

namespace bot::rand {

// This implementation is based on JAX's threefry splitting implementation.
// The number of rounds is set to 20, for a safe margin over the currently
// known lower limit, 13. 20 is also the default in the original authors'
// threefry implementation:
// https://github.com/DEShawResearch/random123/blob/main/include/Random123/threefry.h
//
// Original copyright / license:
// Copyright 2019 The JAX Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
RandKey split_i(RandKey src, u32 idx, u32 idx_upper)
{
    // Rotation distances specified by the Threefry2x32 algorithm.
    u32 rotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};
    u32 x[2];
    u32 ks[3];
    
    // 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
    ks[2] = 0x1BD11BDA;
    
    ks[0] = src.a;
    x[0] = idx;
    ks[2] = ks[2] ^ src.a;
    
    ks[1] = src.b;
    x[1] = idx_upper;
    ks[2] = ks[2] ^ src.b;
    
    auto rotate_left = [](u32 v, u32 distance) {
        return (v << distance) | (v >> (32 - distance));
    };
    
    // Performs a single round of the Threefry2x32 algorithm, with a rotation
    // amount 'rotation'.
    auto round = [&](u32* v, u32 rotation) {
        v[0] += v[1];
        v[1] = rotate_left(v[1], rotation);
        v[1] ^= v[0];
    };
    
    // There are no known statistical flaws with 13 rounds of Threefry2x32.
    // We are conservative and use 20 rounds.
    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1];
BOT_UNROLL
    for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
    }
    
    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 1u;
BOT_UNROLL
    for (int i = 4; i < 8; ++i) {
        round(x, rotations[i]);
    }
    
    x[0] = x[0] + ks[2];
    x[1] = x[1] + ks[0] + 2u;
BOT_UNROLL
    for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
    }
    
    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1] + 3u;
BOT_UNROLL
    for (int i = 4; i < 8; ++i) {
        round(x, rotations[i]);
    }
    
    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 4u;
BOT_UNROLL
    for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
    }

    RandKey out;
    out.a = x[0] + ks[2];
    out.b = x[1] + ks[0] + 5u;

    return out;
}

i32 sampleI32(RandKey k, i32 a, i32 b)
{
    u32 s = (u32)(b - a);

    // Lemire, Fast Random Number Generation in an Interval.
    // Algorithm 5. This is probably non-ideal for GPU but is unbiased.
    u32 x = bits32(k);

#ifdef BOT_GPU
    u32 l = x * s;
    u32 h = __umulhi(x, s);
#else
    u32 l, h;
    {
        u64 tmp = (u64)x * (u64)s;
        l = (u32)tmp;
        h = (u32)(tmp >> 32);
    }
#endif

    if (l < s) [[unlikely]] {
        // 2^32 % s == (2^32 - s) % s == -s % s
        u32 t = (0_u32 - s) % s;

        while (l < t) {
            // This might be suspect: reusing k but we're rejecting the random
            // number k generated, so maybe it's fine...?
            k = split_i(k, 0);
            x = bits32(k);

#ifdef BOT_GPU
            l = x * s;
            h = __umulhi(x, s);
#else
            {
                u64 tmp = (u64)x * (u64)s;
                l = (u32)tmp;
                h = (u32)(tmp >> 32);
            }
#endif
        }
    }

    return (i32)h + a;
}

}
