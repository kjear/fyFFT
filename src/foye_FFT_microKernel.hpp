#ifndef _FOYE_FFT_MICROKERNEL_HPP_
#define _FOYE_FFT_MICROKERNEL_HPP_

namespace fy::fft
{
    struct micro_kernel_common
    {
        static constexpr float s2 = 0.7071067811865475244f;
        static constexpr float c1 = 0.9238795325112867561f;
        static constexpr float s1 = 0.3826834323650897717f;

        [[msvc::forceinline]] static __m256 complex_mul_twiddle(__m256 v, const float* w_n, const float* w_s)
        {
            return fma_complexf32(v, _mm256_load_ps(w_n), _mm256_load_ps(w_s));
        }

        [[msvc::forceinline]] static void radix2_butterfly(__m256& a, __m256& b, __m256 w_n, __m256 w_s)
        {
            __m256 t = fma_complexf32(b, w_n, w_s);
            b = _mm256_sub_ps(a, t);
            a = _mm256_add_ps(a, t);
        }

        [[msvc::forceinline]] static void radix4_butterfly_base(__m256& r0, __m256& r1, __m256& r2, __m256& r3, __m256 mask)
        {
            __m256 t0 = _mm256_add_ps(r0, r2);
            __m256 t1 = _mm256_sub_ps(r0, r2);
            __m256 t2 = _mm256_add_ps(r1, r3);
            __m256 t3 = _mm256_sub_ps(r1, r3);

            r0 = _mm256_add_ps(t0, t2);
            r2 = _mm256_sub_ps(t0, t2);

            __m256 t3_swapped = _mm256_permute_ps(t3, 0xB1);
            __m256 t3_rotated = _mm256_xor_ps(t3_swapped, mask);

            r1 = _mm256_add_ps(t1, t3_rotated);
            r3 = _mm256_sub_ps(t1, t3_rotated);
        }

        template<typename... Args>
        [[msvc::forceinline]] static void scale_all(__m256 scale, Args&... args)
        {
            (..., (args = _mm256_mul_ps(args, scale)));
        }
    };

    struct FFT1D_C2C_kernel_8_Radix_8 : micro_kernel_common
    {
        struct alignas(32) constants_type {
            float w1_n[8];
            float w1_s[8];
            float inv_scale[8];
        };

        static inline const constants_type fwd_consts = {
            {1.f, 0.f,  s2, -s2,  0.f, -1.f, -s2, -s2},
            {0.f, 1.f, -s2,  s2, -1.f,  0.f, -s2, -s2},
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}
        };

        static inline const constants_type inv_consts = {
            {1.f, 0.f,  s2,  s2,  0.f,  1.f, -s2,  s2},
            {0.f, 1.f,  s2,  s2,  1.f,  0.f,  s2, -s2},
            {0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f}
        };

        template<bool invert>
        [[msvc::forceinline]] static __m256 apply_trivial_twiddle_w4(__m256 v)
        {
            __m256 v_swapped = _mm256_permute_ps(v, 0xB1);
            const __m256 mask = invert ? _mm256_setr_ps(-0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f)
                : _mm256_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
            __m256 v_rotated = _mm256_xor_ps(v_swapped, mask);
            return _mm256_blend_ps(v, v_rotated, 0xCC);
        }

        template<bool invert>
        [[msvc::forceinline]] static void compute_inplace(__m256& v0, __m256& v1, const constants_type& C)
        {
            __m256 sum1 = _mm256_add_ps(v0, v1);
            __m256 diff1 = _mm256_sub_ps(v0, v1);

            v1 = complex_mul_twiddle(diff1, C.w1_n, C.w1_s);
            v0 = sum1;

            __m256 v0_swap = _mm256_permute2f128_ps(v0, v0, 0x01);
            __m256 v1_swap = _mm256_permute2f128_ps(v1, v1, 0x01);

            __m256 v0_sum = _mm256_add_ps(v0, v0_swap);
            __m256 v0_diff = _mm256_sub_ps(v0_swap, v0);
            __m256 v1_sum = _mm256_add_ps(v1, v1_swap);
            __m256 v1_diff = _mm256_sub_ps(v1_swap, v1);

            v0_diff = apply_trivial_twiddle_w4<invert>(v0_diff);
            v1_diff = apply_trivial_twiddle_w4<invert>(v1_diff);

            v0 = _mm256_blend_ps(v0_sum, v0_diff, 0xF0);
            v1 = _mm256_blend_ps(v1_sum, v1_diff, 0xF0);

            __m256 v0_perm = _mm256_castpd_ps(_mm256_permute_pd(_mm256_castps_pd(v0), 0x5));
            __m256 v1_perm = _mm256_castpd_ps(_mm256_permute_pd(_mm256_castps_pd(v1), 0x5));

            __m256 v0_final_sum = _mm256_add_ps(v0, v0_perm);
            __m256 v0_final_sub = _mm256_sub_ps(v0_perm, v0);
            __m256 v1_final_sum = _mm256_add_ps(v1, v1_perm);
            __m256 v1_final_sub = _mm256_sub_ps(v1_perm, v1);

            v0 = _mm256_castpd_ps(_mm256_blend_pd(_mm256_castps_pd(v0_final_sum), _mm256_castps_pd(v0_final_sub), 0xA));
            v1 = _mm256_castpd_ps(_mm256_blend_pd(_mm256_castps_pd(v1_final_sum), _mm256_castps_pd(v1_final_sub), 0xA));
        }

        template<bool invert>
        [[msvc::forceinline]] static void forward(const float* input, float* output)
        {
            const constants_type& C = invert ? inv_consts : fwd_consts;
            __m256 v0 = _mm256_load_ps(input);
            __m256 v1 = _mm256_load_ps(input + 8);

            compute_inplace<invert>(v0, v1, C);

            if constexpr (invert) {
                __m256 scale = _mm256_load_ps(C.inv_scale);
                v0 = _mm256_mul_ps(v0, scale);
                v1 = _mm256_mul_ps(v1, scale);
            }

            __m256d t0 = _mm256_unpacklo_pd(_mm256_castps_pd(v0), _mm256_castps_pd(v1));
            __m256d t1 = _mm256_unpackhi_pd(_mm256_castps_pd(v0), _mm256_castps_pd(v1));
            _mm256_store_ps(output, _mm256_castpd_ps(t0));
            _mm256_store_ps(output + 8, _mm256_castpd_ps(t1));
        }
    };

    struct FFT1D_C2C_kernel_16_Radix_4x4 : micro_kernel_common
    {
        struct alignas(32) constants_type
        {
            float w_row1_n[8], w_row1_s[8];
            float w_row2_n[8], w_row2_s[8];
            float w_row3_n[8], w_row3_s[8];
            float inv_scale[8];
            float neg_i_mask[8];
            float pos_i_mask[8];
        };

        static constexpr constants_type fwd_consts = {
            {1.f, 0.f,  c1, -s1,  s2, -s2,  s1, -c1},
            {0.f, 1.f, -s1,  c1, -s2,  s2, -c1,  s1},
            {1.f, 0.f,  s2, -s2,  0.f, -1.f, -s2, -s2},
            {0.f, 1.f, -s2,  s2, -1.f,  0.f, -s2, -s2},
            {1.f, 0.f,  s1, -c1, -s2, -s2, -c1,  s1},
            {0.f, 1.f, -c1,  s1, -s2, -s2,  s1, -c1},
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
            {0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f},
            {-0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f}
        };

        static constexpr constants_type inv_consts = {
            {1.f, 0.f,  c1, s1,  s2, s2,  s1, c1},
            {0.f, 1.f,  s1, c1,  s2, s2,  c1, s1},
            {1.f, 0.f,  s2, s2,  0.f, 1.f, -s2, s2},
            {0.f, 1.f,  s2, s2,  1.f, 0.f,  s2, -s2},
            {1.f, 0.f,  s1, c1, -s2, s2, -c1, -s1},
            {0.f, 1.f,  c1, s1,  s2, -s2, -s1, -c1},
            {0.0625f, 0.0625f, 0.0625f, 0.0625f, 0.0625f, 0.0625f, 0.0625f, 0.0625f},
            {0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f},
            {-0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f}
        };

        template<bool invert>
        [[msvc::forceinline]] static void compute_internal(__m256& r0, __m256& r1, __m256& r2, __m256& r3, const constants_type& C)
        {
            __m256 mask = invert ? _mm256_load_ps(C.pos_i_mask) : _mm256_load_ps(C.neg_i_mask);

            radix4_butterfly_base(r0, r1, r2, r3, mask);

            r1 = fma_complexf32(r1, _mm256_load_ps(C.w_row1_n), _mm256_load_ps(C.w_row1_s));
            r2 = fma_complexf32(r2, _mm256_load_ps(C.w_row2_n), _mm256_load_ps(C.w_row2_s));
            r3 = fma_complexf32(r3, _mm256_load_ps(C.w_row3_n), _mm256_load_ps(C.w_row3_s));

            transpose_4x4_complexf32(r0, r1, r2, r3);
            radix4_butterfly_base(r0, r1, r2, r3, mask);
        }

        template<bool invert>
        [[msvc::forceinline]] void forward(const float* input, float* output)
        {
            constexpr const constants_type& C = invert ? inv_consts : fwd_consts;

            __m256 r0 = _mm256_load_ps(input + 0);
            __m256 r1 = _mm256_load_ps(input + 8);
            __m256 r2 = _mm256_load_ps(input + 16);
            __m256 r3 = _mm256_load_ps(input + 24);

            compute_internal<invert>(r0, r1, r2, r3, C);

            if constexpr (invert)
            {
                scale_all(_mm256_load_ps(C.inv_scale), r0, r1, r2, r3);
            }

            _mm256_store_ps(output + 0, r0);
            _mm256_store_ps(output + 8, r1);
            _mm256_store_ps(output + 16, r2);
            _mm256_store_ps(output + 24, r3);
        }
    };

    struct FFT1D_C2C_kernel_32_Radix_2x4x4 : micro_kernel_common, FFT1D_C2C_kernel_16_Radix_4x4
    {
        using k16 = FFT1D_C2C_kernel_16_Radix_4x4;

        static constexpr float c16 = 0.9807852804032304491f;
        static constexpr float s16 = 0.1950903220161282678f;
        static constexpr float c3_16 = 0.8314696123025452370f;
        static constexpr float s3_16 = 0.5555702330196022247f;

        struct alignas(32) constants_type
        {
            float tw0_norm[8], tw0_swap[8];
            float tw1_norm[8], tw1_swap[8];
            float tw2_norm[8], tw2_swap[8];
            float tw3_norm[8], tw3_swap[8];
            float inv_scale[8];
        };

        static constexpr constants_type fwd_consts = {
            {1.0f, 0.0f,  c16, -s16,  c1, -s1,  c3_16, -s3_16},
            {0.0f, 1.0f, -s16,  c16, -s1,  c1, -s3_16,  c3_16},
            {s2,  -s2,   s3_16,-c3_16, s1, -c1,  s16,  -c16},
            {-s2,  s2,  -c3_16, s3_16,-c1,  s1, -c16,   s16},
            {0.0f,-1.0f, -s16, -c16, -s1, -c1, -s3_16, -c3_16},
            {-1.0f,0.0f, -c16, -s16, -c1, -s1, -c3_16, -s3_16},
            {-s2, -s2,  -c3_16,-s3_16,-c1, -s1, -c16,  -s16},
            {-s2, -s2,  -s3_16,-c3_16,-s1, -c1, -s16,  -c16},
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}
        };

        static constexpr constants_type inv_consts = {
            {1.0f, 0.0f,  c16,  s16,  c1,  s1,  c3_16,  s3_16},
            {0.0f, 1.0f,  s16,  c16,  s1,  c1,  s3_16,  c3_16},
            {s2,   s2,    s3_16, c3_16, s1,  c1,  s16,  c16},
            {s2,   s2,    c3_16, s3_16, c1,  s1,  c16,  s16},
            {0.0f, 1.0f, -s16,  c16, -s1,  c1, -s3_16,  c3_16},
            {1.0f, 0.0f,  c16, -s16,  c1, -s1,  c3_16, -s3_16},
            {-s2,  s2,   -c3_16, s3_16,-c1,  s1, -c16,  s16},
            {s2,  -s2,    s3_16,-c3_16, s1, -c1,  s16, -c16},
            {0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f, 0.03125f}
        };

        template<bool invert>
        [[msvc::forceinline]] void forward(const float* input, float* output)
        {
            const auto& C32 = invert ? inv_consts : fwd_consts;
            const auto& C16 = invert ? k16::inv_consts : k16::fwd_consts;

            __m256 r0 = _mm256_load_ps(input + 0);
            __m256 r1 = _mm256_load_ps(input + 8);
            __m256 r2 = _mm256_load_ps(input + 16);
            __m256 r3 = _mm256_load_ps(input + 24);
            __m256 r4 = _mm256_load_ps(input + 32);
            __m256 r5 = _mm256_load_ps(input + 40);
            __m256 r6 = _mm256_load_ps(input + 48);
            __m256 r7 = _mm256_load_ps(input + 56);

            separate_even_odd(r0, r1);
            separate_even_odd(r2, r3);
            separate_even_odd(r4, r5);
            separate_even_odd(r6, r7);

            compute_internal<invert>(r0, r2, r4, r6, C16);
            compute_internal<invert>(r1, r3, r5, r7, C16);

            radix2_butterfly(r0, r1, _mm256_load_ps(C32.tw0_norm), _mm256_load_ps(C32.tw0_swap));
            radix2_butterfly(r2, r3, _mm256_load_ps(C32.tw1_norm), _mm256_load_ps(C32.tw1_swap));
            radix2_butterfly(r4, r5, _mm256_load_ps(C32.tw2_norm), _mm256_load_ps(C32.tw2_swap));
            radix2_butterfly(r6, r7, _mm256_load_ps(C32.tw3_norm), _mm256_load_ps(C32.tw3_swap));

            if constexpr (invert)
            {
                scale_all(_mm256_load_ps(C32.inv_scale), r0, r1, r2, r3, r4, r5, r6, r7);
            }

            _mm256_store_ps(output + 0, r0);  
            _mm256_store_ps(output + 8, r2);
            _mm256_store_ps(output + 16, r4);
            _mm256_store_ps(output + 24, r6);
            _mm256_store_ps(output + 32, r1);
            _mm256_store_ps(output + 40, r3);
            _mm256_store_ps(output + 48, r5); 
            _mm256_store_ps(output + 56, r7);
        }
    };

    struct FFT1D_C2C_kernel_64_Radix_4x4x4 : micro_kernel_common
    {
        struct alignas(32) constants_type
        {
            float w16_row1_n[8], w16_row1_s[8];
            float w16_row2_n[8], w16_row2_s[8];
            float w16_row3_n[8], w16_row3_s[8];

            float w64_r1_0_n[8], w64_r1_0_s[8];
            float w64_r1_1_n[8], w64_r1_1_s[8];
            float w64_r1_2_n[8], w64_r1_2_s[8];
            float w64_r1_3_n[8], w64_r1_3_s[8];

            float w64_r2_0_n[8], w64_r2_0_s[8];
            float w64_r2_1_n[8], w64_r2_1_s[8];
            float w64_r2_2_n[8], w64_r2_2_s[8];
            float w64_r2_3_n[8], w64_r2_3_s[8];

            float w64_r3_0_n[8], w64_r3_0_s[8];
            float w64_r3_1_n[8], w64_r3_1_s[8];
            float w64_r3_2_n[8], w64_r3_2_s[8];
            float w64_r3_3_n[8], w64_r3_3_s[8];

            float inv_scale[8];
            float neg_i_mask[8];
            float pos_i_mask[8];
        };

        static constexpr constants_type fwd_consts = {
            {1.f, 0.f,  c1, -s1,  s2, -s2,  s1, -c1}, 
            {0.f, 1.f, -s1,  c1, -s2,  s2, -c1,  s1},
            {1.f, 0.f,  s2, -s2,  0.f, -1.f, -s2, -s2}, 
            {0.f, 1.f, -s2,  s2, -1.f,  0.f, -s2, -s2},
            {1.f, 0.f,  s1, -c1, -s2, -s2, -c1,  s1}, 
            {0.f, 1.f, -c1,  s1, -s2, -s2,  s1, -c1},

            {1.000000000f, 0.000000000f, 0.995184727f, -0.098017140f, 0.980785280f, -0.195090322f, 0.956940336f, -0.290284677f},
            {0.000000000f, 1.000000000f, -0.098017140f, 0.995184727f, -0.195090322f, 0.980785280f, -0.290284677f, 0.956940336f},
            {0.923879533f, -0.382683432f, 0.881921264f, -0.471396737f, 0.831469612f, -0.555570233f, 0.773010453f, -0.634393284f},
            {-0.382683432f, 0.923879533f, -0.471396737f, 0.881921264f, -0.555570233f, 0.831469612f, -0.634393284f, 0.773010453f},
            {0.707106781f, -0.707106781f, 0.634393284f, -0.773010453f, 0.555570233f, -0.831469612f, 0.471396737f, -0.881921264f},
            {-0.707106781f, 0.707106781f, -0.773010453f, 0.634393284f, -0.831469612f, 0.555570233f, -0.881921264f, 0.471396737f},
            {0.382683432f, -0.923879533f, 0.290284677f, -0.956940336f, 0.195090322f, -0.980785280f, 0.098017140f, -0.995184727f},
            {-0.923879533f, 0.382683432f, -0.956940336f, 0.290284677f, -0.980785280f, 0.195090322f, -0.995184727f, 0.098017140f},
            {1.000000000f, 0.000000000f, 0.980785280f, -0.195090322f, 0.923879533f, -0.382683432f, 0.831469612f, -0.555570233f},
            {0.000000000f, 1.000000000f, -0.195090322f, 0.980785280f, -0.382683432f, 0.923879533f, -0.555570233f, 0.831469612f},
            {0.707106781f, -0.707106781f, 0.555570233f, -0.831469612f, 0.382683432f, -0.923879533f, 0.195090322f, -0.980785280f},
            {-0.707106781f, 0.707106781f, -0.831469612f, 0.555570233f, -0.923879533f, 0.382683432f, -0.980785280f, 0.195090322f},
            {0.000000000f, -1.000000000f, -0.195090322f, -0.980785280f, -0.382683432f, -0.923879533f, -0.555570233f, -0.831469612f},
            {-1.000000000f, 0.000000000f, -0.980785280f, -0.195090322f, -0.923879533f, -0.382683432f, -0.831469612f, -0.555570233f},
            {-0.707106781f, -0.707106781f, -0.831469612f, -0.555570233f, -0.923879533f, -0.382683432f, -0.980785280f, -0.195090322f},
            {-0.707106781f, -0.707106781f, -0.555570233f, -0.831469612f, -0.382683432f, -0.923879533f, -0.195090322f, -0.980785280f},
            {1.000000000f, 0.000000000f, 0.956940336f, -0.290284677f, 0.831469612f, -0.555570233f, 0.634393284f, -0.773010453f},
            {0.000000000f, 1.000000000f, -0.290284677f, 0.956940336f, -0.555570233f, 0.831469612f, -0.773010453f, 0.634393284f},
            {0.382683432f, -0.923879533f, 0.098017140f, -0.995184727f, -0.195090322f, -0.980785280f, -0.471396737f, -0.881921264f},
            {-0.923879533f, 0.382683432f, -0.995184727f, 0.098017140f, -0.980785280f, -0.195090322f, -0.881921264f, -0.471396737f},
            {-0.707106781f, -0.707106781f, -0.881921264f, -0.471396737f, -0.980785280f, -0.195090322f, -0.995184727f, 0.098017140f},
            {-0.707106781f, -0.707106781f, -0.471396737f, -0.881921264f, -0.195090322f, -0.980785280f, 0.098017140f, -0.995184727f},
            {-0.923879533f, 0.382683432f, -0.773010453f, 0.634393284f, -0.555570233f, 0.831469612f, -0.290284677f, 0.956940336f},
            {0.382683432f, -0.923879533f, 0.634393284f, -0.773010453f, 0.831469612f, -0.555570233f, 0.956940336f, -0.290284677f},

            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
            {0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f},
            {-0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f}
        };

        static constexpr constants_type inv_consts = {
            {1.f, 0.f,  c1, s1,  s2, s2,  s1, c1}, 
            {0.f, 1.f,  s1, c1,  s2, s2,  c1, s1},
            {1.f, 0.f,  s2, s2,  0.f, 1.f, -s2, s2}, 
            {0.f, 1.f,  s2, s2,  1.f, 0.f,  s2, -s2},
            {1.f, 0.f,  s1, c1, -s2, s2, -c1, -s1}, 
            {0.f, 1.f,  c1, s1,  s2, -s2, -s1, -c1},

            {1.000000000f, 0.000000000f, 0.995184727f, 0.098017140f, 0.980785280f, 0.195090322f, 0.956940336f, 0.290284677f},
            {0.000000000f, 1.000000000f, 0.098017140f, 0.995184727f, 0.195090322f, 0.980785280f, 0.290284677f, 0.956940336f},
            {0.923879533f, 0.382683432f, 0.881921264f, 0.471396737f, 0.831469612f, 0.555570233f, 0.773010453f, 0.634393284f},
            {0.382683432f, 0.923879533f, 0.471396737f, 0.881921264f, 0.555570233f, 0.831469612f, 0.634393284f, 0.773010453f},
            {0.707106781f, 0.707106781f, 0.634393284f, 0.773010453f, 0.555570233f, 0.831469612f, 0.471396737f, 0.881921264f},
            {0.707106781f, 0.707106781f, 0.773010453f, 0.634393284f, 0.831469612f, 0.555570233f, 0.881921264f, 0.471396737f},
            {0.382683432f, 0.923879533f, 0.290284677f, 0.956940336f, 0.195090322f, 0.980785280f, 0.098017140f, 0.995184727f},
            {0.923879533f, 0.382683432f, 0.956940336f, 0.290284677f, 0.980785280f, 0.195090322f, 0.995184727f, 0.098017140f},
            {1.000000000f, 0.000000000f, 0.980785280f, 0.195090322f, 0.923879533f, 0.382683432f, 0.831469612f, 0.555570233f},
            {0.000000000f, 1.000000000f, 0.195090322f, 0.980785280f, 0.382683432f, 0.923879533f, 0.555570233f, 0.831469612f},
            {0.707106781f, 0.707106781f, 0.555570233f, 0.831469612f, 0.382683432f, 0.923879533f, 0.195090322f, 0.980785280f},
            {0.707106781f, 0.707106781f, 0.831469612f, 0.555570233f, 0.923879533f, 0.382683432f, 0.980785280f, 0.195090322f},
            {0.000000000f, 1.000000000f, -0.195090322f, 0.980785280f, -0.382683432f, 0.923879533f, -0.555570233f, 0.831469612f},
            {1.000000000f, 0.000000000f, 0.980785280f, -0.195090322f, 0.923879533f, -0.382683432f, 0.831469612f, -0.555570233f},
            {-0.707106781f, 0.707106781f, -0.831469612f, 0.555570233f, -0.923879533f, 0.382683432f, -0.980785280f, 0.195090322f},
            {0.707106781f, -0.707106781f, 0.555570233f, -0.831469612f, 0.382683432f, -0.923879533f, 0.195090322f, -0.980785280f},
            {1.000000000f, 0.000000000f, 0.956940336f, 0.290284677f, 0.831469612f, 0.555570233f, 0.634393284f, 0.773010453f},
            {0.000000000f, 1.000000000f, 0.290284677f, 0.956940336f, 0.555570233f, 0.831469612f, 0.773010453f, 0.634393284f},
            {0.382683432f, 0.923879533f, 0.098017140f, 0.995184727f, -0.195090322f, 0.980785280f, -0.471396737f, 0.881921264f},
            {0.923879533f, 0.382683432f, 0.995184727f, 0.098017140f, 0.980785280f, -0.195090322f, 0.881921264f, -0.471396737f},
            {-0.707106781f, 0.707106781f, -0.881921264f, 0.471396737f, -0.980785280f, 0.195090322f, -0.995184727f, -0.098017140f},
            {0.707106781f, -0.707106781f, 0.471396737f, -0.881921264f, 0.195090322f, -0.980785280f, -0.098017140f, -0.995184727f},
            {-0.923879533f, -0.382683432f, -0.773010453f, -0.634393284f, -0.555570233f, -0.831469612f, -0.290284677f, -0.956940336f},
            {-0.382683432f, -0.923879533f, -0.634393284f, -0.773010453f, -0.831469612f, -0.555570233f, -0.956940336f, -0.290284677f},

            {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
            {0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f},
            {-0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f}
        };

        template<bool invert>
        [[msvc::forceinline]] void forward(const float* input, float* output)
        {
            const constants_type& C = invert ? inv_consts : fwd_consts;

            __m256 rxx[16];
            for (std::size_t i = 0; i < 16; ++i)
            {
                rxx[i] = _mm256_load_ps(input + (i * 8));
            }

            for (std::size_t i = 0; i < 4; ++i)
            {
                radix4_butterfly<invert>(rxx[i], rxx[i + 4], rxx[i + 8], rxx[i + 12], C);
            }

            rxx[4] = fma_complexf32(rxx[4], _mm256_load_ps(C.w64_r1_0_n), _mm256_load_ps(C.w64_r1_0_s));
            rxx[5] = fma_complexf32(rxx[5], _mm256_load_ps(C.w64_r1_1_n), _mm256_load_ps(C.w64_r1_1_s));
            rxx[6] = fma_complexf32(rxx[6], _mm256_load_ps(C.w64_r1_2_n), _mm256_load_ps(C.w64_r1_2_s));
            rxx[7] = fma_complexf32(rxx[7], _mm256_load_ps(C.w64_r1_3_n), _mm256_load_ps(C.w64_r1_3_s));

            rxx[8] = fma_complexf32(rxx[8], _mm256_load_ps(C.w64_r2_0_n), _mm256_load_ps(C.w64_r2_0_s));
            rxx[9] = fma_complexf32(rxx[9], _mm256_load_ps(C.w64_r2_1_n), _mm256_load_ps(C.w64_r2_1_s));
            rxx[10] = fma_complexf32(rxx[10], _mm256_load_ps(C.w64_r2_2_n), _mm256_load_ps(C.w64_r2_2_s));
            rxx[11] = fma_complexf32(rxx[11], _mm256_load_ps(C.w64_r2_3_n), _mm256_load_ps(C.w64_r2_3_s));

            rxx[12] = fma_complexf32(rxx[12], _mm256_load_ps(C.w64_r3_0_n), _mm256_load_ps(C.w64_r3_0_s));
            rxx[13] = fma_complexf32(rxx[13], _mm256_load_ps(C.w64_r3_1_n), _mm256_load_ps(C.w64_r3_1_s));
            rxx[14] = fma_complexf32(rxx[14], _mm256_load_ps(C.w64_r3_2_n), _mm256_load_ps(C.w64_r3_2_s));
            rxx[15] = fma_complexf32(rxx[15], _mm256_load_ps(C.w64_r3_3_n), _mm256_load_ps(C.w64_r3_3_s));

            forward_16_inner<invert>(rxx[0], rxx[1], rxx[2], rxx[3], C);
            forward_16_inner<invert>(rxx[4], rxx[5], rxx[6], rxx[7], C);
            forward_16_inner<invert>(rxx[8], rxx[9], rxx[10], rxx[11], C);
            forward_16_inner<invert>(rxx[12], rxx[13], rxx[14], rxx[15], C);

            if constexpr (invert)
            {
                __m256 inv = _mm256_load_ps(C.inv_scale);
                for (std::size_t i = 0; i < 16; ++i)
                {
                    rxx[i] = _mm256_mul_ps(rxx[i], inv);
                }
            }

            for (std::size_t i = 0; i < 4; ++i)
            {
                transpose_4x4_linear(rxx[i], rxx[i + 4], rxx[i + 8], rxx[i + 12]);

                float* dst = output + (i * 32);

                for (std::size_t j = 0; j < 4; ++j)
                {
                    _mm256_store_ps(dst + (j * 8), rxx[i + (j * 4)]);
                }
            }
        }

    private:
        [[msvc::forceinline]] static void transpose_4x4_linear(__m256& r0, __m256& r1, __m256& r2, __m256& r3)
        {
            __m256d r0d = _mm256_castps_pd(r0);
            __m256d r1d = _mm256_castps_pd(r1);
            __m256d r2d = _mm256_castps_pd(r2);
            __m256d r3d = _mm256_castps_pd(r3);

            __m256d t0 = _mm256_unpacklo_pd(r0d, r1d);
            __m256d t1 = _mm256_unpackhi_pd(r0d, r1d);
            __m256d t2 = _mm256_unpacklo_pd(r2d, r3d);
            __m256d t3 = _mm256_unpackhi_pd(r2d, r3d);

            r0 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x20));
            r1 = _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x20));
            r2 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x31));
            r3 = _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x31));
        }

        template<bool invert>
        [[msvc::forceinline]] static void forward_16_inner(__m256& r0, __m256& r1, __m256& r2, __m256& r3, const constants_type& C)
        {
            radix4_butterfly<invert>(r0, r1, r2, r3, C);

            r1 = fma_complexf32(r1, _mm256_load_ps(C.w16_row1_n), _mm256_load_ps(C.w16_row1_s));
            r2 = fma_complexf32(r2, _mm256_load_ps(C.w16_row2_n), _mm256_load_ps(C.w16_row2_s));
            r3 = fma_complexf32(r3, _mm256_load_ps(C.w16_row3_n), _mm256_load_ps(C.w16_row3_s));

            transpose_4x4_linear(r0, r1, r2, r3);
            radix4_butterfly<invert>(r0, r1, r2, r3, C);
        }

        template<bool invert>
        [[msvc::forceinline]] static void radix4_butterfly(__m256& r0, __m256& r1, __m256& r2, __m256& r3, const constants_type& C)
        {
            __m256 t0 = _mm256_add_ps(r0, r2);
            __m256 t1 = _mm256_sub_ps(r0, r2);
            __m256 t2 = _mm256_add_ps(r1, r3);
            __m256 t3 = _mm256_sub_ps(r1, r3);

            r0 = _mm256_add_ps(t0, t2);
            r2 = _mm256_sub_ps(t0, t2);

            __m256 t3_swapped = _mm256_permute_ps(t3, 0xB1);
            __m256 t3_rotated;

            if constexpr (!invert)
            {
                t3_rotated = _mm256_xor_ps(t3_swapped, _mm256_load_ps(C.neg_i_mask));
                r1 = _mm256_add_ps(t1, t3_rotated);
                r3 = _mm256_sub_ps(t1, t3_rotated);
            }
            else
            {
                t3_rotated = _mm256_xor_ps(t3_swapped, _mm256_load_ps(C.pos_i_mask));
                r1 = _mm256_add_ps(t1, t3_rotated);
                r3 = _mm256_sub_ps(t1, t3_rotated);
            }
        }
    };

    struct FFT1D_C2C_kernel_128_Radix_2x4x4x4 : public FFT1D_C2C_kernel_64_Radix_4x4x4
    {
        using k64 = FFT1D_C2C_kernel_64_Radix_4x4x4;

        struct alignas(32) constants_type
        {
            float w_n[128];
            float w_s[128];
            float half_scale[8];
        };

        static constexpr constants_type fwd_consts = {
            {
                 1.000000f,  0.000000f,  0.998795f, -0.049068f,  0.995185f, -0.098017f,  0.989177f, -0.146730f,
                 0.980785f, -0.195090f,  0.970031f, -0.242980f,  0.956940f, -0.290285f,  0.941544f, -0.336890f,
                 0.923880f, -0.382683f,  0.903989f, -0.427555f,  0.881921f, -0.471397f,  0.857729f, -0.514103f,
                 0.831470f, -0.555570f,  0.803208f, -0.595699f,  0.773010f, -0.634393f,  0.740951f, -0.671559f,
                 0.707107f, -0.707107f,  0.671559f, -0.740951f,  0.634393f, -0.773010f,  0.595699f, -0.803208f,
                 0.555570f, -0.831470f,  0.514103f, -0.857729f,  0.471397f, -0.881921f,  0.427555f, -0.903989f,
                 0.382683f, -0.923880f,  0.336890f, -0.941544f,  0.290285f, -0.956940f,  0.242980f, -0.970031f,
                 0.195090f, -0.980785f,  0.146730f, -0.989177f,  0.098017f, -0.995185f,  0.049068f, -0.998795f,
                 0.000000f, -1.000000f, -0.049068f, -0.998795f, -0.098017f, -0.995185f, -0.146730f, -0.989177f,
                -0.195090f, -0.980785f, -0.242980f, -0.970031f, -0.290285f, -0.956940f, -0.336890f, -0.941544f,
                -0.382683f, -0.923880f, -0.427555f, -0.903989f, -0.471397f, -0.881921f, -0.514103f, -0.857729f,
                -0.555570f, -0.831470f, -0.595699f, -0.803208f, -0.634393f, -0.773010f, -0.671559f, -0.740951f,
                -0.707107f, -0.707107f, -0.740951f, -0.671559f, -0.773010f, -0.634393f, -0.803208f, -0.595699f,
                -0.831470f, -0.555570f, -0.857729f, -0.514103f, -0.881921f, -0.471397f, -0.903989f, -0.427555f,
                -0.923880f, -0.382683f, -0.941544f, -0.336890f, -0.956940f, -0.290285f, -0.970031f, -0.242980f,
                -0.980785f, -0.195090f, -0.989177f, -0.146730f, -0.995185f, -0.098017f, -0.998795f, -0.049068f
            },
            {
                 0.000000f,  1.000000f, -0.049068f,  0.998795f, -0.098017f,  0.995185f, -0.146730f,  0.989177f,
                -0.195090f,  0.980785f, -0.242980f,  0.970031f, -0.290285f,  0.956940f, -0.336890f,  0.941544f,
                -0.382683f,  0.923880f, -0.427555f,  0.903989f, -0.471397f,  0.881921f, -0.514103f,  0.857729f,
                -0.555570f,  0.831470f, -0.595699f,  0.803208f, -0.634393f,  0.773010f, -0.671559f,  0.740951f,
                -0.707107f,  0.707107f, -0.740951f,  0.671559f, -0.773010f,  0.634393f, -0.803208f,  0.595699f,
                -0.831470f,  0.555570f, -0.857729f,  0.514103f, -0.881921f,  0.471397f, -0.903989f,  0.427555f,
                -0.923880f,  0.382683f, -0.941544f,  0.336890f, -0.956940f,  0.290285f, -0.970031f,  0.242980f,
                -0.980785f,  0.195090f, -0.989177f,  0.146730f, -0.995185f,  0.098017f, -0.998795f,  0.049068f,
                -1.000000f,  0.000000f, -0.998795f, -0.049068f, -0.995185f, -0.098017f, -0.989177f, -0.146730f,
                -0.980785f, -0.195090f, -0.970031f, -0.242980f, -0.956940f, -0.290285f, -0.941544f, -0.336890f,
                -0.923880f, -0.382683f, -0.903989f, -0.427555f, -0.881921f, -0.471397f, -0.857729f, -0.514103f,
                -0.831470f, -0.555570f, -0.803208f, -0.595699f, -0.773010f, -0.634393f, -0.740951f, -0.671559f,
                -0.707107f, -0.707107f, -0.671559f, -0.740951f, -0.634393f, -0.773010f, -0.595699f, -0.803208f,
                -0.555570f, -0.831470f, -0.514103f, -0.857729f, -0.471397f, -0.881921f, -0.427555f, -0.903989f,
                -0.382683f, -0.923880f, -0.336890f, -0.941544f, -0.290285f, -0.956940f, -0.242980f, -0.970031f,
                -0.195090f, -0.980785f, -0.146730f, -0.989177f, -0.098017f, -0.995185f, -0.049068f, -0.998795f
            },
            {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}
        };

        static constexpr constants_type inv_consts = {
            {
                 1.000000f,  0.000000f,  0.998795f,  0.049068f,  0.995185f,  0.098017f,  0.989177f,  0.146730f,
                 0.980785f,  0.195090f,  0.970031f,  0.242980f,  0.956940f,  0.290285f,  0.941544f,  0.336890f,
                 0.923880f,  0.382683f,  0.903989f,  0.427555f,  0.881921f,  0.471397f,  0.857729f,  0.514103f,
                 0.831470f,  0.555570f,  0.803208f,  0.595699f,  0.773010f,  0.634393f,  0.740951f,  0.671559f,
                 0.707107f,  0.707107f,  0.671559f,  0.740951f,  0.634393f,  0.773010f,  0.595699f,  0.803208f,
                 0.555570f,  0.831470f,  0.514103f,  0.857729f,  0.471397f,  0.881921f,  0.427555f,  0.903989f,
                 0.382683f,  0.923880f,  0.336890f,  0.941544f,  0.290285f,  0.956940f,  0.242980f,  0.970031f,
                 0.195090f,  0.980785f,  0.146730f,  0.989177f,  0.098017f,  0.995185f,  0.049068f,  0.998795f,
                 0.000000f,  1.000000f, -0.049068f,  0.998795f, -0.098017f,  0.995185f, -0.146730f,  0.989177f,
                -0.195090f,  0.980785f, -0.242980f,  0.970031f, -0.290285f,  0.956940f, -0.336890f,  0.941544f,
                -0.382683f,  0.923880f, -0.427555f,  0.903989f, -0.471397f,  0.881921f, -0.514103f,  0.857729f,
                -0.555570f,  0.831470f, -0.595699f,  0.803208f, -0.634393f,  0.773010f, -0.671559f,  0.740951f,
                -0.707107f,  0.707107f, -0.740951f,  0.671559f, -0.773010f,  0.634393f, -0.803208f,  0.595699f,
                -0.831470f,  0.555570f, -0.857729f,  0.514103f, -0.881921f,  0.471397f, -0.903989f,  0.427555f,
                -0.923880f,  0.382683f, -0.941544f,  0.336890f, -0.956940f,  0.290285f, -0.970031f,  0.242980f,
                -0.980785f,  0.195090f, -0.989177f,  0.146730f, -0.995185f,  0.098017f, -0.998795f,  0.049068f
            },
            {
                 0.000000f,  1.000000f,  0.049068f,  0.998795f,  0.098017f,  0.995185f,  0.146730f,  0.989177f,
                 0.195090f,  0.980785f,  0.242980f,  0.970031f,  0.290285f,  0.956940f,  0.336890f,  0.941544f,
                 0.382683f,  0.923880f,  0.427555f,  0.903989f,  0.471397f,  0.881921f,  0.514103f,  0.857729f,
                 0.555570f,  0.831470f,  0.595699f,  0.803208f,  0.634393f,  0.773010f,  0.671559f,  0.740951f,
                 0.707107f,  0.707107f,  0.740951f,  0.671559f,  0.773010f,  0.634393f,  0.803208f,  0.595699f,
                 0.831470f,  0.555570f,  0.857729f,  0.514103f,  0.881921f,  0.471397f,  0.903989f,  0.427555f,
                 0.923880f,  0.382683f,  0.941544f,  0.336890f,  0.956940f,  0.290285f,  0.970031f,  0.242980f,
                 0.980785f,  0.195090f,  0.989177f,  0.146730f,  0.995185f,  0.098017f,  0.998795f,  0.049068f,
                 1.000000f,  0.000000f,  0.998795f, -0.049068f,  0.995185f, -0.098017f,  0.989177f, -0.146730f,
                 0.980785f, -0.195090f,  0.970031f, -0.242980f,  0.956940f, -0.290285f,  0.941544f, -0.336890f,
                 0.923880f, -0.382683f,  0.903989f, -0.427555f,  0.881921f, -0.471397f,  0.857729f, -0.514103f,
                 0.831470f, -0.555570f,  0.803208f, -0.595699f,  0.773010f, -0.634393f,  0.740951f, -0.671559f,
                 0.707107f, -0.707107f,  0.671559f, -0.740951f,  0.634393f, -0.773010f,  0.595699f, -0.803208f,
                 0.555570f, -0.831470f,  0.514103f, -0.857729f,  0.471397f, -0.881921f,  0.427555f, -0.903989f,
                 0.382683f, -0.923880f,  0.336890f, -0.941544f,  0.290285f, -0.956940f,  0.242980f, -0.970031f,
                 0.195090f, -0.980785f,  0.146730f, -0.989177f,  0.098017f, -0.995185f,  0.049068f, -0.998795f
            },
            {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}
        };

        template<bool invert>
        [[msvc::forceinline]] void forward(const float* input, float* output)
        {
            const constants_type& C = invert ? inv_consts : fwd_consts;

            alignas(32) float temp[256];

            for (std::size_t i = 0; i < 4; ++i)
            {
                std::size_t offset = i * 32;
                std::size_t half_offset = 128;

                __m256 r0 = _mm256_load_ps(input + offset + 0);
                __m256 r1 = _mm256_load_ps(input + offset + 8);
                __m256 r2 = _mm256_load_ps(input + offset + 16);
                __m256 r3 = _mm256_load_ps(input + offset + 24);

                __m256 r4 = _mm256_load_ps(input + offset + half_offset + 0);
                __m256 r5 = _mm256_load_ps(input + offset + half_offset + 8);
                __m256 r6 = _mm256_load_ps(input + offset + half_offset + 16);
                __m256 r7 = _mm256_load_ps(input + offset + half_offset + 24);

                __m256 t0 = _mm256_add_ps(r0, r4);
                __m256 t4 = _mm256_sub_ps(r0, r4);
                __m256 t1 = _mm256_add_ps(r1, r5);
                __m256 t5 = _mm256_sub_ps(r1, r5);
                __m256 t2 = _mm256_add_ps(r2, r6);
                __m256 t6 = _mm256_sub_ps(r2, r6);
                __m256 t3 = _mm256_add_ps(r3, r7);
                __m256 t7 = _mm256_sub_ps(r3, r7);

                __m256 w0_n = _mm256_load_ps(C.w_n + offset + 0);
                __m256 w0_s = _mm256_load_ps(C.w_s + offset + 0);
                __m256 w1_n = _mm256_load_ps(C.w_n + offset + 8);
                __m256 w1_s = _mm256_load_ps(C.w_s + offset + 8);
                __m256 w2_n = _mm256_load_ps(C.w_n + offset + 16);
                __m256 w2_s = _mm256_load_ps(C.w_s + offset + 16);
                __m256 w3_n = _mm256_load_ps(C.w_n + offset + 24);
                __m256 w3_s = _mm256_load_ps(C.w_s + offset + 24);

                t4 = fma_complexf32(t4, w0_n, w0_s);
                t5 = fma_complexf32(t5, w1_n, w1_s);
                t6 = fma_complexf32(t6, w2_n, w2_s);
                t7 = fma_complexf32(t7, w3_n, w3_s);

                _mm256_store_ps(temp + offset + 0, t0);
                _mm256_store_ps(temp + offset + 8, t1);
                _mm256_store_ps(temp + offset + 16, t2);
                _mm256_store_ps(temp + offset + 24, t3);

                _mm256_store_ps(temp + offset + half_offset + 0, t4);
                _mm256_store_ps(temp + offset + half_offset + 8, t5);
                _mm256_store_ps(temp + offset + half_offset + 16, t6);
                _mm256_store_ps(temp + offset + half_offset + 24, t7);
            }


            k64::template forward<invert>(temp, temp);
            k64::template forward<invert>(temp + 128, temp + 128);

            __m256 inv_scale_vec;
            if constexpr (invert) 
            {
                inv_scale_vec = _mm256_load_ps(C.half_scale);
            }

            for (std::size_t i = 0; i < 16; ++i)
            {
                __m256 evens = _mm256_load_ps(temp + i * 8);
                __m256 odds = _mm256_load_ps(temp + 128 + i * 8);

                if constexpr (invert) 
                {
                    evens = _mm256_mul_ps(evens, inv_scale_vec);
                    odds = _mm256_mul_ps(odds, inv_scale_vec);
                }

                __m256d evens_d = _mm256_castps_pd(evens);
                __m256d odds_d = _mm256_castps_pd(odds);

                __m256d t0 = _mm256_unpacklo_pd(evens_d, odds_d);
                __m256d t1 = _mm256_unpackhi_pd(evens_d, odds_d);

                __m256 r0 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t1, 0x20));
                __m256 r1 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t1, 0x31));

                _mm256_store_ps(output + i * 16 + 0, r0);
                _mm256_store_ps(output + i * 16 + 8, r1);
            }
        }
    };

    struct FFT1D_C2C_kernel_256_Radix_2x2x4x4x4 : public FFT1D_C2C_kernel_128_Radix_2x4x4x4
    {
        using k128 = FFT1D_C2C_kernel_128_Radix_2x4x4x4;

        struct alignas(32) constants_type
        {
            float w_n[256];
            float w_s[256];
            float half_scale[8];
        };

        static constexpr constants_type fwd_consts = {
            {
                1.000000f,  0.000000f,  0.999699f, -0.024541f,  0.998795f, -0.049068f,  0.997290f, -0.073565f,
                0.995185f, -0.098017f,  0.992480f, -0.122411f,  0.989177f, -0.146730f,  0.985278f, -0.170962f,
                0.980785f, -0.195090f,  0.975702f, -0.219101f,  0.970031f, -0.242980f,  0.963776f, -0.266713f,
                0.956940f, -0.290285f,  0.949528f, -0.313682f,  0.941544f, -0.336890f,  0.932993f, -0.359895f,
                0.923880f, -0.382683f,  0.914210f, -0.405241f,  0.903989f, -0.427555f,  0.893224f, -0.449611f,
                0.881921f, -0.471397f,  0.870087f, -0.492898f,  0.857729f, -0.514103f,  0.844854f, -0.534998f,
                0.831470f, -0.555570f,  0.817585f, -0.575808f,  0.803208f, -0.595699f,  0.788346f, -0.615232f,
                0.773010f, -0.634393f,  0.757209f, -0.653173f,  0.740951f, -0.671559f,  0.724247f, -0.689541f,
                0.707107f, -0.707107f,  0.689541f, -0.724247f,  0.671559f, -0.740951f,  0.653173f, -0.757209f,
                0.634393f, -0.773010f,  0.615232f, -0.788346f,  0.595699f, -0.803208f,  0.575808f, -0.817585f,
                0.555570f, -0.831470f,  0.534998f, -0.844854f,  0.514103f, -0.857729f,  0.492898f, -0.870087f,
                0.471397f, -0.881921f,  0.449611f, -0.893224f,  0.427555f, -0.903989f,  0.405241f, -0.914210f,
                0.382683f, -0.923880f,  0.359895f, -0.932993f,  0.336890f, -0.941544f,  0.313682f, -0.949528f,
                0.290285f, -0.956940f,  0.266713f, -0.963776f,  0.242980f, -0.970031f,  0.219101f, -0.975702f,
                0.195090f, -0.980785f,  0.170962f, -0.985278f,  0.146730f, -0.989177f,  0.122411f, -0.992480f,
                0.098017f, -0.995185f,  0.073565f, -0.997290f,  0.049068f, -0.998795f,  0.024541f, -0.999699f,
                0.000000f, -1.000000f, -0.024541f, -0.999699f, -0.049068f, -0.998795f, -0.073565f, -0.997290f,
               -0.098017f, -0.995185f, -0.122411f, -0.992480f, -0.146730f, -0.989177f, -0.170962f, -0.985278f,
               -0.195090f, -0.980785f, -0.219101f, -0.975702f, -0.242980f, -0.970031f, -0.266713f, -0.963776f,
               -0.290285f, -0.956940f, -0.313682f, -0.949528f, -0.336890f, -0.941544f, -0.359895f, -0.932993f,
               -0.382683f, -0.923880f, -0.405241f, -0.914210f, -0.427555f, -0.903989f, -0.449611f, -0.893224f,
               -0.471397f, -0.881921f, -0.492898f, -0.870087f, -0.514103f, -0.857729f, -0.534998f, -0.844854f,
               -0.555570f, -0.831470f, -0.575808f, -0.817585f, -0.595699f, -0.803208f, -0.615232f, -0.788346f,
               -0.634393f, -0.773010f, -0.653173f, -0.757209f, -0.671559f, -0.740951f, -0.689541f, -0.724247f,
               -0.707107f, -0.707107f, -0.724247f, -0.689541f, -0.740951f, -0.671559f, -0.757209f, -0.653173f,
               -0.773010f, -0.634393f, -0.788346f, -0.615232f, -0.803208f, -0.595699f, -0.817585f, -0.575808f,
               -0.831470f, -0.555570f, -0.844854f, -0.534998f, -0.857729f, -0.514103f, -0.870087f, -0.492898f,
               -0.881921f, -0.471397f, -0.893224f, -0.449611f, -0.903989f, -0.427555f, -0.914210f, -0.405241f,
               -0.923880f, -0.382683f, -0.932993f, -0.359895f, -0.941544f, -0.336890f, -0.949528f, -0.313682f,
               -0.956940f, -0.290285f, -0.963776f, -0.266713f, -0.970031f, -0.242980f, -0.975702f, -0.219101f,
               -0.980785f, -0.195090f, -0.985278f, -0.170962f, -0.989177f, -0.146730f, -0.992480f, -0.122411f,
               -0.995185f, -0.098017f, -0.997290f, -0.073565f, -0.998795f, -0.049068f, -0.999699f, -0.024541f
            },
            {
                0.000000f,  1.000000f, -0.024541f,  0.999699f, -0.049068f,  0.998795f, -0.073565f,  0.997290f,
               -0.098017f,  0.995185f, -0.122411f,  0.992480f, -0.146730f,  0.989177f, -0.170962f,  0.985278f,
               -0.195090f,  0.980785f, -0.219101f,  0.975702f, -0.242980f,  0.970031f, -0.266713f,  0.963776f,
               -0.290285f,  0.956940f, -0.313682f,  0.949528f, -0.336890f,  0.941544f, -0.359895f,  0.932993f,
               -0.382683f,  0.923880f, -0.405241f,  0.914210f, -0.427555f,  0.903989f, -0.449611f,  0.893224f,
               -0.471397f,  0.881921f, -0.492898f,  0.870087f, -0.514103f,  0.857729f, -0.534998f,  0.844854f,
               -0.555570f,  0.831470f, -0.575808f,  0.817585f, -0.595699f,  0.803208f, -0.615232f,  0.788346f,
               -0.634393f,  0.773010f, -0.653173f,  0.757209f, -0.671559f,  0.740951f, -0.689541f,  0.724247f,
               -0.707107f,  0.707107f, -0.724247f,  0.689541f, -0.740951f,  0.671559f, -0.757209f,  0.653173f,
               -0.773010f,  0.634393f, -0.788346f,  0.615232f, -0.803208f,  0.595699f, -0.817585f,  0.575808f,
               -0.831470f,  0.555570f, -0.844854f,  0.534998f, -0.857729f,  0.514103f, -0.870087f,  0.492898f,
               -0.881921f,  0.471397f, -0.893224f,  0.449611f, -0.903989f,  0.427555f, -0.914210f,  0.405241f,
               -0.923880f,  0.382683f, -0.932993f,  0.359895f, -0.941544f,  0.336890f, -0.949528f,  0.313682f,
               -0.956940f,  0.290285f, -0.963776f,  0.266713f, -0.970031f,  0.242980f, -0.975702f,  0.219101f,
               -0.980785f,  0.195090f, -0.985278f,  0.170962f, -0.989177f,  0.146730f, -0.992480f,  0.122411f,
               -0.995185f,  0.098017f, -0.997290f,  0.073565f, -0.998795f,  0.049068f, -0.999699f,  0.024541f,
               -1.000000f,  0.000000f, -0.999699f, -0.024541f, -0.998795f, -0.049068f, -0.997290f, -0.073565f,
               -0.995185f, -0.098017f, -0.992480f, -0.122411f, -0.989177f, -0.146730f, -0.985278f, -0.170962f,
               -0.980785f, -0.195090f, -0.975702f, -0.219101f, -0.970031f, -0.242980f, -0.963776f, -0.266713f,
               -0.956940f, -0.290285f, -0.949528f, -0.313682f, -0.941544f, -0.336890f, -0.932993f, -0.359895f,
               -0.923880f, -0.382683f, -0.914210f, -0.405241f, -0.903989f, -0.427555f, -0.893224f, -0.449611f,
               -0.881921f, -0.471397f, -0.870087f, -0.492898f, -0.857729f, -0.514103f, -0.844854f, -0.534998f,
               -0.831470f, -0.555570f, -0.817585f, -0.575808f, -0.803208f, -0.595699f, -0.788346f, -0.615232f,
               -0.773010f, -0.634393f, -0.757209f, -0.653173f, -0.740951f, -0.671559f, -0.724247f, -0.689541f,
               -0.707107f, -0.707107f, -0.689541f, -0.724247f, -0.671559f, -0.740951f, -0.653173f, -0.757209f,
               -0.634393f, -0.773010f, -0.615232f, -0.788346f, -0.595699f, -0.803208f, -0.575808f, -0.817585f,
               -0.555570f, -0.831470f, -0.534998f, -0.844854f, -0.514103f, -0.857729f, -0.492898f, -0.870087f,
               -0.471397f, -0.881921f, -0.449611f, -0.893224f, -0.427555f, -0.903989f, -0.405241f, -0.914210f,
               -0.382683f, -0.923880f, -0.359895f, -0.932993f, -0.336890f, -0.941544f, -0.313682f, -0.949528f,
               -0.290285f, -0.956940f, -0.266713f, -0.963776f, -0.242980f, -0.970031f, -0.219101f, -0.975702f,
               -0.195090f, -0.980785f, -0.170962f, -0.985278f, -0.146730f, -0.989177f, -0.122411f, -0.992480f,
               -0.098017f, -0.995185f, -0.073565f, -0.997290f, -0.049068f, -0.998795f, -0.024541f, -0.999699f
            },
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}
        };

        static constexpr constants_type inv_consts = {
            {
                1.000000f,  0.000000f,  0.999699f,  0.024541f,  0.998795f,  0.049068f,  0.997290f,  0.073565f,
                0.995185f,  0.098017f,  0.992480f,  0.122411f,  0.989177f,  0.146730f,  0.985278f,  0.170962f,
                0.980785f,  0.195090f,  0.975702f,  0.219101f,  0.970031f,  0.242980f,  0.963776f,  0.266713f,
                0.956940f,  0.290285f,  0.949528f,  0.313682f,  0.941544f,  0.336890f,  0.932993f,  0.359895f,
                0.923880f,  0.382683f,  0.914210f,  0.405241f,  0.903989f,  0.427555f,  0.893224f,  0.449611f,
                0.881921f,  0.471397f,  0.870087f,  0.492898f,  0.857729f,  0.514103f,  0.844854f,  0.534998f,
                0.831470f,  0.555570f,  0.817585f,  0.575808f,  0.803208f,  0.595699f,  0.788346f,  0.615232f,
                0.773010f,  0.634393f,  0.757209f,  0.653173f,  0.740951f,  0.671559f,  0.724247f,  0.689541f,
                0.707107f,  0.707107f,  0.689541f,  0.724247f,  0.671559f,  0.740951f,  0.653173f,  0.757209f,
                0.634393f,  0.773010f,  0.615232f,  0.788346f,  0.595699f,  0.803208f,  0.575808f,  0.817585f,
                0.555570f,  0.831470f,  0.534998f,  0.844854f,  0.514103f,  0.857729f,  0.492898f,  0.870087f,
                0.471397f,  0.881921f,  0.449611f,  0.893224f,  0.427555f,  0.903989f,  0.405241f,  0.914210f,
                0.382683f,  0.923880f,  0.359895f,  0.932993f,  0.336890f,  0.941544f,  0.313682f,  0.949528f,
                0.290285f,  0.956940f,  0.266713f,  0.963776f,  0.242980f,  0.970031f,  0.219101f,  0.975702f,
                0.195090f,  0.980785f,  0.170962f,  0.985278f,  0.146730f,  0.989177f,  0.122411f,  0.992480f,
                0.098017f,  0.995185f,  0.073565f,  0.997290f,  0.049068f,  0.998795f,  0.024541f,  0.999699f,
                0.000000f,  1.000000f, -0.024541f,  0.999699f, -0.049068f,  0.998795f, -0.073565f,  0.997290f,
               -0.098017f,  0.995185f, -0.122411f,  0.992480f, -0.146730f,  0.989177f, -0.170962f,  0.985278f,
               -0.195090f,  0.980785f, -0.219101f,  0.975702f, -0.242980f,  0.970031f, -0.266713f,  0.963776f,
               -0.290285f,  0.956940f, -0.313682f,  0.949528f, -0.336890f,  0.941544f, -0.359895f,  0.932993f,
               -0.382683f,  0.923880f, -0.405241f,  0.914210f, -0.427555f,  0.903989f, -0.449611f,  0.893224f,
               -0.471397f,  0.881921f, -0.492898f,  0.870087f, -0.514103f,  0.857729f, -0.534998f,  0.844854f,
               -0.555570f,  0.831470f, -0.575808f,  0.817585f, -0.595699f,  0.803208f, -0.615232f,  0.788346f,
               -0.634393f,  0.773010f, -0.653173f,  0.757209f, -0.671559f,  0.740951f, -0.689541f,  0.724247f,
               -0.707107f,  0.707107f, -0.724247f,  0.689541f, -0.740951f,  0.671559f, -0.757209f,  0.653173f,
               -0.773010f,  0.634393f, -0.788346f,  0.615232f, -0.803208f,  0.595699f, -0.817585f,  0.575808f,
               -0.831470f,  0.555570f, -0.844854f,  0.534998f, -0.857729f,  0.514103f, -0.870087f,  0.492898f,
               -0.881921f,  0.471397f, -0.893224f,  0.449611f, -0.903989f,  0.427555f, -0.914210f,  0.405241f,
               -0.923880f,  0.382683f, -0.932993f,  0.359895f, -0.941544f,  0.336890f, -0.949528f,  0.313682f,
               -0.956940f,  0.290285f, -0.963776f,  0.266713f, -0.970031f,  0.242980f, -0.975702f,  0.219101f,
               -0.980785f,  0.195090f, -0.985278f,  0.170962f, -0.989177f,  0.146730f, -0.992480f,  0.122411f,
               -0.995185f,  0.098017f, -0.997290f,  0.073565f, -0.998795f,  0.049068f, -0.999699f,  0.024541f
            },
            {
                0.000000f,  1.000000f,  0.024541f,  0.999699f,  0.049068f,  0.998795f,  0.073565f,  0.997290f,
                0.098017f,  0.995185f,  0.122411f,  0.992480f,  0.146730f,  0.989177f,  0.170962f,  0.985278f,
                0.195090f,  0.980785f,  0.219101f,  0.975702f,  0.242980f,  0.970031f,  0.266713f,  0.963776f,
                0.290285f,  0.956940f,  0.313682f,  0.949528f,  0.336890f,  0.941544f,  0.359895f,  0.932993f,
                0.382683f,  0.923880f,  0.405241f,  0.914210f,  0.427555f,  0.903989f,  0.449611f,  0.893224f,
                0.471397f,  0.881921f,  0.492898f,  0.870087f,  0.514103f,  0.857729f,  0.534998f,  0.844854f,
                0.555570f,  0.831470f,  0.575808f,  0.817585f,  0.595699f,  0.803208f,  0.615232f,  0.788346f,
                0.634393f,  0.773010f,  0.653173f,  0.757209f,  0.671559f,  0.740951f,  0.689541f,  0.724247f,
                0.707107f,  0.707107f,  0.724247f,  0.689541f,  0.740951f,  0.671559f,  0.757209f,  0.653173f,
                0.773010f,  0.634393f,  0.788346f,  0.615232f,  0.803208f,  0.595699f,  0.817585f,  0.575808f,
                0.831470f,  0.555570f,  0.844854f,  0.534998f,  0.857729f,  0.514103f,  0.870087f,  0.492898f,
                0.881921f,  0.471397f,  0.893224f,  0.449611f,  0.903989f,  0.427555f,  0.914210f,  0.405241f,
                0.923880f,  0.382683f,  0.932993f,  0.359895f,  0.941544f,  0.336890f,  0.949528f,  0.313682f,
                0.956940f,  0.290285f,  0.963776f,  0.266713f,  0.970031f,  0.242980f,  0.975702f,  0.219101f,
                0.980785f,  0.195090f,  0.985278f,  0.170962f,  0.989177f,  0.146730f,  0.992480f,  0.122411f,
                0.995185f,  0.098017f,  0.997290f,  0.073565f,  0.998795f,  0.049068f,  0.999699f,  0.024541f,
                1.000000f,  0.000000f,  0.999699f, -0.024541f,  0.998795f, -0.049068f,  0.997290f, -0.073565f,
                0.995185f, -0.098017f,  0.992480f, -0.122411f,  0.989177f, -0.146730f,  0.985278f, -0.170962f,
                0.980785f, -0.195090f,  0.975702f, -0.219101f,  0.970031f, -0.242980f,  0.963776f, -0.266713f,
                0.956940f, -0.290285f,  0.949528f, -0.313682f,  0.941544f, -0.336890f,  0.932993f, -0.359895f,
                0.923880f, -0.382683f,  0.914210f, -0.405241f,  0.903989f, -0.427555f,  0.893224f, -0.449611f,
                0.881921f, -0.471397f,  0.870087f, -0.492898f,  0.857729f, -0.514103f,  0.844854f, -0.534998f,
                0.831470f, -0.555570f,  0.817585f, -0.575808f,  0.803208f, -0.595699f,  0.788346f, -0.615232f,
                0.773010f, -0.634393f,  0.757209f, -0.653173f,  0.740951f, -0.671559f,  0.724247f, -0.689541f,
                0.707107f, -0.707107f,  0.689541f, -0.724247f,  0.671559f, -0.740951f,  0.653173f, -0.757209f,
                0.634393f, -0.773010f,  0.615232f, -0.788346f,  0.595699f, -0.803208f,  0.575808f, -0.817585f,
                0.555570f, -0.831470f,  0.534998f, -0.844854f,  0.514103f, -0.857729f,  0.492898f, -0.870087f,
                0.471397f, -0.881921f,  0.449611f, -0.893224f,  0.427555f, -0.903989f,  0.405241f, -0.914210f,
                0.382683f, -0.923880f,  0.359895f, -0.932993f,  0.336890f, -0.941544f,  0.313682f, -0.949528f,
                0.290285f, -0.956940f,  0.266713f, -0.963776f,  0.242980f, -0.970031f,  0.219101f, -0.975702f,
                0.195090f, -0.980785f,  0.170962f, -0.985278f,  0.146730f, -0.989177f,  0.122411f, -0.992480f,
                0.098017f, -0.995185f,  0.073565f, -0.997290f,  0.049068f, -0.998795f,  0.024541f, -0.999699f
            },
            {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}
        };

        template<bool invert>
        [[msvc::forceinline]] void forward(const float* input, float* output)
        {
            const constants_type& C = invert ? inv_consts : fwd_consts;

            alignas(32) float temp[512];

            for (std::size_t i = 0; i < 32; ++i)
            {
                std::size_t offset = i * 8;
                std::size_t half_offset = 256;

                __m256 r_top = _mm256_load_ps(input + offset);
                __m256 r_bot = _mm256_load_ps(input + offset + half_offset);

                __m256 t_add = _mm256_add_ps(r_top, r_bot);
                __m256 t_sub = _mm256_sub_ps(r_top, r_bot);

                __m256 w_n = _mm256_load_ps(C.w_n + offset);
                __m256 w_s = _mm256_load_ps(C.w_s + offset);

                t_sub = fma_complexf32(t_sub, w_n, w_s);

                _mm256_store_ps(temp + offset, t_add);
                _mm256_store_ps(temp + offset + half_offset, t_sub);
            }

            k128::template forward<invert>(temp, temp);
            k128::template forward<invert>(temp + 256, temp + 256);

            __m256 scale_vec;
            if constexpr (invert)
            {
                scale_vec = _mm256_load_ps(C.half_scale);
            }

            for (std::size_t i = 0; i < 32; ++i)
            {
                __m256 evens = _mm256_load_ps(temp + i * 8);
                __m256 odds = _mm256_load_ps(temp + 256 + i * 8);

                if constexpr (invert) 
                {
                    evens = _mm256_mul_ps(evens, scale_vec);
                    odds = _mm256_mul_ps(odds, scale_vec);
                }

                __m256d ed = _mm256_castps_pd(evens);
                __m256d od = _mm256_castps_pd(odds);

                __m256d t0 = _mm256_unpacklo_pd(ed, od);
                __m256d t1 = _mm256_unpackhi_pd(ed, od);

                _mm256_store_ps(output + i * 16 + 0, _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t1, 0x20)));
                _mm256_store_ps(output + i * 16 + 8, _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t1, 0x31)));
            }
        }
    };
}

namespace fy::fft
{
    template<bool invert>
    static void apply_radix2_single(const float* input, float* output)
    {
        if constexpr (is_avx2_available)
        {
            __m128 in = _mm_load_ps(input);

            __m128 lo = _mm_movelh_ps(in, in);
            __m128 hi = _mm_movehl_ps(in, in);

            __m128 sum = _mm_add_ps(lo, hi);
            __m128 diff = _mm_sub_ps(lo, hi);

            __m128 res = _mm_shuffle_ps(sum, diff, _MM_SHUFFLE(1, 0, 1, 0));
            if constexpr (invert)
            {
                res = _mm_mul_ps(res, _mm_set1_ps(0.5f));
            }

            _mm_store_ps(output, res);
        }
        else
        {
            float r0 = input[0];
            float i0 = input[1];
            float r1 = input[2];
            float i1 = input[3];

            float out_r0 = r0 + r1;
            float out_i0 = i0 + i1;
            float out_r1 = r0 - r1;
            float out_i1 = i0 - i1;

            if constexpr (invert)
            {
                constexpr float inv_2 = 0.5f;
                output[0] = out_r0 * inv_2;
                output[1] = out_i0 * inv_2;
                output[2] = out_r1 * inv_2;
                output[3] = out_i1 * inv_2;
            }
            else
            {
                output[0] = out_r0;
                output[1] = out_i0;
                output[2] = out_r1;
                output[3] = out_i1;
            }
        }
    }

    template<bool invert>
    static void apply_radix3_single(const float* input, float* output)
    {
        constexpr float c_half = 0.5f;
        constexpr float c_sin60 = 0.86602540378443864676f;

        if constexpr (is_avx2_available)
        {
            __m256 v = _mm256_load_ps(input);

            const __m256i idx_0 = _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1);
            const __m256i idx_1 = _mm256_setr_epi32(2, 3, 2, 3, 2, 3, 2, 3);
            const __m256i idx_2 = _mm256_setr_epi32(4, 5, 4, 5, 4, 5, 4, 5);

            __m256 x0 = _mm256_permutevar8x32_ps(v, idx_0);
            __m256 x1 = _mm256_permutevar8x32_ps(v, idx_1);
            __m256 x2 = _mm256_permutevar8x32_ps(v, idx_2);

            __m256 t1 = _mm256_add_ps(x1, x2);
            __m256 m = _mm256_fnmadd_ps(t1, _mm256_set1_ps(c_half), x0);

            __m256 t2 = _mm256_sub_ps(x1, x2);
            __m256 k = _mm256_mul_ps(t2, _mm256_set1_ps(c_sin60));

            __m256 k_swap = _mm256_permute_ps(k, 0xB1);

            __m256 s_mask;
            if constexpr (invert)
            {
                s_mask = _mm256_setr_ps(0.f, 0.f, -1.f, 1.f, 1.f, -1.f, 0.f, 0.f);
            }
            else
            {
                s_mask = _mm256_setr_ps(0.f, 0.f, 1.f, -1.f, -1.f, 1.f, 0.f, 0.f);
            }

            __m256 s_term = _mm256_mul_ps(k_swap, s_mask);

            __m256 result = _mm256_add_ps(m, s_term);

            __m256 X0_val = _mm256_add_ps(x0, t1);
            result = _mm256_blend_ps(result, X0_val, 0x03);

            if constexpr (invert)
            {
                result = _mm256_mul_ps(result, _mm256_set1_ps(1.0f / 3.0f));
            }

            _mm_store_ps(output, _mm256_castps256_ps128(result));
            __m128 hi = _mm256_extractf128_ps(result, 1);
            _mm_store_sd(reinterpret_cast<double*>(output + 4), _mm_castps_pd(hi));
        }
        else
        {
            float r0 = input[0], i0 = input[1];
            float r1 = input[2], i1 = input[3];
            float r2 = input[4], i2 = input[5];

            float t1_r = r1 + r2;
            float t1_i = i1 + i2;

            float m_r = r0 - c_half * t1_r;
            float m_i = i0 - c_half * t1_i;

            float t2_r = r1 - r2;
            float t2_i = i1 - i2;

            float s_r = c_sin60 * t2_r;
            float s_i = c_sin60 * t2_i;

            float out0_r = r0 + t1_r;
            float out0_i = i0 + t1_i;

            float out1_r, out1_i;
            float out2_r, out2_i;

            if constexpr (invert)
            {
                out1_r = m_r - s_i;
                out1_i = m_i + s_r;

                out2_r = m_r + s_i;
                out2_i = m_i - s_r;

                constexpr float inv_3 = 1.0f / 3.0f;
                output[0] = out0_r * inv_3; output[1] = out0_i * inv_3;
                output[2] = out1_r * inv_3; output[3] = out1_i * inv_3;
                output[4] = out2_r * inv_3; output[5] = out2_i * inv_3;
            }
            else
            {
                out1_r = m_r + s_i;
                out1_i = m_i - s_r;

                out2_r = m_r - s_i;
                out2_i = m_i + s_r;

                output[0] = out0_r; output[1] = out0_i;
                output[2] = out1_r; output[3] = out1_i;
                output[4] = out2_r; output[5] = out2_i;
            }
        }
    }

    template<bool invert>
    static void apply_radix4_single(const float* input, float* output)
    {
        if constexpr (is_avx2_available)
        {
            __m256 v = _mm256_load_ps(input);

            __m256d vd = _mm256_castps_pd(v);
            vd = _mm256_permute4x64_pd(vd, _MM_SHUFFLE(3, 1, 2, 0));
            v = _mm256_castpd_ps(vd);

            __m256 t1 = _mm256_permute_ps(v, 0x44);
            __m256 t2 = _mm256_permute_ps(v, 0xEE);

            __m256 s1_sum = _mm256_add_ps(t1, t2);
            __m256 s1_diff = _mm256_sub_ps(t1, t2);

            v = _mm256_shuffle_ps(s1_sum, s1_diff, 0x44);

            __m256 CD = _mm256_permute2f128_ps(v, v, 0x11);
            __m256 CD_swap = _mm256_permute_ps(CD, 0xB1);
            __m256 T_mixed = _mm256_blend_ps(CD, CD_swap, 0xCC);

            const __m256 sign_mask = invert
                ? _mm256_setr_ps(0.f, 0.f, -0.f, 0.f, 0.f, 0.f, -0.f, 0.f)
                : _mm256_setr_ps(0.f, 0.f, 0.f, -0.f, 0.f, 0.f, 0.f, -0.f);

            __m256 T = _mm256_xor_ps(T_mixed, sign_mask);
            __m256 AB = _mm256_permute2f128_ps(v, v, 0x00);

            __m256 out_sum = _mm256_add_ps(AB, T);
            __m256 out_sub = _mm256_sub_ps(AB, T);

            __m256 result = _mm256_permute2f128_ps(out_sum, out_sub, 0x20);

            if constexpr (invert)
            {
                result = _mm256_mul_ps(result, _mm256_set1_ps(0.25f));
            }

            _mm256_store_ps(output, result);
        }
        else
        {
            float r0 = input[0];
            float i0 = input[1];
            float r1 = input[2];
            float i1 = input[3];
            float r2 = input[4];
            float i2 = input[5];
            float r3 = input[6];
            float i3 = input[7];

            float t0_r = r0 + r2;
            float t0_i = i0 + i2;
            float t2_r = r0 - r2;
            float t2_i = i0 - i2;

            float t1_r = r1 + r3;
            float t1_i = i1 + i3;
            float t3_r = r1 - r3;
            float t3_i = i1 - i3;

            float out0_r = t0_r + t1_r;
            float out0_i = t0_i + t1_i;
            float out2_r = t0_r - t1_r;
            float out2_i = t0_i - t1_i;

            float t3_rot_r, t3_rot_i;
            if constexpr (invert)
            {
                t3_rot_r = -t3_i;
                t3_rot_i = t3_r;
            }
            else
            {
                t3_rot_r = t3_i;
                t3_rot_i = -t3_r;
            }

            float out1_r = t2_r + t3_rot_r;
            float out1_i = t2_i + t3_rot_i;
            float out3_r = t2_r - t3_rot_r;
            float out3_i = t2_i - t3_rot_i;

            if constexpr (invert)
            {
                constexpr float inv_4 = 0.25f;
                output[0] = out0_r * inv_4;
                output[1] = out0_i * inv_4;
                output[2] = out1_r * inv_4;
                output[3] = out1_i * inv_4;
                output[4] = out2_r * inv_4;
                output[5] = out2_i * inv_4;
                output[6] = out3_r * inv_4;
                output[7] = out3_i * inv_4;
            }
            else
            {
                output[0] = out0_r;
                output[1] = out0_i;
                output[2] = out1_r;
                output[3] = out1_i;
                output[4] = out2_r;
                output[5] = out2_i;
                output[6] = out3_r;
                output[7] = out3_i;
            }
        }
    }

    template<bool invert>
    static inline void apply_radix5_single(const float* in, float* out)
    {
        constexpr float sign = invert ? +1.0f : -1.0f;

        constexpr float c1 = 0.3090169943749474241f;
        constexpr float s1 = 0.9510565162951535721f;
        constexpr float c2 = -0.8090169943749474241f;
        constexpr float s2 = 0.5877852522924731292f;

        const float w1r = c1, w1i = sign * s1;
        const float w2r = c2, w2i = sign * s2;

        float x0r = in[0];
        float x0i = in[1];
        float x1r = in[2];
        float x1i = in[3];
        float x2r = in[4];
        float x2i = in[5];
        float x3r = in[6];
        float x3i = in[7];
        float x4r = in[8];
        float x4i = in[9];

        float X0r = x0r + x1r + x2r + x3r + x4r;
        float X0i = x0i + x1i + x2i + x3i + x4i;

        auto cmul = [](float ar, float ai, float br, float bi, float& yr, float& yi)
            {
                yr = ar * br - ai * bi;
                yi = ar * bi + ai * br;
            };

        float t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
        cmul(x1r, x1i, w1r, w1i, t1r, t1i);
        cmul(x2r, x2i, w2r, w2i, t2r, t2i);
        cmul(x3r, x3i, w2r, -w2i, t3r, t3i);
        cmul(x4r, x4i, w1r, -w1i, t4r, t4i);
        float X1r = x0r + t1r + t2r + t3r + t4r;
        float X1i = x0i + t1i + t2i + t3i + t4i;

        cmul(x1r, x1i, w2r, w2i, t1r, t1i);
        cmul(x2r, x2i, w1r, -w1i, t2r, t2i);
        cmul(x3r, x3i, w1r, w1i, t3r, t3i);
        cmul(x4r, x4i, w2r, -w2i, t4r, t4i);
        float X2r = x0r + t1r + t2r + t3r + t4r;
        float X2i = x0i + t1i + t2i + t3i + t4i;

        cmul(x1r, x1i, w2r, -w2i, t1r, t1i);
        cmul(x2r, x2i, w1r, w1i, t2r, t2i);
        cmul(x3r, x3i, w1r, -w1i, t3r, t3i);
        cmul(x4r, x4i, w2r, w2i, t4r, t4i);
        float X3r = x0r + t1r + t2r + t3r + t4r;
        float X3i = x0i + t1i + t2i + t3i + t4i;

        cmul(x1r, x1i, w1r, -w1i, t1r, t1i);
        cmul(x2r, x2i, w2r, -w2i, t2r, t2i);
        cmul(x3r, x3i, w2r, w2i, t3r, t3i);
        cmul(x4r, x4i, w1r, w1i, t4r, t4i);
        float X4r = x0r + t1r + t2r + t3r + t4r;
        float X4i = x0i + t1i + t2i + t3i + t4i;

        if constexpr (invert)
        {
            X0r *= 0.2f;
            X0i *= 0.2f;
            X1r *= 0.2f;
            X1i *= 0.2f;
            X2r *= 0.2f;
            X2i *= 0.2f;
            X3r *= 0.2f;
            X3i *= 0.2f;
            X4r *= 0.2f;
            X4i *= 0.2f;
        }

        out[0] = X0r;
        out[1] = X0i;
        out[2] = X1r;
        out[3] = X1i;
        out[4] = X2r;
        out[5] = X2i;
        out[6] = X3r;
        out[7] = X3i;
        out[8] = X4r;
        out[9] = X4i;
    }

    template<bool invert>
    static void apply_radix7_single(const float* in, float* out)
    {
        constexpr float sign = invert ? +1.0f : -1.0f;

        float x0r = in[0];
        float x0i = in[1];
        float x1r = in[2];
        float x1i = in[3];
        float x2r = in[4];
        float x2i = in[5];
        float x3r = in[6];
        float x3i = in[7];
        float x4r = in[8];
        float x4i = in[9];
        float x5r = in[10];
        float x5i = in[11];
        float x6r = in[12];
        float x6i = in[13];

        float a1r = x1r + x6r;
        float a1i = x1i + x6i;
        float a2r = x2r + x5r;
        float a2i = x2i + x5i;
        float a3r = x3r + x4r;
        float a3i = x3i + x4i;

        float b1r = x1r - x6r;
        float b1i = x1i - x6i;
        float b2r = x2r - x5r;
        float b2i = x2i - x5i;
        float b3r = x3r - x4r;
        float b3i = x3i - x4i;

        constexpr float c11 = 0.62348980185873353053f;
        constexpr float s11 = 0.78183148246802980871f;
        constexpr float c12 = -0.22252093395631440429f;
        constexpr float s12 = 0.97492791218182360702f;
        constexpr float c13 = -0.90096886790241912624f;
        constexpr float s13 = 0.43388373911755812048f;

        constexpr float c21 = c12, s21 = s12;
        constexpr float c22 = c13, s22 = -s13;
        constexpr float c23 = c11, s23 = -s11;

        constexpr float c31 = c13, s31 = s13;
        constexpr float c32 = c11, s32 = -s11;
        constexpr float c33 = c12, s33 = s12;

        constexpr float c41 = c31, s41 = -s31;
        constexpr float c42 = c32, s42 = -s32;
        constexpr float c43 = c33, s43 = -s33;

        constexpr float c51 = c21, s51 = -s21;
        constexpr float c52 = c22, s52 = -s22;
        constexpr float c53 = c23, s53 = -s23;

        constexpr float c61 = c11, s61 = -s11;
        constexpr float c62 = c12, s62 = -s12;
        constexpr float c63 = c13, s63 = -s13;

        auto Xk_from_coeff = [&](float c1, float s1, float c2, float s2, float c3, float s3,
            float& Xkr, float& Xki)
            {
                float yr = x0r, yi = x0i;

                yr += a1r * c1 - sign * b1i * s1;
                yi += a1i * c1 + sign * b1r * s1;

                yr += a2r * c2 - sign * b2i * s2;
                yi += a2i * c2 + sign * b2r * s2;

                yr += a3r * c3 - sign * b3i * s3;
                yi += a3i * c3 + sign * b3r * s3;

                Xkr = yr; Xki = yi;
            };

        float X0r = x0r + a1r + a2r + a3r;
        float X0i = x0i + a1i + a2i + a3i;

        float X1r, X1i, X2r, X2i, X3r, X3i, X4r, X4i, X5r, X5i, X6r, X6i;
        Xk_from_coeff(c11, s11, c12, s12, c13, s13, X1r, X1i);
        Xk_from_coeff(c21, s21, c22, s22, c23, s23, X2r, X2i);
        Xk_from_coeff(c31, s31, c32, s32, c33, s33, X3r, X3i);
        Xk_from_coeff(c41, s41, c42, s42, c43, s43, X4r, X4i);
        Xk_from_coeff(c51, s51, c52, s52, c53, s53, X5r, X5i);
        Xk_from_coeff(c61, s61, c62, s62, c63, s63, X6r, X6i);

        if constexpr (invert)
        {
            constexpr float inv7 = 1.0f / 7.0f;
            X0r *= inv7; X0i *= inv7;
            X1r *= inv7; X1i *= inv7;
            X2r *= inv7; X2i *= inv7;
            X3r *= inv7; X3i *= inv7;
            X4r *= inv7; X4i *= inv7;
            X5r *= inv7; X5i *= inv7;
            X6r *= inv7; X6i *= inv7;
        }

        out[0] = X0r;
        out[1] = X0i;
        out[2] = X1r;
        out[3] = X1i;
        out[4] = X2r;
        out[5] = X2i;
        out[6] = X3r;
        out[7] = X3i;
        out[8] = X4r;
        out[9] = X4i;
        out[10] = X5r;
        out[11] = X5i;
        out[12] = X6r;
        out[13] = X6i;
    }
}

namespace fy::fft
{
    template<std::size_t N> struct C2CF32_kernel {};

    template<> struct C2CF32_kernel<8> : FFT1D_C2C_kernel_8_Radix_8 {};
    template<> struct C2CF32_kernel<16> : FFT1D_C2C_kernel_16_Radix_4x4 {};
    template<> struct C2CF32_kernel<32> : FFT1D_C2C_kernel_32_Radix_2x4x4 {};
    template<> struct C2CF32_kernel<64> : FFT1D_C2C_kernel_64_Radix_4x4x4 {};
    template<> struct C2CF32_kernel<128> : FFT1D_C2C_kernel_128_Radix_2x4x4x4 {};
    template<> struct C2CF32_kernel<256> : FFT1D_C2C_kernel_256_Radix_2x2x4x4x4 {};

    template<bool invert>
    [[msvc::forceinline]] bool especially_kernel_dispatch(const float* input, float* output, std::size_t length)
    {
        if (length < 8)
        {
            switch (length)
            {
                case 2: { apply_radix2_single<invert>(input, output); return true; }
                case 3: { apply_radix3_single<invert>(input, output); return true; }
                case 4: { apply_radix4_single<invert>(input, output); return true; }
                default: { return false; }
            }
        }

        if constexpr (is_avx2_available)
        {
            [[msvc::flatten]]
            {
                switch (length)
                {
                    case 8: { C2CF32_kernel<8>{}.template     forward<invert>(input, output); return true; }
                    case 16: { C2CF32_kernel<16>{}.template   forward<invert>(input, output); return true; }
                    case 32: { C2CF32_kernel<32>{}.template   forward<invert>(input, output); return true; }
                    case 64: { C2CF32_kernel<64>{}.template   forward<invert>(input, output); return true; }
                    case 128: { C2CF32_kernel<128>{}.template forward<invert>(input, output); return true; }
                    case 256: { C2CF32_kernel<256>{}.template forward<invert>(input, output); return true; }
                    default: { return false; }
                }
            }
        }

        return false;
    }
}






#endif