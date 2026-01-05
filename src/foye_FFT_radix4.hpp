#ifndef _FOYE_FFT_RADIX4_HPP_
#define _FOYE_FFT_RADIX4_HPP_

static bool is_power_of_four(std::size_t n)
{
    return n > 0 && (n & (n - 1)) == 0 && (n & 0x55555555) != 0;
}

namespace fy::fft::internal_radix4
{
    namespace radix4_lut
    {
        constexpr std::size_t max_table_bits = 16;
        constexpr std::size_t max_table_size = 1 << max_table_bits;
        constexpr std::size_t total_lut_blocks = []() -> std::size_t {
            std::size_t count = 0;
            for (std::size_t p = 2; p <= max_table_bits; p += 2)
            {
                std::size_t stage_size = std::size_t(1) << p;
                count += stage_size / 4;
            }
            return count; 
        }();

        using bit_rev_lut_type = std::array<std::array<std::size_t, max_table_size>, max_table_bits>;
        using butterfly_block_linear = std::array<float, 48>;

        struct alignas(32) lut_container
        {
            alignas(32) std::array<butterfly_block_linear, total_lut_blocks> storage;
            alignas(32) std::array<const butterfly_block_linear*, max_table_bits> offsets;
        };

        static lut_container& compute_butterfly_tables()
        {
            static lut_container container{};
            std::size_t current_offset = 0;

            for (std::size_t p = 2; p <= max_table_bits; p += 2)
            {
                std::size_t stage_size = std::size_t(1) << p;
                std::size_t n_full = stage_size * 4;

                container.offsets[p - 1] = &container.storage[current_offset];

                for (std::size_t k = 0; k < stage_size; k += 4)
                {
                    butterfly_block_linear& block = container.storage[current_offset++];

                    float* w_ptr = block.data();

                    for (std::size_t i = 0; i < 4; ++i)
                    {
                        std::size_t current_k = k + i;
                        double angle_base = -2.0 * std::numbers::pi_v<double> / static_cast<double>(n_full);

                        auto write_sincos_pair = [&](double k_idx, float* dest)
                            {
                                double ang = angle_base * k_idx;
                                float c = static_cast<float>(std::cos(ang));
                                float s = static_cast<float>(std::sin(ang));

                                dest[i * 2 + 0] = c;
                                dest[i * 2 + 1] = s;

                                dest[8 + i * 2 + 0] = s;
                                dest[8 + i * 2 + 1] = c;
                            };

                        write_sincos_pair(static_cast<double>(current_k) * 1.0, w_ptr + 0);
                        write_sincos_pair(static_cast<double>(current_k) * 2.0, w_ptr + 16);
                        write_sincos_pair(static_cast<double>(current_k) * 3.0, w_ptr + 32);
                    }
                }
            }

            return container;
        }

        static bit_rev_lut_type& compute_digit_rev_tables()
        {
            static bit_rev_lut_type tables{};
            for (std::size_t p = 2; p <= max_table_bits; p += 2)
            {
                std::size_t n = std::size_t(1) << p;
                auto& table = tables[p - 1];
                for (std::size_t i = 0; i < n; ++i)
                {
                    std::size_t rev = 0;
                    std::size_t x = i;
                    for (std::size_t k = 0; k < p; k += 2)
                    {
                        rev = (rev << 2) | (x & 3);
                        x >>= 2;
                    }
                    table[i] = rev;
                }
            }

            return tables;
        }

        static const lut_container& butterfly_lut = compute_butterfly_tables();
        static const bit_rev_lut_type& digit_rev_lut = compute_digit_rev_tables();
    }

    template<bool invert_mul>
    [[msvc::forceinline]] static __m256 complex_mul_packed(__m256 v, __m256 w_n, __m256 w_s)
    {
        __m256 v_re = _mm256_moveldup_ps(v);
        __m256 v_im = _mm256_movehdup_ps(v);

        if constexpr (invert_mul)
        {
            const __m256 sign_mask_n = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
            const __m256 sign_mask_s = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);

            w_n = _mm256_xor_ps(w_n, sign_mask_n);
            w_s = _mm256_xor_ps(w_s, sign_mask_s);
        }

        return _mm256_fmaddsub_ps(v_re, w_n, _mm256_mul_ps(v_im, w_s));
    }
    
    template<bool invert>
    [[msvc::forceinline]] static void butterfly_radix4_pure(__m256& x0, __m256& x1, __m256& x2, __m256& x3)
    {
        __m256 s02 = _mm256_add_ps(x0, x2);
        __m256 d02 = _mm256_sub_ps(x0, x2);
        __m256 s13 = _mm256_add_ps(x1, x3);
        __m256 d13 = _mm256_sub_ps(x1, x3);

        x0 = _mm256_add_ps(s02, s13);
        x2 = _mm256_sub_ps(s02, s13);

        __m256 d13_swap = _mm256_permute_ps(d13, 0xB1);
        __m256 d13_rot;

        if constexpr (invert) 
        {
            const __m256 mask = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
            d13_rot = _mm256_xor_ps(d13_swap, mask);
        }
        else 
        {
            const __m256 mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
            d13_rot = _mm256_xor_ps(d13_swap, mask);
        }

        x1 = _mm256_add_ps(d02, d13_rot);
        x3 = _mm256_sub_ps(d02, d13_rot);
    }

    struct stage2_twiddles
    {
        __m256 w1_n, w1_s;
        __m256 w2_n, w2_s;
        __m256 w3_n, w3_s;

        [[msvc::forceinline]] void set_cplx(float* dst, int k)
        {
            double ang = -2.0 * std::numbers::pi_v<double> *k / 16.0;
            dst[0] = (float)std::cos(ang);
            dst[1] = (float)std::sin(ang);
        }

        [[msvc::forceinline]] stage2_twiddles()
        {
            alignas(32) float buf_n[8];
            alignas(32) float buf_s[8];

            for (int i = 0; i < 4; ++i)
            {
                float tmp[2]; 
                set_cplx(tmp, i);
                buf_n[i * 2] = tmp[0]; 
                buf_n[i * 2 + 1] = tmp[1];
                buf_s[i * 2] = tmp[1]; 
                buf_s[i * 2 + 1] = tmp[0];
            }

            w1_n = _mm256_load_ps(buf_n);
            w1_s = _mm256_load_ps(buf_s);

            for (int i = 0; i < 4; ++i)
            {
                float tmp[2]; 
                set_cplx(tmp, i * 2);
                buf_n[i * 2] = tmp[0];
                buf_n[i * 2 + 1] = tmp[1];
                buf_s[i * 2] = tmp[1];
                buf_s[i * 2 + 1] = tmp[0];
            }

            w2_n = _mm256_load_ps(buf_n);
            w2_s = _mm256_load_ps(buf_s);

            for (int i = 0; i < 4; ++i)
            {
                float tmp[2]; 
                set_cplx(tmp, i * 3);
                buf_n[i * 2] = tmp[0]; 
                buf_n[i * 2 + 1] = tmp[1];
                buf_s[i * 2] = tmp[1]; 
                buf_s[i * 2 + 1] = tmp[0];
            }

            w3_n = _mm256_load_ps(buf_n);
            w3_s = _mm256_load_ps(buf_s);
        }
    };

    template<bool invert>
    [[msvc::forceinline]] static void fused_stage1_2_avx(float* data, std::size_t n)
    {
        static const stage2_twiddles tw;
        for (std::size_t i = 0; i < n * 2; i += 32)
        {
            __m256 r0 = _mm256_load_ps(data + i + 0);
            __m256 r1 = _mm256_load_ps(data + i + 8);
            __m256 r2 = _mm256_load_ps(data + i + 16);
            __m256 r3 = _mm256_load_ps(data + i + 24);

            transpose_4x4_complexf32(r0, r1, r2, r3);

            butterfly_radix4_pure<invert>(r0, r1, r2, r3);
            transpose_4x4_complexf32(r0, r1, r2, r3);

            r1 = complex_mul_packed<invert>(r1, tw.w1_n, tw.w1_s);
            r2 = complex_mul_packed<invert>(r2, tw.w2_n, tw.w2_s);
            r3 = complex_mul_packed<invert>(r3, tw.w3_n, tw.w3_s);

            butterfly_radix4_pure<invert>(r0, r1, r2, r3);

            _mm256_store_ps(data + i + 0, r0);
            _mm256_store_ps(data + i + 8, r1);
            _mm256_store_ps(data + i + 16, r2);
            _mm256_store_ps(data + i + 24, r3);
        }
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly_kernel_x4_avx_linear(float* p0, float* p1, float* p2, float* p3, const float* w_ptr)
    {
        __m256 w1_n = _mm256_load_ps(w_ptr + 0);
        __m256 w1_s = _mm256_load_ps(w_ptr + 8);

        __m256 w2_n = _mm256_load_ps(w_ptr + 16);
        __m256 w2_s = _mm256_load_ps(w_ptr + 24);

        __m256 w3_n = _mm256_load_ps(w_ptr + 32);
        __m256 w3_s = _mm256_load_ps(w_ptr + 40);

        if constexpr (invert)
        {
            const __m256 sign_n = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
            const __m256 sign_s = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);

            w1_n = _mm256_xor_ps(w1_n, sign_n); 
            w1_s = _mm256_xor_ps(w1_s, sign_s);
            w2_n = _mm256_xor_ps(w2_n, sign_n); 
            w2_s = _mm256_xor_ps(w2_s, sign_s);
            w3_n = _mm256_xor_ps(w3_n, sign_n);
            w3_s = _mm256_xor_ps(w3_s, sign_s);
        }

        __m256 x0 = _mm256_load_ps(p0);
        __m256 x1 = _mm256_load_ps(p1);
        __m256 x2 = _mm256_load_ps(p2);
        __m256 x3 = _mm256_load_ps(p3);

        __m256 t1 = fma_complexf32(x1, w1_n, w1_s);
        __m256 t2 = fma_complexf32(x2, w2_n, w2_s);
        __m256 t3 = fma_complexf32(x3, w3_n, w3_s);

        __m256 s02 = _mm256_add_ps(x0, t2);
        __m256 d02 = _mm256_sub_ps(x0, t2);
        __m256 s13 = _mm256_add_ps(t1, t3);
        __m256 d13 = _mm256_sub_ps(t1, t3);

        __m256 d13_swap = _mm256_permute_ps(d13, 0xB1);
        __m256 d13_rot;

        if constexpr (invert) 
        {
            const __m256 mask = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
            d13_rot = _mm256_xor_ps(d13_swap, mask);
        }
        else 
        {
            const __m256 mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
            d13_rot = _mm256_xor_ps(d13_swap, mask);
        }

        _mm256_store_ps(p0, _mm256_add_ps(s02, s13));
        _mm256_store_ps(p1, _mm256_add_ps(d02, d13_rot));
        _mm256_store_ps(p2, _mm256_sub_ps(s02, s13));
        _mm256_store_ps(p3, _mm256_sub_ps(d02, d13_rot));
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly_stride1_scalar(float* data, std::size_t length)
    {
        for (std::size_t i = 0; i < length; i += 4)
        {
            float* p = data + i * 2;
            float x0r = p[0];
            float x0i = p[1];
            float x1r = p[2];
            float x1i = p[3];
            float x2r = p[4];
            float x2i = p[5];
            float x3r = p[6];
            float x3i = p[7];

            float b0r = x0r + x2r; 
            float b0i = x0i + x2i;
            float b1r = x0r - x2r; 
            float b1i = x0i - x2i;
            float b2r = x1r + x3r; 
            float b2i = x1i + x3i;
            float b3r = x1r - x3r; 
            float b3i = x1i - x3i;

            float b3_rot_r, b3_rot_i;
            if constexpr (invert) 
            {
                b3_rot_r = -b3i; 
                b3_rot_i = b3r; 
            }
            else 
            {
                b3_rot_r = b3i; 
                b3_rot_i = -b3r; 
            }

            p[0] = b0r + b2r;
            p[1] = b0i + b2i;
            p[2] = b1r + b3_rot_r; 
            p[3] = b1i + b3_rot_i;
            p[4] = b0r - b2r; 
            p[5] = b0i - b2i;
            p[6] = b1r - b3_rot_r; 
            p[7] = b1i - b3_rot_i;
        }
    }

    struct FFT1D_C2C_radix4_invoker
    {
        template<bool invert>
        void dispatch(const float* input, float* output, std::size_t length)
        {
            assert(is_power_of_four(length));

            if (input == output)
            {
                bit_reverse_inplace(output, length);
            }
            else
            {
                bit_reverse_copy(input, output, length);
            }

            std::size_t start_stage_size = 4;
            if (is_avx2_available && (length >= 16))
            {
                fused_stage1_2_avx<invert>(output, length);
                start_stage_size = 16;
            }
            else
            {
                butterfly_stride1_scalar<invert>(output, length);
            }

            for (std::size_t stage_size = start_stage_size; stage_size < length; stage_size *= 4)
            {
                std::size_t group_size = stage_size * 4;

                std::size_t p = 0;
                std::size_t temp = stage_size;
                while (temp >>= 1)
                {
                    p++;
                }

                const auto* lut_ptr = radix4_lut::butterfly_lut.offsets[p - 1];

                for (std::size_t group_start = 0; group_start < length; group_start += group_size)
                {
                    float* base_ptr = output + group_start * 2;
                    std::size_t stride_float = stage_size * 2;

                    float* p0 = base_ptr;
                    float* p1 = base_ptr + stride_float;
                    float* p2 = base_ptr + stride_float * 2;
                    float* p3 = base_ptr + stride_float * 3;

                    std::size_t k = 0;

                    if constexpr (is_avx2_available)
                    {
                        for (; k + 3 < stage_size; k += 4)
                        {
                            butterfly_kernel_x4_avx_linear<invert>(
                                p0, p1, p2, p3,
                                lut_ptr[k / 4].data()
                            );

                            p0 += 8; 
                            p1 += 8; 
                            p2 += 8; 
                            p3 += 8;
                        }
                    }

                    for (; k < stage_size; ++k)
                    {
                        const auto& block = lut_ptr[k / 4];
                        const float* w_data = block.data();

                        std::size_t sub_idx = (k % 4) * 2;

                        float w1_r = w_data[0 + sub_idx];
                        float w1_i = w_data[0 + sub_idx + 1];

                        float w2_r = w_data[16 + sub_idx];
                        float w2_i = w_data[16 + sub_idx + 1];

                        float w3_r = w_data[32 + sub_idx];
                        float w3_i = w_data[32 + sub_idx + 1];

                        if constexpr (invert)
                        {
                            w1_i = -w1_i;
                            w2_i = -w2_i;
                            w3_i = -w3_i;
                        }

                        float x0r = p0[0]; 
                        float x0i = p0[1];
                        float x1r = p1[0]; 
                        float x1i = p1[1];
                        float x2r = p2[0]; 
                        float x2i = p2[1];
                        float x3r = p3[0]; 
                        float x3i = p3[1];

                        float t1r, t1i, t2r, t2i, t3r, t3i;
                        complex_multiply(x1r, x1i, w1_r, w1_i, t1r, t1i);
                        complex_multiply(x2r, x2i, w2_r, w2_i, t2r, t2i);
                        complex_multiply(x3r, x3i, w3_r, w3_i, t3r, t3i);

                        float b0r = x0r + t2r;
                        float b0i = x0i + t2i;
                        float b1r = x0r - t2r; 
                        float b1i = x0i - t2i;
                        float b2r = t1r + t3r; 
                        float b2i = t1i + t3i;
                        float b3r = t1r - t3r; 
                        float b3i = t1i - t3i;

                        float b3_rot_r, b3_rot_i;
                        if constexpr (invert)
                        {
                            b3_rot_r = -b3i; 
                            b3_rot_i = b3r;
                        }
                        else
                        {
                            b3_rot_r = b3i; 
                            b3_rot_i = -b3r;
                        }

                        p0[0] = b0r + b2r;      
                        p0[1] = b0i + b2i;
                        p1[0] = b1r + b3_rot_r; 
                        p1[1] = b1i + b3_rot_i;
                        p2[0] = b0r - b2r;      
                        p2[1] = b0i - b2i;
                        p3[0] = b1r - b3_rot_r; 
                        p3[1] = b1i - b3_rot_i;

                        p0 += 2; 
                        p1 += 2; 
                        p2 += 2; 
                        p3 += 2;
                    }
                }
            }

            if constexpr (invert)
            {
                invert_scale(output, length);
            }
        }

    private:
        [[msvc::forceinline]] static void complex_multiply(
            float ar, float ai, float br, float bi, float& or_, float& oi)
        {
            or_ = ar * br - ai * bi;
            oi = ar * bi + ai * br;
        }

        static void bit_reverse_copy(const float* src, float* dst, std::size_t length)
        {
            int p = 0;
            std::size_t temp = length;
            while (temp >>= 1)
            {
                ++p;
            }

            if (p >= 2 && p <= static_cast<int>(radix4_lut::max_table_bits))
            {
                const auto& table = radix4_lut::digit_rev_lut[p - 1];
                for (std::size_t j = 0; j < length; ++j)
                {
                    std::size_t i = table[j];
                    *(reinterpret_cast<double*>(dst + 2 * j)) = *(reinterpret_cast<const double*>(src + 2 * i));
                }
            }
            else if (length == 1) 
            {
                dst[0] = src[0];
                dst[1] = src[1]; 
            }
        }

        static void bit_reverse_inplace(float* data, std::size_t length)
        {
            int p = 0;
            std::size_t temp = length;
            while (temp >>= 1)
            {
                ++p;
            }

            if (p >= 2 && p <= static_cast<int>(radix4_lut::max_table_bits))
            {
                const auto& table = radix4_lut::digit_rev_lut[p - 1];
                for (std::size_t i = 0; i < length; ++i)
                {
                    std::size_t j = table[i];
                    if (i < j)
                    {
                        double t = reinterpret_cast<double*>(data)[i];
                        reinterpret_cast<double*>(data)[i] = reinterpret_cast<double*>(data)[j];
                        reinterpret_cast<double*>(data)[j] = t;
                    }
                }
            }
        }
    };
}



#endif