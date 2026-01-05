#ifndef _FOYE_FFT_RADIX3_HPP_
#define _FOYE_FFT_RADIX3_HPP_

namespace fy::fft::internal_radix3
{
    namespace radix3_lut
    {
        constexpr std::size_t max_power_of_3 = 11;
        constexpr std::size_t max_table_size = 177147;
        constexpr std::size_t total_rev_size = []() -> std::size_t
            {
                std::size_t total = 0;
                std::size_t n = 3;
                for (std::size_t i = 0; i < max_power_of_3; ++i)
                {
                    total += n;
                    n *= 3;
                }

                return total;
            }();

        struct alignas(64) butterfly_coeffs 
        {
            std::array<float, (max_table_size / 3) * 4 + 4> twiddles;
        };

        struct alignas(64) butterfly_coeffs_packed 
        {
            std::array<float, (max_table_size / 3) * 8 + 32> twiddles_packed;
        };

        struct alignas(64) digit_rev_lut_type
        {
            std::array<std::uint32_t, total_rev_size> data;
            std::array<std::size_t, max_power_of_3 + 1> offsets;
        };

        using butterfly_lut_type = std::array<butterfly_coeffs, max_power_of_3>;
        using butterfly_packed_lut_type = std::array<butterfly_coeffs_packed, max_power_of_3>;

        static butterfly_lut_type& compute_butterfly_tables() 
        {
            static butterfly_lut_type tables{};
            std::size_t n = 3;
            for (std::size_t p = 1; p <= max_power_of_3; ++p) 
            {
                std::size_t stride = n / 3;
                auto& table = tables[p - 1];
                for (std::size_t k = 0; k < stride; ++k) 
                {
                    double angle = -2.0 * std::numbers::pi_v<double> * k / n;
                    table.twiddles[k * 4 + 0] = std::cos(angle);
                    table.twiddles[k * 4 + 1] = std::sin(angle);
                    table.twiddles[k * 4 + 2] = std::cos(2.0 * angle);
                    table.twiddles[k * 4 + 3] = std::sin(2.0 * angle);
                }
                n *= 3;
            }

            return tables;
        }

        static butterfly_packed_lut_type& compute_butterfly_packed_tables() 
        {
            static butterfly_packed_lut_type tables{};
            std::size_t n = 3;
            for (std::size_t p = 1; p <= max_power_of_3; ++p) 
            {
                std::size_t stride = n / 3;
                auto& packed_table = tables[p - 1];
                for (std::size_t k = 0; k < stride; k += 4) 
                {
                    for (std::size_t i = 0; i < 4; ++i) 
                    {
                        if (k + i >= stride)
                        {
                            break;
                        }

                        double angle = -2.0 * std::numbers::pi_v<double> * (k + i) / n;
                        float w1r = std::cos(angle);
                        float w1i = std::sin(angle);
                        float w2r = std::cos(2.0 * angle);
                        float w2i = std::sin(2.0 * angle);

                        std::size_t base = (k / 4) * 32;
                        packed_table.twiddles_packed[base + i * 2 + 0] = w1r;
                        packed_table.twiddles_packed[base + i * 2 + 1] = w1i;
                        packed_table.twiddles_packed[base + 8 + i * 2 + 0] = w1i;
                        packed_table.twiddles_packed[base + 8 + i * 2 + 1] = w1r;
                        packed_table.twiddles_packed[base + 16 + i * 2 + 0] = w2r;
                        packed_table.twiddles_packed[base + 16 + i * 2 + 1] = w2i;
                        packed_table.twiddles_packed[base + 24 + i * 2 + 0] = w2i;
                        packed_table.twiddles_packed[base + 24 + i * 2 + 1] = w2r;
                    }
                }
                n *= 3;
            }
            return tables;
        }

        static digit_rev_lut_type& compute_digit_rev_tables()
        {
            static digit_rev_lut_type lut{};
            std::size_t offset = 0; 
            std::size_t n = 3;
            for (std::size_t p = 1; p <= max_power_of_3; ++p) 
            {
                lut.offsets[p - 1] = offset;
                for (std::size_t i = 0; i < n; ++i) 
                {
                    std::size_t j = 0, temp = i;
                    for (std::size_t k = 0; k < p; ++k) 
                    {
                        j = j * 3 + (temp % 3); 
                        temp /= 3; 
                    }

                    lut.data[offset + i] = static_cast<std::uint32_t>(j);
                }
                offset += n; n *= 3;
            }
            lut.offsets[max_power_of_3] = offset;
            return lut;
        }

        static const butterfly_lut_type& butterfly_lut = compute_butterfly_tables();
        static const butterfly_packed_lut_type& butterfly_packed_lut = compute_butterfly_packed_tables();
        static const digit_rev_lut_type& digit_rev_lut = compute_digit_rev_tables();
    }

    template<bool invert>
    [[msvc::forceinline]] static void radix3_kernel_x8(float* u, float* v, float* w, const float* tw)
    {
        const __m256 half = _mm256_set1_ps(0.5f);

        float s_val_f = invert ? 0.86602540378f : -0.86602540378f;

        const __m256 sn = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
        const __m256 ss = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);

        __m256 x0_0 = _mm256_load_ps(u);
        __m256 x1_0 = _mm256_load_ps(v);
        __m256 x2_0 = _mm256_load_ps(w);

        __m256 x0_1 = _mm256_load_ps(u + 8);
        __m256 x1_1 = _mm256_load_ps(v + 8);
        __m256 x2_1 = _mm256_load_ps(w + 8);

        __m256 w1n_0 = _mm256_load_ps(tw + 0);
        __m256 w1s_0 = _mm256_load_ps(tw + 8);
        __m256 w1n_1 = _mm256_load_ps(tw + 32);
        __m256 w1s_1 = _mm256_load_ps(tw + 40);

        if constexpr (invert)
        {
            w1n_0 = _mm256_xor_ps(w1n_0, sn); 
            w1s_0 = _mm256_xor_ps(w1s_0, ss);
            w1n_1 = _mm256_xor_ps(w1n_1, sn); 
            w1s_1 = _mm256_xor_ps(w1s_1, ss);
        }

        __m256 t1_0 = _mm256_fmaddsub_ps(_mm256_moveldup_ps(x1_0), w1n_0, _mm256_mul_ps(_mm256_movehdup_ps(x1_0), w1s_0));
        __m256 t1_1 = _mm256_fmaddsub_ps(_mm256_moveldup_ps(x1_1), w1n_1, _mm256_mul_ps(_mm256_movehdup_ps(x1_1), w1s_1));

        __m256 w2n_0 = _mm256_load_ps(tw + 16);
        __m256 w2s_0 = _mm256_load_ps(tw + 24);

        __m256 w2n_1 = _mm256_load_ps(tw + 48);
        __m256 w2s_1 = _mm256_load_ps(tw + 56);

        if constexpr (invert) 
        {
            w2n_0 = _mm256_xor_ps(w2n_0, sn); 
            w2s_0 = _mm256_xor_ps(w2s_0, ss);
            w2n_1 = _mm256_xor_ps(w2n_1, sn); 
            w2s_1 = _mm256_xor_ps(w2s_1, ss);
        }

        __m256 t2_0 = _mm256_fmaddsub_ps(_mm256_moveldup_ps(x2_0), w2n_0, _mm256_mul_ps(_mm256_movehdup_ps(x2_0), w2s_0));
        __m256 t2_1 = _mm256_fmaddsub_ps(_mm256_moveldup_ps(x2_1), w2n_1, _mm256_mul_ps(_mm256_movehdup_ps(x2_1), w2s_1));

        __m256 sum_0 = _mm256_add_ps(t1_0, t2_0);
        __m256 sum_1 = _mm256_add_ps(t1_1, t2_1);

        __m256 diff_0 = _mm256_sub_ps(t1_0, t2_0);
        __m256 diff_1 = _mm256_sub_ps(t1_1, t2_1);

        _mm256_store_ps(u, _mm256_add_ps(x0_0, sum_0));
        _mm256_store_ps(u + 8, _mm256_add_ps(x0_1, sum_1));

        __m256 m_0 = _mm256_fnmadd_ps(sum_0, half, x0_0);
        __m256 m_1 = _mm256_fnmadd_ps(sum_1, half, x0_1);

        __m256 s_vec_0 = _mm256_permute_ps(diff_0, 0xB1);
        __m256 s_vec_1 = _mm256_permute_ps(diff_1, 0xB1);

        const __m256 s_val = _mm256_set1_ps(s_val_f);
        s_vec_0 = _mm256_mul_ps(s_vec_0, s_val);
        s_vec_1 = _mm256_mul_ps(s_vec_1, s_val);

        const __m256 neg_real_mask = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
        s_vec_0 = _mm256_xor_ps(s_vec_0, neg_real_mask);
        s_vec_1 = _mm256_xor_ps(s_vec_1, neg_real_mask);

        _mm256_store_ps(v, _mm256_add_ps(m_0, s_vec_0));
        _mm256_store_ps(v + 8, _mm256_add_ps(m_1, s_vec_1));

        _mm256_store_ps(w, _mm256_sub_ps(m_0, s_vec_0));
        _mm256_store_ps(w + 8, _mm256_sub_ps(m_1, s_vec_1));
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly_radix3_stage1_permute_fused(
        const float* src, float* dst, std::size_t n, const std::uint32_t* rev_table)
    {
        constexpr float s_val = invert ? 0.86602540378f : -0.86602540378f;

        std::size_t i = 0;
        if constexpr (is_avx2_available)
        {
            for (; i + 11 < n; i += 12)
            {
                size_t idx0_0 = rev_table[i + 0];
                size_t idx1_0 = rev_table[i + 3];
                size_t idx2_0 = rev_table[i + 6];
                size_t idx3_0 = rev_table[i + 9];

                size_t idx0_1 = rev_table[i + 1];
                size_t idx1_1 = rev_table[i + 4];
                size_t idx2_1 = rev_table[i + 7];
                size_t idx3_1 = rev_table[i + 10];

                size_t idx0_2 = rev_table[i + 2];
                size_t idx1_2 = rev_table[i + 5];
                size_t idx2_2 = rev_table[i + 8];
                size_t idx3_2 = rev_table[i + 11];

                __m256 x0 = _mm256_setr_ps(
                    src[2 * idx0_0], src[2 * idx0_0 + 1], src[2 * idx1_0], src[2 * idx1_0 + 1],
                    src[2 * idx2_0], src[2 * idx2_0 + 1], src[2 * idx3_0], src[2 * idx3_0 + 1]);

                __m256 x1 = _mm256_setr_ps(
                    src[2 * idx0_1], src[2 * idx0_1 + 1], src[2 * idx1_1], src[2 * idx1_1 + 1],
                    src[2 * idx2_1], src[2 * idx2_1 + 1], src[2 * idx3_1], src[2 * idx3_1 + 1]);

                __m256 x2 = _mm256_setr_ps(
                    src[2 * idx0_2], src[2 * idx0_2 + 1], src[2 * idx1_2], src[2 * idx1_2 + 1],
                    src[2 * idx2_2], src[2 * idx2_2 + 1], src[2 * idx3_2], src[2 * idx3_2 + 1]);

                __m256 sr = _mm256_add_ps(x1, x2);
                __m256 si = _mm256_sub_ps(x1, x2);

                __m256 m = _mm256_fnmadd_ps(sr, _mm256_set1_ps(0.5f), x0);

                __m256 diff_swapped = _mm256_permute_ps(si, 0xB1);
                __m256 s_vec = _mm256_mul_ps(diff_swapped, 
                    _mm256_setr_ps(-s_val, s_val, -s_val, s_val, -s_val, s_val, -s_val, s_val));

                __m256 out0 = _mm256_add_ps(x0, sr);
                __m256 out1 = _mm256_add_ps(m, s_vec);
                __m256 out2 = _mm256_sub_ps(m, s_vec);

                __m128 out0_lo = _mm256_castps256_ps128(out0);
                __m128 out0_hi = _mm256_extractf128_ps(out0, 1);
                __m128 out1_lo = _mm256_castps256_ps128(out1);
                __m128 out1_hi = _mm256_extractf128_ps(out1, 1);
                __m128 out2_lo = _mm256_castps256_ps128(out2);
                __m128 out2_hi = _mm256_extractf128_ps(out2, 1);

                float* p = dst + i * 2;
                _mm_store_ps(p + 0, _mm_shuffle_ps(out0_lo, out1_lo, _MM_SHUFFLE(1, 0, 1, 0)));
                _mm_store_ps(p + 4, _mm_shuffle_ps(out2_lo, out0_lo, _MM_SHUFFLE(3, 2, 1, 0)));
                _mm_store_ps(p + 8, _mm_shuffle_ps(out1_lo, out2_lo, _MM_SHUFFLE(3, 2, 3, 2)));
                _mm_store_ps(p + 12, _mm_shuffle_ps(out0_hi, out1_hi, _MM_SHUFFLE(1, 0, 1, 0)));
                _mm_store_ps(p + 16, _mm_shuffle_ps(out2_hi, out0_hi, _MM_SHUFFLE(3, 2, 1, 0)));
                _mm_store_ps(p + 20, _mm_shuffle_ps(out1_hi, out2_hi, _MM_SHUFFLE(3, 2, 3, 2)));
            }
        }

        for (; i < n; i += 3)
        {
            std::size_t idx0 = rev_table[i];
            std::size_t idx1 = rev_table[i + 1];
            std::size_t idx2 = rev_table[i + 2];

            float x0r = src[2 * idx0];   
            float x0i = src[2 * idx0 + 1];

            float x1r = src[2 * idx1];   
            float x1i = src[2 * idx1 + 1];

            float x2r = src[2 * idx2];   
            float x2i = src[2 * idx2 + 1];

            float sr = x1r + x2r;
            float si = x1i + x2i;
            float dr = x1r - x2r;
            float di = x1i - x2i;

            float* p = dst + i * 2;
            p[0] = x0r + sr;
            p[1] = x0i + si;

            float mr = x0r - 0.5f * sr;
            float mi = x0i - 0.5f * si;

            float sx = -s_val * di;
            float sy = s_val * dr;

            p[2] = mr + sx;
            p[3] = mi + sy;
            p[4] = mr - sx;
            p[5] = mi - sy;
        }
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly_radix3_stage1_inplace(float* data, std::size_t n)
    {
        constexpr float s_val = invert ? 0.86602540378f : -0.86602540378f;
        std::size_t i = 0;
        if constexpr (is_avx2_available)
        {
            for (; i + 11 < n; i += 12)
            {
                float* p = data + i * 2;
                __m128 m0 = _mm_load_ps(p + 0);
                __m128 m1 = _mm_load_ps(p + 4);
                __m128 m2 = _mm_load_ps(p + 8);
                __m128 m3 = _mm_load_ps(p + 12);
                __m128 m4 = _mm_load_ps(p + 16);
                __m128 m5 = _mm_load_ps(p + 20);

                __m128 x0_lo = _mm_shuffle_ps(m0, m1, _MM_SHUFFLE(3, 2, 1, 0));
                __m128 x0_hi = _mm_shuffle_ps(m3, m4, _MM_SHUFFLE(3, 2, 1, 0));
                __m256 x0 = _mm256_insertf128_ps(_mm256_castps128_ps256(x0_lo), x0_hi, 1);

                __m128 x1_lo = _mm_shuffle_ps(m0, m2, _MM_SHUFFLE(1, 0, 3, 2));
                __m128 x1_hi = _mm_shuffle_ps(m3, m5, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 x1 = _mm256_insertf128_ps(_mm256_castps128_ps256(x1_lo), x1_hi, 1);

                __m128 x2_lo = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(3, 2, 1, 0));
                __m128 x2_hi = _mm_shuffle_ps(m4, m5, _MM_SHUFFLE(3, 2, 1, 0));
                __m256 x2 = _mm256_insertf128_ps(_mm256_castps128_ps256(x2_lo), x2_hi, 1);

                __m256 sr = _mm256_add_ps(x1, x2);
                __m256 si = _mm256_sub_ps(x1, x2);
                __m256 m = _mm256_fnmadd_ps(sr, _mm256_set1_ps(0.5f), x0);

                __m256 diff_swapped = _mm256_permute_ps(si, 0xB1);
                __m256 s_vec = _mm256_mul_ps(diff_swapped, 
                    _mm256_setr_ps(-s_val, s_val, -s_val, s_val, -s_val, s_val, -s_val, s_val));

                __m256 out0 = _mm256_add_ps(x0, sr);
                __m256 out1 = _mm256_add_ps(m, s_vec);
                __m256 out2 = _mm256_sub_ps(m, s_vec);

                __m128 out0_lo = _mm256_castps256_ps128(out0);
                __m128 out0_hi = _mm256_extractf128_ps(out0, 1);

                __m128 out1_lo = _mm256_castps256_ps128(out1);
                __m128 out1_hi = _mm256_extractf128_ps(out1, 1);

                __m128 out2_lo = _mm256_castps256_ps128(out2);
                __m128 out2_hi = _mm256_extractf128_ps(out2, 1);

                _mm_store_ps(p + 0, _mm_shuffle_ps(out0_lo, out1_lo, _MM_SHUFFLE(1, 0, 1, 0)));
                _mm_store_ps(p + 4, _mm_shuffle_ps(out2_lo, out0_lo, _MM_SHUFFLE(3, 2, 1, 0)));
                _mm_store_ps(p + 8, _mm_shuffle_ps(out1_lo, out2_lo, _MM_SHUFFLE(3, 2, 3, 2)));
                _mm_store_ps(p + 12, _mm_shuffle_ps(out0_hi, out1_hi, _MM_SHUFFLE(1, 0, 1, 0)));
                _mm_store_ps(p + 16, _mm_shuffle_ps(out2_hi, out0_hi, _MM_SHUFFLE(3, 2, 1, 0)));
                _mm_store_ps(p + 20, _mm_shuffle_ps(out1_hi, out2_hi, _MM_SHUFFLE(3, 2, 3, 2)));
            }
        }

        for (; i < n; i += 3)
        {
            float* p = data + i * 2;
            float x0r = p[0]; 
            float x0i = p[1];
            float x1r = p[2]; 
            float x1i = p[3];
            float x2r = p[4]; 
            float x2i = p[5];

            float sr = x1r + x2r; 
            float si = x1i + x2i;
            float dr = x1r - x2r; 
            float di = x1i - x2i;

            p[0] = x0r + sr; 
            p[1] = x0i + si;

            float mr = x0r - 0.5f * sr; 
            float mi = x0i - 0.5f * si;

            float sx = -s_val * di; 
            float sy = s_val * dr;

            p[2] = mr + sx;
            p[3] = mi + sy;
            p[4] = mr - sx; 
            p[5] = mi - sy;
        }
    }

    template<bool invert>
    [[msvc::forceinline]] static void radix3_kernel_x4(float* u, float* v, float* w, const float* tw) 
    {
        __m256 x0 = _mm256_load_ps(u);
        __m256 x1 = _mm256_load_ps(v);
        __m256 x2 = _mm256_load_ps(w);

        __m256 w1n = _mm256_load_ps(tw + 0);  __m256 w1s = _mm256_load_ps(tw + 8);
        __m256 w2n = _mm256_load_ps(tw + 16); __m256 w2s = _mm256_load_ps(tw + 24);

        const __m256 sn = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
        const __m256 ss = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
        if constexpr (invert) 
        {
            w1n = _mm256_xor_ps(w1n, sn); w1s = _mm256_xor_ps(w1s, ss);
            w2n = _mm256_xor_ps(w2n, sn); w2s = _mm256_xor_ps(w2s, ss);
        }

        __m256 t1 = _mm256_fmaddsub_ps(_mm256_moveldup_ps(x1), w1n, _mm256_mul_ps(_mm256_movehdup_ps(x1), w1s));
        __m256 t2 = _mm256_fmaddsub_ps(_mm256_moveldup_ps(x2), w2n, _mm256_mul_ps(_mm256_movehdup_ps(x2), w2s));

        __m256 sum = _mm256_add_ps(t1, t2);
        __m256 diff = _mm256_sub_ps(t1, t2);

        _mm256_store_ps(u, _mm256_add_ps(x0, sum));
        __m256 m = _mm256_fnmadd_ps(sum, _mm256_set1_ps(0.5f), x0);

        float s_val = invert ? 0.86602540378f : -0.86602540378f;
        __m256 s_vec = _mm256_mul_ps(_mm256_permute_ps(diff, 0xB1), _mm256_set1_ps(s_val));

        s_vec = _mm256_xor_ps(s_vec, _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f));

        _mm256_store_ps(v, _mm256_add_ps(m, s_vec));
        _mm256_store_ps(w, _mm256_sub_ps(m, s_vec));
    }

    struct FFT1D_C2C_radix3_invoker
    {
        template<bool invert>
        void dispatch(const float* input, float* output, std::size_t length)
        {
            std::size_t p_total = 0;
            std::size_t temp = length;
            while (temp > 1)
            {
                temp /= 3;
                ++p_total;
            }

            const std::uint32_t* rev_table = &radix3_lut::digit_rev_lut.data[radix3_lut::digit_rev_lut.offsets[p_total - 1]];

            bool stage1_done = false;
            if (input != output)
            {
                butterfly_radix3_stage1_permute_fused<invert>(input, output, length, rev_table);
                stage1_done = true;
            }
            else
            {
                std::uint64_t* ptr = reinterpret_cast<std::uint64_t*>(output);
                for (std::size_t i = 0; i < length; ++i)
                {
                    std::size_t j = rev_table[i];
                    if (i < j)
                    {
                        std::swap(ptr[i], ptr[j]);
                    }
                }

                butterfly_radix3_stage1_inplace<invert>(output, length);
                stage1_done = true;
            }

            float* data = output;

            std::size_t p = 2;
            std::size_t step = 9;
            for (; p <= p_total; ++p)
            {
                std::size_t stride = step / 3;
                const auto& packed_lut = radix3_lut::butterfly_packed_lut[p - 1];
                const auto& normal_lut = radix3_lut::butterfly_lut[p - 1];

                for (std::size_t k = 0; k < length; k += step)
                {
                    float* base = data + 2 * k;
                    std::size_t j = 0;

                    if constexpr (is_avx2_available)
                    {
                        for (; j + 7 < stride; j += 8)
                        {
                            radix3_kernel_x8<invert>(
                                base + 2 * j,
                                base + 2 * (j + stride),
                                base + 2 * (j + 2 * stride),
                                packed_lut.twiddles_packed.data() + (j / 4) * 32
                            );
                        }

                        if (j + 3 < stride)
                        {
                            radix3_kernel_x4<invert>(
                                base + 2 * j,
                                base + 2 * (j + stride),
                                base + 2 * (j + 2 * stride),
                                packed_lut.twiddles_packed.data() + (j / 4) * 32
                            );
                            j += 4;
                        }
                    }

                    for (; j < stride; ++j)
                    {
                        float w1r = normal_lut.twiddles[j * 4 + 0];
                        float w1i = normal_lut.twiddles[j * 4 + 1];
                        float w2r = normal_lut.twiddles[j * 4 + 2];
                        float w2i = normal_lut.twiddles[j * 4 + 3];

                        if constexpr (invert) 
                        {
                            w1i = -w1i; 
                            w2i = -w2i;
                        }

                        float* p0 = base + 2 * j;
                        float* p1 = base + 2 * (j + stride);
                        float* p2 = base + 2 * (j + 2 * stride);

                        float t1r = p1[0] * w1r - p1[1] * w1i;
                        float t1i = p1[0] * w1i + p1[1] * w1r;
                        float t2r = p2[0] * w2r - p2[1] * w2i;
                        float t2i = p2[0] * w2i + p2[1] * w2r;

                        float sr = t1r + t2r;
                        float si = t1i + t2i;
                        float dr = t1r - t2r;
                        float di = t1i - t2i;

                        float x0r = p0[0];
                        float x0i = p0[1];

                        p0[0] = x0r + sr;
                        p0[1] = x0i + si;

                        float mr = x0r - 0.5f * sr;
                        float mi = x0i - 0.5f * si;

                        float s_fac = invert ? 0.86602540378f : -0.86602540378f;
                        float sx = -s_fac * di;
                        float sy = s_fac * dr;

                        p1[0] = mr + sx;
                        p1[1] = mi + sy;
                        p2[0] = mr - sx;
                        p2[1] = mi - sy;
                    }
                }

                step *= 3;
            }

            if constexpr (invert)
            {
                invert_scale(output, length);
            }
        }
    };
}

#endif
