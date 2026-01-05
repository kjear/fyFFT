#ifndef _FOYE_FFT_RADIX7_HPP_
#define _FOYE_FFT_RADIX7_HPP_

namespace fy::fft::internal_radix7
{
    namespace radix7_lut
    {
        constexpr std::size_t max_power_of_7 = 6;
        constexpr std::size_t max_table_size = 117649;
        constexpr std::size_t total_rev_size = 137257;
        constexpr std::size_t total_soa_size = []() -> std::size_t
            {
                std::size_t total_floats = 0;
                std::size_t n = 7;
                for (std::size_t p = 1; p <= max_power_of_7; ++p)
                {
                    std::size_t stride = n / 7;
                    std::size_t blocks = (stride + 3) / 4;
                    total_floats += blocks * 96;
                    n *= 7;
                }
                return total_floats;
            }();

        constexpr float c71 = -0.166666667f;
        constexpr float c72 = 0.790156469f;
        constexpr float c73 = -0.055854267f;
        constexpr float c74 = -0.734302201f;
        constexpr float c75 = 0.781831482f;
        constexpr float c76 = 0.974927912f;
        constexpr float c77 = 0.433883739f;

        struct alignas(64) butterfly_coeffs
        {
            std::array<float, (max_table_size / 7) * 12> twiddles;
        };

        struct alignas(32) butterfly_coeffs_soa
        {
            const float* data;
            std::size_t stride;
        };

        struct alignas(64) digit_rev_lut_type
        {
            std::array<std::uint32_t, total_rev_size> data;
            std::array<std::size_t, max_power_of_7 + 1> offsets;
        };

        using butterfly_lut_type = std::array<butterfly_coeffs, max_power_of_7>;
        using butterfly_lut_soa_type = std::array<butterfly_coeffs_soa, max_power_of_7>;

        static butterfly_lut_type& compute_butterfly_tables()
        {
            static butterfly_lut_type tables{};

            std::size_t n = 7;
            for (std::size_t p = 1; p <= max_power_of_7; ++p)
            {
                std::size_t stride = n / 7;
                auto& table = tables[p - 1];
                for (std::size_t k = 0; k < stride; ++k)
                {
                    for (std::size_t m = 1; m <= 6; ++m)
                    {
                        double angle = -2.0 * std::numbers::pi_v<double> * k * m / n;
                        table.twiddles[k * 12 + (m - 1) * 2 + 0] = static_cast<float>(std::cos(angle));
                        table.twiddles[k * 12 + (m - 1) * 2 + 1] = static_cast<float>(std::sin(angle));
                    }
                }
                n *= 7;
            }

            return tables;
        }

        static butterfly_lut_soa_type& compute_butterfly_tables_soa()
        {
            static alignas(32) std::array<float, total_soa_size> storage{};
            static butterfly_lut_soa_type tables{};

            std::size_t n = 7;
            float* storage_ptr = storage.data();

            for (std::size_t p = 1; p <= max_power_of_7; ++p)
            {
                std::size_t stride = n / 7;
                tables[p - 1].stride = stride;
                tables[p - 1].data = storage_ptr;

                for (std::size_t k_base = 0; k_base < stride; k_base += 4)
                {
                    for (std::size_t m = 1; m <= 6; ++m)
                    {
                        float re[8] = { 0 }, im[8] = { 0 };
                        for (std::size_t i = 0; i < 4; ++i)
                        {
                            std::size_t k = k_base + i;
                            if (k < stride)
                            {
                                double angle = -2.0 * std::numbers::pi_v<double> * k * m / n;
                                re[i] = static_cast<float>(std::cos(angle));
                                im[i] = static_cast<float>(std::sin(angle));
                                re[i + 4] = re[i];
                                im[i + 4] = im[i];
                            }
                            else
                            {
                                re[i] = 1.0f; 
                                im[i] = 0.0f;
                                re[i + 4] = 1.0f; 
                                im[i + 4] = 0.0f;
                            }
                        }

                        std::copy(std::begin(re), std::end(re), storage_ptr); 
                        storage_ptr += 8;

                        std::copy(std::begin(im), std::end(im), storage_ptr); 
                        storage_ptr += 8;
                    }
                }
                n *= 7;
            }

            return tables;
        }

        static digit_rev_lut_type& compute_digit_rev_tables()
        {
            static digit_rev_lut_type lut{};

            std::size_t offset = 0;
            std::size_t n = 7;
            for (std::size_t p = 1; p <= max_power_of_7; ++p)
            {
                lut.offsets[p - 1] = offset;
                for (std::size_t i = 0; i < n; ++i)
                {
                    std::size_t j = 0, temp = i;
                    for (std::size_t k = 0; k < p; ++k)
                    {
                        j = j * 7 + (temp % 7);
                        temp /= 7;
                    }
                    lut.data[offset + i] = static_cast<std::uint32_t>(j);
                }
                offset += n; n *= 7;
            }

            lut.offsets[max_power_of_7] = offset;
            return lut;
        }

        static const butterfly_lut_type& butterfly_lut = compute_butterfly_tables();
        static const butterfly_lut_soa_type& butterfly_lut_soa = compute_butterfly_tables_soa();
        static const digit_rev_lut_type& digit_rev_lut = compute_digit_rev_tables();
    }

    template<bool invert>
    [[msvc::forceinline]] static void radix7_kernel_winograd(float* p0, float* p1, float* p2, float* p3, float* p4, float* p5, float* p6)
    {
        float r0 = p0[0]; 
        float i0 = p0[1];

        float s1r = p1[0] + p6[0];
        float s1i = p1[1] + p6[1];
        float d1r = p1[0] - p6[0]; 
        float d1i = p1[1] - p6[1];

        float s2r = p2[0] + p5[0]; 
        float s2i = p2[1] + p5[1];
        float d2r = p2[0] - p5[0];
        float d2i = p2[1] - p5[1];

        float s3r = p3[0] + p4[0];
        float s3i = p3[1] + p4[1];
        float d3r = p3[0] - p4[0]; 
        float d3i = p3[1] - p4[1];

        float t0r = s1r + s2r + s3r;
        float t0i = s1i + s2i + s3i;

        p0[0] = r0 + t0r;
        p0[1] = i0 + t0i;

        float zr = r0 + t0r * radix7_lut::c71;
        float zi = i0 + t0i * radix7_lut::c71;

        float m1r = s1r * radix7_lut::c72 + s2r * radix7_lut::c73 + s3r * radix7_lut::c74;
        float m1i = s1i * radix7_lut::c72 + s2i * radix7_lut::c73 + s3i * radix7_lut::c74;

        float m2r = s1r * radix7_lut::c73 + s2r * radix7_lut::c74 + s3r * radix7_lut::c72;
        float m2i = s1i * radix7_lut::c73 + s2i * radix7_lut::c74 + s3i * radix7_lut::c72;

        float m3r = s1r * radix7_lut::c74 + s2r * radix7_lut::c72 + s3r * radix7_lut::c73;
        float m3i = s1i * radix7_lut::c74 + s2i * radix7_lut::c72 + s3i * radix7_lut::c73;

        float n1r = d1r * radix7_lut::c75 + d2r * radix7_lut::c76 + d3r * radix7_lut::c77;
        float n1i = d1i * radix7_lut::c75 + d2i * radix7_lut::c76 + d3i * radix7_lut::c77;

        float n2r = d1r * radix7_lut::c76 - d2r * radix7_lut::c77 - d3r * radix7_lut::c75;
        float n2i = d1i * radix7_lut::c76 - d2i * radix7_lut::c77 - d3i * radix7_lut::c75;

        float n3r = d1r * radix7_lut::c77 - d2r * radix7_lut::c75 + d3r * radix7_lut::c76;
        float n3i = d1i * radix7_lut::c77 - d2i * radix7_lut::c75 + d3i * radix7_lut::c76;

        constexpr float s_fac = invert ? 1.0f : -1.0f;
        n1r *= s_fac;
        n1i *= s_fac;
        n2r *= s_fac; 
        n2i *= s_fac;
        n3r *= s_fac; 
        n3i *= s_fac;

        p1[0] = zr + m1r - n1i; 
        p1[1] = zi + m1i + n1r;
        p6[0] = zr + m1r + n1i; 
        p6[1] = zi + m1i - n1r;

        p2[0] = zr + m2r - n2i; 
        p2[1] = zi + m2i + n2r;
        p5[0] = zr + m2r + n2i; 
        p5[1] = zi + m2i - n2r;

        p3[0] = zr + m3r - n3i; 
        p3[1] = zi + m3i + n3r;
        p4[0] = zr + m3r + n3i; 
        p4[1] = zi + m3i - n3r;
    }

    template<bool invert>
    [[msvc::forceinline]] static void radix7_kernel_winograd_interleaved(
        __m256& u0, __m256& u1, __m256& u2, __m256& u3,
        __m256& u4, __m256& u5, __m256& u6)
    {
        const __m256 vc71 = _mm256_set1_ps(radix7_lut::c71);
        const __m256 vc72 = _mm256_set1_ps(radix7_lut::c72);
        const __m256 vc73 = _mm256_set1_ps(radix7_lut::c73);
        const __m256 vc74 = _mm256_set1_ps(radix7_lut::c74);
        const __m256 vc75 = _mm256_set1_ps(radix7_lut::c75);
        const __m256 vc76 = _mm256_set1_ps(radix7_lut::c76);
        const __m256 vc77 = _mm256_set1_ps(radix7_lut::c77);

        __m256 s1 = _mm256_add_ps(u1, u6);
        __m256 d1 = _mm256_sub_ps(u1, u6);
        __m256 s2 = _mm256_add_ps(u2, u5);
        __m256 d2 = _mm256_sub_ps(u2, u5);
        __m256 s3 = _mm256_add_ps(u3, u4);
        __m256 d3 = _mm256_sub_ps(u3, u4);

        __m256 t0 = _mm256_add_ps(s1, _mm256_add_ps(s2, s3));

        __m256 z = _mm256_add_ps(u0, _mm256_mul_ps(t0, vc71));
        u0 = _mm256_add_ps(u0, t0);

        __m256 m1 = _mm256_fmadd_ps(s1, vc72, _mm256_fmadd_ps(s2, vc73, _mm256_mul_ps(s3, vc74)));
        __m256 m2 = _mm256_fmadd_ps(s1, vc73, _mm256_fmadd_ps(s2, vc74, _mm256_mul_ps(s3, vc72)));
        __m256 m3 = _mm256_fmadd_ps(s1, vc74, _mm256_fmadd_ps(s2, vc72, _mm256_mul_ps(s3, vc73)));

        __m256 n1 = _mm256_fmadd_ps(d1, vc75, _mm256_fmadd_ps(d2, vc76, _mm256_mul_ps(d3, vc77)));

        __m256 inner_n2 = _mm256_fmadd_ps(d2, vc77, _mm256_mul_ps(d3, vc75));
        __m256 n2 = _mm256_fmsub_ps(d1, vc76, inner_n2);

        __m256 n3 = _mm256_fmadd_ps(d1, vc77, _mm256_fnmadd_ps(d2, vc75, _mm256_mul_ps(d3, vc76)));

        if constexpr (!invert)
        {
            __m256 vneg = _mm256_set1_ps(-1.0f);
            n1 = _mm256_mul_ps(n1, vneg);
            n2 = _mm256_mul_ps(n2, vneg);
            n3 = _mm256_mul_ps(n3, vneg);
        }

        __m256 n1_swp = _mm256_permute_ps(n1, 0xB1);
        __m256 n2_swp = _mm256_permute_ps(n2, 0xB1);
        __m256 n3_swp = _mm256_permute_ps(n3, 0xB1);

        const __m256 neg_mask_re = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);

        __m256 n1_rot = _mm256_xor_ps(n1_swp, neg_mask_re);
        __m256 n2_rot = _mm256_xor_ps(n2_swp, neg_mask_re);
        __m256 n3_rot = _mm256_xor_ps(n3_swp, neg_mask_re);

        __m256 zm1 = _mm256_add_ps(z, m1);
        __m256 zm2 = _mm256_add_ps(z, m2);
        __m256 zm3 = _mm256_add_ps(z, m3);

        u1 = _mm256_add_ps(zm1, n1_rot);
        u6 = _mm256_sub_ps(zm1, n1_rot);

        u2 = _mm256_add_ps(zm2, n2_rot);
        u5 = _mm256_sub_ps(zm2, n2_rot);

        u3 = _mm256_add_ps(zm3, n3_rot);
        u4 = _mm256_sub_ps(zm3, n3_rot);
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly_radix7_stage1_permute_vectorized(
        const float* src, float* dst, std::size_t n, const std::uint32_t* rev_table)
    {
        std::size_t i = 0;

        if (n >= 28)
        {
            std::size_t n_vec = n - 28;

            for (; i <= n_vec; i += 28)
            {
                const std::uint32_t* r = rev_table + i;

                __m128i v_idx0 = _mm_set_epi32(r[21], r[14], r[7], r[0]);
                __m128i v_idx1 = _mm_set_epi32(r[22], r[15], r[8], r[1]);
                __m128i v_idx2 = _mm_set_epi32(r[23], r[16], r[9], r[2]);
                __m128i v_idx3 = _mm_set_epi32(r[24], r[17], r[10], r[3]);
                __m128i v_idx4 = _mm_set_epi32(r[25], r[18], r[11], r[4]);
                __m128i v_idx5 = _mm_set_epi32(r[26], r[19], r[12], r[5]);
                __m128i v_idx6 = _mm_set_epi32(r[27], r[20], r[13], r[6]);

                __m256 u0 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx0, 8));
                __m256 u1 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx1, 8));
                __m256 u2 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx2, 8));
                __m256 u3 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx3, 8));
                __m256 u4 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx4, 8));
                __m256 u5 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx5, 8));
                __m256 u6 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx6, 8));

                radix7_kernel_winograd_interleaved<invert>(u0, u1, u2, u3, u4, u5, u6);

                transpose_4x4_complexf32(u0, u1, u2, u3);
                __m256 u7 = _mm256_setzero_ps();
                transpose_4x4_complexf32(u4, u5, u6, u7);

                _mm256_store_ps(dst + i * 2, u0);
                _mm_store_ps(dst + i * 2 + 8, _mm256_castps256_ps128(u4));
                _mm_store_sd(reinterpret_cast<double*>(dst + i * 2 + 12), _mm_castps_pd(_mm256_extractf128_ps(u4, 1)));

                _mm256_store_ps(dst + (i + 7) * 2, u1);
                _mm_store_ps(dst + (i + 7) * 2 + 8, _mm256_castps256_ps128(u5));
                _mm_store_sd(reinterpret_cast<double*>(dst + (i + 7) * 2 + 12), _mm_castps_pd(_mm256_extractf128_ps(u5, 1)));

                _mm256_store_ps(dst + (i + 14) * 2, u2);
                _mm_store_ps(dst + (i + 14) * 2 + 8, _mm256_castps256_ps128(u6));
                _mm_store_sd(reinterpret_cast<double*>(dst + (i + 14) * 2 + 12), _mm_castps_pd(_mm256_extractf128_ps(u6, 1)));

                _mm256_store_ps(dst + (i + 21) * 2, u3);
                _mm_store_ps(dst + (i + 21) * 2 + 8, _mm256_castps256_ps128(u7));
                _mm_store_sd(reinterpret_cast<double*>(dst + (i + 21) * 2 + 12), _mm_castps_pd(_mm256_extractf128_ps(u7, 1)));
            }
        }

        for (; i < n; i += 7)
        {
            alignas(32) float p[7][2];
            for (std::size_t k = 0; k < 7; ++k)
            {
                std::size_t idx = rev_table[i + k];
                p[k][0] = src[2 * idx];
                p[k][1] = src[2 * idx + 1];
            }

            radix7_kernel_winograd<invert>(p[0], p[1], p[2], p[3], p[4], p[5], p[6]);

            float* d = dst + i * 2;
            for (std::size_t k = 0; k < 7; ++k)
            {
                d[2 * k] = p[k][0];
                d[2 * k + 1] = p[k][1];
            }
        }
    }

    template<bool invert>
    [[msvc::forceinline]] static void apply_twiddle_interleaved(__m256& v, const float* w_ptr)
    {
        __m128 w_re_xmm = _mm_load_ps(w_ptr);
        __m128 w_im_xmm = _mm_load_ps(w_ptr + 8);

        __m128 re_lo = _mm_unpacklo_ps(w_re_xmm, w_re_xmm);
        __m128 re_hi = _mm_unpackhi_ps(w_re_xmm, w_re_xmm);
        __m256 w_re = _mm256_insertf128_ps(_mm256_castps128_ps256(re_lo), re_hi, 1);

        __m128 im_lo = _mm_unpacklo_ps(w_im_xmm, w_im_xmm);
        __m128 im_hi = _mm_unpackhi_ps(w_im_xmm, w_im_xmm);
        __m256 w_im = _mm256_insertf128_ps(_mm256_castps128_ps256(im_lo), im_hi, 1);

        __m256 v_swap = _mm256_permute_ps(v, 0xB1);

        __m256 t1 = _mm256_mul_ps(v, w_re);
        __m256 t2 = _mm256_mul_ps(v_swap, w_im);

        if constexpr (!invert)
        {
            v = _mm256_addsub_ps(t1, t2);
        }
        else
        {
            __m256 t2_neg = _mm256_xor_ps(t2, _mm256_set1_ps(-0.0f));
            v = _mm256_addsub_ps(t1, t2_neg);
        }
    }

    template<bool invert>
    [[msvc::forceinline]] void mul_twiddle(float* p, int tw_idx, const float* tw)
    {
        float pr = p[0];
        float pi = p[1];

        float wr = tw[tw_idx];
        float wi = tw[tw_idx + 1];
        if constexpr (invert)
        {
            wi = -wi;
        }

        p[0] = pr * wr - pi * wi;
        p[1] = pr * wi + pi * wr;
    }

    struct FFT1D_C2C_radix7_invoker
    {
        template<bool invert>
        [[msvc::forceinline]] void dispatch(const float* input, float* output, std::size_t length)
        {
            std::size_t p_total = 0;
            std::size_t temp = length;
            while (temp > 1)
            {
                temp /= 7;
                ++p_total;
            }

            const std::uint32_t* rev_table = &radix7_lut::digit_rev_lut.data[
                radix7_lut::digit_rev_lut.offsets[p_total - 1]];

            if (((input != output)) && is_avx2_available)
            {
                butterfly_radix7_stage1_permute_vectorized<invert>(input, output, length, rev_table);
            }
            else
            {
                if (input != output)
                {
                    std::memcpy(output, input, length * sizeof(float) * 2);
                }

                for (std::size_t i = 0; i < length; ++i)
                {
                    std::size_t j = rev_table[i];
                    if (i < j)
                    {
                        std::swap(reinterpret_cast<std::uint64_t*>(output)[i],
                            reinterpret_cast<std::uint64_t*>(output)[j]);
                    }
                }

                for (std::size_t i = 0; i < length; i += 7)
                {
                    float* p = output + i * 2;
                    radix7_kernel_winograd<invert>(p, p + 2, p + 4, p + 6, p + 8, p + 10, p + 12);
                }
            }

            std::size_t step = 49;
            for (std::size_t p = 2; p <= p_total; ++p)
            {
                std::size_t stride = step / 7;
                const auto& lut_soa = radix7_lut::butterfly_lut_soa[p - 1];
                const float* tw_ptr = lut_soa.data;

                for (std::size_t k = 0; k < length; k += step)
                {
                    float* base = output + 2 * k;
                    std::size_t j = 0;

                    if constexpr (is_avx2_available)
                    {
                        if (stride >= 4)
                        {
                            constexpr std::size_t floats_per_m_vector = 16;
                            constexpr std::size_t floats_per_block = 96;

                            for (; j <= stride - 4; j += 4)
                            {
                                float* ptr0 = base + 2 * j;
                                float* ptr1 = base + 2 * (j + stride);
                                float* ptr2 = base + 2 * (j + 2 * stride);
                                float* ptr3 = base + 2 * (j + 3 * stride);
                                float* ptr4 = base + 2 * (j + 4 * stride);
                                float* ptr5 = base + 2 * (j + 5 * stride);
                                float* ptr6 = base + 2 * (j + 6 * stride);

                                __m256 u0 = _mm256_load_ps(ptr0);
                                __m256 u1 = _mm256_load_ps(ptr1);
                                __m256 u2 = _mm256_load_ps(ptr2);
                                __m256 u3 = _mm256_load_ps(ptr3);
                                __m256 u4 = _mm256_load_ps(ptr4);
                                __m256 u5 = _mm256_load_ps(ptr5);
                                __m256 u6 = _mm256_load_ps(ptr6);

                                const float* block_base = tw_ptr + (j / 4) * floats_per_block;

                                apply_twiddle_interleaved<invert>(u1, block_base + 0 * floats_per_m_vector);
                                apply_twiddle_interleaved<invert>(u2, block_base + 1 * floats_per_m_vector);
                                apply_twiddle_interleaved<invert>(u3, block_base + 2 * floats_per_m_vector);
                                apply_twiddle_interleaved<invert>(u4, block_base + 3 * floats_per_m_vector);
                                apply_twiddle_interleaved<invert>(u5, block_base + 4 * floats_per_m_vector);
                                apply_twiddle_interleaved<invert>(u6, block_base + 5 * floats_per_m_vector);

                                radix7_kernel_winograd_interleaved<invert>(u0, u1, u2, u3, u4, u5, u6);

                                _mm256_store_ps(ptr0, u0);
                                _mm256_store_ps(ptr1, u1);
                                _mm256_store_ps(ptr2, u2);
                                _mm256_store_ps(ptr3, u3);
                                _mm256_store_ps(ptr4, u4);
                                _mm256_store_ps(ptr5, u5);
                                _mm256_store_ps(ptr6, u6);
                            }
                        }
                    }

                    const auto& lut_aos = radix7_lut::butterfly_lut[p - 1];
                    for (; j < stride; ++j)
                    {
                        const float* tw = &lut_aos.twiddles[j * 12];
                        float* p0 = base + 2 * j;
                        float* p1 = base + 2 * (j + stride);
                        float* p2 = base + 2 * (j + 2 * stride);
                        float* p3 = base + 2 * (j + 3 * stride);
                        float* p4 = base + 2 * (j + 4 * stride);
                        float* p5 = base + 2 * (j + 5 * stride);
                        float* p6 = base + 2 * (j + 6 * stride);

                        mul_twiddle<invert>(p1, 0, tw);
                        mul_twiddle<invert>(p2, 2, tw);
                        mul_twiddle<invert>(p3, 4, tw);
                        mul_twiddle<invert>(p4, 6, tw);
                        mul_twiddle<invert>(p5, 8, tw);
                        mul_twiddle<invert>(p6, 10, tw);

                        radix7_kernel_winograd<invert>(p0, p1, p2, p3, p4, p5, p6);
                    }
                }
                step *= 7;
            }

            if constexpr (invert)
            {
                invert_scale(output, length);
            }
        }
    };
}

#endif