#ifndef _FOYE_FFT_RADIX5_HPP_
#define _FOYE_FFT_RADIX5_HPP_

namespace fy::fft::internal_radix5
{
    namespace radix5_lut
    {
        constexpr std::size_t max_power_of_5 = 7;
        constexpr std::size_t max_table_size = 78125;
        constexpr std::size_t total_rev_size = 97655;
        constexpr std::size_t total_soa_size = []() -> std::size_t
            {
                std::size_t total_floats = 0;
                std::size_t n = 5;
                for (std::size_t p = 1; p <= max_power_of_5; ++p)
                {
                    std::size_t stride = n / 5;
                    std::size_t blocks = (stride + 3) / 4;
                    total_floats += blocks * 64;
                    n *= 5;
                }
                return total_floats;
            }();

        constexpr float c51 = -1.2500000000f;
        constexpr float c52 = 0.55901699437f;
        constexpr float c53 = 0.95105651629f;
        constexpr float c54 = 0.58778525229f;

        struct alignas(64) butterfly_coeffs
        {
            std::array<float, (max_table_size / 5) * 8> twiddles;
        };

        struct alignas(32) butterfly_coeffs_soa
        {
            const float* data;
            std::size_t stride;
        };

        struct alignas(64) digit_rev_lut_type
        {
            std::array<std::uint32_t, total_rev_size> data;
            std::array<std::size_t, max_power_of_5 + 1> offsets;
        };

        using butterfly_lut_type = std::array<butterfly_coeffs, max_power_of_5>;
        using butterfly_lut_soa_type = std::array<butterfly_coeffs_soa, max_power_of_5>;

        static butterfly_lut_type& compute_butterfly_tables()
        {
            static butterfly_lut_type tables{};
            std::size_t n = 5;
            for (std::size_t p = 1; p <= max_power_of_5; ++p)
            {
                std::size_t stride = n / 5;
                auto& table = tables[p - 1];
                for (std::size_t k = 0; k < stride; ++k)
                {
                    for (int m = 1; m <= 4; ++m)
                    {
                        double angle = -2.0 * std::numbers::pi_v<double> *k * m / n;
                        table.twiddles[k * 8 + (m - 1) * 2 + 0] = static_cast<float>(std::cos(angle));
                        table.twiddles[k * 8 + (m - 1) * 2 + 1] = static_cast<float>(std::sin(angle));
                    }
                }
                n *= 5;
            }
            return tables;
        }

        static butterfly_lut_soa_type& compute_butterfly_tables_soa()
        {
            static alignas(32) std::array<float, total_soa_size> storage{};
            static butterfly_lut_soa_type tables{};

            std::size_t n = 5;
            float* storage_ptr = storage.data();

            for (std::size_t p = 1; p <= max_power_of_5; ++p)
            {
                std::size_t stride = n / 5;

                tables[p - 1].stride = stride;
                tables[p - 1].data = storage_ptr;

                for (std::size_t k_base = 0; k_base < stride; k_base += 4)
                {
                    for (std::size_t m = 1; m <= 4; ++m)
                    {
                        float re[8] = { 0 }, im[8] = { 0 };

                        for (std::size_t i = 0; i < 4; ++i)
                        {
                            std::size_t k = k_base + i;
                            if (k < stride)
                            {
                                double angle = -2.0 * std::numbers::pi_v<double> *k * m / n;
                                re[i] = static_cast<float>(std::cos(angle));
                                im[i] = static_cast<float>(std::sin(angle));

                                re[i + 4] = re[i];
                                im[i + 4] = im[i];
                            }
                            else
                            {
                                re[i] = 1.0f; im[i] = 0.0f;
                                re[i + 4] = 1.0f; im[i + 4] = 0.0f;
                            }
                        }

                        std::copy(std::begin(re), std::end(re), storage_ptr);
                        storage_ptr += 8;

                        std::copy(std::begin(im), std::end(im), storage_ptr);
                        storage_ptr += 8;
                    }
                }
                n *= 5;
            }
            return tables;
        }

        static digit_rev_lut_type& compute_digit_rev_tables()
        {
            static digit_rev_lut_type lut{};
            std::size_t offset = 0;
            std::size_t n = 5;
            for (std::size_t p = 1; p <= max_power_of_5; ++p)
            {
                lut.offsets[p - 1] = offset;
                for (std::size_t i = 0; i < n; ++i)
                {
                    std::size_t j = 0, temp = i;
                    for (std::size_t k = 0; k < p; ++k)
                    {
                        j = j * 5 + (temp % 5);
                        temp /= 5;
                    }
                    lut.data[offset + i] = static_cast<std::uint32_t>(j);
                }
                offset += n; n *= 5;
            }
            lut.offsets[max_power_of_5] = offset;
            return lut;
        }

        static const butterfly_lut_type& butterfly_lut = compute_butterfly_tables();
        static const butterfly_lut_soa_type& butterfly_lut_soa = compute_butterfly_tables_soa();
        static const digit_rev_lut_type& digit_rev_lut = compute_digit_rev_tables();
    }

    template<bool invert>
    [[msvc::forceinline]] static void radix5_kernel_winograd(float* p0, float* p1, float* p2, float* p3, float* p4)
    {
        float tr1 = p1[0] + p4[0];
        float ti1 = p1[1] + p4[1];
        float tr2 = p2[0] + p3[0];
        float ti2 = p2[1] + p3[1];
        float tr3 = p1[0] - p4[0];
        float ti3 = p1[1] - p4[1];
        float tr4 = p2[0] - p3[0];
        float ti4 = p2[1] - p3[1];

        float tr5 = tr1 + tr2;
        float ti5 = ti1 + ti2;
        float x0r = p0[0];
        float x0i = p0[1];

        p0[0] = x0r + tr5;
        p0[1] = x0i + ti5;

        float m1r = tr5 * radix5_lut::c51;
        float m1i = ti5 * radix5_lut::c51;
        float m2r = (tr1 - tr2) * radix5_lut::c52;
        float m2i = (ti1 - ti2) * radix5_lut::c52;

        float r_mid = p0[0] + m1r;
        float i_mid = p0[1] + m1i;

        float r1 = r_mid + m2r;
        float i1 = i_mid + m2i;
        float r2 = r_mid - m2r;
        float i2 = i_mid - m2i;

        constexpr float s_fac = invert ? -1.0f : 1.0f;
        float s1r = (tr3 * radix5_lut::c53 + tr4 * radix5_lut::c54) * s_fac;
        float s1i = (ti3 * radix5_lut::c53 + ti4 * radix5_lut::c54) * s_fac;
        float s2r = (tr3 * radix5_lut::c54 - tr4 * radix5_lut::c53) * s_fac;
        float s2i = (ti3 * radix5_lut::c54 - ti4 * radix5_lut::c53) * s_fac;

        p1[0] = r1 + s1i;
        p1[1] = i1 - s1r;
        p4[0] = r1 - s1i;
        p4[1] = i1 + s1r;
        p2[0] = r2 + s2i;
        p2[1] = i2 - s2r;
        p3[0] = r2 - s2i;
        p3[1] = i2 + s2r;
    }

    template<bool invert>
    [[msvc::forceinline]] static void radix5_kernel_winograd_avx2_interleaved(
        __m256& u0, __m256& u1, __m256& u2, __m256& u3, __m256& u4)
    {
        const __m256 vc51 = _mm256_set1_ps(radix5_lut::c51);
        const __m256 vc52 = _mm256_set1_ps(radix5_lut::c52);
        const __m256 vc53 = _mm256_set1_ps(radix5_lut::c53);
        const __m256 vc54 = _mm256_set1_ps(radix5_lut::c54);

        __m256 tr1 = _mm256_add_ps(u1, u4);
        __m256 tr2 = _mm256_add_ps(u2, u3);
        __m256 tr3 = _mm256_sub_ps(u1, u4);
        __m256 tr4 = _mm256_sub_ps(u2, u3);

        __m256 tr5 = _mm256_add_ps(tr1, tr2);

        u0 = _mm256_add_ps(u0, tr5);

        __m256 m1 = _mm256_mul_ps(tr5, vc51);
        __m256 m2 = _mm256_mul_ps(_mm256_sub_ps(tr1, tr2), vc52);

        __m256 mid = _mm256_add_ps(u0, m1);
        __m256 r1_tmp = _mm256_add_ps(mid, m2);
        __m256 r2_tmp = _mm256_sub_ps(mid, m2);

        __m256 s1 = _mm256_fmadd_ps(tr3, vc53, _mm256_mul_ps(tr4, vc54));
        __m256 s2 = _mm256_fmsub_ps(tr3, vc54, _mm256_mul_ps(tr4, vc53));

        if constexpr (invert) 
        {
            __m256 vneg = _mm256_set1_ps(-1.0f);
            s1 = _mm256_mul_ps(s1, vneg);
            s2 = _mm256_mul_ps(s2, vneg);
        }

        __m256 s1_swp = _mm256_permute_ps(s1, 0xB1);
        __m256 s2_swp = _mm256_permute_ps(s2, 0xB1);

        const __m256 neg_mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);

        __m256 s1_rot = _mm256_xor_ps(s1_swp, neg_mask);
        __m256 s2_rot = _mm256_xor_ps(s2_swp, neg_mask);

        u1 = _mm256_add_ps(r1_tmp, s1_rot);
        u4 = _mm256_sub_ps(r1_tmp, s1_rot);

        u2 = _mm256_add_ps(r2_tmp, s2_rot);
        u3 = _mm256_sub_ps(r2_tmp, s2_rot);
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly_radix5_stage1_permute(
        const float* src, float* dst, std::size_t n, const std::uint32_t* rev_table)
    {
        std::size_t i = 0;
        std::size_t n_vec = n - 5;

        if constexpr (is_avx2_available)
        {
            for (; i < n_vec; i += 20)
            {
                const std::uint32_t* r = rev_table + i;

                __m128i v_idx0 = _mm_set_epi32(r[15], r[10], r[5], r[0]);
                __m128i v_idx1 = _mm_set_epi32(r[16], r[11], r[6], r[1]);
                __m128i v_idx2 = _mm_set_epi32(r[17], r[12], r[7], r[2]);
                __m128i v_idx3 = _mm_set_epi32(r[18], r[13], r[8], r[3]);
                __m128i v_idx4 = _mm_set_epi32(r[19], r[14], r[9], r[4]);

                __m256 u0 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx0, 8));
                __m256 u1 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx1, 8));
                __m256 u2 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx2, 8));
                __m256 u3 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx3, 8));
                __m256 u4 = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(src), v_idx4, 8));

                radix5_kernel_winograd_avx2_interleaved<invert>(u0, u1, u2, u3, u4);

                transpose_4x4_complexf32(u0, u1, u2, u3);

                _mm256_store_ps(dst + i * 2, u0);
                _mm_store_sd(reinterpret_cast<double*>(dst + i * 2 + 8), _mm_castps_pd(_mm256_castps256_ps128(u4)));

                _mm256_store_ps(dst + i * 2 + 10, u1);
                _mm_storeh_pd(reinterpret_cast<double*>(dst + i * 2 + 18), _mm_castps_pd(_mm256_castps256_ps128(u4)));

                _mm256_store_ps(dst + i * 2 + 20, u2);
                _mm_store_sd(reinterpret_cast<double*>(dst + i * 2 + 28), _mm_castps_pd(_mm256_extractf128_ps(u4, 1)));

                _mm256_store_ps(dst + i * 2 + 30, u3);
                _mm_storeh_pd(reinterpret_cast<double*>(dst + i * 2 + 38), _mm_castps_pd(_mm256_extractf128_ps(u4, 1)));
            }
        }
        
        for (; i < n; i += 5)
        {
            std::size_t idx0 = rev_table[i];
            std::size_t idx1 = rev_table[i + 1];
            std::size_t idx2 = rev_table[i + 2];
            std::size_t idx3 = rev_table[i + 3];
            std::size_t idx4 = rev_table[i + 4];

            float p0[2] = { src[2 * idx0], src[2 * idx0 + 1] };
            float p1[2] = { src[2 * idx1], src[2 * idx1 + 1] };
            float p2[2] = { src[2 * idx2], src[2 * idx2 + 1] };
            float p3[2] = { src[2 * idx3], src[2 * idx3 + 1] };
            float p4[2] = { src[2 * idx4], src[2 * idx4 + 1] };

            radix5_kernel_winograd<invert>(p0, p1, p2, p3, p4);

            float* d = dst + i * 2;

            d[0] = p0[0];
            d[1] = p0[1];
            d[2] = p1[0]; 
            d[3] = p1[1];
            d[4] = p2[0]; 
            d[5] = p2[1];
            d[6] = p3[0]; 
            d[7] = p3[1];
            d[8] = p4[0]; 
            d[9] = p4[1];
        }
    }

    template<bool invert>
    [[msvc::forceinline]] static void apply_twiddle_interleaved(__m256& v, const float* w_ptr)
    {
        __m128 w_re_xmm = _mm_loadu_ps(w_ptr);
        __m128 w_im_xmm = _mm_loadu_ps(w_ptr + 8);

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

    struct FFT1D_C2C_radix5_invoker
    {
        template<bool invert>
        void dispatch(const float* input, float* output, std::size_t length)
        {
            std::size_t p_total = 0;
            std::size_t temp = length;
            while (temp > 1) 
            {
                temp /= 5; 
                ++p_total; 
            }

            const std::uint32_t* rev_table = &radix5_lut::digit_rev_lut.data[
                radix5_lut::digit_rev_lut.offsets[p_total - 1]];

            if (input != output)
            {
                butterfly_radix5_stage1_permute<invert>(input, output, length, rev_table);
            }
            else
            {
                for (std::size_t i = 0; i < length; ++i)
                {
                    std::size_t j = rev_table[i];
                    if (i < j)
                    {
                        std::swap(reinterpret_cast<std::uint64_t*>(output)[i],
                            reinterpret_cast<std::uint64_t*>(output)[j]);
                    }
                }
                for (std::size_t i = 0; i < length; i += 5) 
                {
                    float* p = output + i * 2;
                    radix5_kernel_winograd<invert>(p, p + 2, p + 4, p + 6, p + 8);
                }
            }

            std::size_t step = 25;
            for (std::size_t p = 2; p <= p_total; ++p)
            {
                std::size_t stride = step / 5;
                const auto& lut_soa = radix5_lut::butterfly_lut_soa[p - 1];
                const float* tw_ptr = lut_soa.data;

                for (std::size_t k = 0; k < length; k += step)
                {
                    float* base = output + 2 * k;
                    std::size_t j = 0;

                    if constexpr (is_avx2_available)
                    {
                        if (stride >= 4)
                        {
                            for (; j <= stride - 4; j += 4)
                            {
                                float* ptr0 = base + 2 * j;
                                float* ptr1 = base + 2 * (j + stride);
                                float* ptr2 = base + 2 * (j + 2 * stride);
                                float* ptr3 = base + 2 * (j + 3 * stride);
                                float* ptr4 = base + 2 * (j + 4 * stride);

                                __m256 u0 = _mm256_load_ps(ptr0);
                                __m256 u1 = _mm256_load_ps(ptr1);
                                __m256 u2 = _mm256_load_ps(ptr2);
                                __m256 u3 = _mm256_load_ps(ptr3);
                                __m256 u4 = _mm256_load_ps(ptr4);

                                const float* block_tw = tw_ptr + (j / 4) * 64;

                                apply_twiddle_interleaved<invert>(u1, block_tw + 0);
                                apply_twiddle_interleaved<invert>(u2, block_tw + 16);
                                apply_twiddle_interleaved<invert>(u3, block_tw + 32);
                                apply_twiddle_interleaved<invert>(u4, block_tw + 48);

                                radix5_kernel_winograd_avx2_interleaved<invert>(u0, u1, u2, u3, u4);

                                _mm256_store_ps(ptr0, u0);
                                _mm256_store_ps(ptr1, u1);
                                _mm256_store_ps(ptr2, u2);
                                _mm256_store_ps(ptr3, u3);
                                _mm256_store_ps(ptr4, u4);
                            }
                        }
                    }
                    
                    const auto& lut_aos = radix5_lut::butterfly_lut[p - 1];
                    for (; j < stride; ++j)
                    {
                        const float* tw = &lut_aos.twiddles[j * 8];
                        float* p0 = base + 2 * j;
                        float* p1 = base + 2 * (j + stride);
                        float* p2 = base + 2 * (j + 2 * stride);
                        float* p3 = base + 2 * (j + 3 * stride);
                        float* p4 = base + 2 * (j + 4 * stride);

                        mul_twiddle<invert>(p1, 0, tw);
                        mul_twiddle<invert>(p2, 2, tw);
                        mul_twiddle<invert>(p3, 4, tw);
                        mul_twiddle<invert>(p4, 6, tw);

                        radix5_kernel_winograd<invert>(p0, p1, p2, p3, p4);
                    }
                }
                step *= 5;
            }

            if constexpr (invert)
            {
                invert_scale(output, length);
            }
        }
    };
}

#endif