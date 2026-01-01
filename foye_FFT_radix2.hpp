#ifndef _FOYE_FFT_RADIX2_HPP_
#define _FOYE_FFT_RADIX2_HPP_

#include <cmath>
#include <algorithm>
#include <limits>
#include <immintrin.h>
#include <array>
#include <assert.h>
#include <memory>
#include <numbers>
#include <thread>

static bool is_power_of_two(std::size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

static std::size_t next_pow2(std::size_t v)
{
    if (v == 0)
    {
        return 1;
    }

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

namespace fy::fft::internal_radix2
{
    namespace radix2_lut
    {
        constexpr std::size_t max_table_bits = 16;
        constexpr std::size_t max_table_size = 1 << max_table_bits;
        constexpr std::size_t total_packed_floats = (1 << (max_table_bits + 2));

        struct alignas(64) master_butterfly_coeffs
        {
            std::array<float, max_table_size> twiddles;
        };

        struct alignas(64) packed_storage
        {
            std::array<float, total_packed_floats> data;
            std::array<const float*, max_table_bits> stage_offsets;
        };

        using bit_rev_lut_type = std::array<std::array<std::size_t, max_table_size>, max_table_bits>;

        static const master_butterfly_coeffs& compute_scalar_table()
        {
            static master_butterfly_coeffs table{};
            constexpr std::size_t n = max_table_size;
            for (std::size_t k = 0; k < n / 2; ++k)
            {
                double angle = -2.0 * std::numbers::pi_v<double> *static_cast<double>(k) / static_cast<double>(n);
                table.twiddles[k * 2] = static_cast<float>(std::cos(angle));
                table.twiddles[k * 2 + 1] = static_cast<float>(std::sin(angle));
            }
            return table;
        }

        static const packed_storage& compute_packed_table()
        {
            static packed_storage storage{};
            std::size_t current_offset = 0;

            for (std::size_t p = 1; p <= max_table_bits; ++p)
            {
                std::size_t n = std::size_t(1) << p;

                storage.stage_offsets[p - 1] = &storage.data[current_offset];

                if (n >= 8)
                {
                    for (std::size_t k = 0; k < n / 2; k += 4)
                    {
                        std::size_t base_idx = current_offset + (k / 4) * 16;

                        for (std::size_t i = 0; i < 4; ++i)
                        {
                            double angle = -2.0 * std::numbers::pi_v<double> *static_cast<double>(k + i) / static_cast<double>(n);
                            float wr = static_cast<float>(std::cos(angle));
                            float wi = static_cast<float>(std::sin(angle));

                            storage.data[base_idx + i * 2 + 0] = wr;
                            storage.data[base_idx + i * 2 + 1] = wi;
                            storage.data[base_idx + 8 + i * 2 + 0] = wi;
                            storage.data[base_idx + 8 + i * 2 + 1] = wr;
                        }
                    }

                    current_offset += (n * 2);
                }
            }

            return storage;
        }

        static bit_rev_lut_type& compute_bit_rev_tables()
        {
            static bit_rev_lut_type tables{};
            for (std::size_t p = 1; p <= max_table_bits; ++p)
            {
                std::size_t n = std::size_t(1) << p;
                auto& table = tables[p - 1];
                for (std::size_t i = 0; i < n; ++i)
                {
                    std::size_t j = 0;
                    std::size_t m = i;
                    for (std::size_t k = 1; k < n; k <<= 1) 
                    {
                        j = (j << 1) | (m & 1); 
                        m >>= 1; 
                    }

                    table[i] = j;
                }
            }
            return tables;
        }

        static const master_butterfly_coeffs& scalar_lut = compute_scalar_table();
        static const packed_storage& packed_lut = compute_packed_table();
        static const bit_rev_lut_type& bit_rev_lut = compute_bit_rev_tables();
    }
    
    template<bool invert>
    [[msvc::forceinline]] static void butterfly_kernel_x4_packed(float* u_ptr, float* v_ptr, const float* w_ptr)
    {
        __m256 u = _mm256_load_ps(u_ptr);
        __m256 v = _mm256_load_ps(v_ptr);

        __m256 w_normal = _mm256_load_ps(w_ptr);
        __m256 w_swapped = _mm256_load_ps(w_ptr + 8);

        if constexpr (invert)
        {
            const __m256 sign_mask_normal = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
            const __m256 sign_mask_swapped = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);

            w_normal = _mm256_xor_ps(w_normal, sign_mask_normal);
            w_swapped = _mm256_xor_ps(w_swapped, sign_mask_swapped);
        }

        __m256 v_re = _mm256_moveldup_ps(v);
        __m256 v_im = _mm256_movehdup_ps(v);

        __m256 vw = _mm256_fmaddsub_ps(v_re, w_normal, _mm256_mul_ps(v_im, w_swapped));

        __m256 u_out = _mm256_add_ps(u, vw);
        __m256 v_out = _mm256_sub_ps(u, vw);

        _mm256_store_ps(u_ptr, u_out);
        _mm256_store_ps(v_ptr, v_out);
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly_kernel_x16_packed_interleaved(
        float* u_ptr, float* v_ptr, const float* w_ptr)
    {
        __m256 u0 = _mm256_load_ps(u_ptr + 0);
        __m256 u1 = _mm256_load_ps(u_ptr + 8);
        __m256 u2 = _mm256_load_ps(u_ptr + 16);
        __m256 u3 = _mm256_load_ps(u_ptr + 24);

        __m256 v0 = _mm256_load_ps(v_ptr + 0);
        __m256 v1 = _mm256_load_ps(v_ptr + 8);
        __m256 v2 = _mm256_load_ps(v_ptr + 16);
        __m256 v3 = _mm256_load_ps(v_ptr + 24);

        __m256 w0_n = _mm256_load_ps(w_ptr + 0);
        __m256 w0_s = _mm256_load_ps(w_ptr + 8);

        __m256 w1_n = _mm256_load_ps(w_ptr + 16);
        __m256 w1_s = _mm256_load_ps(w_ptr + 24);

        __m256 w2_n = _mm256_load_ps(w_ptr + 32);
        __m256 w2_s = _mm256_load_ps(w_ptr + 40);

        __m256 w3_n = _mm256_load_ps(w_ptr + 48);
        __m256 w3_s = _mm256_load_ps(w_ptr + 56);

        if constexpr (invert)
        {
            const __m256 sign_n = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
            const __m256 sign_s = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);

            w0_n = _mm256_xor_ps(w0_n, sign_n); w0_s = _mm256_xor_ps(w0_s, sign_s);
            w1_n = _mm256_xor_ps(w1_n, sign_n); w1_s = _mm256_xor_ps(w1_s, sign_s);
            w2_n = _mm256_xor_ps(w2_n, sign_n); w2_s = _mm256_xor_ps(w2_s, sign_s);
            w3_n = _mm256_xor_ps(w3_n, sign_n); w3_s = _mm256_xor_ps(w3_s, sign_s);
        }

        __m256 v0_re = _mm256_moveldup_ps(v0);
        __m256 v1_re = _mm256_moveldup_ps(v1);
        __m256 v2_re = _mm256_moveldup_ps(v2);
        __m256 v3_re = _mm256_moveldup_ps(v3);

        __m256 v0_im = _mm256_movehdup_ps(v0);
        __m256 v1_im = _mm256_movehdup_ps(v1);
        __m256 v2_im = _mm256_movehdup_ps(v2);
        __m256 v3_im = _mm256_movehdup_ps(v3);

        __m256 vw0 = _mm256_fmaddsub_ps(v0_re, w0_n, _mm256_mul_ps(v0_im, w0_s));
        __m256 vw1 = _mm256_fmaddsub_ps(v1_re, w1_n, _mm256_mul_ps(v1_im, w1_s));
        __m256 vw2 = _mm256_fmaddsub_ps(v2_re, w2_n, _mm256_mul_ps(v2_im, w2_s));
        __m256 vw3 = _mm256_fmaddsub_ps(v3_re, w3_n, _mm256_mul_ps(v3_im, w3_s));

        _mm256_store_ps(u_ptr + 0, _mm256_add_ps(u0, vw0));
        _mm256_store_ps(v_ptr + 0, _mm256_sub_ps(u0, vw0));

        _mm256_store_ps(u_ptr + 8, _mm256_add_ps(u1, vw1));
        _mm256_store_ps(v_ptr + 8, _mm256_sub_ps(u1, vw1));

        _mm256_store_ps(u_ptr + 16, _mm256_add_ps(u2, vw2));
        _mm256_store_ps(v_ptr + 16, _mm256_sub_ps(u2, vw2));

        _mm256_store_ps(u_ptr + 24, _mm256_add_ps(u3, vw3));
        _mm256_store_ps(v_ptr + 24, _mm256_sub_ps(u3, vw3));
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly(float* data, std::size_t start, std::size_t len, int p)
    {
        std::size_t half_len = len / 2;
        float* u_ptr = data + start * 2;
        float* v_ptr = u_ptr + half_len * 2;
        std::size_t j = 0;

        if constexpr (is_avx2_available)
        {
            const float* w_ptr = radix2_lut::packed_lut.stage_offsets[p - 1];
            for (; j + 15 < half_len; j += 16)
            {
                butterfly_kernel_x16_packed_interleaved<invert>(u_ptr, v_ptr, w_ptr);

                u_ptr += 32;
                v_ptr += 32;
                w_ptr += 64;
            }

            for (; j + 3 < half_len; j += 4)
            {
                butterfly_kernel_x4_packed<invert>(u_ptr, v_ptr, w_ptr);
                u_ptr += 8;
                v_ptr += 8;
                w_ptr += 16;
            }
        }

        const float* master_twiddles = radix2_lut::scalar_lut.twiddles.data();

        std::size_t stride = radix2_lut::max_table_size / len;
        const float* twiddle_ptr = master_twiddles + j * stride * 2;

        std::size_t stride_x2 = stride * 2;
        for (; j < half_len; ++j)
        {
            float w_real = twiddle_ptr[0];
            float w_imag = twiddle_ptr[1];
            if constexpr (invert)
            {
                w_imag = -w_imag;
            }

            float v_real = v_ptr[0];
            float v_imag = v_ptr[1];

            float vw_real = v_real * w_real - v_imag * w_imag;
            float vw_imag = v_real * w_imag + v_imag * w_real;

            float u_real = u_ptr[0];
            float u_imag = u_ptr[1];

            u_ptr[0] = u_real + vw_real;
            u_ptr[1] = u_imag + vw_imag;
            v_ptr[0] = u_real - vw_real;
            v_ptr[1] = u_imag - vw_imag;

            u_ptr += 2;
            v_ptr += 2;

            twiddle_ptr += stride_x2;
        }
    }

    [[msvc::forceinline]] static __m256 butterfly_stage1_kernel(__m256 v)
    {
        __m256 t1 = _mm256_permute_ps(v, 0x44);
        __m256 t2 = _mm256_permute_ps(v, 0xEE);
        __m256 sum = _mm256_add_ps(t1, t2);
        __m256 diff = _mm256_sub_ps(t1, t2);
        return _mm256_shuffle_ps(sum, diff, 0x44);
    }

    [[msvc::forceinline]] static __m256 butterfly_stage2_kernel(__m256 v, __m256 rot_mask)
    {
        __m256 right = _mm256_permute2f128_ps(v, v, 0x11);

        __m256 right_swapped = _mm256_permute_ps(right, 0xB1);
        __m256 right_mixed = _mm256_blend_ps(right, right_swapped, 0xCC);
        __m256 right_twiddled = _mm256_xor_ps(right_mixed, rot_mask);

        __m256 sum = _mm256_add_ps(v, right_twiddled);
        __m256 diff = _mm256_sub_ps(v, right_twiddled);

        return _mm256_permute2f128_ps(sum, diff, 0x20);
    }

    template<bool invert>
    [[msvc::forceinline]] static void butterfly_fused_first_3_stages(float* data, std::size_t n)
    {
        constexpr float s2 = 0.7071067811865475244f;

        const __m256 stage2_rot_mask = invert
            ? _mm256_setr_ps(0.0f, 0.0f, -0.0f, 0.0f, 0.0f, 0.0f, -0.0f, 0.0f)
            : _mm256_setr_ps(0.0f, 0.0f, 0.0f, -0.0f, 0.0f, 0.0f, 0.0f, -0.0f);

        const __m256 w_stage3 = invert
            ? _mm256_setr_ps(1.0f, 0.0f, s2, s2, 0.0f, 1.0f, -s2, s2)
            : _mm256_setr_ps(1.0f, 0.0f, s2, -s2, 0.0f, -1.0f, -s2, -s2);

        const std::size_t length = n * 2;

        for (std::size_t i = 0; i < length; i += 16)
        {
            __m256 y0 = _mm256_load_ps(data + i);
            __m256 y1 = _mm256_load_ps(data + i + 8);

            y0 = butterfly_stage1_kernel(y0);
            y1 = butterfly_stage1_kernel(y1);

            y0 = butterfly_stage2_kernel(y0, stage2_rot_mask);
            y1 = butterfly_stage2_kernel(y1, stage2_rot_mask);

            __m256 y1_re = _mm256_moveldup_ps(y1);
            __m256 y1_im = _mm256_movehdup_ps(y1);

            __m256 w_swapped = _mm256_permute_ps(w_stage3, 0xB1);
            __m256 term2 = _mm256_mul_ps(y1_im, w_swapped);

            __m256 y1_twiddled = _mm256_fmaddsub_ps(y1_re, w_stage3, term2);

            __m256 out0 = _mm256_add_ps(y0, y1_twiddled);
            __m256 out1 = _mm256_sub_ps(y0, y1_twiddled);

            _mm256_store_ps(data + i, out0);
            _mm256_store_ps(data + i + 8, out1);
        }
    }

    [[msvc::forceinline]] static __m256 gather_4_complex(const float* src, const std::size_t* indices, std::size_t k)
    {
        std::size_t i0 = indices[k + 0];
        std::size_t i1 = indices[k + 1];
        std::size_t i2 = indices[k + 2];
        std::size_t i3 = indices[k + 3];

        __m128d d0 = _mm_load_sd(reinterpret_cast<const double*>(src + i0 * 2));
        __m128d d1 = _mm_load_sd(reinterpret_cast<const double*>(src + i1 * 2));
        __m128d d2 = _mm_load_sd(reinterpret_cast<const double*>(src + i2 * 2));
        __m128d d3 = _mm_load_sd(reinterpret_cast<const double*>(src + i3 * 2));

        __m128 f0 = _mm_castpd_ps(d0);
        __m128 f1 = _mm_castpd_ps(d1);
        __m128 f2 = _mm_castpd_ps(d2);
        __m128 f3 = _mm_castpd_ps(d3);

        __m128 v01 = _mm_movelh_ps(f0, f1);
        __m128 v23 = _mm_movelh_ps(f2, f3);

        return _mm256_insertf128_ps(_mm256_castps128_ps256(v01), v23, 1);
    }

    template<bool invert>
    [[msvc::forceinline]] static void process_block_fused_gather(const float* src, float* dst,
        std::size_t offset, std::size_t length,
        const std::size_t* bit_rev_table)
    {
        constexpr float s2 = 0.7071067811865475244f;

        const __m256 stage2_rot_mask = invert
            ? _mm256_setr_ps(0.0f, 0.0f, -0.0f, 0.0f, 0.0f, 0.0f, -0.0f, 0.0f)
            : _mm256_setr_ps(0.0f, 0.0f, 0.0f, -0.0f, 0.0f, 0.0f, 0.0f, -0.0f);

        const __m256 w_stage3 = invert
            ? _mm256_setr_ps(1.0f, 0.0f, s2, s2, 0.0f, 1.0f, -s2, s2)
            : _mm256_setr_ps(1.0f, 0.0f, s2, -s2, 0.0f, -1.0f, -s2, -s2);

        for (std::size_t i = 0; i < length; i += 8)
        {
            const std::size_t* current_indices = bit_rev_table + offset + i;

            __m256 y0 = gather_4_complex(src, current_indices, 0);
            __m256 y1 = gather_4_complex(src, current_indices, 4);

            y0 = butterfly_stage1_kernel(y0);
            y1 = butterfly_stage1_kernel(y1);
            y0 = butterfly_stage2_kernel(y0, stage2_rot_mask);
            y1 = butterfly_stage2_kernel(y1, stage2_rot_mask);

            __m256 y1_re = _mm256_moveldup_ps(y1);
            __m256 y1_im = _mm256_movehdup_ps(y1);

            __m256 term1 = _mm256_mul_ps(y1_re, w_stage3);
            __m256 w_swapped = _mm256_permute_ps(w_stage3, 0xB1);
            __m256 term2 = _mm256_mul_ps(y1_im, w_swapped);
            __m256 y1_twiddled = _mm256_addsub_ps(term1, term2);

            __m256 out0 = _mm256_add_ps(y0, y1_twiddled);
            __m256 out1 = _mm256_sub_ps(y0, y1_twiddled);

            _mm256_store_ps(dst + i * 2, out0);
            _mm256_store_ps(dst + i * 2 + 8, out1);
        }

        std::size_t p = 4;
        for (std::size_t len = 16; len <= length; len <<= 1)
        {
            for (std::size_t k = 0; k < length; k += len)
            {
                butterfly<invert>(dst + k * 2, 0, len, p);
            }

            ++p;
        }
    }

    struct FFT1D_C2C_radix2_invoker
    {
    private:
        static constexpr std::size_t L1_CACHE_THRESHOLD = 4096;

        static void bit_reverse_inplace(float* data, std::size_t length)
        {
            std::size_t p = 0;
            std::size_t temp = length;
            while (temp >>= 1)
            {
                ++p;
            }

            if (p >= 1 && p <= static_cast<std::size_t>(radix2_lut::max_table_bits))
            {
                const std::array<std::size_t, radix2_lut::max_table_size> & table = radix2_lut::bit_rev_lut[p - 1];
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

        static void bit_reverse_copy(const float* src, float* dst, std::size_t length)
        {
            std::size_t p = 0;
            std::size_t temp = length;
            while (temp >>= 1)
            {
                ++p;
            }

            if (p >= 1 && p <= static_cast<std::size_t>(radix2_lut::max_table_bits))
            {
                const std::array<std::size_t, radix2_lut::max_table_size>& table = radix2_lut::bit_rev_lut[p - 1];
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

        template<bool invert>
        static void recursive_fft_out_of_place(const float* src, float* dst, std::size_t offset,
            std::size_t n, const std::size_t* bit_rev_table)
        {
            if (n <= L1_CACHE_THRESHOLD)
            {
                if constexpr (is_avx2_available) 
                {
                    process_block_fused_gather<invert>(src, dst + offset * 2, offset, n, bit_rev_table);
                }
                else
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        std::size_t src_idx = bit_rev_table[offset + i];
                        dst[(offset + i) * 2 + 0] = src[src_idx * 2 + 0];
                        dst[(offset + i) * 2 + 1] = src[src_idx * 2 + 1];
                    }

                    process_block_in_l1<invert>(dst + offset * 2, n);
                }
                return;
            }

            recursive_fft_out_of_place<invert>(src, dst, offset, n / 2, bit_rev_table);
            recursive_fft_out_of_place<invert>(src, dst, offset + n / 2, n / 2, bit_rev_table);

            std::size_t p = 0;
            std::size_t temp = n;
            while (temp >>= 1)
            {
                ++p;
            }

            butterfly<invert>(dst + offset * 2, 0, n, p);
        }

        template<bool invert>
        [[msvc::forceinline]] static void process_block_in_l1(float* data, std::size_t length)
        {
            if constexpr (is_avx2_available)
            {
                butterfly_fused_first_3_stages<invert>(data, length);
            }
            else
            {
                for (std::size_t i = 0; i < length; i += 2)
                {
                    butterfly<invert>(data + i * 2, 0, 2, 1);
                }

                if (length >= 4)
                {
                    for (std::size_t i = 0; i < length; i += 4)
                    {
                        butterfly<invert>(data + i * 2, 0, 4, 2);
                    }
                }

                if (length >= 8)
                {
                    for (std::size_t i = 0; i < length; i += 8)
                    {
                        butterfly<invert>(data + i * 2, 0, 8, 3);
                    }
                }
            }

            std::size_t p = 4;
            for (std::size_t len = 16; len <= length; len <<= 1)
            {
                for (std::size_t i = 0; i < length; i += len)
                {
                    butterfly<invert>(data + i * 2, 0, len, p);
                }
                ++p;
            }
        }

        template<bool invert>
        static void recursive_fft_stage(float* data, std::size_t n)
        {
            if (n <= L1_CACHE_THRESHOLD)
            {
                process_block_in_l1<invert>(data, n);
                return;
            }

            recursive_fft_stage<invert>(data, n / 2);
            recursive_fft_stage<invert>(data + n, n / 2);

            std::size_t p = 0;
            std::size_t temp = n;
            while (temp >>= 1)
            {
                ++p;
            }

            butterfly<invert>(data, 0, n, p);
        }

        template<bool invert>
        static void recursive_fft_in_place(float* data, std::size_t n)
        {
            if (n <= L1_CACHE_THRESHOLD)
            {
                if constexpr (is_avx2_available)
                {
                    butterfly_fused_first_3_stages<invert>(data, n);
                }
                else
                {
                    for (std::size_t i = 0; i < n; i += 2)
                    {
                        butterfly<invert>(data + i * 2, 0, 2, 1);
                    }

                    if (n >= 4)
                    {
                        for (std::size_t i = 0; i < n; i += 4)
                        {
                            butterfly<invert>(data + i * 2, 0, 4, 2);
                        }
                    }

                    if (n >= 8)
                    {
                        for (std::size_t i = 0; i < n; i += 8)
                        {
                            butterfly<invert>(data + i * 2, 0, 8, 3);
                        }
                    }
                }

                std::size_t p = 4;
                for (std::size_t len = 16; len <= n; len <<= 1)
                {
                    for (std::size_t i = 0; i < n; i += len)
                    {
                        butterfly<invert>(data + i * 2, 0, len, p);
                    }
                    ++p;
                }
                return;
            }

            recursive_fft_in_place<invert>(data, n / 2);
            recursive_fft_in_place<invert>(data + n, n / 2);

            std::size_t p = 0;
            std::size_t temp = n;
            while (temp >>= 1)
            {
                ++p;
            }

            butterfly<invert>(data, 0, n, p);
        }

    public:
        template<bool invert>
        void forward(const float* input, float* output, std::size_t length)
        {
            assert(is_power_of_two(length));
            if (!especially_kernel_dispatch<invert>(input, output, length))
            {
                assert(length >= 8);

                std::size_t p_total = 0;
                std::size_t temp = length;
                while (temp >>= 1)
                {
                    ++p_total;
                }

                const auto& table = radix2_lut::bit_rev_lut[p_total - 1];

                if (input != output)
                {
                    recursive_fft_out_of_place<invert>(input, output, 0, length, table.data());
                }
                else
                {
                    bit_reverse_inplace(output, length);
                    recursive_fft_in_place<invert>(output, length);
                }

                if constexpr (invert)
                {
                    invert_scale(output, length);
                }
            }
        }
    };

    struct FFT1D_R2C_radix2_invoker
    {
    public:
        void forward(const float* input, float* output, std::size_t n)
        {
            assert((n & (n - 1)) == 0);
            assert(n <= radix2_lut::max_table_size);
            assert(n >= 8);

            FFT1D_C2C_radix2_invoker c2c;
            c2c.forward<false>(input, output, n / 2);

            float* z = output;

            float z0_r = z[0];
            float z0_i = z[1];

            z[0] = z0_r + z0_i;
            z[1] = 0.0f;
            z[n] = z0_r - z0_i;
            z[n + 1] = 0.0f;

            std::size_t n_2 = n / 2;
            std::size_t n_4 = n / 4;

            const std::size_t stride = radix2_lut::max_table_size / n;
            const float* twiddle_ptr = radix2_lut::scalar_lut.twiddles.data();

            twiddle_ptr += stride * 2;

            for (std::size_t k = 1; k <= n_4; ++k)
            {
                std::size_t idx1 = 2 * k;
                std::size_t idx2 = 2 * (n_2 - k);

                float z1_r = z[idx1];
                float z1_i = z[idx1 + 1];
                float z2_r = z[idx2];
                float z2_i = z[idx2 + 1];

                float a_r = z1_r + z2_r;
                float a_i = z1_i - z2_i;

                float b_r = z1_r - z2_r;
                float b_i = z1_i + z2_i;

                float w_r = twiddle_ptr[0];
                float w_i = twiddle_ptr[1];

                float tmp_r = w_r * b_i + w_i * b_r;
                float tmp_i = -w_r * b_r + w_i * b_i;

                z[idx1] = 0.5f * (a_r + tmp_r);
                z[idx1 + 1] = 0.5f * (a_i + tmp_i);

                z[idx2] = 0.5f * (a_r - tmp_r);
                z[idx2 + 1] = -(0.5f * (a_i - tmp_i));

                twiddle_ptr += stride * 2;
            }
        }

        void backward(const float* input, float* output, std::size_t n)
        {
            assert((n & (n - 1)) == 0);
            assert(n <= radix2_lut::max_table_size);
            assert(n >= 8);

            float* z = output;
            const float* x = input;

            float x0_r = x[0];
            float xny_r = x[n];

            z[0] = 0.5f * (x0_r + xny_r);
            z[1] = 0.5f * (x0_r - xny_r);

            std::size_t n_2 = n / 2;
            std::size_t n_4 = n / 4;

            const std::size_t stride = radix2_lut::max_table_size / n;
            const float* twiddle_ptr = radix2_lut::scalar_lut.twiddles.data();
            twiddle_ptr += stride * 2;

            for (std::size_t k = 1; k <= n_4; ++k)
            {
                std::size_t idx1 = 2 * k;
                std::size_t idx2 = 2 * (n_2 - k);

                float x1_r = x[idx1];
                float x1_i = x[idx1 + 1];
                float x2_r = x[idx2];
                float x2_i = x[idx2 + 1];

                float a_r = x1_r + x2_r;
                float a_i = x1_i - x2_i;

                float b_r = x1_r - x2_r;
                float b_i = x1_i + x2_i;

                float w_r = twiddle_ptr[0];
                float w_i_lut = twiddle_ptr[1];

                float w_i = -w_i_lut;

                float tmp_r = -(w_r * b_i + w_i * b_r);
                float tmp_i = w_r * b_r - w_i * b_i;

                z[idx1] = 0.5f * (a_r + tmp_r);
                z[idx1 + 1] = 0.5f * (a_i + tmp_i);

                z[idx2] = 0.5f * (a_r - tmp_r);
                z[idx2 + 1] = -(0.5f * (a_i - tmp_i));

                twiddle_ptr += stride * 2;
            }

            FFT1D_C2C_radix2_invoker c2c;
            c2c.forward<true>(z, output, n / 2);
        }
    };

    struct complex_block_transposer
    {
        static constexpr std::size_t BLOCK_SIZE = 64;

        [[msvc::forceinline]] static void transpose4x4_complex(__m256d& r0, __m256d& r1, __m256d& r2, __m256d& r3)
        {
            __m256d t0 = _mm256_unpacklo_pd(r0, r1);
            __m256d t1 = _mm256_unpackhi_pd(r0, r1);
            __m256d t2 = _mm256_unpacklo_pd(r2, r3);
            __m256d t3 = _mm256_unpackhi_pd(r2, r3);
            
            r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
            r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
        }

        template<bool is_aligned>
        static void transpose_matrix_to_linear(std::size_t rows, std::size_t block_width,
            const float* src_ptr, std::size_t src_stride, float* tmp_ptr)
        {
            if constexpr (is_avx2_available)
            {
                for (std::size_t r = 0; r < rows; r += 8)
                {
                    for (std::size_t c = 0; c < block_width; c += 4)
                    {
                        if (c + 3 >= block_width)
                        {
                            for (std::size_t cc = c; cc < block_width; ++cc)
                            {
                                for (std::size_t rr = r; rr < r + 8; ++rr) 
                                {
                                    std::size_t s_idx = rr * src_stride + cc * 2;
                                    std::size_t d_idx = (cc * rows + rr) * 2;
                                    tmp_ptr[d_idx] = src_ptr[s_idx];
                                    tmp_ptr[d_idx + 1] = src_ptr[s_idx + 1];
                                }
                            }
                            continue;
                        }

                        const float* src_block = src_ptr + r * src_stride + c * 2;
                        __m256d row_vector[8];

                        for (std::size_t rv_i = 0; rv_i < 8; ++rv_i)
                        {
                            const double* p = reinterpret_cast<const double*>(src_block + rv_i * src_stride);
                            if constexpr (is_aligned)
                            {
                                row_vector[rv_i] = _mm256_load_pd(p);
                            }
                            else
                            {
                                row_vector[rv_i] = _mm256_loadu_pd(p);
                            }
                        }

                        transpose4x4_complex(row_vector[0], row_vector[1], row_vector[2], row_vector[3]);
                        transpose4x4_complex(row_vector[4], row_vector[5], row_vector[6], row_vector[7]);

                        for (int i = 0; i < 4; ++i)
                        {
                            float* dst_col = tmp_ptr + (c + i) * rows * 2 + r * 2;
                            __m256d p1 = (i == 0 ? row_vector[0] : (i == 1 ? row_vector[1] : (i == 2 ? row_vector[2] : row_vector[3])));
                            __m256d p2 = (i == 0 ? row_vector[4] : (i == 1 ? row_vector[5] : (i == 2 ? row_vector[6] : row_vector[7])));

                            _mm256_store_pd(reinterpret_cast<double*>(dst_col), p1);
                            _mm256_store_pd(reinterpret_cast<double*>(dst_col + 8), p2);
                        }
                    }
                }
            }
            else
            {
                for (std::size_t c = 0; c < block_width; ++c)
                {
                    for (std::size_t r = 0; r < rows; ++r)
                    {
                        std::size_t s_idx = r * src_stride + c * 2;
                        std::size_t d_idx = (c * rows + r) * 2;
                        tmp_ptr[d_idx] = src_ptr[s_idx];
                        tmp_ptr[d_idx + 1] = src_ptr[s_idx + 1];
                    }
                }
            }
        }

        template<bool is_aligned>
        static void transpose_linear_to_matrix(std::size_t rows, std::size_t block_width,
            const float* tmp_ptr, float* dst_ptr, std::size_t dst_stride)
        {
            if constexpr (is_avx2_available)
            {
                for (std::size_t r = 0; r < rows; r += 8)
                {
                    for (std::size_t c = 0; c < block_width; c += 4)
                    {
                        if (c + 3 >= block_width)
                        {
                            for (std::size_t cc = c; cc < block_width; ++cc)
                            {
                                for (std::size_t rr = r; rr < r + 8; ++rr)
                                {
                                    std::size_t s_idx = (cc * rows + rr) * 2;
                                    std::size_t d_idx = rr * dst_stride + cc * 2;
                                    dst_ptr[d_idx] = tmp_ptr[s_idx];
                                    dst_ptr[d_idx + 1] = tmp_ptr[s_idx + 1];
                                }
                            }
                            continue;
                        }

                        __m256d v[8];
                        for (int i = 0; i < 4; ++i)
                        {
                            const double* src_col = reinterpret_cast<const double*>(tmp_ptr + (c + i) * rows * 2 + r * 2);
                            v[i + 0] = _mm256_load_pd(src_col + 0);
                            v[i + 4] = _mm256_load_pd(src_col + 4);
                        }

                        transpose4x4_complex(v[0], v[1], v[2], v[3]);
                        transpose4x4_complex(v[4], v[5], v[6], v[7]);

                        float* dst_block = dst_ptr + r * dst_stride + c * 2;
                        for (int i = 0; i < 8; ++i)
                        {
                            double* d = reinterpret_cast<double*>(dst_block + i * dst_stride);
                            if constexpr (is_aligned)
                            {
                                _mm256_store_pd(d, v[i]);
                            }
                            else
                            {
                                _mm256_storeu_pd(d, v[i]);
                            }
                        }
                    }
                }
            }
            else
            {
                for (std::size_t c = 0; c < block_width; ++c)
                {
                    for (std::size_t r = 0; r < rows; ++r)
                    {
                        std::size_t s_idx = (c * rows + r) * 2;
                        std::size_t d_idx = r * dst_stride + c * 2;
                        dst_ptr[d_idx] = tmp_ptr[s_idx];
                        dst_ptr[d_idx + 1] = tmp_ptr[s_idx + 1];
                    }
                }
            }
        }
    };

    struct FFT2D_C2C_radix2_invoker
    {
        float* tmp;
        bool ownflag;
        std::size_t rows_;
        std::size_t cols_;
        FFT1D_C2C_radix2_invoker fft1d;

        static std::size_t calc_external_buffer_needs_elements(std::size_t rows, std::size_t cols) 
        {
            return rows * complex_block_transposer::BLOCK_SIZE * 2; 
        }

        FFT2D_C2C_radix2_invoker(const FFT2D_C2C_radix2_invoker&) = delete;
        FFT2D_C2C_radix2_invoker& operator = (const FFT2D_C2C_radix2_invoker&) = delete;
        FFT2D_C2C_radix2_invoker(FFT2D_C2C_radix2_invoker&&) = delete;
        FFT2D_C2C_radix2_invoker& operator = (FFT2D_C2C_radix2_invoker&&) = delete;

        FFT2D_C2C_radix2_invoker(std::size_t rows, std::size_t cols, void* external_buffer = nullptr)
            : rows_(rows), cols_(cols), fft1d(), ownflag(external_buffer == nullptr)
        {
            assert(is_power_of_two(rows));
            assert(rows <= radix2_lut::max_table_size);
            assert(rows >= 8);

            assert(is_power_of_two(cols));
            assert(cols <= radix2_lut::max_table_size);
            assert(cols >= 8);

            std::size_t buffer_size = rows_ * complex_block_transposer::BLOCK_SIZE * 2 * sizeof(float);
            if (ownflag)
            {
                tmp = reinterpret_cast<float*>(_aligned_malloc(buffer_size, 64));
            }
            else
            {
                tmp = reinterpret_cast<float*>(external_buffer);
            }
        }

        ~FFT2D_C2C_radix2_invoker() 
        {
            if (ownflag && tmp)
            {
                _aligned_free(tmp);
                tmp = nullptr;
            }
        }

        template<bool invert>
        void forward(const float* input, float* output)
        {
            for (std::size_t r = 0; r < rows_; ++r)
            {
                const float* src_row = input + r * cols_ * 2;
                float* dst_row = output + r * cols_ * 2;
                fft1d.template forward<invert>(src_row, dst_row, cols_);
            }

            for (std::size_t c_block = 0; c_block < cols_; c_block += complex_block_transposer::BLOCK_SIZE)
            {
                std::size_t current_block_width = std::min(complex_block_transposer::BLOCK_SIZE, cols_ - c_block);

                float* matrix_block_ptr = output + c_block * 2;

                complex_block_transposer::transpose_matrix_to_linear<true>(rows_, 
                    current_block_width, matrix_block_ptr, cols_ * 2, tmp);

                for (std::size_t i = 0; i < current_block_width; ++i)
                {
                    float* col_ptr = tmp + i * rows_ * 2;
                    fft1d.template forward<invert>(col_ptr, col_ptr, rows_);
                }

                complex_block_transposer::transpose_linear_to_matrix<true>(rows_, 
                    current_block_width, tmp, matrix_block_ptr, cols_ * 2);
            }
        }
    };


    struct FFT2D_R2C_radix2_invoker
    {
        float* tmp;
        float* backward_store;
        bool own_tmp;
        bool own_backward_store;
        std::size_t rows_;
        std::size_t cols_;
        FFT1D_R2C_radix2_invoker fft1d_r2c;
        FFT1D_C2C_radix2_invoker fft1d_c2c;

        static std::size_t external_forward_buffer_needs_elements(std::size_t rows) { return rows * complex_block_transposer::BLOCK_SIZE * 2; }
        static std::size_t external_backward_buffer_needs_elements(std::size_t rows, std::size_t cols) { return rows * (cols / 2 + 1) * 2; }

        FFT2D_R2C_radix2_invoker(const FFT2D_R2C_radix2_invoker&) = delete;
        FFT2D_R2C_radix2_invoker& operator = (const FFT2D_R2C_radix2_invoker&) = delete;
        FFT2D_R2C_radix2_invoker(FFT2D_R2C_radix2_invoker&&) = delete;
        FFT2D_R2C_radix2_invoker& operator = (FFT2D_R2C_radix2_invoker&&) = delete;

        FFT2D_R2C_radix2_invoker(std::size_t rows, std::size_t cols, void* fwd = nullptr, void* bwd = nullptr)
            : rows_(rows), cols_(cols), fft1d_r2c(), fft1d_c2c()
        {
            assert(is_power_of_two(rows));
            assert(rows <= radix2_lut::max_table_size);
            assert(rows >= 8);

            assert(is_power_of_two(cols));
            assert(cols <= radix2_lut::max_table_size);
            assert(cols >= 8);

            std::size_t fwd_bytes = external_forward_buffer_needs_elements(rows_) * sizeof(float);
            if (!fwd) 
            {
                tmp = reinterpret_cast<float*>(_aligned_malloc(fwd_bytes, 64));
                own_tmp = true; 
            }
            else
            {
                tmp = reinterpret_cast<float*>(fwd);
                own_tmp = false; 
            }

            std::size_t bwd_bytes = external_backward_buffer_needs_elements(rows_, cols_) * sizeof(float);
            if (!bwd) 
            {
                backward_store = reinterpret_cast<float*>(_aligned_malloc(bwd_bytes, 64)); 
                own_backward_store = true;
            }
            else 
            {
                backward_store = reinterpret_cast<float*>(bwd);
                own_backward_store = false; 
            }
        }

        ~FFT2D_R2C_radix2_invoker() 
        {
            if (own_tmp && tmp)
            {
                _aligned_free(tmp);
                tmp = nullptr;
            }

            if (own_backward_store && backward_store)
            {
                _aligned_free(backward_store);
                backward_store = nullptr;
            }
        }

        template<bool invert>
        void process_columns_blocked(const float* src_base, float* dst_base, std::size_t complex_cols)
        {
            std::size_t row_stride = complex_cols * 2;

            for (std::size_t c_block = 0; c_block < complex_cols; c_block += complex_block_transposer::BLOCK_SIZE)
            {
                std::size_t current_block_width = std::min(complex_block_transposer::BLOCK_SIZE, complex_cols - c_block);
                const float* src_ptr = src_base + c_block * 2;
                float* dst_ptr = dst_base + c_block * 2;

                complex_block_transposer::transpose_matrix_to_linear<false>(
                    rows_, current_block_width,
                    src_ptr, row_stride,
                    tmp
                );

                for (std::size_t i = 0; i < current_block_width; ++i)
                {
                    fft1d_c2c.forward<invert>(tmp + i * rows_ * 2, tmp + i * rows_ * 2, rows_);
                }

                complex_block_transposer::transpose_linear_to_matrix<false>(
                    rows_, current_block_width,
                    tmp,
                    dst_ptr, row_stride
                );
            }
        }

        void forward(const float* input, float* output)
        {
            [[msvc::flatten]]
            {
                std::size_t complex_cols = cols_ / 2 + 1;
                for (std::size_t r = 0; r < rows_; ++r)
                {
                    fft1d_r2c.forward(input + r * cols_, output + r * complex_cols * 2, cols_);
                }
                process_columns_blocked<false>(output, output, complex_cols);
            }
        }

        void backward(const float* input, float* output)
        {
            [[msvc::flatten]]
            {
                std::size_t complex_cols = cols_ / 2 + 1;
                process_columns_blocked<true>(input, backward_store, complex_cols);
                for (std::size_t r = 0; r < rows_; ++r)
                {
                    fft1d_r2c.backward(backward_store + r * complex_cols * 2, output + r * cols_, cols_);
                }
            }
        }
    };
}

namespace fy::fft
{
    using fy::fft::internal_radix2::FFT1D_C2C_radix2_invoker;
    using fy::fft::internal_radix2::FFT1D_R2C_radix2_invoker;

    using fy::fft::internal_radix2::FFT2D_C2C_radix2_invoker;
    using fy::fft::internal_radix2::FFT2D_R2C_radix2_invoker;
}

#endif