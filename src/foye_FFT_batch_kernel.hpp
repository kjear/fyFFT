#ifndef _FOYE_FFT_BATCH_KERNEL_HPP_
#define _FOYE_FFT_BATCH_KERNEL_HPP_

namespace fy::fft
{
    template<bool invert>
    [[msvc::forceinline]] static void apply_radix2_batch(const float* input, float* output, std::size_t batch)
    {
        std::size_t b = 0;
        if constexpr (is_avx2_available)
        {
            const __m256 v_half = _mm256_set1_ps(0.5f);

            for (; b + 1 < batch; b += 2)
            {
                __m256 in8 = _mm256_loadu_ps(input + b * 4);

                __m256 lo = _mm256_shuffle_ps(in8, in8, _MM_SHUFFLE(1, 0, 1, 0));
                __m256 hi = _mm256_shuffle_ps(in8, in8, _MM_SHUFFLE(3, 2, 3, 2));

                __m256 sum = _mm256_add_ps(lo, hi);
                __m256 diff = _mm256_sub_ps(lo, hi);

                __m256 res = _mm256_shuffle_ps(sum, diff, _MM_SHUFFLE(1, 0, 1, 0));

                if constexpr (invert)
                {
                    res = _mm256_mul_ps(res, v_half);
                }

                _mm256_storeu_ps(output + b * 4, res);
            }
        }

        for (; b < batch; ++b)
        {
            apply_radix2_single<invert>(
                input + b * 4,
                output + b * 4);
        }
    }

    template<bool invert>
    [[msvc::forceinline]] static void apply_radix3_batch(const float* input, float* output, std::size_t batch)
    {
        std::size_t b = 0;
        auto radix3_2batch_kernel_avx2 = [](const __m256 v0, const __m256 v1, float* out12) -> void
            {
                const __m128 a = _mm256_castps256_ps128(v0);
                const __m128 b = _mm256_extractf128_ps(v0, 1);
                const __m128 c = _mm256_castps256_ps128(v1);

                const __m128 f0_x0 = _mm_movelh_ps(a, a);
                const __m128 f0_x1 = _mm_movehl_ps(a, a);
                const __m128 f0_x2 = _mm_movelh_ps(b, b);

                const __m128 f1_x0 = _mm_movehl_ps(b, b);
                const __m128 f1_x1 = _mm_movelh_ps(c, c);
                const __m128 f1_x2 = _mm_movehl_ps(c, c);

                auto pack2 = [](__m128 p0, __m128 p1) -> __m256 
                {
                    __m128 lo = _mm_shuffle_ps(p0, p1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m128 hi = lo;
                    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
                };

                __m256 x0 = pack2(f0_x0, f1_x0);
                __m256 x1 = pack2(f0_x1, f1_x1);
                __m256 x2 = pack2(f0_x2, f1_x2);

                __m256 sr = _mm256_add_ps(x1, x2);
                __m256 dr = _mm256_sub_ps(x1, x2);

                __m256 X0 = _mm256_add_ps(x0, sr);

                const __m256 half = _mm256_set1_ps(0.5f);
                __m256 m = _mm256_fnmadd_ps(sr, half, x0);

                constexpr float s = 0.86602540378443864676f;
                const __m256 s_fac = _mm256_set1_ps(invert ? s : -s);

                __m256 dr_sw = _mm256_permute_ps(dr, 0xB1);
                __m256 t = _mm256_mul_ps(dr_sw, s_fac);
                const __m256 neg_real_mask = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
                __m256 s_vec = _mm256_xor_ps(t, neg_real_mask);

                __m256 X1 = _mm256_add_ps(m, s_vec);
                __m256 X2 = _mm256_sub_ps(m, s_vec);

                if constexpr (invert)
                {
                    const __m256 inv3 = _mm256_set1_ps(1.0f / 3.0f);
                    X0 = _mm256_mul_ps(X0, inv3);
                    X1 = _mm256_mul_ps(X1, inv3);
                    X2 = _mm256_mul_ps(X2, inv3);
                }

                __m128 X0_lo = _mm256_castps256_ps128(X0);
                __m128 X1_lo = _mm256_castps256_ps128(X1);
                __m128 X2_lo = _mm256_castps256_ps128(X2);

                __m128 f0X0_f0X1 = _mm_movelh_ps(X0_lo, X1_lo);
                __m128 f0X2_f1X0 = _mm_shuffle_ps(X2_lo, X0_lo, _MM_SHUFFLE(3, 2, 1, 0));
                __m256 out0_8 = _mm256_insertf128_ps(_mm256_castps128_ps256(f0X0_f0X1), f0X2_f1X0, 1);
                _mm256_store_ps(out12 + 0, out0_8);

                __m128 f1X1_f1X2 = _mm_movehl_ps(X2_lo, X1_lo);
                _mm_store_ps(out12 + 8, f1X1_f1X2);
            };

        if constexpr (is_avx2_available)
        {
            for (; b + 2 < batch; b += 2)
            {
                const float* in = input + b * 6;
                float* out = output + b * 6;

                __m256 v0 = _mm256_load_ps(in + 0);
                __m256 v1 = _mm256_load_ps(in + 8);
                radix3_2batch_kernel_avx2(v0, v1, out);
            }

            const std::size_t rem = batch - b;
            if (rem == 0)
            {
                return;
            }
            else if (rem == 1)
            {
                apply_radix3_single<invert>(input + b * 6, output + b * 6);
                return;
            }
            else
            {
                const float* in = input + b * 6;
                float* out = output + b * 6;

                __m128 a0 = _mm_load_ps(in + 0);
                __m128 a1 = _mm_load_ps(in + 4);
                __m128 a2 = _mm_load_ps(in + 8);

                __m256 v0 = _mm256_insertf128_ps(_mm256_castps128_ps256(a0), a1, 1);
                __m256 v1 = _mm256_insertf128_ps(_mm256_castps128_ps256(a2), _mm_setzero_ps(), 1);
                radix3_2batch_kernel_avx2(v0, v1, out);
            }
        }
        else
        {
            for (; b < batch; ++b)
            {
                apply_radix3_single<invert>(input + b * 6, output + b * 6);
            }
        }
    }

    template<bool invert>
    [[msvc::forceinline]] static void apply_radix4_batch(
        const float* input,
        float* output,
        std::size_t batch)
    {
        std::size_t b = 0;
        if constexpr (is_avx2_available)
        {
            const __m256 v_neg_mask = invert
                ? _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f)
                : _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);

            for (; b + 1 < batch; b += 2)
            {
                __m256 in0 = _mm256_load_ps(input + b * 8);
                __m256 in1 = _mm256_load_ps(input + (b + 1) * 8);

                __m256 v_low = _mm256_permute2f128_ps(in0, in1, 0x20);
                __m256 v_high = _mm256_permute2f128_ps(in0, in1, 0x31);

                __m256 a_even = _mm256_add_ps(v_low, v_high);
                __m256 a_odd = _mm256_sub_ps(v_low, v_high);

                __m256 a_even_swap = _mm256_permute_ps(a_even, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 a_odd_swap = _mm256_permute_ps(a_odd, _MM_SHUFFLE(1, 0, 3, 2));

                __m256 X0_raw = _mm256_add_ps(a_even, a_even_swap);
                __m256 X2_raw = _mm256_sub_ps(a_even, a_even_swap);

                __m256 rot = _mm256_permute_ps(a_odd_swap, _MM_SHUFFLE(2, 3, 0, 1));
                rot = _mm256_xor_ps(rot, v_neg_mask);

                __m256 X1_raw = _mm256_add_ps(a_odd, rot);
                __m256 X3_raw = _mm256_sub_ps(a_odd, rot);

                if constexpr (invert)
                {
                    const __m256 v_inv = _mm256_set1_ps(0.25f);
                    X0_raw = _mm256_mul_ps(X0_raw, v_inv);
                    X1_raw = _mm256_mul_ps(X1_raw, v_inv);
                    X2_raw = _mm256_mul_ps(X2_raw, v_inv);
                    X3_raw = _mm256_mul_ps(X3_raw, v_inv);
                }

                __m256 T1 = _mm256_shuffle_ps(X0_raw, X1_raw, _MM_SHUFFLE(1, 0, 1, 0));
                __m256 T2 = _mm256_shuffle_ps(X2_raw, X3_raw, _MM_SHUFFLE(1, 0, 1, 0));

                __m256 out0 = _mm256_permute2f128_ps(T1, T2, 0x20);
                __m256 out1 = _mm256_permute2f128_ps(T1, T2, 0x31);

                _mm256_store_ps(output + b * 8, out0);
                _mm256_store_ps(output + (b + 1) * 8, out1);
            }
        }

        const __m128 v_inv_sse = _mm_set1_ps(0.25f);
        const __m128 v_neg_mask_sse = invert
            ? _mm_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f)
            : _mm_setr_ps(0.0f, -0.0f, 0.0f, -0.0f);

        for (; b < batch; ++b)
        {
            __m256 v = _mm256_load_ps(input + b * 8);

            __m128 low = _mm256_castps256_ps128(v);
            __m128 high = _mm256_extractf128_ps(v, 1);

            __m128 a_even = _mm_add_ps(low, high);
            __m128 a_odd = _mm_sub_ps(low, high);

            __m128 a0 = _mm_movelh_ps(a_even, a_even);
            __m128 a1 = _mm_movehl_ps(a_even, a_even);
            __m128 X0 = _mm_add_ps(a0, a1);
            __m128 X2 = _mm_sub_ps(a0, a1);

            __m128 a2 = _mm_movelh_ps(a_odd, a_odd);
            __m128 a3 = _mm_movehl_ps(a_odd, a_odd);

            __m128 a3_rot = _mm_shuffle_ps(a3, a3, _MM_SHUFFLE(2, 3, 0, 1));
            a3_rot = _mm_xor_ps(a3_rot, v_neg_mask_sse);

            __m128 X1 = _mm_add_ps(a2, a3_rot);
            __m128 X3 = _mm_sub_ps(a2, a3_rot);

            if constexpr (invert)
            {
                X0 = _mm_mul_ps(X0, v_inv_sse);
                X1 = _mm_mul_ps(X1, v_inv_sse);
                X2 = _mm_mul_ps(X2, v_inv_sse);
                X3 = _mm_mul_ps(X3, v_inv_sse);
            }

            float* out_ptr = output + b * 8;
            _mm_storel_pi(reinterpret_cast<__m64*>(out_ptr + 0), X0);
            _mm_storel_pi(reinterpret_cast<__m64*>(out_ptr + 2), X1);
            _mm_storel_pi(reinterpret_cast<__m64*>(out_ptr + 4), X2);
            _mm_storel_pi(reinterpret_cast<__m64*>(out_ptr + 6), X3);
        }
    }

    template<bool invert>
    static void radix5_4batch_kernel(const float* in, float* out)
    {
        constexpr float c72 = 0.3090169943749474241f;
        constexpr float s72 = 0.9510565162951535721f;
        constexpr float c144 = -0.8090169943749474241f;
        constexpr float s144 = 0.5877852522924731292f;

        const __m256 vc72 = _mm256_set1_ps(c72);
        const __m256 vc144 = _mm256_set1_ps(c144);
        const __m256 vs72 = _mm256_set1_ps(s72);
        const __m256 vs144 = _mm256_set1_ps(s144);
        const __m256 inv5 = _mm256_set1_ps(0.2f);

        const __m256 sign_all_mask = _mm256_set1_ps(-0.0f);

        auto load4_complex_aos_interleaved = [](const float* base,
            std::size_t stride_floats,
            std::size_t complex_index) -> __m256
        {
            const float* p0 = base + 0 * stride_floats + complex_index * 2;
            const float* p1 = base + 1 * stride_floats + complex_index * 2;
            const float* p2 = base + 2 * stride_floats + complex_index * 2;
            const float* p3 = base + 3 * stride_floats + complex_index * 2;

            const __m128 vzero = _mm_setzero_ps();
            __m128 a0 = _mm_loadl_pi(vzero, reinterpret_cast<const __m64*>(p0));
            __m128 a1 = _mm_loadl_pi(vzero, reinterpret_cast<const __m64*>(p1));
            __m128 a2 = _mm_loadl_pi(vzero, reinterpret_cast<const __m64*>(p2));
            __m128 a3 = _mm_loadl_pi(vzero, reinterpret_cast<const __m64*>(p3));

            __m128 lo = _mm_movelh_ps(a0, a1);
            __m128 hi = _mm_movelh_ps(a2, a3);
            return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
        };

        auto store4_complex_aos_interleaved = [](float* base,
            std::size_t stride_floats,
            std::size_t complex_index,
            __m256 v) -> void
        {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);

            float* p0 = base + 0 * stride_floats + complex_index * 2;
            float* p1 = base + 1 * stride_floats + complex_index * 2;
            float* p2 = base + 2 * stride_floats + complex_index * 2;
            float* p3 = base + 3 * stride_floats + complex_index * 2;

            _mm_storel_pi(reinterpret_cast<__m64*>(p0), lo);
            _mm_storeh_pi(reinterpret_cast<__m64*>(p1), lo);
            _mm_storel_pi(reinterpret_cast<__m64*>(p2), hi);
            _mm_storeh_pi(reinterpret_cast<__m64*>(p3), hi);
        };

        __m256 x0 = load4_complex_aos_interleaved(in, 10, 0);
        __m256 x1 = load4_complex_aos_interleaved(in, 10, 1);
        __m256 x2 = load4_complex_aos_interleaved(in, 10, 2);
        __m256 x3 = load4_complex_aos_interleaved(in, 10, 3);
        __m256 x4 = load4_complex_aos_interleaved(in, 10, 4);

        __m256 a = _mm256_add_ps(x1, x4);
        __m256 b = _mm256_add_ps(x2, x3);
        __m256 c = _mm256_sub_ps(x1, x4);
        __m256 d = _mm256_sub_ps(x2, x3);

        __m256 P = _mm256_fmadd_ps(a, vc72, _mm256_mul_ps(b, vc144));
        __m256 R = _mm256_fmadd_ps(a, vc144, _mm256_mul_ps(b, vc72));
        __m256 Q1 = _mm256_fmadd_ps(c, vs72, _mm256_mul_ps(d, vs144));
        __m256 Q2 = _mm256_fmsub_ps(c, vs144, _mm256_mul_ps(d, vs72));

        const __m256 mask = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
        __m256 iQ1 = _mm256_xor_ps(_mm256_permute_ps(Q1, 0xB1), mask);
        __m256 iQ2 = _mm256_xor_ps(_mm256_permute_ps(Q2, 0xB1), mask);

        if constexpr (!invert)
        {
            iQ1 = _mm256_xor_ps(iQ1, sign_all_mask);
            iQ2 = _mm256_xor_ps(iQ2, sign_all_mask);
        }

        __m256 X0 = _mm256_add_ps(x0, _mm256_add_ps(a, b));

        __m256 base1 = _mm256_add_ps(x0, P);
        __m256 X1 = _mm256_add_ps(base1, iQ1);
        __m256 X4 = _mm256_sub_ps(base1, iQ1);

        __m256 base2 = _mm256_add_ps(x0, R);
        __m256 X2 = _mm256_add_ps(base2, iQ2);
        __m256 X3 = _mm256_sub_ps(base2, iQ2);

        if constexpr (invert)
        {
            X0 = _mm256_mul_ps(X0, inv5);
            X1 = _mm256_mul_ps(X1, inv5);
            X2 = _mm256_mul_ps(X2, inv5);
            X3 = _mm256_mul_ps(X3, inv5);
            X4 = _mm256_mul_ps(X4, inv5);
        }

        store4_complex_aos_interleaved(out, 10, 0, X0);
        store4_complex_aos_interleaved(out, 10, 1, X1);
        store4_complex_aos_interleaved(out, 10, 2, X2);
        store4_complex_aos_interleaved(out, 10, 3, X3);
        store4_complex_aos_interleaved(out, 10, 4, X4);
    }

    template<bool invert>
    static void apply_radix5_batch(const float* input, float* output, std::size_t batch)
    {
        std::size_t b = 0;
        if constexpr (is_avx2_available)
        {
            for (; b + 3 < batch; b += 4)
            {
                radix5_4batch_kernel<invert>(input + b * 10, output + b * 10);
            }
        }

        for (; b < batch; ++b)
        {
            apply_radix5_single<invert>(input + b * 10, output + b * 10);
        }
    }

    static __m128 rep_reim(__m128 reim_low2)
    {
        return _mm_movelh_ps(reim_low2, reim_low2);
    }

    static __m256 pack2_complex_rep(__m128 a_reim, __m128 b_reim)
    {
        __m128 a = rep_reim(a_reim);
        __m128 b = rep_reim(b_reim);
        return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
    }

    static __m256 load2_complex_aos_rep(const float* in0, const float* in1, int n)
    {
        const float* p0 = in0 + 2 * n;
        const float* p1 = in1 + 2 * n;

        __m128 z = _mm_setzero_ps();
        __m128 a0 = _mm_loadl_pi(z, reinterpret_cast<const __m64*>(p0));
        __m128 a1 = _mm_loadl_pi(z, reinterpret_cast<const __m64*>(p1));
        return pack2_complex_rep(a0, a1);
    }

    static void store2_complex_aos_from_rep(float* out0, float* out1, int k, __m256 vrep)
    {
        __m128 lo = _mm256_castps256_ps128(vrep);
        __m128 hi = _mm256_extractf128_ps(vrep, 1);

        float* p0 = out0 + 2 * k;
        float* p1 = out1 + 2 * k;
        _mm_storel_pi(reinterpret_cast<__m64*>(p0), lo);
        _mm_storel_pi(reinterpret_cast<__m64*>(p1), hi);
    }

    static __m256 cmul_fast_rep(__m256 a, __m256 b)
    {
        __m256 ar = _mm256_permute_ps(a, 0xA0);
        __m256 ai = _mm256_permute_ps(a, 0xF5);
        __m256 b_sw = _mm256_permute_ps(b, 0xB1);

        __m256 x = _mm256_mul_ps(ar, b);
        __m256 y = _mm256_mul_ps(ai, b_sw);
        return _mm256_addsub_ps(x, y);
    }

    static void cmadd_fast_rep(__m256& acc, __m256 x, __m256 w)
    {
        acc = _mm256_add_ps(acc, cmul_fast_rep(x, w));
    }

    template<bool invert>
    static __m256 Wrep(int m)
    {
        constexpr float c[7] = {
            1.0f,
            0.62348980185873353053f,
           -0.22252093395631440429f,
           -0.90096886790241912624f,
           -0.90096886790241912624f,
           -0.22252093395631440429f,
            0.62348980185873353053f
        };
        constexpr float s[7] = {
            0.0f,
            0.78183148246802980871f,
            0.97492791218182360702f,
            0.43388373911755812048f,
           -0.43388373911755812048f,
           -0.97492791218182360702f,
           -0.78183148246802980871f
        };
        constexpr float sign = invert ? +1.0f : -1.0f;

        __m128 t = _mm_setr_ps(c[m], sign * s[m], c[m], sign * s[m]);
        return _mm256_insertf128_ps(_mm256_castps128_ps256(t), t, 1);
    }

    template<bool invert>
    static void radix7_2batch_kernel_avx2(const float* in0, const float* in1, float* out0, float* out1)
    {
        __m256 x0 = load2_complex_aos_rep(in0, in1, 0);
        __m256 x1 = load2_complex_aos_rep(in0, in1, 1);
        __m256 x2 = load2_complex_aos_rep(in0, in1, 2);
        __m256 x3 = load2_complex_aos_rep(in0, in1, 3);
        __m256 x4 = load2_complex_aos_rep(in0, in1, 4);
        __m256 x5 = load2_complex_aos_rep(in0, in1, 5);
        __m256 x6 = load2_complex_aos_rep(in0, in1, 6);

        const __m256 w0 = Wrep<invert>(0);
        const __m256 w1 = Wrep<invert>(1);
        const __m256 w2 = Wrep<invert>(2);
        const __m256 w3 = Wrep<invert>(3);
        const __m256 w4 = Wrep<invert>(4);
        const __m256 w5 = Wrep<invert>(5);
        const __m256 w6 = Wrep<invert>(6);

        auto scale_store = [&](int k, __m256 Y)
            {
                if constexpr (invert)
                {
                    const __m256 inv7 = _mm256_set1_ps(1.0f / 7.0f);
                    Y = _mm256_mul_ps(Y, inv7);
                }
                store2_complex_aos_from_rep(out0, out1, k, Y);
            };

        {
            __m256 Y = _mm256_setzero_ps();
            cmadd_fast_rep(Y, x0, w0);
            cmadd_fast_rep(Y, x1, w0);
            cmadd_fast_rep(Y, x2, w0);
            cmadd_fast_rep(Y, x3, w0);
            cmadd_fast_rep(Y, x4, w0); 
            cmadd_fast_rep(Y, x5, w0);
            cmadd_fast_rep(Y, x6, w0);
            scale_store(0, Y);
        }

        {
            __m256 Y = _mm256_setzero_ps();
            cmadd_fast_rep(Y, x0, w0); 
            cmadd_fast_rep(Y, x1, w1);
            cmadd_fast_rep(Y, x2, w2);
            cmadd_fast_rep(Y, x3, w3);
            cmadd_fast_rep(Y, x4, w4);
            cmadd_fast_rep(Y, x5, w5);
            cmadd_fast_rep(Y, x6, w6);
            scale_store(1, Y);
        }

        {
            __m256 Y = _mm256_setzero_ps();
            cmadd_fast_rep(Y, x0, w0); 
            cmadd_fast_rep(Y, x1, w2); 
            cmadd_fast_rep(Y, x2, w4);
            cmadd_fast_rep(Y, x3, w6); 
            cmadd_fast_rep(Y, x4, w1); 
            cmadd_fast_rep(Y, x5, w3);
            cmadd_fast_rep(Y, x6, w5);
            scale_store(2, Y);
        }

        {
            __m256 Y = _mm256_setzero_ps();
            cmadd_fast_rep(Y, x0, w0);
            cmadd_fast_rep(Y, x1, w3);
            cmadd_fast_rep(Y, x2, w6);
            cmadd_fast_rep(Y, x3, w2); 
            cmadd_fast_rep(Y, x4, w5); 
            cmadd_fast_rep(Y, x5, w1);
            cmadd_fast_rep(Y, x6, w4);
            scale_store(3, Y);
        }

        {
            __m256 Y = _mm256_setzero_ps();
            cmadd_fast_rep(Y, x0, w0); 
            cmadd_fast_rep(Y, x1, w4);
            cmadd_fast_rep(Y, x2, w1);
            cmadd_fast_rep(Y, x3, w5);
            cmadd_fast_rep(Y, x4, w2); 
            cmadd_fast_rep(Y, x5, w6);
            cmadd_fast_rep(Y, x6, w3);
            scale_store(4, Y);
        }

        {
            __m256 Y = _mm256_setzero_ps();
            cmadd_fast_rep(Y, x0, w0);
            cmadd_fast_rep(Y, x1, w5); 
            cmadd_fast_rep(Y, x2, w3);
            cmadd_fast_rep(Y, x3, w1); 
            cmadd_fast_rep(Y, x4, w6); 
            cmadd_fast_rep(Y, x5, w4);
            cmadd_fast_rep(Y, x6, w2);
            scale_store(5, Y);
        }

        {
            __m256 Y = _mm256_setzero_ps();
            cmadd_fast_rep(Y, x0, w0); 
            cmadd_fast_rep(Y, x1, w6);
            cmadd_fast_rep(Y, x2, w5);
            cmadd_fast_rep(Y, x3, w4); 
            cmadd_fast_rep(Y, x4, w3); 
            cmadd_fast_rep(Y, x5, w2);
            cmadd_fast_rep(Y, x6, w1);
            scale_store(6, Y);
        }
    }

    struct v2x256 { __m256 v01; __m256 v23; };

    static v2x256 load4_complex_aos_rep_4batch(const float* base, std::size_t stride_floats, int n)
    {
        const float* p0 = base + 0 * stride_floats + 2 * n;
        const float* p1 = base + 1 * stride_floats + 2 * n;
        const float* p2 = base + 2 * stride_floats + 2 * n;
        const float* p3 = base + 3 * stride_floats + 2 * n;

        __m128 z = _mm_setzero_ps();
        __m128 a0 = _mm_loadl_pi(z, reinterpret_cast<const __m64*>(p0));
        __m128 a1 = _mm_loadl_pi(z, reinterpret_cast<const __m64*>(p1));
        __m128 a2 = _mm_loadl_pi(z, reinterpret_cast<const __m64*>(p2));
        __m128 a3 = _mm_loadl_pi(z, reinterpret_cast<const __m64*>(p3));

        __m128 r0 = rep_reim(a0);
        __m128 r1 = rep_reim(a1);
        __m128 r2 = rep_reim(a2);
        __m128 r3 = rep_reim(a3);

        v2x256 out;
        out.v01 = _mm256_insertf128_ps(_mm256_castps128_ps256(r0), r1, 1);
        out.v23 = _mm256_insertf128_ps(_mm256_castps128_ps256(r2), r3, 1);
        return out;
    }

    static void store4_complex_aos_from_rep_4batch(float* base, std::size_t stride_floats, int k, v2x256 v)
    {
        __m128 lo01 = _mm256_castps256_ps128(v.v01);
        __m128 hi01 = _mm256_extractf128_ps(v.v01, 1);
        __m128 lo23 = _mm256_castps256_ps128(v.v23);
        __m128 hi23 = _mm256_extractf128_ps(v.v23, 1);

        float* p0 = base + 0 * stride_floats + 2 * k;
        float* p1 = base + 1 * stride_floats + 2 * k;
        float* p2 = base + 2 * stride_floats + 2 * k;
        float* p3 = base + 3 * stride_floats + 2 * k;

        _mm_storel_pi(reinterpret_cast<__m64*>(p0), lo01);
        _mm_storel_pi(reinterpret_cast<__m64*>(p1), hi01);
        _mm_storel_pi(reinterpret_cast<__m64*>(p2), lo23);
        _mm_storel_pi(reinterpret_cast<__m64*>(p3), hi23);
    }

    template<bool invert>
    static void radix7_4batch_kernel_avx2(const float* in, float* out)
    {
        constexpr std::size_t stride = 14;

        v2x256 x0 = load4_complex_aos_rep_4batch(in, stride, 0);
        v2x256 x1 = load4_complex_aos_rep_4batch(in, stride, 1);
        v2x256 x2 = load4_complex_aos_rep_4batch(in, stride, 2);
        v2x256 x3 = load4_complex_aos_rep_4batch(in, stride, 3);
        v2x256 x4 = load4_complex_aos_rep_4batch(in, stride, 4);
        v2x256 x5 = load4_complex_aos_rep_4batch(in, stride, 5);
        v2x256 x6 = load4_complex_aos_rep_4batch(in, stride, 6);

        const __m256 w0 = Wrep<invert>(0);
        const __m256 w1 = Wrep<invert>(1);
        const __m256 w2 = Wrep<invert>(2);
        const __m256 w3 = Wrep<invert>(3);
        const __m256 w4 = Wrep<invert>(4);
        const __m256 w5 = Wrep<invert>(5);
        const __m256 w6 = Wrep<invert>(6);

        auto cmadd2 = [&](v2x256& acc, const v2x256& x, __m256 w)
            {
                acc.v01 = _mm256_add_ps(acc.v01, cmul_fast_rep(x.v01, w));
                acc.v23 = _mm256_add_ps(acc.v23, cmul_fast_rep(x.v23, w));
            };

        auto scale_store = [&](int k, v2x256 Y)
            {
                if constexpr (invert)
                {
                    const __m256 inv7 = _mm256_set1_ps(1.0f / 7.0f);
                    Y.v01 = _mm256_mul_ps(Y.v01, inv7);
                    Y.v23 = _mm256_mul_ps(Y.v23, inv7);
                }
                store4_complex_aos_from_rep_4batch(out, stride, k, Y);
            };

        {
            v2x256 Y{ _mm256_setzero_ps(), _mm256_setzero_ps() };
            cmadd2(Y, x0, w0);
            cmadd2(Y, x1, w0);
            cmadd2(Y, x2, w0); 
            cmadd2(Y, x3, w0);
            cmadd2(Y, x4, w0); 
            cmadd2(Y, x5, w0); 
            cmadd2(Y, x6, w0);
            scale_store(0, Y);
        }

        {
            v2x256 Y{ _mm256_setzero_ps(), _mm256_setzero_ps() };
            cmadd2(Y, x0, w0); 
            cmadd2(Y, x1, w1); 
            cmadd2(Y, x2, w2); 
            cmadd2(Y, x3, w3);
            cmadd2(Y, x4, w4); 
            cmadd2(Y, x5, w5); 
            cmadd2(Y, x6, w6);
            scale_store(1, Y);
        }

        {
            v2x256 Y{ _mm256_setzero_ps(), _mm256_setzero_ps() };
            cmadd2(Y, x0, w0); 
            cmadd2(Y, x1, w2); 
            cmadd2(Y, x2, w4);
            cmadd2(Y, x3, w6);
            cmadd2(Y, x4, w1);
            cmadd2(Y, x5, w3);
            cmadd2(Y, x6, w5);
            scale_store(2, Y);
        }

        {
            v2x256 Y{ _mm256_setzero_ps(), _mm256_setzero_ps() };
            cmadd2(Y, x0, w0);
            cmadd2(Y, x1, w3); 
            cmadd2(Y, x2, w6); 
            cmadd2(Y, x3, w2);
            cmadd2(Y, x4, w5);
            cmadd2(Y, x5, w1); 
            cmadd2(Y, x6, w4);
            scale_store(3, Y);
        }

        {
            v2x256 Y{ _mm256_setzero_ps(), _mm256_setzero_ps() };
            cmadd2(Y, x0, w0); 
            cmadd2(Y, x1, w4); 
            cmadd2(Y, x2, w1); 
            cmadd2(Y, x3, w5);
            cmadd2(Y, x4, w2); 
            cmadd2(Y, x5, w6);
            cmadd2(Y, x6, w3);
            scale_store(4, Y);
        }

        {
            v2x256 Y{ _mm256_setzero_ps(), _mm256_setzero_ps() };
            cmadd2(Y, x0, w0); 
            cmadd2(Y, x1, w5); 
            cmadd2(Y, x2, w3); 
            cmadd2(Y, x3, w1);
            cmadd2(Y, x4, w6); 
            cmadd2(Y, x5, w4); 
            cmadd2(Y, x6, w2);
            scale_store(5, Y);
        }

        {
            v2x256 Y{ _mm256_setzero_ps(), _mm256_setzero_ps() };
            cmadd2(Y, x0, w0);
            cmadd2(Y, x1, w6); 
            cmadd2(Y, x2, w5); 
            cmadd2(Y, x3, w4);
            cmadd2(Y, x4, w3); 
            cmadd2(Y, x5, w2);
            cmadd2(Y, x6, w1);
            scale_store(6, Y);
        }
    }

    template<bool invert>
    [[msvc::forceinline]] static void apply_radix7_batch(const float* input, float* output, std::size_t batch)
    {
        std::size_t b = 0;

        if constexpr (is_avx2_available)
        {
            for (; b + 3 < batch; b += 4)
            {
                radix7_4batch_kernel_avx2<invert>(input + b * 14, output + b * 14);
            }
            for (; b + 1 < batch; b += 2)
            {
                const float* in0 = input + (b + 0) * 14;
                const float* in1 = input + (b + 1) * 14;
                float* out0 = output + (b + 0) * 14;
                float* out1 = output + (b + 1) * 14;
                radix7_2batch_kernel_avx2<invert>(in0, in1, out0, out1);
            }
        }

        for (; b < batch; ++b)
        {
            apply_radix7_single<invert>(input + b * 14, output + b * 14);
        }
    }

    template<bool invert>
    [[msvc::forceinline]] bool especially_batch_kernel_dispatch(std::size_t radix, const float* input, float* output, std::size_t batch)
    {
        const bool probe_only = (input == nullptr || output == nullptr);
        switch (radix)
        {
            case 2: { if (!probe_only) { apply_radix2_batch<invert>(input, output, batch); } return true; }
            case 3: { if (!probe_only) { apply_radix3_batch<invert>(input, output, batch); } return true; }
            case 4: { if (!probe_only) { apply_radix4_batch<invert>(input, output, batch); } return true; }
            case 5: { if (!probe_only) { apply_radix5_batch<invert>(input, output, batch); } return true; }
            case 7: { if (!probe_only) { apply_radix7_batch<invert>(input, output, batch); } return true; }
            default:{ return false; }
        }
    }
}

#endif