#ifndef _FOYE_FFT_DFT_HPP_
#define _FOYE_FFT_DFT_HPP_

namespace fy::fft::dft
{
    [[msvc::forceinline]] static std::complex<float> dft_inner_product(
        const std::complex<float>* in,
        const std::complex<float>* twiddle_row,
        std::size_t N)
    {
        std::complex<float> sum = { 0.0f, 0.0f };
        std::size_t i = 0;

        if constexpr (is_avx2_available)
        {
            __m256 v_sum = _mm256_setzero_ps();

            for (; i + 3 < N; i += 4)
            {
                __m256 v_in = _mm256_load_ps(reinterpret_cast<const float*>(&in[i]));
                __m256 v_tw = _mm256_load_ps(reinterpret_cast<const float*>(&twiddle_row[i]));

                __m256 v_tw_swap = _mm256_permute_ps(v_tw, 0xB1);
                v_sum = fma_complexf32_accumulate(v_sum, v_in, v_tw, v_tw_swap);
            }

            __m256 v_high = _mm256_permute2f128_ps(v_sum, v_sum, 0x01);
            __m256 v_reduced = _mm256_add_ps(v_sum, v_high);

            __m128 v_low = _mm256_castps256_ps128(v_reduced);
            __m128 v_swapped = _mm_shuffle_ps(v_low, v_low, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 v_high_128 = _mm_movehl_ps(v_low, v_low);
            v_low = _mm_add_ps(v_low, v_high_128);

            float real_sum = _mm_cvtss_f32(v_low);
            float imag_sum = _mm_cvtss_f32(_mm_shuffle_ps(v_low, v_low, 0x01));

            sum = std::complex<float>(real_sum, imag_sum);
        }

        for (; i < N; ++i)
        {
            std::complex<float> a = in[i];
            std::complex<float> b = twiddle_row[i];
            float re = a.real() * b.real() - a.imag() * b.imag();
            float im = a.real() * b.imag() + a.imag() * b.real();
            sum += std::complex<float>(re, im);
        }

        return sum;
    }

    namespace dft_manager
    {
        struct plan
        {
            std::size_t N;
            float inv_N_val;

            fy::fft::aligned_vector_type<std::complex<float>> twiddles_fwd;
            fy::fft::aligned_vector_type<std::complex<float>> twiddles_bwd;

            plan(std::size_t length) : N(length)
            {
                inv_N_val = 1.0f / static_cast<float>(N);
                std::size_t total_size = N * N;

                twiddles_fwd.resize(total_size);
                twiddles_bwd.resize(total_size);

                const double angle_base = -2.0 * std::numbers::pi_v<double> / static_cast<double>(N);

                for (std::size_t k = 0; k < N; ++k)
                {
                    for (std::size_t n = 0; n < N; ++n)
                    {
                        double angle = angle_base * static_cast<double>(k * n);

                        float c = static_cast<float>(std::cos(angle));
                        float s = static_cast<float>(std::sin(angle));

                        std::size_t idx = k * N + n;

                        twiddles_fwd[idx] = { c, s };
                        twiddles_bwd[idx] = { c, -s };
                    }
                }
            }

            [[msvc::forceinline]] const std::complex<float>* acquired_fwd_row(std::size_t k) const
            {
                return twiddles_fwd.data() + k * N;
            }

            [[msvc::forceinline]] const std::complex<float>* acquired_bwd_row(std::size_t k) const
            {
                return twiddles_bwd.data() + k * N;
            }
        };
    }

    struct FFT1D_C2C_dft_invoker : basic_unsmooth_path_invoker
    {
        dft_manager::plan plan;

        FFT1D_C2C_dft_invoker(std::size_t length) : plan(length) { }

        void forward(const float* input, float* output) override
        {
            dispatch<false>(input, output);
        }

        void backward(const float* input, float* output) override
        {
            dispatch<true>(input, output);
        }

        template<bool invert>
        void dispatch(const float* input, float* output)
        {
            const std::complex<float>* in_c = reinterpret_cast<const std::complex<float>*>(input);
            std::complex<float>* out_c = reinterpret_cast<std::complex<float>*>(output);

            std::size_t N = plan.N;

            for (std::size_t k = 0; k < N; ++k)
            {
                const std::complex<float>* row_ptr = invert 
                    ? plan.acquired_bwd_row(k) 
                    : plan.acquired_fwd_row(k);

                std::complex<float> val = dft_inner_product(in_c, row_ptr, N);
                if constexpr (invert)
                {
                    out_c[k] = val * plan.inv_N_val;
                }
                else
                {
                    out_c[k] = val;
                }
            }
        }

        static bool is_supported(std::size_t)
        {
            return true;
        }
    };
}

#endif