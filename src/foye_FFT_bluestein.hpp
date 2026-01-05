#ifndef _FOYE_FFT_BLUESTEIN_HPP_
#define _FOYE_FFT_BLUESTEIN_HPP_

namespace fy::fft::bluestein
{
    template<bool conjugate>
    std::complex<float> calculate_chirp_scalar(std::size_t n, std::size_t N)
    {
        const double PI = std::numbers::pi_v<double>;
        const std::uint64_t two_N = 2 * N;
        std::uint64_t sq = static_cast<std::uint64_t>(n) * n;
        double angle = PI * static_cast<double>(sq % two_N) / static_cast<double>(N);
        if constexpr (conjugate)
        {
            angle = -angle;
        }
        return { static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle)) };
    }

    [[msvc::forceinline]] void compute_chirp_avx2(std::size_t n_start, std::size_t N,
        __m256& v_sin, __m256& v_cos)
    {
        alignas(32) float angles[8];
        const std::uint64_t two_N = 2 * N;
        const double inv_N = 1.0 / static_cast<double>(N);
        const double PI = std::numbers::pi_v<double>;

        std::uint64_t sq = (static_cast<std::uint64_t>(n_start) * n_start) % two_N;
        std::uint64_t step = (2 * n_start + 1) % two_N;

        for (std::size_t i = 0; i < 8; ++i)
        {
            angles[i] = static_cast<float>(PI * static_cast<double>(sq) * inv_N);

            sq += step;
            if (sq >= two_N)
            {
                sq -= two_N;
            }

            step += 2;
            if (step >= two_N)
            {
                step -= two_N;
            }
        }

        __m256 v_angle = _mm256_load_ps(angles);
        sincos_avx2(v_angle, &v_sin, &v_cos);
    }

    [[msvc::forceinline]] static void mul_conj_chirp_avx2_optimized(float* io_ptr, __m256 v_chirp)
    {
        __m256 v_in = _mm256_load_ps(io_ptr);
        __m256 v_u = _mm256_moveldup_ps(v_chirp);
        __m256 v_v = _mm256_movehdup_ps(v_chirp);
        __m256 v_in_swap = _mm256_permute_ps(v_in, 0xB1);

        __m256 term2 = _mm256_mul_ps(v_in_swap, v_v);
        __m256 res = _mm256_fmsubadd_ps(v_in, v_u, term2);
        _mm256_store_ps(io_ptr, res);
    }

    [[msvc::forceinline]] static void mul_conj_input_conj_chirp_avx2(float* io_ptr, __m256 v_chirp)
    {
        __m256 v_in = _mm256_load_ps(io_ptr);
        __m256 v_u = _mm256_moveldup_ps(v_chirp);
        __m256 v_v = _mm256_movehdup_ps(v_chirp);
        __m256 v_in_swap = _mm256_permute_ps(v_in, 0xB1);

        __m256 t1 = _mm256_mul_ps(v_in, v_u);
        __m256 t2 = _mm256_mul_ps(v_in_swap, v_v);
        __m256 prod = _mm256_addsub_ps(t1, t2);

        static const __m256 conj_mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
        _mm256_store_ps(io_ptr, _mm256_xor_ps(prod, conj_mask));
    }

    struct plan
    {
        std::size_t N;
        std::size_t M;

        smooth_path_invoker::execute_function_type forward_ptr;
        smooth_path_invoker::execute_function_type backward_ptr;

        fy::fft::aligned_vector_type<unsigned char> data_buffer;

        std::size_t offset_chirp_table;
        std::size_t offset_kernel_fft;
        std::size_t offset_workspace;

        plan(std::size_t length) : N(length)
        {
            M = find_bluestein_M(2 * N - 1);

            forward_ptr = smooth_path_invoker::kernel<false>(M);
            backward_ptr = smooth_path_invoker::kernel<true>(M);

            std::size_t current_offset = 0;

            offset_chirp_table = align_offset(current_offset, 32);
            current_offset = offset_chirp_table + N * sizeof(std::complex<float>);

            offset_kernel_fft = align_offset(current_offset, 32);
            current_offset = offset_kernel_fft + M * sizeof(std::complex<float>);

            offset_workspace = align_offset(current_offset, 64);
            current_offset = offset_workspace + M * sizeof(std::complex<float>);

            data_buffer.resize(current_offset);

            std::complex<float>* ptr_chirp = reinterpret_cast<std::complex<float>*>(data_buffer.data() + offset_chirp_table);
            std::complex<float>* ptr_kernel = reinterpret_cast<std::complex<float>*>(data_buffer.data() + offset_kernel_fft);
            std::complex<float>* ptr_work = reinterpret_cast<std::complex<float>*>(data_buffer.data() + offset_workspace);

            {
                std::size_t n = 0;
                if constexpr (is_avx2_available)
                {
                    for (; n + 8 <= N; n += 8)
                    {
                        __m256 v_sin, v_cos;
                        compute_chirp_avx2(n, N, v_sin, v_cos);

                        __m256 v_lo = _mm256_unpacklo_ps(v_cos, v_sin);
                        __m256 v_hi = _mm256_unpackhi_ps(v_cos, v_sin);

                        _mm256_store_ps(reinterpret_cast<float*>(ptr_chirp + n), _mm256_permute2f128_ps(v_lo, v_hi, 0x20));
                        _mm256_store_ps(reinterpret_cast<float*>(ptr_chirp + n + 4), _mm256_permute2f128_ps(v_lo, v_hi, 0x31));
                    }
                }
                for (; n < N; ++n)
                {
                    ptr_chirp[n] = calculate_chirp_scalar<false>(n, N);
                }
            }

            {
                std::memcpy(ptr_work, ptr_chirp, N * sizeof(std::complex<float>));
                if (M > N)
                {
                    std::memset(ptr_work + N, 0, (M - N) * sizeof(std::complex<float>));
                }
                for (std::size_t i = 1; i < N; ++i)
                {
                    ptr_work[M - i] = ptr_chirp[i];
                }

                forward_ptr(reinterpret_cast<float*>(ptr_work), reinterpret_cast<float*>(ptr_work));

                std::memcpy(ptr_kernel, ptr_work, M * sizeof(std::complex<float>));
            }

            std::memset(ptr_work, 0, M * sizeof(std::complex<float>));
        }

        [[msvc::forceinline]] const std::complex<float>* acquired_chirp_table() const
        {
            return reinterpret_cast<const std::complex<float>*>(data_buffer.data() + offset_chirp_table);
        }

        [[msvc::forceinline]] const std::complex<float>* acquired_kernel() const
        {
            return reinterpret_cast<const std::complex<float>*>(data_buffer.data() + offset_kernel_fft);
        }

        [[msvc::forceinline]] std::complex<float>* acquired_workspace()
        {
            return reinterpret_cast<std::complex<float>*>(data_buffer.data() + offset_workspace);
        }

        static std::size_t find_bluestein_M(std::size_t target)
        {
            for (std::size_t m : smooth_path_invoker::available_smooth_length)
            {
                if (m >= target)
                {
                    return m;
                }
            }
            return 0;
        }
    };

    struct FFT1D_bluestein_invoker : basic_unsmooth_path_invoker
    {
        static constexpr std::size_t min_length = 2;
        static constexpr std::size_t max_length = 39062;

        plan _plan;

        static const std::vector<std::size_t>& all_processable_length()
        {
            static const std::vector<std::size_t> lengths = []() {
                std::vector<std::size_t> res;
                std::size_t limit = smooth_path_invoker::available_smooth_length.back();
                for (std::size_t n = 2; n <= limit; ++n)
                {
                    if (!smooth_path_invoker::is_directly_supported(n) && is_processable(n))
                    {
                        res.push_back(n);
                    }
                }

                return res;
                }();

            return lengths;
        }

        static bool is_processable(std::size_t n)
        {
            return n <= max_length;
        }

        FFT1D_bluestein_invoker(std::size_t length) : _plan(length)
        {
            FOYE_FFT_ASSERT_ASSUME((length <= max_length));
            FOYE_FFT_ASSERT_ASSUME(FFT1D_bluestein_invoker::is_processable(length));
        }

        ~FFT1D_bluestein_invoker() override {}

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
            if (!smooth_path_invoker::is_directly_supported(_plan.N))
            {
                std::complex<float>* workspace = _plan.acquired_workspace();
                const std::complex<float>* kernel = _plan.acquired_kernel();
                const std::complex<float>* chirp = _plan.acquired_chirp_table();

                pre_process_input<invert>(
                    reinterpret_cast<const std::complex<float>*>(input),
                    workspace,
                    chirp);

                float* workspace_f = reinterpret_cast<float*>(workspace);
                _plan.forward_ptr(workspace_f, workspace_f);

                frequency_domain_convolution_complexf32<invert>(
                    reinterpret_cast<float*>(workspace),
                    reinterpret_cast<const float*>(kernel),
                    _plan.M,
                    invert ? (1.0f / static_cast<float>(_plan.N)) : 1.0f);

                _plan.backward_ptr(workspace_f, workspace_f);

                post_process_output<invert>(
                    reinterpret_cast<const std::complex<float>*>(workspace),
                    reinterpret_cast<std::complex<float>*>(output),
                    chirp);
            }
        }

    protected:
        template<bool invert>
        void pre_process_input(const std::complex<float>* input, 
            std::complex<float>* bufA, const std::complex<float>* table_ptr)
        {
            std::size_t n = 0;
            float* bufA_ptr = reinterpret_cast<float*>(bufA);
            const float* table_f_ptr = reinterpret_cast<const float*>(table_ptr);

            if constexpr (is_avx2_available)
            {
                for (; n + 4 <= _plan.N; n += 4)
                {
                    __m256 v_chirp = _mm256_load_ps(table_f_ptr + 2 * n);
                    _mm256_store_ps(bufA_ptr + 2 * n,
                        _mm256_load_ps(reinterpret_cast<const float*>(input + n)));

                    if constexpr (!invert)
                    {
                        mul_conj_chirp_avx2_optimized(bufA_ptr + 2 * n, v_chirp);
                    }
                    else
                    {
                        mul_conj_input_conj_chirp_avx2(bufA_ptr + 2 * n, v_chirp);
                    }
                }
            }

            for (; n < _plan.N; ++n)
            {
                std::complex<float> val = input[n];
                std::complex<float> chirp = table_ptr[n];
                if constexpr (!invert)
                {
                    bufA[n] = val * std::conj(chirp);
                }
                else
                {
                    bufA[n] = std::conj(val) * std::conj(chirp);
                }
            }

            if (_plan.M > _plan.N)
            {
                std::memset(bufA + _plan.N, 0, (_plan.M - _plan.N) * sizeof(std::complex<float>));
            }
        }

        template<bool invert>
        void post_process_output(const std::complex<float>* bufA, 
            std::complex<float>* output, const std::complex<float>* table_ptr)
        {
            std::size_t k = 0;
            float* out_ptr = reinterpret_cast<float*>(output);
            const float* buf_ptr = reinterpret_cast<const float*>(bufA);
            const float* table_f_ptr = reinterpret_cast<const float*>(table_ptr);

            const __m256 conj_mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);

            if constexpr (is_avx2_available)
            {
                for (; k + 4 <= _plan.N; k += 4)
                {
                    __m256 v_chirp = _mm256_loadu_ps(table_f_ptr + 2 * k);
                    _mm256_store_ps(out_ptr + 2 * k, _mm256_load_ps(buf_ptr + 2 * k));

                    mul_conj_chirp_avx2_optimized(out_ptr + 2 * k, v_chirp);
                    if constexpr (invert)
                    {
                        __m256 res = _mm256_loadu_ps(out_ptr + 2 * k);
                        res = _mm256_xor_ps(res, conj_mask);
                        _mm256_store_ps(out_ptr + 2 * k, res);
                    }
                }
            }
            for (; k < _plan.N; ++k)
            {
                std::complex<float> val = bufA[k] * std::conj(table_ptr[k]);
                if constexpr (invert)
                {
                    output[k] = std::conj(val);
                }
                else
                {
                    output[k] = val;
                }
            }
        }
    };
}

#endif
