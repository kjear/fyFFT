#ifndef _FOYE_FFT_RADER_HPP_
#define _FOYE_FFT_RADER_HPP_

namespace fy::fft::rader
{
    [[msvc::forceinline]] static void prepare_time_domain_convolution_kernel_from_angle(
        std::size_t L,
        const float* angle_lut,
        std::complex<float>* kernel_time,
        std::complex<float>* kernel_time_inv)
    {
        std::size_t i = 0;

        if constexpr (is_avx2_available)
        {
            const __m256 v_neg_zero = _mm256_set1_ps(-0.0f);

            for (; i + 7 < L; i += 8)
            {
                __m256 v_angle = _mm256_load_ps(angle_lut + i);

                __m256 v_sin, v_cos;
                fy::fft::sincos_avx2(v_angle, &v_sin, &v_cos);

                __m256 v_lo = _mm256_unpacklo_ps(v_cos, v_sin);
                __m256 v_hi = _mm256_unpackhi_ps(v_cos, v_sin);

                __m256 v_res0 = _mm256_permute2f128_ps(v_lo, v_hi, 0x20);
                __m256 v_res1 = _mm256_permute2f128_ps(v_lo, v_hi, 0x31);

                _mm256_store_ps(reinterpret_cast<float*>(&kernel_time[i]), v_res0);
                _mm256_store_ps(reinterpret_cast<float*>(&kernel_time[i + 4]), v_res1);

                __m256 v_sin_neg = _mm256_xor_ps(v_sin, v_neg_zero);

                v_lo = _mm256_unpacklo_ps(v_cos, v_sin_neg);
                v_hi = _mm256_unpackhi_ps(v_cos, v_sin_neg);

                v_res0 = _mm256_permute2f128_ps(v_lo, v_hi, 0x20);
                v_res1 = _mm256_permute2f128_ps(v_lo, v_hi, 0x31);
                _mm256_store_ps(reinterpret_cast<float*>(&kernel_time_inv[i]), v_res0);
                _mm256_store_ps(reinterpret_cast<float*>(&kernel_time_inv[i + 4]), v_res1);
            }
        }

        for (; i < L; ++i)
        {
            const float a = angle_lut[i];
            const float c = std::cos(a);
            const float s = std::sin(a);

            kernel_time[i] = { c,  s };
            kernel_time_inv[i] = { c, -s };
        }
    }

    struct scatter_item
    {
        std::uint32_t out;
        std::uint32_t src;
    };

    struct gather_item
    {
        std::uint32_t in;
        std::uint32_t dst;
    };

    [[msvc::forceinline]] static void rader_output_scatter_bucketed(
        const std::complex<float>* tmp_out,
        std::complex<float>* out_c,
        const scatter_item* items,
        std::size_t L)
    {
        for (std::size_t k = 0; k < L; ++k)
        {
            const auto out_idx = static_cast<std::size_t>(items[k].out);
            const auto src_idx = static_cast<std::size_t>(items[k].src);
            out_c[out_idx] = tmp_out[src_idx];
        }
    }

    [[msvc::forceinline]] static std::complex<float> rader_input_gather_bucketed(
        const std::complex<float>* in_c,
        std::complex<float>* work_c,
        const gather_item* items,
        std::size_t L)
    {
        std::complex<float> sum_x = { 0.0f, 0.0f };

        for (std::size_t k = 0; k < L; ++k)
        {
            const std::size_t in_idx = static_cast<std::size_t>(items[k].in);
            const std::size_t dst_idx = static_cast<std::size_t>(items[k].dst);

            const std::complex<float> v = in_c[in_idx];
            work_c[dst_idx] = v;
            sum_x += v;
        }

        return sum_x;
    }

    [[msvc::forceinline]] static void rader_output_compute_tmp(
        const std::complex<float>* work_c,
        std::complex<float> x0_term,
        std::complex<float>* tmp_out,
        std::size_t L,
        float output_scale)
    {
        const std::complex<float> x0_scaled = x0_term * output_scale;

        std::size_t i = 0;
        if constexpr (is_avx2_available)
        {
            const __m256 v_scale = _mm256_set1_ps(output_scale);
            const __m256 v_x0_scaled = _mm256_setr_ps(
                x0_scaled.real(), x0_scaled.imag(),
                x0_scaled.real(), x0_scaled.imag(),
                x0_scaled.real(), x0_scaled.imag(),
                x0_scaled.real(), x0_scaled.imag());

            for (; i + 3 < L; i += 4)
            {
                __m256 v_w1 = _mm256_load_ps(reinterpret_cast<const float*>(&work_c[i]));
                __m256 v_w2 = _mm256_loadu_ps(reinterpret_cast<const float*>(&work_c[i + L]));
                __m256 v_sum = _mm256_add_ps(v_w1, v_w2);
                __m256 v_res = _mm256_fmadd_ps(v_sum, v_scale, v_x0_scaled);
                _mm256_store_ps(reinterpret_cast<float*>(&tmp_out[i]), v_res);
            }
        }

        for (; i < L; ++i)
        {
            std::complex<float> conv_val = work_c[i] + work_c[i + L];
            float r = std::fma(conv_val.real(), output_scale, x0_scaled.real());
            float im = std::fma(conv_val.imag(), output_scale, x0_scaled.imag());
            tmp_out[i] = { r, im };
        }
    }

    struct plan
    {
        std::size_t N;
        std::size_t L;
        std::size_t M;

        float inv_N_val;

        smooth_path_invoker::execute_function_type forward_ptr;
        smooth_path_invoker::execute_function_type backward_ptr;

        fy::fft::aligned_vector_type<unsigned char> data_buffer;

        std::size_t offset_lut_in;
        std::size_t offset_lut_out;
        std::size_t offset_k_fwd;
        std::size_t offset_k_bwd;
        std::size_t offset_workspace;
        std::size_t offset_angle_lut;
        std::size_t offset_g_pow;
        std::size_t offset_out_tmp;
        std::size_t offset_gather_items;
        std::size_t offset_scatter_items;

        plan(std::size_t length) : N(length), L(length - 1), M(smooth_path_invoker::next_smooth(2 * (length - 1) - 1))
        {
            FOYE_FFT_ASSERT_ASSUME(M != 0);

            inv_N_val = 1.0f / static_cast<float>(N);
            forward_ptr = smooth_path_invoker::kernel<false>(M);
            backward_ptr = smooth_path_invoker::kernel<true>(M);

            FOYE_FFT_ASSERT_ASSUME(forward_ptr != nullptr);
            FOYE_FFT_ASSERT_ASSUME(backward_ptr != nullptr);

            std::size_t current_offset = 0;
            offset_lut_in = align_offset(current_offset, 32);
            current_offset = offset_lut_in + L * sizeof(std::int32_t);

            offset_lut_out = align_offset(current_offset, 32);
            current_offset = offset_lut_out + L * sizeof(std::int32_t);

            offset_k_fwd = align_offset(current_offset, 32);
            current_offset = offset_k_fwd + M * sizeof(std::complex<float>);

            offset_k_bwd = align_offset(current_offset, 32);
            current_offset = offset_k_bwd + M * sizeof(std::complex<float>);

            offset_g_pow = align_offset(current_offset, alignof(std::size_t));
            current_offset = offset_g_pow + L * sizeof(std::size_t);

            offset_angle_lut = align_offset(current_offset, 32);
            current_offset = offset_angle_lut + L * sizeof(float);

            offset_out_tmp = align_offset(current_offset, 64);
            current_offset = offset_out_tmp + L * sizeof(std::complex<float>);

            offset_gather_items = align_offset(current_offset, 64);
            current_offset = offset_gather_items + L * sizeof(gather_item);

            offset_scatter_items = align_offset(current_offset, 64);
            current_offset = offset_scatter_items + L * sizeof(scatter_item);

            offset_workspace = align_offset(current_offset, 64);
            const std::size_t total_size = offset_workspace + M * sizeof(std::complex<float>);

            data_buffer.resize(total_size);

            std::int32_t* ptr_lut_in = reinterpret_cast<std::int32_t*>(data_buffer.data() + offset_lut_in);
            std::int32_t* ptr_lut_out = reinterpret_cast<std::int32_t*>(data_buffer.data() + offset_lut_out);

            std::complex<float>* pk_fwd = reinterpret_cast<std::complex<float>*>(data_buffer.data() + offset_k_fwd);
            std::complex<float>* pk_bwd = reinterpret_cast<std::complex<float>*>(data_buffer.data() + offset_k_bwd);

            std::size_t* ptr_g_pow = reinterpret_cast<std::size_t*>(data_buffer.data() + offset_g_pow);
            float* angle_lut = reinterpret_cast<float*>(data_buffer.data() + offset_angle_lut);

            const std::size_t g = find_primitive_root(N);
            FOYE_FFT_ASSERT_ASSUME(g != 0);

            std::size_t cur = 1;
            std::size_t cur_inv = 1;
            const std::size_t g_inv = power_mod(g, N - 2, N);

            for (std::size_t i = 0; i < L; ++i)
            {
                ptr_g_pow[i] = cur;
                ptr_lut_in[i] = static_cast<std::int32_t>(cur_inv);
                ptr_lut_out[i] = static_cast<std::int32_t>(cur);

                cur = (cur * g) % N;
                cur_inv = (cur_inv * g_inv) % N;
            }

            const float angle_scale = static_cast<float>(-2.0 * std::numbers::pi_v<double> / double(N));
            for (std::size_t i = 0; i < L; ++i)
            {
                angle_lut[i] = angle_scale * static_cast<float>(ptr_g_pow[i]);
            }

            auto* g_items = reinterpret_cast<gather_item*>(data_buffer.data() + offset_gather_items);
            for (std::size_t i = 0; i < L; ++i)
            {
                g_items[i].in = static_cast<std::uint32_t>(ptr_lut_in[i]);
                g_items[i].dst = static_cast<std::uint32_t>(i);
            }

            std::sort(g_items, g_items + L,
                [](const gather_item& a, const gather_item& b)
                {
                    const std::uint32_t cla = a.in >> 3;
                    const std::uint32_t clb = b.in >> 3;
                    if (cla != clb)
                    {
                        return cla < clb;
                    }

                    return a.in < b.in;
                });

            scatter_item* s_items = reinterpret_cast<scatter_item*>(data_buffer.data() + offset_scatter_items);
            for (std::size_t i = 0; i < L; ++i)
            {
                s_items[i].out = static_cast<std::uint32_t>(ptr_lut_out[i]);
                s_items[i].src = static_cast<std::uint32_t>(i);
            }

            std::sort(s_items, s_items + L,
                [](const scatter_item& a, const scatter_item& b)
                {
                    const std::uint32_t cla = a.out >> 3;
                    const std::uint32_t clb = b.out >> 3;
                    if (cla != clb)
                    {
                        return cla < clb;
                    }

                    return a.out < b.out;
                });

            prepare_time_domain_convolution_kernel_from_angle(
                L, angle_lut, pk_fwd, pk_bwd);

            forward_ptr(reinterpret_cast<float*>(pk_fwd), reinterpret_cast<float*>(pk_fwd));
            forward_ptr(reinterpret_cast<float*>(pk_bwd), reinterpret_cast<float*>(pk_bwd));
        }

        [[msvc::forceinline]] const gather_item* acquired_gather_items() const
        {
            return reinterpret_cast<const gather_item*>(data_buffer.data() + offset_gather_items);
        }

        [[msvc::forceinline]] const scatter_item* acquired_scatter_items() const
        {
            return reinterpret_cast<const scatter_item*>(data_buffer.data() + offset_scatter_items);
        }

        [[msvc::forceinline]] std::complex<float>* acquired_out_tmp()
        {
            return reinterpret_cast<std::complex<float>*>(data_buffer.data() + offset_out_tmp);
        }

        [[msvc::forceinline]] const std::size_t* acquired_g_pow() const
        {
            return reinterpret_cast<const std::size_t*>(data_buffer.data() + offset_g_pow);
        }

        [[msvc::forceinline]] const std::int32_t* acquired_lut_in() const
        {
            return reinterpret_cast<const std::int32_t*>(data_buffer.data() + offset_lut_in);
        }

        [[msvc::forceinline]] const std::int32_t* acquired_lut_out() const
        {
            return reinterpret_cast<const std::int32_t*>(data_buffer.data() + offset_lut_out);
        }

        [[msvc::forceinline]] const std::complex<float>* acquired_forward_kernel() const
        {
            return reinterpret_cast<const std::complex<float>*>(data_buffer.data() + offset_k_fwd);
        }

        [[msvc::forceinline]] const std::complex<float>* acquired_backward_kernel() const
        {
            return reinterpret_cast<const std::complex<float>*>(data_buffer.data() + offset_k_bwd);
        }

        [[msvc::forceinline]] std::complex<float>* acquired_workspace()
        {
            return reinterpret_cast<std::complex<float>*>(data_buffer.data() + offset_workspace);
        }
    };

    struct FFT1D_C2C_rader_invoker : basic_unsmooth_path_invoker
    {
        plan plan_;

        FFT1D_C2C_rader_invoker(std::size_t length) : plan_(length) { }

        static bool is_processable(std::size_t length)
        {
            static const std::vector<std::size_t>& processable_lengths = all_processable_length();
            return std::binary_search(processable_lengths.begin(), processable_lengths.end(), length);
        }

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

            std::complex<float>* work_c = plan_.acquired_workspace();

            const std::complex<float> x0_term = in_c[0];

            std::complex<float> sum_x = rader_input_gather_bucketed(
                in_c,
                work_c,
                plan_.acquired_gather_items(),
                plan_.L);

            std::memset(work_c + plan_.L, 0, (plan_.M - plan_.L) * sizeof(std::complex<float>));

            sum_x += x0_term;

            float* f_work = reinterpret_cast<float*>(work_c);

            plan_.forward_ptr(f_work, f_work);

            const float* kernel_ptr = reinterpret_cast<const float*>(
                invert 
                ? plan_.acquired_backward_kernel() 
                : plan_.acquired_forward_kernel());

            fy::fft::frequency_domain_convolution_complexf32<false>(f_work, kernel_ptr, plan_.M);

            plan_.backward_ptr(f_work, f_work);

            const float scale = invert ? plan_.inv_N_val : 1.0f;
            out_c[0] = sum_x * scale;

            std::complex<float>* tmp_out = plan_.acquired_out_tmp();
            rader_output_compute_tmp(work_c, x0_term, tmp_out, plan_.L, scale);

            rader_output_scatter_bucketed(
                tmp_out,
                out_c,
                plan_.acquired_scatter_items(),
                plan_.L);
        }

        static const std::vector<std::size_t>& all_processable_length()
        {
            static const std::vector<std::size_t> lengths = []() {
                std::vector<std::size_t> res;
                std::size_t limit = smooth_path_invoker::available_smooth_length.back();
                for (std::size_t n = 2; n <= limit; ++n)
                {
                    if (!smooth_path_invoker::is_directly_supported(n) && is_prime(n))
                    {
                        if (smooth_path_invoker::next_smooth(2 * n - 3) != 0)
                        {
                            res.push_back(n);
                        }
                    }
                }

                return res;
                }();

            return lengths;
        }
    };

}


#endif
