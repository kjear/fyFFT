#ifndef _FOYE_FFT_MIXRADIX_HPP_
#define _FOYE_FFT_MIXRADIX_HPP_

namespace fy::fft::mix_radix
{
    template<bool invert>
    [[msvc::forceinline]] static __m256 complex_mul_avx2(__m256 data, __m256 twiddle)
    {
        __m256 tr = _mm256_permute_ps(twiddle, _MM_SHUFFLE(2, 2, 0, 0));
        __m256 ti = _mm256_permute_ps(twiddle, _MM_SHUFFLE(3, 3, 1, 1));

        if constexpr (invert)
        {
            ti = _mm256_xor_ps(ti, _mm256_set1_ps(-0.0f));
        }

        __m256 data_swapped = _mm256_permute_ps(data, _MM_SHUFFLE(2, 3, 0, 1));

        __m256 mul1 = _mm256_mul_ps(data, tr);
        __m256 mul2 = _mm256_mul_ps(data_swapped, ti);
        return _mm256_addsub_ps(mul1, mul2);
    }

    [[msvc::forceinline]] __m256 load_twiddle_vec(
        std::size_t index,
        std::size_t r_index,
        std::size_t stride,
        std::size_t tw_mod,
        const float* twiddle_table)
    {
        const std::uint64_t row_s = static_cast<std::uint64_t>(r_index) * static_cast<std::uint64_t>(stride);
        const std::uint64_t mod = static_cast<std::uint64_t>(tw_mod);

        const std::uint64_t k0 = (static_cast<std::uint64_t>(index + 0) * row_s) % mod;
        const std::uint64_t k1 = (static_cast<std::uint64_t>(index + 1) * row_s) % mod;
        const std::uint64_t k2 = (static_cast<std::uint64_t>(index + 2) * row_s) % mod;
        const std::uint64_t k3 = (static_cast<std::uint64_t>(index + 3) * row_s) % mod;

        const std::size_t idx0 = static_cast<std::size_t>(k0) * 2;
        const std::size_t idx1 = static_cast<std::size_t>(k1) * 2;
        const std::size_t idx2 = static_cast<std::size_t>(k2) * 2;
        const std::size_t idx3 = static_cast<std::size_t>(k3) * 2;

        return _mm256_setr_ps(
            twiddle_table[idx0], twiddle_table[idx0 + 1],
            twiddle_table[idx1], twiddle_table[idx1 + 1],
            twiddle_table[idx2], twiddle_table[idx2 + 1],
            twiddle_table[idx3], twiddle_table[idx3 + 1]
        );
    }

    template<bool invert>
    [[msvc::forceinline]] static void fused_transpose_apply_twiddles(
        const float* src, float* dst,
        std::size_t rows, std::size_t cols,
        const float* twiddle_table,
        std::size_t stride,
        std::size_t tw_mod)
    {
        std::size_t r_limit = 0;
        std::size_t c_limit = 0;

        if constexpr (is_avx2_available)
        {
            r_limit = (rows / 4) * 4;
            c_limit = (cols / 4) * 4;

            for (std::size_t r = 0; r + 3 < rows; r += 4)
            {
                for (std::size_t c = 0; c + 3 < cols; c += 4)
                {
                    __m256 row0 = _mm256_load_ps(src + (r + 0) * cols * 2 + c * 2);
                    __m256 row1 = _mm256_load_ps(src + (r + 1) * cols * 2 + c * 2);
                    __m256 row2 = _mm256_load_ps(src + (r + 2) * cols * 2 + c * 2);
                    __m256 row3 = _mm256_load_ps(src + (r + 3) * cols * 2 + c * 2);

                    __m256 tw0 = load_twiddle_vec(c, r + 0, stride, tw_mod, twiddle_table);
                    __m256 tw1 = load_twiddle_vec(c, r + 1, stride, tw_mod, twiddle_table);
                    __m256 tw2 = load_twiddle_vec(c, r + 2, stride, tw_mod, twiddle_table);
                    __m256 tw3 = load_twiddle_vec(c, r + 3, stride, tw_mod, twiddle_table);

                    row0 = complex_mul_avx2<invert>(row0, tw0);
                    row1 = complex_mul_avx2<invert>(row1, tw1);
                    row2 = complex_mul_avx2<invert>(row2, tw2);
                    row3 = complex_mul_avx2<invert>(row3, tw3);

                    transpose_4x4(row0, row1, row2, row3);

                    _mm256_store_ps(dst + (c + 0) * rows * 2 + r * 2, row0);
                    _mm256_store_ps(dst + (c + 1) * rows * 2 + r * 2, row1);
                    _mm256_store_ps(dst + (c + 2) * rows * 2 + r * 2, row2);
                    _mm256_store_ps(dst + (c + 3) * rows * 2 + r * 2, row3);
                }
            }
        }

        for (std::size_t i = 0; i < rows; ++i)
        {
            const std::uint64_t row_s = static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(stride);
            const std::uint64_t mod = static_cast<std::uint64_t>(tw_mod);

            const std::size_t j_begin = (i < r_limit) ? c_limit : 0;

            for (std::size_t j = j_begin; j < cols; ++j)
            {
                const std::size_t src_idx = i * cols * 2 + j * 2;
                const std::size_t dst_idx = j * rows * 2 + i * 2;

                const std::uint64_t k = (static_cast<std::uint64_t>(j) * row_s) % mod;
                const std::size_t tw_idx = static_cast<std::size_t>(k) * 2;

                float sr = src[src_idx];
                float si = src[src_idx + 1];
                float wr = twiddle_table[tw_idx];
                float wi = twiddle_table[tw_idx + 1];
                if constexpr (invert)
                {
                    wi = -wi;
                }

                dst[dst_idx] = sr * wr - si * wi;
                dst[dst_idx + 1] = sr * wi + si * wr;
            }
        }
    }

    namespace mixed_radix_manager
    {
        static constexpr std::size_t maximum_memory_bytes_for_twiddles_map = 16 * 1024 * 1024;

        struct plan_node
        {
            std::size_t n;
            std::size_t scratch_size_floats = 0;

            bool is_leaf;

            smooth_path_invoker::execute_function_type kernel_fwd = nullptr;
            smooth_path_invoker::execute_function_type kernel_bwd = nullptr;

            std::size_t n1 = 0;
            std::size_t n2 = 0;

            std::shared_ptr<plan_node> sub_plan_n1 = nullptr;
            std::shared_ptr<plan_node> sub_plan_n2 = nullptr;
        };

        using aligned_vector_t = std::vector<float, fy::fft::allocator<float>>;
        using kernel_ptr_type = std::shared_ptr<aligned_vector_t>;
        using cache_mutex_type = std::conditional_t<enable_thread_safe, std::mutex, fy::fft::null_mutex>;
        using plan_ptr_type = std::shared_ptr<plan_node>;

        static inline cache_mutex_type _mutex;
        static inline cache_mutex_type _cache_mutex;
        static inline std::size_t _cache_current_bytes = 0;
        static inline std::unordered_map<std::size_t, kernel_ptr_type> _twiddle_cache;
        static inline std::unordered_map<std::size_t, plan_ptr_type> _plan_cache;

        static std::size_t calculate_scratch_size(std::size_t n, std::size_t n1, std::size_t n2,
            std::size_t sub_req_n1, std::size_t sub_req_n2)
        {
            if (smooth_path_invoker::is_directly_supported(n))
            {
                return 0;
            }

            std::size_t current_level = (n * 4 + 7) & ~std::size_t(7);
            return current_level + std::max(sub_req_n1, sub_req_n2);
        }

        static kernel_ptr_type compute_twiddles(std::size_t n)
        {
            auto ptr = std::make_shared<aligned_vector_t>(n * 2);
            float* data = ptr->data();
            double angle_step = -2.0 * std::numbers::pi_v<double> / static_cast<double>(n);

            std::size_t k = 0;
            if constexpr (is_avx2_available)
            {
                if (n >= 8)
                {
                    __m256 step_vec = _mm256_set1_ps(static_cast<float>(angle_step));
                    __m256 index_offsets = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
                    for (; k + 7 < n; k += 8)
                    {
                        __m256 k_vec = _mm256_set1_ps(static_cast<float>(k));
                        __m256 angles = _mm256_mul_ps(_mm256_add_ps(k_vec, index_offsets), step_vec);
                        __m256 sin_vals, cos_vals;
                        sincos_avx2(angles, &sin_vals, &cos_vals);

                        __m256 lo = _mm256_unpacklo_ps(cos_vals, sin_vals);
                        __m256 hi = _mm256_unpackhi_ps(cos_vals, sin_vals);
                        _mm256_store_ps(data + k * 2, _mm256_permute2f128_ps(lo, hi, 0x20));
                        _mm256_store_ps(data + k * 2 + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
                    }
                }
            }

            for (; k < n; ++k)
            {
                double angle = angle_step * static_cast<double>(k);
                data[k * 2] = static_cast<float>(std::cos(angle));
                data[k * 2 + 1] = static_cast<float>(std::sin(angle));
            }

            return ptr;
        }

        static kernel_ptr_type acquired_twiddles(std::size_t n)
        {
            std::lock_guard<cache_mutex_type> lock(_cache_mutex);

            auto it = _twiddle_cache.find(n);
            if (it != _twiddle_cache.end())
            {
                return it->second;
            }

            std::size_t required_bytes = n * 2 * sizeof(float);
            if (required_bytes > maximum_memory_bytes_for_twiddles_map)
            {
                return compute_twiddles(n);
            }

            while (_cache_current_bytes + required_bytes > maximum_memory_bytes_for_twiddles_map &&
                !_twiddle_cache.empty())
            {
                auto erase_it = _twiddle_cache.begin();
                _cache_current_bytes -= erase_it->second->capacity() * sizeof(float);
                _twiddle_cache.erase(erase_it);
            }

            auto new_ptr = compute_twiddles(n);
            _twiddle_cache[n] = new_ptr;
            _cache_current_bytes += required_bytes;

            return new_ptr;
        }

        static std::size_t compute_n1_logic(std::size_t n)
        {
            std::size_t fallback_factor = 0;
            const auto& smooth_lengths = smooth_path_invoker::available_smooth_length;
            for (auto it = smooth_lengths.rbegin(); it != smooth_lengths.rend(); ++it)
            {
                std::size_t f = *it;
                if (f < n && n % f == 0)
                {
                    std::size_t leaf_candidate = n / f;
                    if (smooth_path_invoker::is_directly_supported(leaf_candidate))
                    {
                        return f;
                    }

                    if (fallback_factor == 0)
                    {
                        fallback_factor = f;
                    }
                }
            }

            if (fallback_factor != 0)
            {
                return fallback_factor;
            }

            if (n % 8 == 0)
            {
                return n / 8;
            }

            if (n % 4 == 0)
            {
                return n / 4;
            }

            if (n % 2 == 0)
            {
                return n / 2;
            }

            return 0;
        }

        static plan_ptr_type build_plan_recursive(std::size_t n)
        {
            auto node = std::make_shared<plan_node>();
            node->n = n;

            if (smooth_path_invoker::is_directly_supported(n))
            {
                node->is_leaf = true;
                node->kernel_fwd = smooth_path_invoker::kernel<false>(n);
                node->kernel_bwd = smooth_path_invoker::kernel<true>(n);
                node->scratch_size_floats = 0;
            }
            else
            {
                node->is_leaf = false;
                node->n1 = compute_n1_logic(n);
                node->n2 = n / node->n1;

                node->sub_plan_n1 = build_plan_recursive(node->n1);
                node->sub_plan_n2 = build_plan_recursive(node->n2);

                node->scratch_size_floats = calculate_scratch_size(n, node->n1, node->n2,
                    node->sub_plan_n1->scratch_size_floats,
                    node->sub_plan_n2->scratch_size_floats);
            }
            return node;
        }

        static plan_ptr_type acquired_plan(std::size_t n)
        {
            std::lock_guard<cache_mutex_type> lock(_mutex);
            auto it = _plan_cache.find(n);
            if (it != _plan_cache.end())
            {
                return it->second;
            }

            auto new_plan = build_plan_recursive(n);
            _plan_cache[n] = new_plan;
            return new_plan;
        }
    }

    struct FFT1D_mixed_radix_invoker : basic_unsmooth_path_invoker
    {
    private:
        static constexpr std::size_t MAX_BUFFER_SIZE = 262144;

        struct workspace_type
        {
            alignas(64) std::uint64_t buffer[MAX_BUFFER_SIZE];
        };

        static inline thread_local workspace_type workspace;
        std::size_t N;
        mixed_radix_manager::kernel_ptr_type _twiddles_ptr;
        mixed_radix_manager::plan_ptr_type _plan;

    public:
        static std::array<std::size_t, 10> get_decomposition_plan(std::size_t n)
        {
            std::array<std::size_t, 10> plan;
            plan.fill(0);

            std::size_t current_n = n;
            std::size_t idx = 0;

            while (!smooth_path_invoker::is_directly_supported(current_n))
            {
                std::size_t n1 = mixed_radix_manager::compute_n1_logic(current_n);
                plan[idx++] = n1;
                current_n /= n1;
            }

            if (idx < 10)
            {
                plan[idx] = current_n;
            }

            return plan;
        }

        static const std::vector<std::size_t>& all_processable_length()
        {
            static std::vector<std::size_t> result = []() -> std::vector<std::size_t> {
                std::vector<std::size_t> result_;
                for (std::size_t i = 6; i < 78125; ++i)
                {
                    if (is_processable(i) && (!smooth_path_invoker::is_directly_supported(i)))
                    {
                        result_.push_back(i);
                    }
                }

                return result_;
                }();

            return result;
        }

        static bool is_processable(std::size_t n)
        {
            while (n > 1)
            {
                bool found = false;
                for (std::int64_t i = static_cast<std::int64_t>(
                    smooth_path_invoker::available_smooth_length.size()) - 1;
                    i >= 0; --i)
                {
                    std::size_t f = smooth_path_invoker::available_smooth_length[i];
                    if (f <= n && n % f == 0)
                    {
                        n /= f;
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    return false;
                }
            }

            return true;
        }

        FFT1D_mixed_radix_invoker(std::size_t length) : N(length)
        {
            FOYE_FFT_ASSERT_ASSUME(length >= 6);
            FOYE_FFT_ASSERT_ASSUME(length < 78125);
            FOYE_FFT_ASSERT_ASSUME((FFT1D_mixed_radix_invoker::is_processable(length)));

            _plan = mixed_radix_manager::acquired_plan(length);
            _twiddles_ptr = mixed_radix_manager::acquired_twiddles(length);
        }

        template<bool invert>
        void dispatch(const float* input, float* output)
        {
            std::size_t required_floats = _plan->scratch_size_floats;

            float* buffer_ptr;
            alignas(64) float stack_buf[2048];
            if (required_floats <= 2048)
            {
                buffer_ptr = stack_buf;
            }
            else
            {
                buffer_ptr = reinterpret_cast<float*>(workspace.buffer);
            }

            FOYE_FFT_ASSERT_ASSUME((reinterpret_cast<std::uintptr_t>(buffer_ptr) & 31) == 0);
            execute_plan<invert>(*_plan, input, output, buffer_ptr, _twiddles_ptr->data(), 1);
        }

        ~FFT1D_mixed_radix_invoker() override { }

        void forward(const float* input, float* output) override
        {
            dispatch<false>(input, output);
        }

        void backward(const float* input, float* output) override
        {
            dispatch<true>(input, output);
        }

    private:
        static std::size_t required_buffer_floats(std::size_t n)
        {
            if (smooth_path_invoker::is_directly_supported(n))
            {
                return 0;
            }

            std::size_t n1 = mixed_radix_manager::compute_n1_logic(n);
            std::size_t n2 = n / n1;
            std::size_t current_level = (n * 4 + 7) & ~std::size_t(7);
            return current_level + std::max(
                required_buffer_floats(n1),
                required_buffer_floats(n2));
        }

        template<bool invert>
        void execute_plan(const mixed_radix_manager::plan_node& node,
            const float* input, float* output,
            float* buffer_ptr, const float* twiddle_base, std::size_t stride)
        {
            if (node.is_leaf)
            {
                if constexpr (invert)
                {
                    node.kernel_bwd(input, output);
                }
                else
                {
                    node.kernel_fwd(input, output);
                }

                return;
            }

            const std::size_t n = node.n;
            const std::size_t n1 = node.n1;
            const std::size_t n2 = node.n2;

            float* buf_level_1 = buffer_ptr;
            const std::size_t offset1 = (n * 2 + 7) & ~std::size_t(7);
            const std::size_t offset2 = (n * 4 + 7) & ~std::size_t(7);
            float* buf_level_2 = buffer_ptr + offset1;
            float* next_recursion_buffer = buffer_ptr + offset2;

            FOYE_FFT_ASSERT_ASSUME((reinterpret_cast<std::uintptr_t>(buf_level_1) & 31) == 0);
            FOYE_FFT_ASSERT_ASSUME((reinterpret_cast<std::uintptr_t>(buf_level_2) & 31) == 0);
            FOYE_FFT_ASSERT_ASSUME((reinterpret_cast<std::uintptr_t>(next_recursion_buffer) & 31) == 0);

            transpose_matrix(input, buf_level_1, n2, n1);

            {
                const auto& sub_plan_n2 = *node.sub_plan_n2;
                if (sub_plan_n2.is_leaf)
                {
                    assert((n1 * 2) == 2 * n1);
                    if (!fy::fft::especially_batch_kernel_dispatch<invert>(n2, buf_level_1, buf_level_1, n1))
                    {
                        auto kern = invert ? sub_plan_n2.kernel_bwd : sub_plan_n2.kernel_fwd;
                        for (std::size_t i = 0; i < n1; ++i)
                        {
                            float* row_ptr = buf_level_1 + i * n2 * 2;
                            kern(row_ptr, row_ptr);
                        }
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < n1; ++i)
                    {
                        float* row_ptr = buf_level_1 + i * n2 * 2;
                        execute_plan<invert>(sub_plan_n2,
                            row_ptr, row_ptr,
                            next_recursion_buffer, twiddle_base, stride * n1);
                    }
                }
            }

            fused_transpose_apply_twiddles<invert>(
                buf_level_1, buf_level_2,
                n1, n2,
                twiddle_base,
                stride,
                this->N);

            {
                const auto& sub_plan_n1 = *node.sub_plan_n1;
                if (sub_plan_n1.is_leaf)
                {
                    assert((n1 * 2) == 2 * n1);
                    if (!fy::fft::especially_batch_kernel_dispatch<invert>(n1, buf_level_2, buf_level_2, n2))
                    {
                        auto kern = invert ? sub_plan_n1.kernel_bwd : sub_plan_n1.kernel_fwd;
                        for (std::size_t j = 0; j < n2; ++j)
                        {
                            float* row_ptr = buf_level_2 + j * n1 * 2;
                            kern(row_ptr, row_ptr);
                        }
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < n2; ++j)
                    {
                        float* row_ptr = buf_level_2 + j * n1 * 2;
                        execute_plan<invert>(sub_plan_n1,
                            row_ptr, row_ptr,
                            next_recursion_buffer, twiddle_base, stride * n2);
                    }
                }
            }

            transpose_matrix(buf_level_2, output, n2, n1);
        }
    };
}

#endif
