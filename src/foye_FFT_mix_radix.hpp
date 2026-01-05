#ifndef _FOYE_FFT_MIXRADIX_HPP_
#define _FOYE_FFT_MIXRADIX_HPP_

namespace fy::fft::mix_radix
{
    [[msvc::forceinline]] static __m256 complex_mul_avx2(__m256 data, __m256 tw)
    {
        __m256 wr = _mm256_moveldup_ps(tw);
        __m256 wi = _mm256_movehdup_ps(tw);

        __m256 data_swapped = _mm256_permute_ps(data, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 c = _mm256_mul_ps(data_swapped, wi);
        return _mm256_fmaddsub_ps(data, wr, c);
    }

    static void fused_transpose_apply_twiddles(
        const float* src,
        float* dst,
        std::size_t rows,
        std::size_t cols,
        const float* tw_ptr)
    {
        std::size_t r4 = rows & ~std::size_t(3);
        std::size_t c4 = cols & ~std::size_t(3);

        if constexpr (is_avx2_available)
        {
            for (std::size_t r = 0; r < r4; r += 4)
            {
                for (std::size_t c = 0; c < c4; c += 4)
                {
                    __m256 r0 = _mm256_load_ps(src + (r + 0) * cols * 2 + c * 2);
                    __m256 r1 = _mm256_load_ps(src + (r + 1) * cols * 2 + c * 2);
                    __m256 r2 = _mm256_load_ps(src + (r + 2) * cols * 2 + c * 2);
                    __m256 r3 = _mm256_load_ps(src + (r + 3) * cols * 2 + c * 2);

                    r0 = complex_mul_avx2(r0, _mm256_load_ps(tw_ptr + 0));
                    r1 = complex_mul_avx2(r1, _mm256_load_ps(tw_ptr + 8));
                    r2 = complex_mul_avx2(r2, _mm256_load_ps(tw_ptr + 16));
                    r3 = complex_mul_avx2(r3, _mm256_load_ps(tw_ptr + 24));
                    tw_ptr += 32;

                    transpose_4x4_complexf32(r0, r1, r2, r3);

                    _mm256_store_ps(dst + (c + 0) * rows * 2 + r * 2, r0);
                    _mm256_store_ps(dst + (c + 1) * rows * 2 + r * 2, r1);
                    _mm256_store_ps(dst + (c + 2) * rows * 2 + r * 2, r2);
                    _mm256_store_ps(dst + (c + 3) * rows * 2 + r * 2, r3);
                }

                for (std::size_t c = c4; c < cols; ++c)
                {
                    for (std::size_t bi = 0; bi < 4; ++bi)
                    {
                        std::size_t si = (r + bi) * cols * 2 + c * 2;
                        std::size_t di = c * rows * 2 + (r + bi) * 2;

                        float sr = src[si], si2 = src[si + 1];
                        float wr = *tw_ptr++, wi = *tw_ptr++;

                        dst[di] = sr * wr - si2 * wi;
                        dst[di + 1] = sr * wi + si2 * wr;
                    }
                }
            }
        }

        std::size_t r_start = (is_avx2_available ? r4 : 0);
        for (std::size_t r = r_start; r < rows; ++r)
        {
            for (std::size_t c = 0; c < cols; ++c)
            {
                std::size_t si = r * cols * 2 + c * 2;
                std::size_t di = c * rows * 2 + r * 2;

                float sr = src[si], si2 = src[si + 1];
                float wr = *tw_ptr++, wi = *tw_ptr++;

                dst[di] = sr * wr - si2 * wi;
                dst[di + 1] = sr * wi + si2 * wr;
            }
        }
    }

    struct buffer_reference
    {
        enum Location { external_in, external_out, workspace };
        Location loc;
        std::size_t offset;

        float* resolve(float* in, float* out, float* buf) const
        {
            if (loc == external_in) return in;
            if (loc == external_out) return out;
            return buf + offset;
        }
    };

    enum class operator_type
    {
        TRANSPOSE,
        LEAF_BATCH_FFT,
        FUSED_TWIDDLE_TRANSPOSE
    };

    struct flatten_operator
    {
        operator_type type;
        buffer_reference src;
        buffer_reference dst;
        std::size_t n1, n2;
        const float* twiddle_ptr = nullptr;
        smooth_path_invoker::execute_function_type leaf_kernel = nullptr;
        bool (*batch_dispatch)(std::size_t, const float*, float*, std::size_t) = nullptr;
    };

    struct plan_node
    {
        std::size_t n = 0;
        std::size_t scratch_size_floats = 0;
        bool is_leaf = false;

        smooth_path_invoker::execute_function_type kernel_fwd = nullptr;
        smooth_path_invoker::execute_function_type kernel_bwd = nullptr;

        std::size_t n1 = 0, n2 = 0;
        std::unique_ptr<plan_node> sub_plan_n1 = nullptr;
        std::unique_ptr<plan_node> sub_plan_n2 = nullptr;

        bool n1_support_batch = false;
        bool n2_support_batch = false;

        std::unique_ptr<aligned_vector_type<float>> twiddle_fwd = nullptr;
        std::unique_ptr<aligned_vector_type<float>> twiddle_bwd = nullptr;
    };

    static std::size_t calculate_scratch_size(
        std::size_t n,
        std::size_t n1,
        std::size_t n2,
        std::size_t sub1,
        std::size_t sub2)
    {
        if (smooth_path_invoker::is_directly_supported(n))
        {
            return 0;
        }

        std::size_t here = (n * 4 + 7) & ~std::size_t(7);
        std::size_t child = std::max(
            ((n1 * 4 + 7) & ~std::size_t(7)) + sub1,
            ((n2 * 4 + 7) & ~std::size_t(7)) + sub2);

        return here + child;
    }

    static std::unique_ptr<aligned_vector_type<float>> compute_twiddles(std::size_t n)
    {
        auto ptr = std::make_unique<aligned_vector_type<float>>(n * 2);
        float* data = ptr->data();

        double angle_step = -2.0 * std::numbers::pi_v<double> / static_cast<double>(n);
        std::size_t k = 0;

        if constexpr (is_avx2_available)
        {
            if (n >= 8)
            {
                __m256 step_vec = _mm256_set1_ps(static_cast<float>(angle_step));
                __m256 index_offsets = _mm256_setr_ps(
                    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

                for (; k + 7 < n; k += 8)
                {
                    __m256 k_vec = _mm256_set1_ps(static_cast<float>(k));
                    __m256 angles =
                        _mm256_mul_ps(_mm256_add_ps(k_vec, index_offsets), step_vec);

                    __m256 sin_vals, cos_vals;
                    sincos_avx2(angles, &sin_vals, &cos_vals);

                    __m256 lo = _mm256_unpacklo_ps(cos_vals, sin_vals);
                    __m256 hi = _mm256_unpackhi_ps(cos_vals, sin_vals);

                    _mm256_store_ps(data + k * 2,
                        _mm256_permute2f128_ps(lo, hi, 0x20));
                    _mm256_store_ps(data + k * 2 + 8,
                        _mm256_permute2f128_ps(lo, hi, 0x31));
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

    static std::unique_ptr<aligned_vector_type<float>> compute_twiddle_matrix(
        std::size_t N, std::size_t n1, std::size_t n2,
        std::size_t stride, const float* base)
    {
        auto mat = std::make_unique<aligned_vector_type<float>>(n1 * n2 * 2);
        float* out = mat->data();

        auto store_tw = [&](std::size_t r, std::size_t c)
            {
                std::size_t k = (r * stride * c) % N;
                *out++ = base[k * 2];
                *out++ = base[k * 2 + 1];
            };

        std::size_t r4 = n1 & ~std::size_t(3);
        std::size_t c4 = n2 & ~std::size_t(3);

        for (std::size_t r = 0; r < r4; r += 4)
        {
            for (std::size_t c = 0; c < c4; c += 4)
            {
                for (std::size_t bi = 0; bi < 4; ++bi)
                {
                    for (std::size_t bj = 0; bj < 4; ++bj)
                    {
                        store_tw(r + bi, c + bj);
                    }
                }
            }

            for (std::size_t c = c4; c < n2; ++c)
            {
                for (std::size_t bi = 0; bi < 4; ++bi)
                {
                    store_tw(r + bi, c);
                }
            }
        }

        for (std::size_t r = r4; r < n1; ++r)
        {
            for (std::size_t c = 0; c < n2; ++c)
            {
                store_tw(r, c);
            }
        }

        return mat;
    }

    static std::size_t compute_n1_logic(std::size_t n)
    {
        const auto& smooth_lengths = smooth_path_invoker::available_smooth_length;
        std::size_t best_n1 = 0;
        double best_cost = std::numeric_limits<double>::infinity();

        for (std::size_t f : smooth_lengths)
        {
            if (f >= n)
            {
                continue;
            }

            if (n % f != 0)
            {
                continue;
            }

            std::size_t n1 = f;
            std::size_t n2 = n / f;

            bool n1_supported = smooth_path_invoker::is_directly_supported(n1);
            bool n2_supported = smooth_path_invoker::is_directly_supported(n2);
            bool n1_batch = especially_batch_kernel_dispatch<false>(n1, nullptr, nullptr, n2);
            bool n2_batch = especially_batch_kernel_dispatch<false>(n2, nullptr, nullptr, n1);

            double imbalance_factor = std::abs(std::log2((double)n1) - std::log2((double)n2));

            int priority = 0;
            if ((n1_supported && n1_batch) || (n2_supported && n2_batch))
            {
                priority = 2;
            }
            else if (n1_supported || n2_supported)
            {
                priority = 1;
            }
            else
            {
                priority = 0;
            }

            double cost = static_cast<double>(n);
            cost *= (1.0 + imbalance_factor * 0.2);

            if (n1_batch)
            {
                cost *= 0.85;
            }
            if (n2_batch)
            {
                cost *= 0.85;
            }

            if (priority > 0 || best_n1 == 0)
            {
                if (priority > 0)
                {
                    if (priority > ((best_n1 == 0) ? -1 : 0))
                    {
                        best_n1 = n1;
                        best_cost = cost;
                    }
                    else if (priority == ((best_n1 == 0) ? -1 : 0) && cost < best_cost)
                    {
                        best_n1 = n1;
                        best_cost = cost;
                    }
                }
                else
                {
                    if (cost < best_cost)
                    {
                        best_n1 = n1;
                        best_cost = cost;
                    }
                }
            }
        }

        if (best_n1 == 0)
        {
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

        return best_n1;
    }

    struct plan
    {
        std::size_t N;
        std::unique_ptr<plan_node> root;
        std::unique_ptr<aligned_vector_type<float>> base_twiddles;
        aligned_vector_type<float> workspace;
        aligned_vector_type<flatten_operator> ops_fwd;
        aligned_vector_type<flatten_operator> ops_bwd;

        plan(std::size_t n) : N(n)
        {
            root = build(n);
            base_twiddles = compute_twiddles(n);
            populate(*root, 1);

            std::size_t scratch_used = 0;
            flatten(root.get(), false,
                { buffer_reference::external_in, 0 },
                { buffer_reference::external_out, 0 },
                0, scratch_used, ops_fwd);

            scratch_used = 0;
            flatten(root.get(), true,
                { buffer_reference::external_in, 0 },
                { buffer_reference::external_out, 0 },
                0, scratch_used, ops_bwd);

            if (scratch_used > 0)
            {
                workspace.resize(scratch_used * sizeof(float));
            }
        }

    private:
        void flatten(const plan_node* node, bool invert,
            buffer_reference in, buffer_reference out,
            std::size_t buf_offset,
            std::size_t& max_scratch,
            aligned_vector_type<flatten_operator>& ops)
        {
            if (node->is_leaf)
            {
                flatten_operator op;
                op.type = operator_type::LEAF_BATCH_FFT;
                op.src = in;
                op.dst = out;
                op.n1 = node->n;
                op.n2 = 1;
                op.leaf_kernel = invert ? node->kernel_bwd : node->kernel_fwd;
                ops.push_back(op);
                return;
            }

            std::size_t n = node->n;
            std::size_t n1 = node->n1;
            std::size_t n2 = node->n2;

            std::size_t b1_off = buf_offset;
            std::size_t b2_off = buf_offset + ((n * 2 + 7) & ~std::size_t(7));
            std::size_t next_off = buf_offset + ((n * 4 + 7) & ~std::size_t(7));
            max_scratch = std::max(max_scratch, next_off);

            ops.push_back({ operator_type::TRANSPOSE, in, {buffer_reference::workspace, b1_off}, n2, n1 });

            if (node->sub_plan_n2->is_leaf && node->n2_support_batch)
            {
                flatten_operator op;
                op.type = operator_type::LEAF_BATCH_FFT;
                op.src = { buffer_reference::workspace, b1_off };
                op.dst = { buffer_reference::workspace, b1_off };
                op.n1 = n2;
                op.n2 = n1;
                op.batch_dispatch = invert
                    ? especially_batch_kernel_dispatch<true>
                    : especially_batch_kernel_dispatch<false>;
                ops.push_back(op);
            }
            else
            {
                for (std::size_t i = 0; i < n1; ++i)
                {
                    flatten(node->sub_plan_n2.get(), invert,
                        { buffer_reference::workspace, b1_off + i * n2 * 2 },
                        { buffer_reference::workspace, b1_off + i * n2 * 2 },
                        next_off, max_scratch, ops);
                }
            }

            flatten_operator fused;
            fused.type = operator_type::FUSED_TWIDDLE_TRANSPOSE;
            fused.src = { buffer_reference::workspace, b1_off };
            fused.dst = { buffer_reference::workspace, b2_off };
            fused.n1 = n1;
            fused.n2 = n2;
            fused.twiddle_ptr = (invert ? node->twiddle_bwd : node->twiddle_fwd)->data();
            ops.push_back(fused);

            if (node->sub_plan_n1->is_leaf && node->n1_support_batch)
            {
                flatten_operator op;
                op.type = operator_type::LEAF_BATCH_FFT;
                op.src = { buffer_reference::workspace, b2_off };
                op.dst = { buffer_reference::workspace, b2_off };
                op.n1 = n1;
                op.n2 = n2;
                op.batch_dispatch = invert
                    ? especially_batch_kernel_dispatch<true>
                    : especially_batch_kernel_dispatch<false>;
                ops.push_back(op);
            }
            else
            {
                for (std::size_t j = 0; j < n2; ++j)
                {
                    flatten(node->sub_plan_n1.get(), invert,
                        { buffer_reference::workspace, b2_off + j * n1 * 2 },
                        { buffer_reference::workspace, b2_off + j * n1 * 2 },
                        next_off, max_scratch, ops);
                }
            }

            ops.push_back({ operator_type::TRANSPOSE, {buffer_reference::workspace, b2_off}, out, n2, n1 });
        }

        void populate(plan_node& node, std::size_t stride)
        {
            if (node.is_leaf)
            {
                return;
            }

            auto fwd = compute_twiddle_matrix(N, node.n1, node.n2, stride, base_twiddles->data());
            auto bwd = std::make_unique<aligned_vector_type<float>>(*fwd);
            for (std::size_t i = 1; i < bwd->size(); i += 2)
            {
                (*bwd)[i] = -(*bwd)[i];
            }

            node.twiddle_fwd = std::move(fwd);
            node.twiddle_bwd = std::move(bwd);

            populate(*node.sub_plan_n2, stride * node.n1);
            populate(*node.sub_plan_n1, stride * node.n2);
        }

        static std::unique_ptr<plan_node> build(std::size_t n)
        {
            auto node = std::make_unique<plan_node>();
            node->n = n;

            if (smooth_path_invoker::is_directly_supported(n))
            {
                node->is_leaf = true;
                node->kernel_fwd = smooth_path_invoker::kernel<false>(n);
                node->kernel_bwd = smooth_path_invoker::kernel<true>(n);
            }
            else
            {
                node->n1 = compute_n1_logic(n);
                node->n2 = n / node->n1;

                node->sub_plan_n1 = build(node->n1);
                node->sub_plan_n2 = build(node->n2);

                node->n1_support_batch = especially_batch_kernel_dispatch<false>(node->n1, nullptr, nullptr, node->n2);
                node->n2_support_batch = especially_batch_kernel_dispatch<false>(node->n2, nullptr, nullptr, node->n1);

                node->scratch_size_floats = calculate_scratch_size(
                    n, node->n1, node->n2,
                    node->sub_plan_n1->scratch_size_floats,
                    node->sub_plan_n2->scratch_size_floats);
            }

            return node;
        }
    };

    struct FFT1D_mixed_radix_invoker : basic_unsmooth_path_invoker
    {
    private:
        plan _plan;

        void execute_operators(const aligned_vector_type<flatten_operator>& ops, const float* in, float* out)
        {
            float* buf = reinterpret_cast<float*>(_plan.workspace.data());
            float* in_ptr = const_cast<float*>(in);

            for (const fy::fft::mix_radix::flatten_operator& op : ops)
            {
                float* src = op.src.resolve(in_ptr, out, buf);
                float* dst = op.dst.resolve(in_ptr, out, buf);

                switch (op.type)
                {
                case operator_type::TRANSPOSE:
                {
                    transpose_matrix_complexf32(src, dst, op.n1, op.n2);
                    break;
                }

                case operator_type::LEAF_BATCH_FFT:
                {
                    if (op.batch_dispatch)
                    {
                        op.batch_dispatch(op.n1, src, dst, op.n2);
                    }
                    else
                    {
                        op.leaf_kernel(src, dst);
                    }
                    break;
                }

                case operator_type::FUSED_TWIDDLE_TRANSPOSE:
                {
                    fused_transpose_apply_twiddles(src, dst, op.n1, op.n2, op.twiddle_ptr);
                    break;
                }
                }
            }
        }

    public:
        static std::array<std::size_t, 10> get_decomposition_plan(std::size_t n)
        {
            std::array<std::size_t, 10> plan;
            plan.fill(0);

            std::size_t current_n = n;
            std::size_t idx = 0;

            while (!smooth_path_invoker::is_directly_supported(current_n))
            {
                std::size_t n1 = compute_n1_logic(current_n);
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

        FFT1D_mixed_radix_invoker(std::size_t n) : _plan(n) {}

        void forward(const float* in, float* out) override
        {
            execute_operators(_plan.ops_fwd, in, out);
        }

        void backward(const float* in, float* out) override
        {
            execute_operators(_plan.ops_bwd, in, out);
        }

        template<bool invert>
        void dispatch(const float* in, float* out)
        {
            execute_operators(invert ? _plan.ops_bwd : _plan.ops_fwd, in, out);
        }
    };
}

#endif