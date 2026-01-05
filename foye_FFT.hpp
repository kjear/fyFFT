#ifndef _FOYE_FFT_HPP_
#define _FOYE_FFT_HPP_

#include <cmath>
#include <algorithm>
#include <limits>
#include <immintrin.h>
#include <array>
#include <assert.h>
#include <memory>
#include <numbers>
#include <numeric>
#include <unordered_map>
#include <new>
#include <bit>
#include <bitset>
#include <mutex>
#include <complex>
#include <variant>

#include <Windows.h>
#undef min
#undef max

#define FOYE_FFT_THREAD_SAFE_ENABLE 0

#if defined(FOYE_FFT_THREAD_SAFE_ENABLE) && (FOYE_FFT_THREAD_SAFE_ENABLE)
inline constexpr bool enable_thread_safe = true;
#elif (!defined(FOYE_FFT_THREAD_SAFE_ENABLE)) || (!FOYE_FFT_THREAD_SAFE_ENABLE)
inline constexpr bool enable_thread_safe = false;
#endif

#if (defined(__AVX2__) && (__AVX2__)) || defined(FOYE_FFT_FORCE_ENABLE_AVX2)
inline constexpr bool is_avx2_available = true;
#else
inline constexpr bool is_avx2_available = false;
#endif

#if defined(_DEBUG)
#define FOYE_FFT_ASSERT_ASSUME(expr) assert(expr)
#else
#define FOYE_FFT_ASSERT_ASSUME(expr) __assume(expr)
#endif

#define FOYE_FFT_DISABLE_NON_STANDARD_FP_EXCEPTION (_mm_setcsr(_mm_getcsr() | 0x8040))

#define FOYE_MXCSR_DAZ   (1u << 6)
#define FOYE_MXCSR_FTZ   (1u << 15)
#define FOYE_MXCSR_NON_STANDARD_FP \
    (FOYE_MXCSR_DAZ | FOYE_MXCSR_FTZ)

namespace fy::fft
{
	struct MXCSR
	{
		MXCSR() : __foye_fp_saved_csr(_mm_getcsr()) {}
		~MXCSR() { _mm_setcsr(__foye_fp_saved_csr); }
	private:
		unsigned int __foye_fp_saved_csr;
	};
}

#include "src\foye_FFT_utility.hpp"
#include "src\foye_FFT_microKernel.hpp"
#include "src\foye_FFT_batch_kernel.hpp"

#include "src\foye_FFT_radix2.hpp"
#include "src\foye_FFT_radix3.hpp"
#include "src\foye_FFT_radix4.hpp"
#include "src\foye_FFT_radix5.hpp"
#include "src\foye_FFT_radix7.hpp"

#include "src\foye_FFT_radix_wrap.hpp"
#include "src\foye_FFT_bluestein.hpp"
#include "src\foye_FFT_mix_radix.hpp"
#include "src\foye_FFT_rader.hpp"
#include "src\foye_FFT_dft.hpp"

namespace fy::fft
{
    using fy::fft::mix_radix::FFT1D_mixed_radix_invoker;
    using fy::fft::rader::FFT1D_C2C_rader_invoker;
    using fy::fft::bluestein::FFT1D_bluestein_invoker;
    using fy::fft::dft::FFT1D_C2C_dft_invoker;

    constexpr std::size_t min_input_length = 2;
    constexpr std::size_t max_input_length = 80'000;

    struct FFT1D_C2C_invoker
    {
    private:
        using invoker_variant_type = std::variant<
            std::monostate, 
            FFT1D_mixed_radix_invoker, FFT1D_C2C_rader_invoker, 
            FFT1D_bluestein_invoker, FFT1D_C2C_dft_invoker>;

        invoker_variant_type invoker;
        std::size_t N;

    public:
        FFT1D_C2C_invoker(std::size_t length) : N(length)
        {
            FOYE_FFT_ASSERT_ASSUME((length >= min_input_length));
            FOYE_FFT_ASSERT_ASSUME((length <= max_input_length));

            if (smooth_path_invoker::is_directly_supported(length))
            {
                invoker = std::monostate{};
            }
            else if (FFT1D_mixed_radix_invoker::is_processable(length))
            {
                invoker = FFT1D_mixed_radix_invoker(length);
            }
            else if (FFT1D_C2C_rader_invoker::is_processable(length))
            {
                invoker = FFT1D_C2C_rader_invoker(length);
            }
            else if (FFT1D_bluestein_invoker::is_processable(length))
            {
                invoker = FFT1D_bluestein_invoker(length);
            }
            else
            {
                invoker = FFT1D_C2C_dft_invoker(length);
            }
        };

        template<bool invert>
        void dispatch(const float* input, float* output)
        {
            std::visit(
            [&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::monostate>)
                {
                    smooth_path_dispatch<invert>(input, output, N);
                }
                else
                {
                    if constexpr (invert)
                    {
                        arg.backward(input, output);
                    }
                    else
                    {
                        arg.forward(input, output);
                    }
                }
            }, invoker);
        }

        enum class FFT_execution_strategy
        {
            kernel_direct,
            mix_radix,
            rader,
            bluestein,
            DFT
        };

        static FFT_execution_strategy acquired_execution_strategy(std::size_t length)
        {
            if (smooth_path_invoker::is_directly_supported(length))
            {
                return FFT_execution_strategy::kernel_direct;
            }
            else if (FFT1D_mixed_radix_invoker::is_processable(length))
            {
                return FFT_execution_strategy::mix_radix;
            }
            else if (FFT1D_C2C_rader_invoker::is_processable(length))
            {
                return FFT_execution_strategy::rader;
            }
            else if (FFT1D_bluestein_invoker::is_processable(length))
            {
                return FFT_execution_strategy::bluestein;
            }
            else
            {
                return FFT_execution_strategy::DFT;
            }
        }
    };
}

#endif