#ifndef _FOYE_FFT_UTILITY_HPP_
#define _FOYE_FFT_UTILITY_HPP_

namespace fy::fft
{
	struct basic_unsmooth_path_invoker
	{
		virtual ~basic_unsmooth_path_invoker() = default;
		virtual void forward(const float* input, float* output) {}
		virtual void backward(const float* input, float* output) {}
	};

	template<typename T, std::size_t alignment = is_avx2_available ? 32 : alignof(T)>
	struct allocator
	{
		using value_type = T;
		using pointer = value_type*;
		using size_type = std::size_t;

		template <typename U>
		struct rebind
		{
			using other = allocator<U, alignment>;
		};

		allocator() noexcept = default;

		template<typename U>
		allocator(const allocator<U, alignment>&) noexcept {}

		[[nodiscard]] pointer allocate(std::size_t n)
#if !defined(_DEBUG)
			noexcept
#endif
		{
			pointer ptr;
#if defined(_DEBUG)
			if (n > std::size_t(-1) / sizeof(value_type))
			{
				throw std::bad_alloc();
			}
#endif
			ptr = reinterpret_cast<pointer>(
				_aligned_malloc(n * sizeof(value_type), alignment));
#if defined(_DEBUG)
			if (!ptr)
			{
				throw std::bad_alloc();
			}
#endif
			return ptr;
		}

		void deallocate(pointer p, std::size_t) noexcept
		{
			if (p)
			{
				_aligned_free(reinterpret_cast<void*>(p));
			}
		}

		bool operator==(const allocator&) const noexcept { return true; }
		bool operator!=(const allocator&) const noexcept { return false; }
	};

	[[msvc::forceinline]] static std::size_t align_offset(std::size_t offset, std::size_t alignment)
	{
		return (offset + alignment - 1) & ~(alignment - 1);
	}

	template<typename T>
	using aligned_vector_type = std::vector<T, fy::fft::allocator<T>>;

	[[msvc::forceinline]] static void transpose_4x4_complexf32(__m256& r0, __m256& r1, __m256& r2, __m256& r3)
	{
		__m256d r0d = _mm256_castps_pd(r0);
		__m256d r1d = _mm256_castps_pd(r1);
		__m256d r2d = _mm256_castps_pd(r2);
		__m256d r3d = _mm256_castps_pd(r3);

		__m256d t0 = _mm256_unpacklo_pd(r0d, r1d);
		__m256d t1 = _mm256_unpackhi_pd(r0d, r1d);
		__m256d t2 = _mm256_unpacklo_pd(r2d, r3d);
		__m256d t3 = _mm256_unpackhi_pd(r2d, r3d);

		r0 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x20));
		r1 = _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x20));
		r2 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x31));
		r3 = _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x31));
	}

	[[msvc::forceinline]] static void transpose_matrix_complexf32(const float* src, float* dst, std::size_t rows, std::size_t cols)
	{
		constexpr std::size_t BLOCK_SIZE = 32;

		const std::size_t src_stride = cols * 2;
		const std::size_t dst_stride = rows * 2;

		for (std::size_t i = 0; i < rows; i += BLOCK_SIZE)
		{
			for (std::size_t j = 0; j < cols; j += BLOCK_SIZE)
			{
				std::size_t blk_rows = std::min(BLOCK_SIZE, rows - i);
				std::size_t blk_cols = std::min(BLOCK_SIZE, cols - j);
				std::size_t ii = 0;

				if constexpr (is_avx2_available)
				{
					for (; ii + 4 <= blk_rows; ii += 4)
					{
						std::size_t jj = 0;

						for (; jj + 8 <= blk_cols; jj += 8)
						{
							const float* src_ptr = src + ((i + ii) * cols + (j + jj)) * 2;
							float* dst_ptr = dst + ((j + jj) * rows + (i + ii)) * 2;

							__m256 a0 = _mm256_load_ps(src_ptr + 0 * src_stride);
							__m256 a1 = _mm256_load_ps(src_ptr + 1 * src_stride);
							__m256 a2 = _mm256_load_ps(src_ptr + 2 * src_stride);
							__m256 a3 = _mm256_load_ps(src_ptr + 3 * src_stride);

							__m256 b0 = _mm256_load_ps(src_ptr + 0 * src_stride + 8);
							__m256 b1 = _mm256_load_ps(src_ptr + 1 * src_stride + 8);
							__m256 b2 = _mm256_load_ps(src_ptr + 2 * src_stride + 8);
							__m256 b3 = _mm256_load_ps(src_ptr + 3 * src_stride + 8);

							transpose_4x4_complexf32(a0, a1, a2, a3);
							transpose_4x4_complexf32(b0, b1, b2, b3);

							_mm256_store_ps(dst_ptr + 0 * dst_stride, a0);
							_mm256_store_ps(dst_ptr + 1 * dst_stride, a1);
							_mm256_store_ps(dst_ptr + 2 * dst_stride, a2);
							_mm256_store_ps(dst_ptr + 3 * dst_stride, a3);

							_mm256_store_ps(dst_ptr + 4 * dst_stride, b0);
							_mm256_store_ps(dst_ptr + 5 * dst_stride, b1);
							_mm256_store_ps(dst_ptr + 6 * dst_stride, b2);
							_mm256_store_ps(dst_ptr + 7 * dst_stride, b3);
						}

						for (; jj + 4 <= blk_cols; jj += 4)
						{
							const float* src_ptr = src + ((i + ii) * cols + (j + jj)) * 2;
							float* dst_ptr = dst + ((j + jj) * rows + (i + ii)) * 2;

							__m256 r0 = _mm256_load_ps(src_ptr + 0 * src_stride);
							__m256 r1 = _mm256_load_ps(src_ptr + 1 * src_stride);
							__m256 r2 = _mm256_load_ps(src_ptr + 2 * src_stride);
							__m256 r3 = _mm256_load_ps(src_ptr + 3 * src_stride);

							transpose_4x4_complexf32(r0, r1, r2, r3);

							_mm256_store_ps(dst_ptr + 0 * dst_stride, r0);
							_mm256_store_ps(dst_ptr + 1 * dst_stride, r1);
							_mm256_store_ps(dst_ptr + 2 * dst_stride, r2);
							_mm256_store_ps(dst_ptr + 3 * dst_stride, r3);
						}

						for (; jj < blk_cols; ++jj)
						{
							for (std::size_t k = 0; k < 4; ++k)
							{
								dst[((j + jj) * rows + (i + ii + k)) * 2 + 0] = src[((i + ii + k) * cols + (j + jj)) * 2 + 0];
								dst[((j + jj) * rows + (i + ii + k)) * 2 + 1] = src[((i + ii + k) * cols + (j + jj)) * 2 + 1];
							}
						}
					}
				}

				for (; ii < blk_rows; ++ii)
				{
					for (std::size_t jj = 0; jj < blk_cols; ++jj)
					{
						dst[((j + jj) * rows + (i + ii)) * 2 + 0] = src[((i + ii) * cols + (j + jj)) * 2 + 0];
						dst[((j + jj) * rows + (i + ii)) * 2 + 1] = src[((i + ii) * cols + (j + jj)) * 2 + 1];
					}
				}
			}
		}
	}

	[[msvc::forceinline]] static void separate_even_odd(__m256& a, __m256& b)
	{
		__m256d ad = _mm256_castps_pd(a);
		__m256d bd = _mm256_castps_pd(b);
		__m256d evens_mixed = _mm256_unpacklo_pd(ad, bd);
		__m256d odds_mixed = _mm256_unpackhi_pd(ad, bd);
		a = _mm256_castpd_ps(_mm256_permute4x64_pd(evens_mixed, 0xD8));
		b = _mm256_castpd_ps(_mm256_permute4x64_pd(odds_mixed, 0xD8));
	}

	[[msvc::forceinline]] static __m256 fma_complexf32(__m256 v, __m256 w_normal, __m256 w_swapped)
	{
		__m256 v_re = _mm256_moveldup_ps(v);
		__m256 v_im = _mm256_movehdup_ps(v);
		return _mm256_fmaddsub_ps(v_re, w_normal, _mm256_mul_ps(v_im, w_swapped));
	}

	[[msvc::forceinline]] static __m256 fma_complexf32_accumulate(
		__m256 accumulator,
		__m256 v,
		__m256 w_normal,
		__m256 w_swapped)
	{
		__m256 v_re = _mm256_moveldup_ps(v);
		__m256 v_im = _mm256_movehdup_ps(v);
		return _mm256_fmaddsub_ps(v_re, w_normal, _mm256_fmaddsub_ps(v_im, w_swapped, accumulator));
	}

	template<bool ApplyScale>
	[[msvc::forceinline]] static void frequency_domain_convolution_complexf32(
		float* data, const float* kernel, std::size_t M, float scale_factor = 1.0f)
	{
		std::size_t i = 0;
		__m256 v_scale;
		if constexpr (ApplyScale)
		{
			v_scale = _mm256_set1_ps(scale_factor);
		}

		for (; i + 8 <= 2 * M; i += 8)
		{
			__m256 v_data = _mm256_load_ps(data + i);
			__m256 v_kern = _mm256_load_ps(kernel + i);

			__m256 v_data_im_re = _mm256_permute_ps(v_data, 0xB1);
			__m256 v_kern_re = _mm256_moveldup_ps(v_kern);
			__m256 v_kern_im = _mm256_movehdup_ps(v_kern);

			__m256 prod = _mm256_mul_ps(v_data, v_kern_re);
			__m256 term2 = _mm256_mul_ps(v_data_im_re, v_kern_im);

			__m256 res = _mm256_addsub_ps(prod, term2);

			if constexpr (ApplyScale)
			{
				res = _mm256_mul_ps(res, v_scale);
			}

			_mm256_store_ps(data + i, res);
		}

		for (; i < 2 * M; i += 2)
		{
			float a = data[i];
			float b = data[i + 1];
			float c = kernel[i];
			float d = kernel[i + 1];

			float re = a * c - b * d;
			float im = a * d + b * c;

			if constexpr (ApplyScale)
			{
				re *= scale_factor;
				im *= scale_factor;
			}

			data[i] = re;
			data[i + 1] = im;
		}
	}

	[[msvc::forceinline]] static void sincos_avx2(__m256 input, __m256* sin_result, __m256* cos_result)
	{
		constexpr std::uint32_t SIGN_BIT_MASK = 0x80000000;
		constexpr std::uint32_t INV_SIGN_BIT_MASK = ~SIGN_BIT_MASK;

		constexpr float PI_OVER_4_RECIPROCAL = 1.27323954473516f;

		constexpr float REDUCE_COEFF1 = -0.78515625f;
		constexpr float REDUCE_COEFF2 = -2.4187564849853515625e-4f;
		constexpr float REDUCE_COEFF3 = -3.77489497744594108e-8f;

		constexpr float COS_COEFF0 = 2.443315711809948E-005f;
		constexpr float COS_COEFF1 = -1.388731625493765E-003f;
		constexpr float COS_COEFF2 = 4.166664568298827E-002f;
		constexpr float HALF = 0.5f;
		constexpr float ONE = 1.0f;

		constexpr float SIN_COEFF0 = -1.9515295891E-4f;
		constexpr float SIN_COEFF1 = 8.3321608736E-3f;
		constexpr float SIN_COEFF2 = -1.6666654611E-1f;

		constexpr int INTEGER_ONE = 1;
		constexpr int INTEGER_TWO = 2;
		constexpr int INTEGER_FOUR = 4;
		constexpr int SHIFT_29 = 29;

		__m256 inv_sign_mask = _mm256_set1_ps(std::bit_cast<float>(INV_SIGN_BIT_MASK));
		__m256 sign_mask = _mm256_set1_ps(std::bit_cast<float>(SIGN_BIT_MASK));
		__m256 pi_over_4_inv = _mm256_set1_ps(PI_OVER_4_RECIPROCAL);

		__m256i pi32_1 = _mm256_set1_epi32(INTEGER_ONE);
		__m256i pi32_2 = _mm256_set1_epi32(INTEGER_TWO);
		__m256i pi32_4 = _mm256_set1_epi32(INTEGER_FOUR);
		__m256i pi32_inv1 = _mm256_set1_epi32(~INTEGER_ONE);

		__m256 x = input;
		__m256 sign_bit_sin = _mm256_and_ps(x, sign_mask);

		x = _mm256_and_ps(x, inv_sign_mask);

		__m256 y = _mm256_mul_ps(x, pi_over_4_inv);
		__m256i quadrant_int = _mm256_cvttps_epi32(y);

		quadrant_int = _mm256_add_epi32(quadrant_int, pi32_1);
		quadrant_int = _mm256_and_si256(quadrant_int, pi32_inv1);
		y = _mm256_cvtepi32_ps(quadrant_int);

		__m256 dp1 = _mm256_set1_ps(REDUCE_COEFF1);
		__m256 dp2 = _mm256_set1_ps(REDUCE_COEFF2);
		__m256 dp3 = _mm256_set1_ps(REDUCE_COEFF3);

		x = _mm256_fmadd_ps(y, dp1, x);
		x = _mm256_fmadd_ps(y, dp2, x);
		x = _mm256_fmadd_ps(y, dp3, x);

		__m256 z = _mm256_mul_ps(x, x);

		__m256 cos_p = _mm256_set1_ps(COS_COEFF0);
		cos_p = _mm256_fmadd_ps(cos_p, z, _mm256_set1_ps(COS_COEFF1));
		cos_p = _mm256_fmadd_ps(cos_p, z, _mm256_set1_ps(COS_COEFF2));
		cos_p = _mm256_mul_ps(cos_p, z);
		cos_p = _mm256_mul_ps(cos_p, z);
		cos_p = _mm256_sub_ps(cos_p, _mm256_mul_ps(z, _mm256_set1_ps(HALF)));
		cos_p = _mm256_add_ps(cos_p, _mm256_set1_ps(ONE));

		__m256 sin_p = _mm256_set1_ps(SIN_COEFF0);
		sin_p = _mm256_fmadd_ps(sin_p, z, _mm256_set1_ps(SIN_COEFF1));
		sin_p = _mm256_fmadd_ps(sin_p, z, _mm256_set1_ps(SIN_COEFF2));
		sin_p = _mm256_mul_ps(sin_p, z);
		sin_p = _mm256_fmadd_ps(sin_p, x, x);

		__m256i swap_cond = _mm256_and_si256(quadrant_int, pi32_2);
		swap_cond = _mm256_cmpeq_epi32(swap_cond, _mm256_setzero_si256());
		__m256 swap_mask = _mm256_castsi256_ps(swap_cond);

		__m256 sin_out = _mm256_blendv_ps(cos_p, sin_p, swap_mask);
		__m256 cos_out = _mm256_blendv_ps(sin_p, cos_p, swap_mask);

		__m256i sin_sign_flip = _mm256_and_si256(quadrant_int, pi32_4);
		sin_sign_flip = _mm256_slli_epi32(sin_sign_flip, SHIFT_29);

		__m256i cos_sign_flip = _mm256_add_epi32(quadrant_int, pi32_2);
		cos_sign_flip = _mm256_and_si256(cos_sign_flip, pi32_4);
		cos_sign_flip = _mm256_slli_epi32(cos_sign_flip, SHIFT_29);

		sin_out = _mm256_xor_ps(sin_out, _mm256_castsi256_ps(sin_sign_flip));
		sin_out = _mm256_xor_ps(sin_out, sign_bit_sin);

		cos_out = _mm256_xor_ps(cos_out, _mm256_castsi256_ps(cos_sign_flip));

		*sin_result = sin_out;
		*cos_result = cos_out;
	}

	[[msvc::forceinline]] static void invert_scale(float* data, std::size_t length)
	{
		const std::size_t total_floats = length * 2;
		const float inv = 1.0f / static_cast<float>(length);

		std::size_t i = 0;

		if constexpr (is_avx2_available)
		{
			while (i < total_floats && ((reinterpret_cast<std::uintptr_t>(data + i) & 0x1F) != 0))
			{
				data[i] *= inv;
				i++;
			}

			if (i + 8 <= total_floats)
			{
				__m256 v_inv = _mm256_set1_ps(inv);
				for (; i + 8 <= total_floats; i += 8)
				{
					__m256 v_data = _mm256_load_ps(data + i);
					__m256 v_result = _mm256_mul_ps(v_data, v_inv);
					_mm256_store_ps(data + i, v_result);
				}
			}
		}

		for (; i < total_floats; ++i)
		{
			data[i] *= inv;
		}
	}

	static constexpr bool is_prime(std::size_t num)
	{
		if (num <= 1)
		{
			return false;
		}

		if (num == 2 || num == 3)
		{
			return true;
		}

		if (num % 2 == 0)
		{
			return false;
		}

		for (std::size_t i = 3; i * i <= num; i += 2)
		{
			if (num % i == 0)
			{
				return false;
			}
		}

		return true;
	}

	static std::size_t power_mod(std::size_t base, std::size_t exp, std::size_t mod)
	{
		std::size_t res = 1;
		base %= mod;
		while (exp > 0)
		{
			if (exp % 2 == 1)
			{
				res = (res * base) % mod;
			}

			base = (base * base) % mod;
			exp /= 2;
		}
		return res;
	}

	static std::size_t find_primitive_root(std::size_t N)
	{
		if (N == 2)
		{
			return 1;
		}

		std::size_t phi = N - 1;
		std::size_t n = phi;

		std::size_t factors[32];
		std::size_t factor_count = 0;

		for (std::size_t i = 2; i * i <= n; ++i)
		{
			if (n % i == 0)
			{
				factors[factor_count++] = i;
				while (n % i == 0)
				{
					n /= i;
				}
			}
		}
		if (n > 1)
		{
			factors[factor_count++] = n;
		}

		for (std::size_t res = 2; res <= N; ++res)
		{
			bool ok = true;
			for (std::size_t i = 0; i < factor_count; ++i)
			{
				if (power_mod(res, phi / factors[i], N) == 1)
				{
					ok = false;
					break;
				}
			}

			if (ok)
			{
				return res;
			}
		}

		return 0;
	}
}

#endif