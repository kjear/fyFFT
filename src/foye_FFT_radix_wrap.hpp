#ifndef _FOYE_FFT_RADIX_WRAP_HPP_
#define _FOYE_FFT_RADIX_WRAP_HPP_

namespace fy::fft
{
	struct smooth_path_invoker
	{
		using execute_function_type = void(*)(const float*, float*);
		using functions_map_type = std::unordered_map<std::size_t, execute_function_type>;

		using radix2_invoker = fy::fft::internal_radix2::FFT1D_C2C_radix2_invoker;
		using radix3_invoker = fy::fft::internal_radix3::FFT1D_C2C_radix3_invoker;
		using radix4_invoker = fy::fft::internal_radix4::FFT1D_C2C_radix4_invoker;
		using radix5_invoker = fy::fft::internal_radix5::FFT1D_C2C_radix5_invoker;
		using radix7_invoker = fy::fft::internal_radix7::FFT1D_C2C_radix7_invoker;

		smooth_path_invoker()
		{
			smooth_path_invoker::append<radix2_invoker, available_radix2_kernel_sizes, false>(forward_map);
			smooth_path_invoker::append<radix4_invoker, available_radix4_kernel_sizes, false>(forward_map);
			smooth_path_invoker::append<radix3_invoker, available_radix3_kernel_sizes, false>(forward_map);
			smooth_path_invoker::append<radix5_invoker, available_radix5_kernel_sizes, false>(forward_map);
			smooth_path_invoker::append<radix7_invoker, available_radix7_kernel_sizes, false>(forward_map);

			smooth_path_invoker::append<radix2_invoker, available_radix2_kernel_sizes, true>(backward_map);
			smooth_path_invoker::append<radix4_invoker, available_radix4_kernel_sizes, true>(backward_map);
			smooth_path_invoker::append<radix3_invoker, available_radix3_kernel_sizes, true>(backward_map);
			smooth_path_invoker::append<radix5_invoker, available_radix5_kernel_sizes, true>(backward_map);
			smooth_path_invoker::append<radix7_invoker, available_radix7_kernel_sizes, true>(backward_map);
		}

		template<bool invert>
		static execute_function_type kernel(std::size_t length)
		{
			const smooth_path_invoker& instance = smooth_path_invoker::acquired();
			const functions_map_type& map = invert ? instance.backward_map : instance.forward_map;
			auto it = map.find(length);
			return (it != map.end()) ? it->second : nullptr;
		}

		static constexpr bool is_directly_supported(std::size_t length)
		{
			return std::binary_search(available_smooth_length.begin(), available_smooth_length.end(), length);
		}

		template<bool invert>
		static bool dispatch(const float* input, float* output, std::size_t length)
		{
			const smooth_path_invoker& instance = smooth_path_invoker::acquired();
			const functions_map_type& map = invert ? instance.backward_map : instance.forward_map;
			auto it = map.find(length);
			if (it != map.end())
			{
				it->second(input, output);
				return true;
			}

			return false;
		}

		template<bool invert>
		static bool available(std::size_t length)
		{
			const smooth_path_invoker& instance = smooth_path_invoker::acquired();
			const functions_map_type& map = invert ? instance.backward_map : instance.forward_map;
			return map.contains(length);
		}

		static bool available(std::size_t length, bool invert)
		{
			if (invert) return smooth_path_invoker::available<true>(length);
			else        return smooth_path_invoker::available<false>(length);
		}

		static std::size_t next_smooth(std::size_t length)
		{
			auto it = std::lower_bound(available_smooth_length.begin(), available_smooth_length.end(), length);
			return (it == available_smooth_length.end()) ? 0 : *it;
		}

		static const smooth_path_invoker& acquired()
		{
			static smooth_path_invoker instance;
			return instance;
		}

		static constexpr std::array<std::size_t, 16> available_radix2_kernel_sizes = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
		static constexpr std::array<std::size_t, 10> available_radix3_kernel_sizes = { 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049 };
		static constexpr std::array<std::size_t, 8> available_radix4_kernel_sizes = { 4, 16, 64, 256, 1024, 4096, 16384, 65536 };
		static constexpr std::array<std::size_t, 7> available_radix5_kernel_sizes = { 5, 25, 125, 625, 3125, 15625, 78125 };
		static constexpr std::array<std::size_t, 5> available_radix7_kernel_sizes = { 7, 49, 343, 2401, 16807 };

		static constexpr std::array<std::size_t, 38> available_smooth_length = {
			2, 3, 4, 5, 7, 8, 9, 16, 25, 27, 32, 49, 64, 81,
			125, 128, 243, 256, 343, 512, 625, 729, 1024,
			2048, 2187, 2401, 3125, 4096, 6561, 8192, 15625,
			16384, 16807, 19683, 32768, 59049, 65536, 78125
		};

	private:
		functions_map_type forward_map;
		functions_map_type backward_map;

		template<typename Invoker, const auto& Sizes, bool Invert, std::size_t... Is>
		static void append_impl(functions_map_type& map, std::index_sequence<Is...>)
		{
			(map.insert_or_assign(std::size_t{ Sizes[Is] },
				[](const float* in, float* out) { Invoker{}.template dispatch<Invert>(in, out, Sizes[Is]); }), ...);
		}

		template<typename Invoker, const auto& Sizes, bool Invert>
		static void append(functions_map_type& map)
		{
			append_impl<Invoker, Sizes, Invert>(map, std::make_index_sequence<Sizes.size()>{});
		}
	};

	template<bool invert>
	static bool smooth_path_dispatch(const float* input, float* output, std::size_t length)
	{
		return invert
			? smooth_path_invoker::dispatch<true>(input, output, length)
			: smooth_path_invoker::dispatch<false>(input, output, length);
	}
}

#endif