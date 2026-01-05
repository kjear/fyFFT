#include <random>
#include <malloc.h>

#include "foye_FFT.hpp"

int main()
{
    using cf32 = std::complex<float>;

    std::size_t length = 1024;

    cf32* input = reinterpret_cast<cf32*>(_aligned_malloc(sizeof(cf32) * length, 32));
    cf32* result_for_FFT = reinterpret_cast<cf32*>(_aligned_malloc(sizeof(cf32) * length, 32));
    cf32* result_for_IFFT = reinterpret_cast<cf32*>(_aligned_malloc(sizeof(cf32) * length, 32));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (std::size_t i = 0; i < length; ++i)
    {
        input[i] = { static_cast<float>(dis(gen)), static_cast<float>(dis(gen)) };
    }

    std::memset(result_for_FFT, 0, sizeof(cf32) * length);
    std::memset(result_for_IFFT, 0, sizeof(cf32) * length);

    fy::fft::FFT1D_C2C_invoker invoker{ length };

    invoker.template dispatch<false>(
        reinterpret_cast<float*>(input),
        reinterpret_cast<float*>(result_for_FFT));

    invoker.template dispatch<false>(
        reinterpret_cast<float*>(input),
        reinterpret_cast<float*>(result_for_IFFT));

    return 0;
}