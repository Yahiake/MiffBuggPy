#pragma once
#define NOMINMAX
#include "CDSPResampler.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace Miffbuggpy
{

/**
    host rate -> 48 kHz -> (model step per sample) -> host rate, mono.
    All allocation happens in prepare(); process() is allocation-free.
*/
class FixedRateResampler
{
public:
    static constexpr double kModelRate = 48000.0;
    static constexpr int kChunk = 256;                 // max host samples per r8brain call
    static constexpr int kRingSize = 1 << 15;          // must be a power of two
    static constexpr int kRingMask = kRingSize - 1;

    void prepare (double hostRate)
    {
        hostRateStored = hostRate;
        maxModelBlock = (int) std::ceil ((double) kChunk * kModelRate / hostRate) + 64;

        toModel = std::make_unique<r8b::CDSPResampler24> (hostRate, kModelRate, kChunk);
        toHost  = std::make_unique<r8b::CDSPResampler24> (kModelRate, hostRate, maxModelBlock);

        hostBuf.assign ((size_t) kChunk, 0.0);
        modelBuf.assign ((size_t) maxModelBlock, 0.0);
        ring.assign ((size_t) kRingSize, 0.0f);

        primeZeros = measureWorstCaseDeficit() + 16;
        reset();
    }

    void release()
    {
        toModel.reset();
        toHost.reset();
    }

    void reset()
    {
        if (toModel != nullptr) toModel->clear();
        if (toHost != nullptr)  toHost->clear();

        std::fill (ring.begin(), ring.end(), 0.0f);
        rd = 0;
        wr = primeZeros;       // FIFO starts holding `primeZeros` zeros
        count = primeZeros;
        underruns = 0;
    }

    /** Host-rate latency to report with setLatencySamples(). */
    int getLatencySamples() const noexcept { return primeZeros; }

    /** Should stay 0. Useful in a debug build / test. */
    int getUnderrunCount() const noexcept { return underruns; }

    /** `in` and `out` may be the same pointer. `step` runs the model at 48 kHz: float -> float. */
    template <typename Step>
    void process (const float* in, float* out, int numSamples, Step&& step)
    {
        int done = 0;

        while (done < numSamples)
        {
            const int n = std::min (kChunk, numSamples - done);

            for (int i = 0; i < n; ++i)
                hostBuf[(size_t) i] = in != nullptr ? (double) in[done + i] : 0.0;

            double* op = nullptr;
            const int m = std::min (toModel->process (hostBuf.data(), n, op), maxModelBlock);

            if (m > 0)
            {
                for (int i = 0; i < m; ++i)
                    modelBuf[(size_t) i] = (double) step ((float) op[i]);

                double* hp = nullptr;
                const int k = toHost->process (modelBuf.data(), m, hp);
                pushHost (hp, k);
            }

            for (int i = 0; i < n; ++i)
                out[done + i] = popHost();

            done += n;
        }
    }

private:
    void pushHost (const double* p, int k) noexcept
    {
        for (int i = 0; i < k; ++i)
        {
            if (count >= kRingSize) { rd = (rd + 1) & kRingMask; --count; }   // drop oldest
            ring[(size_t) wr] = (float) p[i];
            wr = (wr + 1) & kRingMask;
            ++count;
        }
    }

    float popHost() noexcept
    {
        if (count == 0) { ++underruns; return 0.0f; }
        const float v = ring[(size_t) rd];
        rd = (rd + 1) & kRingMask;
        --count;
        return v;
    }

    /** Pushes 1 s of zeros through both stages and returns max (consumed - produced). */
    int measureWorstCaseDeficit()
    {
        toModel->clear();
        toHost->clear();

        long long consumed = 0, produced = 0, worst = 0;
        const long long total = (long long) hostRateStored;   // ~1 second

        while (consumed < total)
        {
            const int n = 64;
            std::fill (hostBuf.begin(), hostBuf.begin() + n, 0.0);
            consumed += n;

            double* op = nullptr;
            const int m = std::min (toModel->process (hostBuf.data(), n, op), maxModelBlock);

            if (m > 0)
            {
                std::fill (modelBuf.begin(), modelBuf.begin() + m, 0.0);
                double* hp = nullptr;
                produced += toHost->process (modelBuf.data(), m, hp);
            }

            worst = std::max (worst, consumed - produced);
        }

        return (int) worst;
    }

    std::unique_ptr<r8b::CDSPResampler24> toModel, toHost;
    std::vector<double> hostBuf, modelBuf;
    std::vector<float> ring;

    double hostRateStored = 48000.0;
    int maxModelBlock = 0;
    int primeZeros = 0;
    int rd = 0, wr = 0, count = 0, underruns = 0;
};

} // namespace Miffbuggpy
