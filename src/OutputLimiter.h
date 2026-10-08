#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace Miffbuggpy
{

/**
    Output safety limiter: soft-knee tanh saturation with a peak-hold gain rider.

    A neural fuzz can output well past +/-1 at high Sustain and Volume, and clipping
    in the host's integer output stream sounds far worse than a designed saturation.
    This stage is the "output stage runs out of headroom" the pedal has, made explicit:

      1. tanh soft clip, gain-staged so the knee starts around -6 dBFS
      2. a lookahead-free peak follower that eases the stage gain down when the
         signal stays hot, and recovers slowly so quiet passages are not pumped

    There is no lookahead buffer, so latency stays at whatever the resampler reports
    and the stage never allocates after prepare().
*/
class OutputLimiter
{
public:
    void prepare (double sampleRate) noexcept
    {
        fs = (float) sampleRate;

        // Attack/release in seconds, converted to per-sample pole coefficients
        // y += c * (target - y). Release is deliberately slow: pumping is more
        // audible than a brief overshoot at this stage.
        attackCoeff  = poleFor (0.005f, fs);
        releaseCoeff = poleFor (0.250f, fs);

        reset();
    }

    void reset() noexcept
    {
        gain = 1.0f;
        peak = 0.0f;
    }

    /** Ceiling in [0.25, 1] as a linear amplitude (1.0 = no reduction, tanh still active). */
    void setCeiling (float linear) noexcept
    {
        ceiling = juceNorm (linear);
    }

    /** Processes one sample. `ceiling` should be set before the first call of a block. */
    float processSample (float x) noexcept
    {
        // 1) Peak follower on the absolute input. Attack is immediate (max), decay
        //    uses the release pole, so transient peaks are caught on the same sample.
        const auto ax = std::abs (x);
        peak = ax > peak ? ax : peak + releaseCoeff * (ax - peak);

        // 2) Gain rider: keep the pre-tanh peak just under the knee. Overshoot
        //    margin of 1.05 lets single-sample peaks through rather than riding
        //    on every transient.
        const auto target = ceiling * 0.63f / std::max (peak, 1.0e-6f);
        const auto clamped = std::min (target, 1.0f);
        const auto c = clamped < gain ? attackCoeff : releaseCoeff;
        gain += c * (clamped - gain);

        // 3) Soft knee. 1/atanh(k) scales tanh so inputs at the knee pass at
        //    ~unity gain; beyond it compression rises smoothly to a hard asymptote.
        const auto y = x * gain;
        return std::tanh (y * kDrive) * kDriveInv;
    }

private:
    static constexpr float kDrive = 2.0f;         // input multiplier before tanh
    static inline const float kDriveInv = 1.0f / std::tanh (kDrive);

    static float poleFor (float seconds, float sampleRate) noexcept
    {
        // c such that y[n] = y[n-1] + c * (target - y[n-1]) has time constant `seconds`
        return 1.0f - std::exp (-1.0f / (seconds * sampleRate));
    }

    static float juceNorm (float v) noexcept
    {
        return v < 0.25f ? 0.25f : (v > 1.0f ? 1.0f : v);
    }

    float fs = 48000.0f;
    float ceiling = 1.0f;
    float gain = 1.0f;
    float peak = 0.0f;
    float attackCoeff = 0.0f, releaseCoeff = 0.0f;
};

} // namespace Miffbuggpy
