#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

#include <RTNeural.h>
#include "FixedRateResampler.h"

#include <array>
#include <atomic>
#include <vector>

namespace Miffbuggpy
{

// ===========================================================================
// Network geometry.
//
// These mirror model.json exactly and are checked against the embedded file at
// construction time, so a retrained network with a different shape fails loudly
// at load instead of silently reading past the end of a buffer.
// ===========================================================================
/**
    Knobs the network is conditioned on. Volume is NOT among them.

    Volume is a pure post-gain on this circuit, measured rather than assumed: holding
    Sustain and Tone fixed, rendering at Volume 0.4, 0.6 and 0.8 reproduced the Volume
    0.2 render times exactly 2, 3 and 4, with residual ESR of -180 dB at high Sustain
    and -85 dB at worst -- roughly 4000x finer than this model resolves. The render
    database therefore pins Volume at 0.2 and the network never sees it, and the plugin
    applies the measured law as gain on the output instead. Feeding Volume in as a
    constant channel would be worse than omitting it: a feature that never varies carries
    no information, so its weights would just sit at their initialisation.
*/
inline constexpr int kNumKnobs = 2;
inline constexpr int kHiddenSize = 128;

/** One audio sample plus one value per conditioned knob. */
inline constexpr int kInputSize = 1 + kNumKnobs;
inline constexpr int kOutputSize = 1;

/** Where every knob starts, and what a freshly loaded preset means by "default". */
inline constexpr float kDefaultKnob = 0.5f;

// ===========================================================================
// Volume.
//
// The pot law was measured on this netlist rather than assumed: with Sustain and Tone
// held fixed, output gain is 5 * Volume, so Volume 0.2 is unity and the knob spans 0 to 5.
// Applied literally, a plugin knob centred at 0.5 would sit at +8 dB, which is not what
// anyone expects from a knob that starts in the middle. So the law is renormalised to be
// unity at the default position:
//
//     gain(v) = kVolumeGainAtFull * v      ->  0 at v=0, 1 at v=0.5, 2 at v=1
//
// That is the measured curve divided by 2.5 throughout, so its shape is the circuit's and
// only its absolute scale is chosen. The +6 dB above centre is kept because the pedal does
// really get louder past unity, up to where the output stage runs out of headroom.
//
// Accuracy of treating Volume as a plain gain: worst case -85 dB, typically -180 dB. That
// error is systematic across every sample, which is why it is stated rather than hidden --
// it is far below the ~1% ESR the network operates at, but it is not zero.
// ===========================================================================
inline constexpr float kVolumeGainAtFull = 2.0f;

/** The gain to apply to the network output for a Volume knob position in [0, 1]. */
inline constexpr float volumeGainFor (float v) noexcept
{
    // Spelled with comparisons rather than juce::jlimit because jlimit is not constexpr,
    // and the clamp is written out here so the value cannot go negative or run past the
    // knob's range even if a host sends something outside [0, 1].
    return kVolumeGainAtFull * (v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v));
}

// ===========================================================================
// Parameters.
//
// Order here is the order the knobs appear on a real Big Muff Pi, left to
// right. It is deliberately NOT the order the network expects them in; see
// kFeatureSlot below.
// ===========================================================================
enum KnobParam
{
    kSustain = 0,
    kTone = 1,
    kVolume = 2,
    kNumParams = 3
};

// Which feature slot each UI knob is written into. -1 means the knob does not reach the
// network at all.
//
//   feature slot 0 = audio
//   feature slot 1 = Sustain
//   feature slot 2 = Tone
//   Volume (-1)    = applied as pure output gain
inline constexpr std::array<int, kNumParams> kFeatureSlot { 1, 2, -1 };

static_assert (kInputSize == 1 + kNumKnobs, "Feature vector must be audio plus one slot per knob.");
static_assert (kNumParams == kNumKnobs + 1,
               "There is one knob more than there are conditioned inputs: Volume is applied "
               "as gain and never reaches the network.");

// ===========================================================================
// The network.
//
// model.json declares three layers, but RTNeural needs four: layer 0 is a dense
// with "activation":"tanh", and ModelT::parseJson treats that as an assertion
// that the *next* template layer is the matching activation. Omitting the
// TanhActivationT therefore fails to load rather than loading a wrong network,
// which is the behaviour we want.
// ===========================================================================
using Network = RTNeural::ModelT<float,
                                 kInputSize,
                                 kOutputSize,
                                 RTNeural::DenseT<float, kInputSize, kHiddenSize>,
                                 RTNeural::TanhActivationT<float, kHiddenSize>,
                                 RTNeural::GRULayerT<float, kHiddenSize, kHiddenSize>,
                                 RTNeural::DenseT<float, kHiddenSize, kOutputSize>>;

// ===========================================================================
// AudioProcessor
// ===========================================================================
class PluginProcessor : public juce::AudioProcessor,
                        private juce::AudioProcessorValueTreeState::Listener
{
public:
    PluginProcessor();
    ~PluginProcessor() override;

    // --- lifecycle ---------------------------------------------------------
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    // --- processing --------------------------------------------------------
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // --- editor ------------------------------------------------------------
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    // --- identity ----------------------------------------------------------
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override;

    // --- programs ----------------------------------------------------------
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return "Default"; }
    void changeProgramName (int, const juce::String&) override {}

    // --- state -------------------------------------------------------------
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // --- parameters --------------------------------------------------------
    juce::AudioProcessorValueTreeState& getState() noexcept { return parameters; }

    /** True when the embedded network loaded and validated. */
    bool isModelValid() const noexcept { return modelValid; }

    /** Populated only when modelValid is false. */
    const juce::String& getModelError() const noexcept { return modelError; }

    /** Ground-truth validation ESR carried over from the export, for display. */
    double getExportValEsr() const noexcept { return exportValEsr; }

    /** Knob ordering the embedded model expects, for display and for tests. */
    const juce::StringArray& getModelKnobOrder() const noexcept { return modelKnobOrder; }

    /** Returns current host sample rate. */
    double getHostSampleRate() const noexcept { return currentHostRate; }

    /** True when host sample rate differs from the native 48 kHz model training rate. */
    bool isResamplingActive() const noexcept { return needsResampling; }

private:
    // -- network ------------------------------------------------------------
    /** Declares knobs and DSP parameters. */
    static juce::AudioProcessorValueTreeState::ParameterLayout makeLayout();

    /** Host thread: a parameter moved. Stores to an atomic for the audio thread. */
    void parameterChanged (const juce::String& parameterID, float newValue) override;

    /** Parses and validates the embedded model.json. Off the audio thread. */
    bool loadModel();

    /** Discards the network's recurrent state by reloading from the embedded source. */
    void resetModelState();

    Network model;
    bool modelValid = false;
    juce::String modelError;
    juce::StringArray modelKnobOrder;
    double exportValEsr = 0.0;

    // -- parameters ---------------------------------------------------------
    juce::AudioProcessorValueTreeState parameters;

    /** Target values, written by the host thread and read by the audio thread. */
    std::array<std::atomic<float>, kNumParams> targets { { kDefaultKnob, kDefaultKnob, kDefaultKnob } };

    float smoothingCoeff = 1.0f;

    static constexpr const char* kParamId[kNumParams] { "sustain", "tone", "volume" };

    // -- Resampling & transparent DC blocking ------------------------------
    juce::dsp::FirstOrderTPTFilter<float> dcBlocker;

    double currentHostRate = 48000.0;
    bool needsResampling = false;

	FixedRateResampler resampler;
    std::array<float, kNumParams> smoothedParams { { kDefaultKnob, kDefaultKnob, kDefaultKnob } };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginProcessor)
};

} // namespace Miffbuggpy