#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <RTNeural.h>

#include <array>
#include <atomic>

namespace Miffbuggpy
{

// ===========================================================================
// Network geometry.
//
// These mirror model.json exactly and are checked against the embedded file at
// construction time, so a retrained network with a different shape fails loudly
// at load instead of silently reading past the end of a buffer.
// ===========================================================================
inline constexpr int kNumKnobs = 3;
inline constexpr int kHiddenSize = 128;

/** One audio sample plus one value per knob. */
inline constexpr int kInputSize = 1 + kNumKnobs;
inline constexpr int kOutputSize = 1;

/** Where every knob starts, and what a freshly loaded preset means by "default". */
inline constexpr float kDefaultKnob = 0.5f;

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

// The exported model's metadata declares its conditioning order as
// ["Volume", "Sustain", "Tone"], and the training code wrote column order to
// match. Feeding them in the UI order instead would silently produce a model
// whose knobs are cross-wired, which is exactly the kind of bug that survives a
// clean ESR number because the network still fits the training data.
//
//   feature slot 0 = audio
//   feature slot 1 = Volume
//   feature slot 2 = Sustain
//   feature slot 3 = Tone
inline constexpr std::array<int, kNumParams> kFeatureSlot { 2, 3, 1 };

static_assert (kInputSize == 1 + kNumKnobs, "Feature vector must be audio plus one slot per knob.");

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

private:
    // -- network ------------------------------------------------------------
    /** Declares the three knobs, in the order they appear on the pedal. */
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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginProcessor)
};

} // namespace Miffbuggpy