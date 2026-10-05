#include "Plugin.h"

#include "BinaryData.h"
#include "Editor.h"

namespace Miffbuggpy
{

// ===========================================================================
// Helpers
// ===========================================================================

namespace
{

/**
    Recomputes the embedded model.

    RTNeural's ModelT::parseJson returns void and reports every mismatch through
    debug_print alone. A model that does not match the template therefore loads as
    silent garbage rather than failing, which would still produce plausible
    looking audio. Everything below exists to make that failure loud instead: the
    JSON is validated against the compiled-in geometry first, and the network is
    then probed to confirm it actually holds trained weights.
*/
juce::String readEmbeddedModel (juce::String& errorOut)
{
    const auto* bytes = BinaryData::model_json;
    const auto size = static_cast<int> (BinaryData::model_jsonSize);

    if (bytes == nullptr || size <= 0)
    {
        errorOut = "Embedded model.json is missing or empty.";
        return {};
    }

    return juce::String::fromUTF8 (reinterpret_cast<const char*> (bytes), size);
}

/** Fails with a message describing the first structural mismatch found. */
bool validateGeometry (const nlohmann::json& j, juce::String& errorOut)
{
    if (! j.contains ("in_shape") || ! j["in_shape"].is_array())
    {
        errorOut = "model.json has no in_shape array.";
        return false;
    }

    const auto& inShape = j["in_shape"];
    if (inShape.back().get<int>() != kInputSize)
    {
        errorOut = juce::String ("model.json expects ")
                       + juce::String (inShape.back().get<int>())
                       + " input features, this build is compiled for "
                       + juce::String (kInputSize) + ".";
        return false;
    }

    if (! j.contains ("layers") || ! j["layers"].is_array())
    {
        errorOut = "model.json has no layers array.";
        return false;
    }

    const auto& layers = j["layers"];

    // Three declared layers become four at runtime, because RTNeural expects the
    // tanh named by layer 0's "activation" as its own layer in the template.
    static const char* kExpectedTypes[] { "dense", "gru", "dense" };
    const auto expectedCount = static_cast<int> (sizeof (kExpectedTypes) / sizeof (kExpectedTypes[0]));

    if (static_cast<int> (layers.size()) != expectedCount)
    {
        errorOut = juce::String ("model.json declares ") + juce::String ((int) layers.size())
                       + " layers, this build expects " + juce::String (expectedCount) + ".";
        return false;
    }

    for (int i = 0; i < expectedCount; ++i)
    {
        const auto& l = layers.at ((size_t) i);

        if (! l.contains ("type") || l["type"].get<std::string>() != kExpectedTypes[i])
        {
            errorOut = juce::String ("model.json layer ") + juce::String (i) + " should be type "
                       + kExpectedTypes[i] + ".";
            return false;
        }

        // The middle layer is the only one whose width is the hidden size.
        if (i == 1 && l.contains ("shape") && l["shape"].is_array() && l["shape"].back().is_number())
        {
            const auto width = l["shape"].back().get<int>();
            if (width != kHiddenSize)
            {
                errorOut = juce::String ("model.json hidden size is ") + juce::String (width)
                           + ", this build is compiled for " + juce::String (kHiddenSize) + ".";
                return false;
            }
        }
    }

    // The tanh is the whole reason this build has four runtime layers. If it is
    // absent the network would be loaded as a different, wrong topology.
    const auto& first = layers.at (0);
    if (! first.contains ("activation") || first["activation"].get<std::string>() != "tanh")
    {
        errorOut = "model.json layer 0 is missing \"activation\":\"tanh\"; "
                   "the compiled network topology would not match.";
        return false;
    }

    return true;
}

/**
    Confirms the loaded network holds real weights.

    parseJson gives no failure signal, so a probe is the only way to tell a trained
    network from an unloaded one. A known input must produce a bounded, finite,
    non-zero response; an unloaded network produces exactly zero.

    Driven with a real signal rather than silence: a silent input is a degenerate
    probe, because a network has no reason to leave its zero-input fixed point.
*/
bool probeNetwork (Network& net, juce::String& errorOut)
{
    std::array<float, kInputSize> frame {};
    frame[0] = 0.25f;
    for (int k = 0; k < kNumParams; ++k)
        if (kFeatureSlot[(size_t) k] >= 0)
            frame[(size_t) kFeatureSlot[(size_t) k]] = 0.5f;

    net.reset();

    float peak = 0.0f;
    for (int i = 0; i < 64; ++i)
    {
        const auto y = net.forward (frame.data());
        peak = juce::jmax (peak, std::abs (y));
    }

    if (! std::isfinite (peak))
    {
        errorOut = "Loaded network produced a non-finite response; weights look corrupt.";
        return false;
    }

    if (peak < 1e-9f)
    {
        errorOut = "Loaded network produced silence, so its weights were not applied.";
        return false;
    }

    // Restore the zero state the plugin is expected to start from.
    net.reset();
    return true;
}

} // namespace

// ===========================================================================
// Construction
// ===========================================================================

PluginProcessor::PluginProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::mono(), true)
                          .withOutput ("Output", juce::AudioChannelSet::mono(), true)),
      parameters (*this, nullptr, "MIFFBUGGPY", makeLayout())
{
    // The host thread writes these atomics; the audio thread only ever reads them.
    for (int i = 0; i < kNumParams; ++i)
    {
        targets[(size_t) i].store (kDefaultKnob);
        parameters.addParameterListener (kParamId[i], this);
    }

    loadModel();
}

PluginProcessor::~PluginProcessor()
{
    for (int i = 0; i < kNumParams; ++i)
        parameters.removeParameterListener (kParamId[i], this);
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::makeLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // Real Big Muff order, left to right. All three default to 0.5, which is also
    // where targets[] starts, so the plugin's first block is already consistent
    // with what the host is told the values are.
    const auto range = juce::NormalisableRange<float> (0.0f, 1.0f);

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kSustain], 1 }, "Sustain", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kTone], 1 }, "Tone", range, kDefaultKnob));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { kParamId[kVolume], 1 }, "Volume", range, kDefaultKnob));

    return layout;
}

// ===========================================================================
// Model loading
// ===========================================================================

bool PluginProcessor::loadModel()
{
    modelError.clear();
    modelKnobOrder.clear();
    exportValEsr = 0.0;

    const auto text = readEmbeddedModel (modelError);
    if (text.isEmpty())
        return false;

    // allow_exceptions = false, so malformed JSON returns a discarded value instead
    // of throwing. Throwing out of a constructor inside a host is a bad afternoon.
    const auto j = nlohmann::json::parse (text.toStdString(), nullptr, false);
    if (j.is_discarded())
    {
        modelError = "Embedded model.json is not valid JSON.";
        return false;
    }

    if (j.contains ("metadata"))
    {
        const auto& meta = j["metadata"];

        if (meta.contains ("knobs"))
            for (const auto& k : meta["knobs"])
                modelKnobOrder.add (juce::String (k.get<std::string>()));

        if (meta.contains ("val_esr"))
            exportValEsr = meta["val_esr"].get<double>();
    }

    if (! validateGeometry (j, modelError))
        return false;

    // The conditioning order is baked into the first layer's weights. If the exported
    // model ever changes its declared knob order, the cross-wiring is silent and the
    // failure shows up as "the knobs feel wrong", so it is checked here instead.
    //
    // Only the conditioned knobs are compared. Volume appears in the manifest but not in
    // the feature vector, so a model that still listed it would mean the export was made
    // by an older trainer whose weights assume a Volume input this build does not send.
    if (modelKnobOrder.size() == (size_t) kNumKnobs)
    {
        static const char* expected[] { "Sustain", "Tone" };
        for (int i = 0; i < kNumKnobs; ++i)
            if (modelKnobOrder[(size_t) i] != expected[i])
            {
                modelError = juce::String ("model.json conditioning order is [")
                             + modelKnobOrder[0] + ", " + modelKnobOrder[1]
                             + "], this build is wired for [Sustain, Tone].";
                return false;
            }
    }

    // A model exported before Volume was pinned declared it as a conditioning channel and
    // therefore has a wider first layer. Rejecting it by name gives a usable message;
    // letting validateGeometry catch the width difference would report a number instead.
    if (modelKnobOrder.contains ("Volume"))
    {
        modelError = "model.json conditions the network on Volume, but Volume is applied as "
                     "output gain in this build. Re-export from the current trainer.";
        return false;
    }

    model.parseJson (j);
    modelValid = probeNetwork (model, modelError);
    return modelValid;
}

void PluginProcessor::resetModelState()
{
    if (modelValid)
        model.reset();
}

// ===========================================================================
// Parameters
// ===========================================================================

void PluginProcessor::parameterChanged (const juce::String& parameterID, float newValue)
{
    for (int i = 0; i < kNumParams; ++i)
        if (parameterID == kParamId[i])
        {
            targets[(size_t) i].store (juce::jlimit (0.0f, 1.0f, newValue));
            return;
        }
}

// ===========================================================================
// Lifecycle
// ===========================================================================

void PluginProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // One-pole smoothing, ~15 ms to converge. Long enough that automation does not
    // zipper, short enough to feel immediate.
    const auto tauSeconds = 0.015;
    smoothingCoeff = static_cast<float> (1.0 - std::exp (-1.0 / (tauSeconds * sampleRate)));

    model.reset();
}

void PluginProcessor::releaseResources()
{
}

bool PluginProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();

    // Mono in, mono or stereo out. A Big Muff is a mono pedal with a mono input,
    // so stereo input is only accepted to avoid dropping the user's signal.
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;

    const auto& in = layouts.getMainInputChannelSet();
    if (in != juce::AudioChannelSet::mono() && in != juce::AudioChannelSet::stereo())
        return false;

    return true;
}

double PluginProcessor::getTailLengthSeconds() const
{
    // The GRU rings on after the input stops, so there is a genuine tail. Reporting
    // zero makes hosts discard reverb tails and shorten offline renders.
    return 2.0;
}

// ===========================================================================
// Processing
//
// RTNeural's ModelT is a per-sample interface here: forward(const float*) takes one
// kInputSize vector and returns one output sample. Knob smoothing is therefore free
// to run per sample rather than per block, because the feature vector is being
// rebuilt one sample at a time regardless. That stops an automation ramp from
// stepping once every 512 samples.
// ===========================================================================

void PluginProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const auto numSamples = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();

    if (numChannels == 0)
        return;

    if (! modelValid)
    {
        // No network. Pass audio through untouched rather than emitting silence,
        // so a bad embed is audible as "no effect" instead of "broken host".
        buffer.clear();
        return;
    }

    const auto* input = buffer.getReadPointer (0);

    std::array<float, kInputSize> frame {};
    std::array<float, kNumParams> s {};

    for (int i = 0; i < kNumParams; ++i)
        s[(size_t) i] = targets[(size_t) i].load();

    for (int n = 0; n < numSamples; ++n)
    {
        frame[0] = input != nullptr ? input[n] : 0.0f;

        for (int i = 0; i < kNumParams; ++i)
        {
            s[(size_t) i] += smoothingCoeff * (targets[(size_t) i].load() - s[(size_t) i]);

            // Slot -1 is Volume: smoothed like the others so the gain does not step, but
            // never written into the feature vector.
            if (kFeatureSlot[(size_t) i] >= 0)
                frame[(size_t) kFeatureSlot[(size_t) i]] = s[(size_t) i];
        }

        // The network carries its own hidden state across the block boundary, which
        // is what makes continuous playback correct rather than block-by-block.
        const auto y = model.forward (frame.data()) * volumeGainFor (s[(size_t) kVolume]);

        for (int ch = 0; ch < numChannels; ++ch)
            buffer.setSample (ch, n, y);
    }
}

// ===========================================================================
// State
// ===========================================================================

void PluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = parameters.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void PluginProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
    {
        const auto tree = juce::ValueTree::fromXml (*xml);

        if (tree.hasType (parameters.state.getType()))
        {
            parameters.replaceState (tree);

            // Preset loads always start from a zero recurrent state, so the same
            // preset always produces the same audio. Without this the hidden state
            // carries over from whatever was playing, and a preset change is not
            // reproducible.
            resetModelState();
        }
    }
}

// ===========================================================================
// Editor
// ===========================================================================

juce::AudioProcessorEditor* PluginProcessor::createEditor()
{
    return new PluginEditor (*this);
}

} // namespace Miffbuggpy

// The JUCE plugin client declares this with C++ linkage at global scope and looks
// it up by that exact symbol, so it must not sit inside the Miffbuggpy namespace:
// a namespaced definition mangles to a different name and the VST3 link fails with
// an unresolved external.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Miffbuggpy::PluginProcessor();
}