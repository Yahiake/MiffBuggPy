#pragma once

#include "Plugin.h"

namespace Miffbuggpy
{

// ===========================================================================
// Editor
//
// Three rotary knobs in the order they appear on the pedal, plus a status line.
// Deliberately no editor-side audio: the editor never touches the network, so
// what is measured is what a host runs.
// ===========================================================================
// Timer is mixed in rather than held as a member because startTimer/stopTimer are
// protected: a Timer owned by value is unusable from outside its own class.
class PluginEditor : public juce::AudioProcessorEditor,
                     private juce::Timer
{
public:
    explicit PluginEditor (PluginProcessor&);
    ~PluginEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // -- layout -------------------------------------------------------------
    static constexpr int kNumKnobsShown = 3;
    static constexpr int kMargin = 18;
    static constexpr int kHeaderHeight = 34;
    static constexpr int kStatusHeight = 46;
    static constexpr int kPreferredHeight = 300;
    static constexpr int kPreferredWidth = 420;

    void layoutKnobs();

    // -- content ------------------------------------------------------------
    PluginProcessor& processor;

    std::array<juce::Slider, kNumKnobsShown> knobs;
    std::array<juce::Label, kNumKnobsShown> knobLabels;

    /** Held so the knobs stay bound to the processor's parameters. */
    std::array<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>, kNumKnobsShown> knobAttachment;

    juce::Label title;
    juce::Label status;

    /** Repaints the status line without touching parameters. */
    void timerCallback() override;
    void refreshStatus();

    /**
        A dark, pedal-style knob: body, pointer, and a value arc.

        LookAndFeel_V4 rather than LookAndFeel, because LookAndFeel itself declares
        several methods pure virtual (lasso, level meter, ticks, focus outline and
        others). Deriving from it directly produces an abstract class that cannot be
        instantiated; V4 is the concrete base intended for this.
    */
    struct KnobLook : juce::LookAndFeel_V4
    {
        KnobLook();

        void drawRotarySlider (juce::Graphics&, int x, int y, int width, int height,
                               float sliderPosProportional, float rotaryStartAngle,
                               float rotaryEndAngle, juce::Slider&) override;
    } knobLook;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};

} // namespace Miffbuggpy