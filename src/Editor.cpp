#include "Editor.h"

namespace Miffbuggpy
{

namespace
{
// Not constexpr: juce::Colour's constructor is not a constant expression, so a
// constexpr instance would be rejected. inline const avoids the static init
// order problem these have across translation units.
inline const juce::Colour kPanel { 0xff1c1e20 };
inline const juce::Colour kPanelEdge { 0xff000000 };
inline const juce::Colour kText { 0xffd8d4c8 };
inline const juce::Colour kTextDim { 0xff8a8578 };
inline const juce::Colour kKnobBody { 0xff2b2e31 };
inline const juce::Colour kKnobPointer { 0xffc8a24a };
inline const juce::Colour kArc { 0xff7d8f6a };
inline const juce::Colour kArcOff { 0xff33363a };
inline const juce::Colour kBad { 0xffd4705a };

/**
    Maps a knob slot onto the parameter ID the processor registered.

    The slots are the pedal's physical left-to-right order, which is not the
    processor's enum order for Volume, so this switch is the one place the two
    orders are reconciled. Going through the enum keeps the two in sync if the
    parameter list ever grows.
*/
const char* paramIdFor (int i)
{
    switch (i)
    {
        case kSustain: return "sustain";
        case kTone: return "tone";
        default: return "volume";
    }
}
} // namespace

// ===========================================================================
// KnobLook
// ===========================================================================

PluginEditor::KnobLook::KnobLook()
{
    setColour (juce::ResizableWindow::backgroundColourId, juce::Colours::transparentBlack);
}

void PluginEditor::KnobLook::drawRotarySlider (juce::Graphics& g,
                                               int x, int y, int width, int height,
                                               float sliderPosProportional,
                                               float rotaryStartAngle,
                                               float rotaryEndAngle,
                                               juce::Slider&)
{
    const auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat().reduced (4.0f);
    const auto centre = bounds.getCentre();
    const auto radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f - 6.0f;
    if (radius <= 2.0f)
        return;

    const auto thickness = juce::jmax (2.5f, radius * 0.16f);
    const auto angle = rotaryStartAngle
                     + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

    // Unfilled portion of the arc first, then the filled part over the top.
    juce::Path track;
    track.addCentredArc (centre.x, centre.y, radius - thickness * 0.5f,
                         radius - thickness * 0.5f, 0.0f,
                         rotaryStartAngle, rotaryEndAngle, true);
    g.setColour (kArcOff);
    g.strokePath (track, juce::PathStrokeType (thickness, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    if (sliderPosProportional > 0.0f)
    {
        juce::Path fill;
        fill.addCentredArc (centre.x, centre.y, radius - thickness * 0.5f,
                            radius - thickness * 0.5f, 0.0f,
                            rotaryStartAngle, angle, true);
        g.setColour (kArc);
        g.strokePath (fill, juce::PathStrokeType (thickness, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    // Body
    g.setColour (kKnobBody);
    g.fillEllipse (centre.x - radius * 0.78f, centre.y - radius * 0.78f,
                   radius * 1.56f, radius * 1.56f);
    g.setColour (kPanelEdge);
    g.drawEllipse (centre.x - radius * 0.78f, centre.y - radius * 0.78f,
                   radius * 1.56f, radius * 1.56f, 1.0f);

    // Pointer
    juce::Path pointer;
    const auto pointerLength = radius * 0.62f;
    const auto pointerWidth = juce::jmax (1.5f, radius * 0.10f);
    pointer.addRoundedRectangle (-pointerWidth * 0.5f, -pointerLength,
                                 pointerWidth, pointerLength * 0.8f,
                                 pointerWidth * 0.5f);
    pointer.applyTransform (juce::AffineTransform::rotation (angle).translated (centre));
    g.setColour (kKnobPointer);
    g.fillPath (pointer);
}

// ===========================================================================
// Editor
// ===========================================================================

PluginEditor::PluginEditor (PluginProcessor& p)
    : AudioProcessorEditor (&p), processor (p)
{
    static const char* kNames[kNumKnobsShown] { "SUSTAIN", "TONE", "VOLUME" };

    for (int i = 0; i < kNumKnobsShown; ++i)
    {
        auto& knob = knobs[(size_t) i];

        knob.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        knob.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 70, 18);
        knob.setRange (0.0, 1.0, 0.0);
        knob.setDoubleClickReturnValue (true, 0.5);
        knob.setLookAndFeel (&knobLook);

        auto attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
            processor.getState(),
            juce::String (paramIdFor (i)),
            knob);
        knobAttachment[i] = std::move (attachment);

        knob.setColour (juce::Slider::textBoxTextColourId, kText);
        knob.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        knob.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);

        knobLabels[(size_t) i].setText (kNames[i], juce::dontSendNotification);
        knobLabels[(size_t) i].setJustificationType (juce::Justification::centred);
        knobLabels[(size_t) i].setColour (juce::Label::textColourId, kTextDim);
        knobLabels[(size_t) i].setFont (juce::Font (11.0f, juce::Font::bold));
        addAndMakeVisible (knobLabels[(size_t) i]);

        addAndMakeVisible (knob);
    }

    title.setText ("MIFFBUGGPY", juce::dontSendNotification);
    title.setJustificationType (juce::Justification::centredLeft);
    title.setColour (juce::Label::textColourId, kText);
    title.setFont (juce::Font (19.0f, juce::Font::bold));
    addAndMakeVisible (title);

    status.setJustificationType (juce::Justification::centredLeft);
    status.setFont (juce::Font (11.0f));
    addAndMakeVisible (status);

    refreshStatus();
    startTimer (500);

    setResizable (true, true);
    setResizeLimits (320, 240, 900, 700);
    setSize (kPreferredWidth, kPreferredHeight);
}

PluginEditor::~PluginEditor()
{
    stopTimer();

    for (auto& knob : knobs)
        knob.setLookAndFeel (nullptr);
}

void PluginEditor::paint (juce::Graphics& g)
{
    g.fillAll (kPanel);

    // Thin inner bevel so the panel does not read as a flat rectangle.
    g.setColour (kPanelEdge.withAlpha (0.6f));
    g.drawRect (getLocalBounds(), 1);
}

void PluginEditor::resized()
{
    auto area = getLocalBounds().reduced (kMargin);

    title.setBounds (area.removeFromTop (kHeaderHeight));
    area.removeFromTop (6);

    status.setBounds (area.removeFromBottom (kStatusHeight));
    area.removeFromBottom (4);

    layoutKnobs();
}

void PluginEditor::layoutKnobs()
{
    const auto area = getLocalBounds().reduced (kMargin).withTrimmedTop (kHeaderHeight + 6)
                          .withTrimmedBottom (kStatusHeight + 4)
                          .withTrimmedTop (6);

    const auto slot = area.getWidth() / (float) kNumKnobsShown;

    for (int i = 0; i < kNumKnobsShown; ++i)
    {
        auto slotArea = area.withX ((int) (slot * i)).withWidth ((int) slot);

        knobLabels[(size_t) i].setBounds (slotArea.removeFromBottom (22));

        // Square-ish knob area centred in the remaining space.
        const auto side = juce::jmin (slotArea.getWidth() - 12, slotArea.getHeight());
        knobs[(size_t) i].setBounds (slotArea.withSizeKeepingCentre (side, side));
    }
}

void PluginEditor::refreshStatus()
{
    if (! processor.isModelValid())
    {
        status.setColour (juce::Label::textColourId, kBad);
        status.setText ("MODEL FAILED TO LOAD\n" + processor.getModelError(), juce::dontSendNotification);
        return;
    }

    const auto order = processor.getModelKnobOrder();
    juce::String orderText;
    for (int i = 0; i < order.size(); ++i)
        orderText += (i > 0 ? juce::String (", ") : juce::String()) + order[i];

    // Volume is named explicitly rather than being absent from the line, because its
    // absence from `order` is a design decision and a reader of the UI has no way to
    // tell it apart from a knob that was simply forgotten.
    juce::String rateText;
    if (processor.isResamplingActive())
        rateText = juce::String (processor.getHostSampleRate(), 0) + " Hz (Resampled to 48 kHz)";
    else
        rateText = "48000 Hz (Native)";

    status.setColour (juce::Label::textColourId, kTextDim);
    status.setText ("Network sees: " + orderText
                      + juce::String ("\nVolume: output gain | Host: ") + rateText,
                    juce::dontSendNotification);
}

void PluginEditor::timerCallback()
{
    refreshStatus();
}

} // namespace Miffbuggpy
