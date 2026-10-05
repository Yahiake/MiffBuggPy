MIFFBUGGPY
==========
A neural emulation of an Electro-Harmonix Big Muff Pi fuzz pedal.

What this is
------------
MiffBuggPy is not a guitar pedal circuit. It is a neural network that has been
trained to imitate the behaviour of a LiveSPICE simulation of a Big Muff Pi
circuit, across all three of the pedal's controls at once.

The ground truth is simulation, not hardware. The reference is the SPICE model,
so when this plugin and the circuit model disagree, the model is right. No
physical Big Muff Pi was measured for this project.

What it is accurate to
----------------------
The network reproduces the LiveSPICE model of the circuit within a measured
error, in the region of knob and signal space covered by the training data.
Outside that region it extrapolates, and extrapolated behaviour is not
guaranteed to be musical. The Status readout in the plugin shows the validation
Error Signal Ratio the network achieved when it was exported.

Controls
--------
SUSTAIN  Fuzz amount, matching the pedal's own control.
TONE     The passive tone stack, bass to treble.
VOLUME   Output level, as on the pedal.

Bypassing is transparent: the plugin passes audio through unmodified.

Requirements
------------
Windows (x64), macOS (Universal: Apple Silicon arm64 & Intel x86_64), or Linux (x64).
VST3, Audio Unit (macOS), or Standalone application. No installation step,
no system drivers, no network access at runtime.

Licence
-------
Released under the GNU General Public License v3.0, which covers the
JUCE framework this plugin is built on. See LICENSE for the full text.