#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "DiodeClipper.h"

//==============================================================================
/**
    Parameter identifiers

    Parameters in JUCE are instances of the class "AudioProcessorParameter".
    There are different cubclasses for floating-point parameters, interger
    parameters, boolean parameters, and so on.

    Each Parameter needs to have a unique name that identifies it. In JUCE this
    identifier is a string, or more precisely, a "juce::String" object.

    As of JUCE 7, the full parameter identifier is an instance of
    "juce::ParameterID". This is a small class that combines the string
    identifier with a version number. This version number is necessary in
    particular for Audio Unit plug-ins. It ensures that you can safely add new
    backwards compatible with existing music projects.

    The following codes define a new namespace containing one "juce::ParameterID"
    object for each parameter definitions. To get the identifier for the "gain"
    parameter, for example, you can now write "MyParameterID::gain". The number
    "1" corresponds to the parameter version number.
*/
namespace MyParameterID
{
    #define PARAMETER_ID(str) const juce::ParameterID str(#str, 1);
    
    PARAMETER_ID(diode)
    PARAMETER_ID(config)
    
    PARAMETER_ID(drive)
    PARAMETER_ID(tone)
    PARAMETER_ID(mix)
    PARAMETER_ID(level)

    #undef PARAMETER_ID
}

//==============================================================================
class AudioPluginAudioProcessor  : public  juce::AudioProcessor,
                                   private juce::ValueTree::Listener
{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

private:
    //==============================================================================
    // JUCE has a powerful class called "ValueTree", a tree structure that can
    // contain objects of any type. The APVTS uses such a "ValueTree" to hold the
    // plug-in's parameters.
    // One of its useful features is that the "ValueTree" allows you to connect
    // listeners to nodes in the tree, for receiving notifications when something in
    // the tree changes. It can also serialize the tree to XML, supports undo/redo
    // functionality, automatically manages the lifetimes of nodes through reference
    // counting and more. It's usually added to the public section, so that the
    // editor class that class that handles the UI can also access it.
    juce::AudioProcessorValueTreeState apvts { *this, nullptr, "PARAMETERS", createParameterLayout() };

    //==============================================================================
    // "createParameterLayout()" will be called by APVTS. Inside this method is
    // where you will instantiate all the AudioParameter objects.
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    //==============================================================================
    // This method is used to grab the parameter with the identifier from APVTS, and
    // casts it to an destination AudioParameter object.
    template<typename T>
    inline static void castParameter(juce::AudioProcessorValueTreeState& apvts, const juce::ParameterID& id, T& destination) {
        destination = dynamic_cast<T>(apvts.getParameter(id.getParamID()));
        jassert(destination); // parameter does not exist or wrong type
    }

    //==============================================================================
    // Whenever the APVTS notifies the listener that a parameter has received a new
    // value, all this does is set the "parametersChanged" boolean to true. This is
    // thread-safe, since it's an atomic variable.
    std::atomic<bool> parametersChanged {false};
    juce::String lastChangedParamID {};
    void valueTreePropertyChanged(juce::ValueTree& tree, const juce::Identifier&) override {
        // DBG("Detected Parameter Change.");
        parametersChanged.store(true);
        lastChangedParamID = tree.getProperty("id").toString();
    }

    //==============================================================================
    // This "updateParameters()" method is where you'll do all the necessary
    // calculations using the new parameter values.
    void updateParameters();
    void refreshParameters();
    
    //==============================================================================
    // "apvts.getRawParameterValue(...)->load()" performs a relatively inefficient
    // parameter lookup by comparing string, since parameter identifiers are
    // "juce::String" objects under the hood. You'll be doing this lookup hundreds
    // of times per second, since it happens at the start of "processBlock()".
    // It's more efficient use the AudioParameter object directly. The following
    // variables are used to catch the parameters coming frome apvts. Each of them
    // are one by one connected to the parameters we defined at the begining
    // namespace "MyParameterID".
    juce::AudioParameterChoice* diodeParameter;
    juce::AudioParameterChoice* configParameter;
    juce::AudioParameterFloat* driveParameter;
    juce::AudioParameterFloat* toneParameter;
    juce::AudioParameterFloat* mixParameter;
    juce::AudioParameterFloat* levelParameter;

    //==============================================================================
    int pDiode;
    int pConfig;
    float pDrive;
    float pTone;
    float pMix;
    float pOutputGain_dB;
    
    //==============================================================================
    float fs;
    DiodeType diode;
    DiodeConfig config;
    float drive;
    float tone;
    float mix;
    float outputGain;
    
    float x {};
    float y {};
    float x_last {};
    float y_last {};
    
    //==============================================================================
    float vs_r; // resistance of resistor R1
    
    DiodeClipper diodeClipper {48000.0f};
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
