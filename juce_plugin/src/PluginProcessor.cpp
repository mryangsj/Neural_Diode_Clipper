#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       )
{
    //==============================================================================
    // Grab the parameter with the identifier from the APVTS and cast it to an
    // AudioParameter obeject, so that we can quickly lookup the value in the later.
    castParameter(apvts, MyParameterID::diode, diodeParameter);
    castParameter(apvts, MyParameterID::config, configParameter);
    castParameter(apvts, MyParameterID::drive, driveParameter);
    castParameter(apvts, MyParameterID::tone, toneParameter);
    castParameter(apvts, MyParameterID::mix, mixParameter);
    castParameter(apvts, MyParameterID::level, levelParameter);

    //==============================================================================
    // Listening to parameter changees
    apvts.state.addListener(this);
    
    //==============================================================================
    // initialize diode clipper
}

//==============================================================================
AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
    apvts.state.removeListener(this);
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

//==============================================================================
bool AudioPluginAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

//==============================================================================
bool AudioPluginAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

//==============================================================================
bool AudioPluginAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

//==============================================================================
double AudioPluginAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

//==============================================================================
int AudioPluginAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

//==============================================================================
int AudioPluginAudioProcessor::getCurrentProgram()
{
    return 0;
}

//==============================================================================
void AudioPluginAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

//==============================================================================
void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    juce::ignoreUnused (sampleRate, samplesPerBlock);

    //==============================================================================
    fs = float(sampleRate);
    diodeClipper.set_fs(fs);

    // This forces updateParameters() to be executed the very first time processBlock
    // is called.
    refreshParameters();
//    parametersChanged.store(true);
}

//==============================================================================
void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

//==============================================================================
bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

//==============================================================================
void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages)
{
    //==============================================================================
    juce::ignoreUnused (midiMessages);
    
    //==============================================================================
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();
    auto bufferSize = buffer.getNumSamples();
    
    //==============================================================================
    // In case we have more outputs than inputs, this code clears any output channels
    // that didn't contain input data, (because these aren't guaranteed to be
    // empty - they may contain garbage).
    // This is here to avoid people getting screaming feedback when they first compile
    // a plugin, but obviously you don't need to keep this code if your algorithm always
    // overwrites all the output channels.
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, bufferSize);
    
    
    //==============================================================================
    // These two lines of code do a thread-safe check to see whether
    // "parametersChanged" is true. If so, it calls "updateParameters()" method to
    // perform the parameter calculations. It also immediately sets
    // "parametersChanged" back to false.
    // This "check if true and set back to false" is a single atomic operation.
    // It either succeeds or it fails. If the operation fails, some other thread was
    // trying to write to "parametersChanged" while "processBlock()" was trying to
    // read it. This is not a big deal. You'll miss the parameter update in this
    // block, but most likely the operation will succeed in one of the next blocks
    // and "updateParameters()" will eventually get called.
    bool expected = true;
    if (parametersChanged.compare_exchange_strong(expected, false)) { updateParameters(); }
    
    //==============================================================================
    // This is the place where you'd normally do the guts of your plugin's
    // audio processing...
    // Make sure to reset the state if your inner loop is processing
    // the samples and the outer loop is handling the channels.
    // Alternatively, you can process the samples with the channels
    // interleaved by keeping the same state.
//    float signal_tot {};
//    float signal_mean {};
//    float signal_out {};
    
    for (int i {}; i < bufferSize; i++) {
        //for monon (low cpu computation)
//        for (int ch_num {}; ch_num < totalNumInputChannels; ch_num++) {
//            const auto* ch_in = buffer.getReadPointer(ch_num);
//            signal_tot += ch_in[i];
//        }
//
//        signal_mean = signal_tot / totalNumInputChannels;
//        signal_out = outputGain * diodeClipper.forward(drive * signal_mean);
//        signal_tot = 0.0f;
//
//        for (int ch_num {}; ch_num < totalNumOutputChannels; ch_num++) {
//            auto* ch_out = buffer.getWritePointer(ch_num);
//            ch_out[i] = signal_out;
//        }
        
        
        // for stereo (high cpu computation)
        for (int ch_num {}; ch_num < totalNumInputChannels; ch_num++) {
            auto* ch = buffer.getWritePointer(ch_num);
            
            x = diodeClipper.forward(drive * ch[i]);
            y = x - x_last + (0.9995f * y_last);
            x_last = x;
            y_last = y;
            
            ch[i] = outputGain * ((1.0f - mix) * ch[i] + (mix * y));
        }
    }
}
    
//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    //return new AudioPluginAudioProcessorEditor (*this);
    
    auto editor = new juce::GenericAudioProcessorEditor(*this);
//    editor->setSize(400, 300);
    return editor;
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    
    // Save plugin's current state and load it back the next time the plug-in is used.
    copyXmlToBinary(*apvts.copyState().createXml(), destData);
    //DBG(apvts.copyState().toXmlString());
}

//==============================================================================
void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    //==============================================================================
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    
    // You are provided with a block of binary data. First, this calls
    // "getXmlFromeBinary()" to parse the binary data into an XML document. Then it
    // verifies this XML contains the <Parameters> tag. And finally, it calls
    // "apvts.replaceState()" to update the values of all the parameters to the new
    // values.
    
    // After loading and restoring the plugin's state, you set the
    // "parametersChanged" boolean to true to singal "processBlock()" that it should
    // call "updateParameters()" again.
    
    // There is no way to know exactly when "setStateInformation()" may be called in
    // the lifetime of the plug-in, so you should assume it could happen at any time,
    // even if the plug-in is busy processing audio. Hence, it's necessary to
    // recalculate anything that depends on these parameter values the very next
    // time "processBlock()" is called.
    
    // The APVTS "replaceState()" method is thread-safe, by the way. That's good
    // thing too, because there is no gurantee that "getStateInformation()" or
    // "setStateInformation()" will be called from any particular thread, it doesn't
    // have to be the UI thread.
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml.get() != nullptr && xml->hasTagName(apvts.state.getType())) {
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
        parametersChanged.store(true);
    }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout AudioPluginAudioProcessor::createParameterLayout()
{
    //==============================================================================
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    
    //==============================================================================
    // add "DIODE" parameter
    const juce::StringArray diodeChoices {"1N34A", "1N4148"};
    layout.add(std::make_unique<juce::AudioParameterChoice>
               (MyParameterID::diode,
                "DIODE",
                diodeChoices,
                1));
    
    //==============================================================================
    // add "CONFIG" parameter
    const juce::StringArray configChoices {"1U1D", "1U2D"};
    layout.add(std::make_unique<juce::AudioParameterChoice>
               (MyParameterID::config,
                "CONFIG",
                configChoices,
                0));
    
    //==============================================================================
    // add "DRIVE" parameter
    const juce::NormalisableRange<float> range_drive(-25.0f, 25.0f, 1e-1f);
    layout.add(std::make_unique<juce::AudioParameterFloat>
               (MyParameterID::drive,
                "DRIVE",
                range_drive,
                16.0f,
                juce::AudioParameterFloatAttributes().withLabel("dB")));
    
    //==============================================================================
    // add "TONE" parameter
    const juce::NormalisableRange<float> range_tone(0.0f, 1.0f, 1e-2f);
    layout.add(std::make_unique<juce::AudioParameterFloat>
               (MyParameterID::tone,
                "TONE",
                range_tone,
                0.70f));
    
    //==============================================================================
    // add "MIX" parameter
    const juce::NormalisableRange<float> range_mix(0.0f, 100.0f, 1.0f);
    layout.add(std::make_unique<juce::AudioParameterFloat>
               (MyParameterID::mix,
                "MIX",
                range_mix,
                60.0f,
                juce::AudioParameterFloatAttributes().withLabel("%")));
    
    //==============================================================================
    // add "LEVEL" parameter
    const juce::NormalisableRange<float> range_gain(-12.0f, 12.0f, 1e-1f);
    layout.add(std::make_unique<juce::AudioParameterFloat>
               (MyParameterID::level,
                "LEVEL",
                range_gain,
                0.0f,
                juce::AudioParameterFloatAttributes().withLabel("dB")));
    
    //==============================================================================
    return layout;
}

//==============================================================================
void AudioPluginAudioProcessor::updateParameters()
{
    if (lastChangedParamID == MyParameterID::diode.getParamID() ||
        lastChangedParamID == MyParameterID::config.getParamID()) {
        pDiode = diodeParameter->getIndex();
        pConfig = configParameter->getIndex();
        diode = static_cast<DiodeType>(pDiode);
        config = static_cast<DiodeConfig>(pConfig);
        diodeClipper.set_model(diode, config);
    } else if (lastChangedParamID == MyParameterID::drive.getParamID()) {
        pDrive = driveParameter->get();
        drive = juce::Decibels::decibelsToGain(pDrive);
    } else if (lastChangedParamID == MyParameterID::tone.getParamID()) {
        pTone = toneParameter->get();
        diodeClipper.set_vs_r((-90e3f * pTone + 100e3f));
    } else if (lastChangedParamID == MyParameterID::mix.getParamID()) {
        pMix = mixParameter->get();
        mix = pMix / 100.0f;
    } else if (lastChangedParamID == MyParameterID::level.getParamID()) {
        pOutputGain_dB = levelParameter->get();
        outputGain = juce::Decibels::decibelsToGain(pOutputGain_dB);
    }
}

//==============================================================================
void AudioPluginAudioProcessor::refreshParameters()
{
    pDiode = diodeParameter->getIndex();
    pConfig = configParameter->getIndex();
    diode = static_cast<DiodeType>(pDiode);
    config = static_cast<DiodeConfig>(pConfig);
    diodeClipper.set_model(diode, config);
    
    pDrive = driveParameter->get();
    drive = juce::Decibels::decibelsToGain(pDrive);
    
    pTone = toneParameter->get();
    diodeClipper.set_vs_r((-90e3f * pTone + 100e3f));
    
    pMix = mixParameter->get();
    mix = pMix / 100.0f;
    
    pOutputGain_dB = levelParameter->get();
    outputGain = juce::Decibels::decibelsToGain(pOutputGain_dB);
}
