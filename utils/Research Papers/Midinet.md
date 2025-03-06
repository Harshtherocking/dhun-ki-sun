**MIDINET: A CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORK FOR SYMBOLIC-DOMAIN MUSIC GENERATION** 

\-SNG

1. PDF-[https://arxiv.org/abs/1703.10847](https://arxiv.org/abs/1703.10847)  
2. **Abstract:**  
   This paper presents *MidiNet*, a novel convolutional generative adversarial network (CNN-GAN) model for generating melodies in the symbolic domain (MIDI format). Unlike most music generation models based on recurrent neural networks (RNNs), MidiNet uses CNNs for faster training and parallelization. The model can generate melodies one bar at a time while considering prior musical contexts like chord progressions and preceding melodies. The authors also introduce a conditioning mechanism for greater creative control and perform user studies comparing MidiNet’s output with Google's MelodyRNN models.  
   The paper proposes *MidiNet*, a convolutional GAN model for generating symbolic music (MIDI). It improves upon existing models by leveraging CNNs instead of RNNs, offering faster and more parallelizable training. MidiNet includes a novel conditioning mechanism to generate melodies influenced by prior bars or chord sequences. The model is evaluated through user studies comparing its output with MelodyRNN, demonstrating that MidiNet produces more interesting and creative melodies while maintaining similar levels of realism and pleasantness. MidiNet’s flexible architecture can be extended to multi-track music generation, offering a versatile tool for algorithmic composition.  
3. **Tools required:**  
   1. Deep Learning Framework: TensorFlow (for implementing CNN and GAN architectures).  
   2. Dataset: 526 MIDI tabs (preprocessed from 1,022 MIDI files) from TheoryTab for training the model.  
   3. Models Used:  
      1. Generator CNN (for creating melodies).  
      2. Discriminator CNN (for distinguishing real from generated melodies).  
      3. Conditioner CNN (to integrate prior musical context).  
   4. GAN Techniques:  
      1. Transposed Convolutions (for upsampling).  
      2. Feature Matching (to stabilize GAN training).  
      3. Label Smoothing (to prevent overfitting).  
   5. Evaluation Method: User study comparing the output of MidiNet and MelodyRNN across three metrics—pleasantness, realism, and creativity.  
    


   

   