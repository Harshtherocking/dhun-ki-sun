       **Classical Music Generation using RNN**

Link : [Classical Music Generation using RNN.pdf](http://www.ir.juit.ac.in:8080/jspui/bitstream/123456789/3580/1/Classical%20Music%20Generation%20using%20RNN.pdf)  
        

### **Abstract**

To create progressive music based on existing melodies, the network should consider repeated melodies and overlapping timesteps. Using Recurrent Neural Networks (RNNs), each node will evaluate information from previous nodes and pass its learnings to the next node. In RNNs, there is no direction constraint.

### **Introduction**

**Concept of Music:**

* **Most songs have three or more keys simultaneously.**  
* **Digital partitions can encrypt music into plain text file formats for easy differentiation and use.**

**Automatic Music Generation:**

* **Models like WaveNet and Long Short-Term Memory (LSTM) are used to generate music automatically.**  
* **LSTM, a type of RNN, captures long-term dependencies in sequences, making it suitable for predicting successive amplitude values from input sound waves.**  
* **Metrics based on musical theory (e.g., note repetition, motif notes) can assess the quality of generated music mathematically.**

### **Machine Learning**

**Types of Learning:**

1. **Supervised Learning: Uses labeled data.**  
2. **Unsupervised Learning: Uses unlabeled data.**  
3. **Reinforcement Learning: Reward-based learning, works on feedback principles.**

### **Research Article**

**Integration of classical music into music schools by analyzing various RNN structures. Use of GRU (Gated Recurrent Unit) layers.**

### **System Development**

**LSTM vs. Basic Neural Networks:**

* **LSTMs have a chain-like topology with different repeating module structures.**  
* **Basic neural networks multiply input by weights and use a sigmoid function to normalize.**

**Recurrent Neural Networks:**

* **Similar to convolutional neural networks (CNNs) for image recognition.**  
* **Convolution involves replacing each pixel with a weighted sum of surrounding pixels.**

### **Inspiration from RNN-RBM Combination**

* **Biaxial RNN: Utilizes time and note axes, with each recurring layer converting input into output and sending connections along these axes.**


 **Page 37 \- (all below that i am not able to understand)**  
 **Nоrmаl соmmuniсаtiоn аllоws раtterns аt а time, but we dо they hаve nо wаy оf getting gооd сhоrds: the оutрut оf eасh nоte is соmрletely indeрendent оf аll the оutрut оf the nоte. Here we саn find insрirаtiоn in the RNN-RBM соmbinаtiоn аbоve: let the first раrt оf оur netwоrk fасe the mоment, аnd let the seсоnd раrt сreаte beаutiful sоngs.**

 **The sоlutiоn I hаve deсided tо tаke with me is whаt I саll “biаxiаl RNN”.**   
**The ideа is thаt we hаve twо аxis (аnd оne mосk аxis): there is а time аxis аnd а nоte аxis (аnd а рseudо-аxis соmрutаtiоn аxis). Eасh reсurring lаyer соnverts the inрut intо the оutрut, аnd sends а reсurring соnneсtiоn with оne оf these аxes. But there is nо reаsоn why everyоne shоuld роst а соnneсtiоn with the sаme**

