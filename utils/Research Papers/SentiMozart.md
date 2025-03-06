**SentiMozart: Music Generation based on Emotions**  
               **\- by Vishal**

**Research Paper** **Link**\- [https://www.scitepress.org/papers/2018/65977/65977.pdf](https://www.scitepress.org/papers/2018/65977/65977.pdf)

A **github repo** is found dealing with the **recognition of Music and emotion**s- [https://github.com/danyalimran93/Music-Emotion-Recognition](https://github.com/danyalimran93/Music-Emotion-Recognition)

---

**Summary of this Paper-**

This research presents a framework that generates music based on the detected emotion of a person’s facial expression.

* The proposed framework consists of two models:   
  * An **Image Classification Model,** which categorizes facial expressions into **seven major sentiment** classes (**Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**) using a **Convolutional Neural Network (CNN)**,   
  * A Music Generation Model, which employs a **Doubly Stacked Long Short-Term Memory (LSTM)** network to generate corresponding music.


* To evaluate the effectiveness of the system, the **emotional Mean Opinion Score (MOS)** is used as a metric.  
    
* Experimental results demonstrate a strong correlation between the generated music and the intended emotion, indicating the potential of this framework for applications in multimedia, entertainment, and personalized user experiences.


---

**How it was demonstrated-**

### **1\. Dataset Collection & Preparation**

* A dataset of **35,887 grayscale images** (48×48 pixels) was used for **facial expression classification**.  
* **200 MIDI music files** per emotion (Happy, Sad, Neutral).  
* Some emotions were merged for music generation:  
  * *Happy & Surprise → Happy*  
  * *Angry, Fear, Disgust → Sad*  
  * *Neutral remained unchanged*

  ### **2\. Image Classification Model**

* A **CNN model** classified facial expressions into **7 sentiment classes**.  
* **75.01% accuracy** is achieved in classification.

  ### **3\. Music Generation Model**

* A **Doubly Stacked LSTM** generates music based on classified emotions.  
* This model trained on **2000 epochs** with MIDI sequences as input.

  ### **4\. Evaluation using Mean Opinion Score (MOS)**

* A **scale of 0 to 10** was used:  
  * **0 \= Sad**  
  * **5 \= Neutral**  
  * **10 \= Happy**  
* The **average scores** of facial emotion ratings and corresponding music ratings were plotted.

  ### **5\. Correlation Calculation**

* The correlation between the **image MOS and music MOS** was computed, resulting in a **high correlation of 0.93**.  
* This indicates that the generated music successfully reflected the intended emotions from facial expressions.

This high correlation score suggests that this model effectively translates emotions into music, making it a reliable approach to be used for our project. 

---

Model’s mentioned in the paper-  
Google Magenta 