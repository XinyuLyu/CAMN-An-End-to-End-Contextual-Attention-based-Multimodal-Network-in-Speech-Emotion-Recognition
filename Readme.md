# EMNLP 2019 Abstract
## **CAMN**: An End-to-End Contextual Attention based Multimodal Network in Speech Emotion Recognition

Context-dependent multimodal emotion recognition is challenging because:   
1. most current researches model the modality-specific feature extraction and modality fusion separately in both the training and inference stage, which **ignore the model complexity**, prevent the global optimal feature representation, and increase the difficulty of application;   
2. the contextual information learning strategies that rely on recurrent neural networks **limit modeling efficiency** due to the sequential computation.   

To address the above issues, we introduce an **end-to-end** multimodal network that **integrates the modality-specific feature extraction module and modality fusion module**.   

Our model only relies on **attention mechanism** to learn the modality-specific representation and fusion representation simultaneously, which allows the **integral optimization**, computes the **contextual features in parallel**, and **reduces the model complexity**.   

Our model achieves state-of-the-art performs on two published multimodal emotion recognition datasets: *IEMOCAP* and *MELD*. The result shows the effectiveness and efficiency of the proposed CAMN model.  