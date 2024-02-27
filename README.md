# 2D-CNN-LSTM

This code is implementation of following paper:
> Zhao, J., Mao, X., & Chen, L. (2019). Speech emotion recognition using deep 1D & 2D CNN LSTM networks. Biomedical Signal Processing and Control, 47, 312-323.

&nbsp;&nbsp;I implementated 2D CNN LSTM model for Speech Emotion Recognition Zhao et al. (2019) [1]

&nbsp;&nbsp;This paper gives us intuitive model that extracts features of speech data with popular deep learning techniques: local features from CNN, and global features from LSTM.

&nbsp;&nbsp;I also implented 1D CNN LSTM [here][1dgithub]. The paper highlights that the 2D CNN LSTM network 'outperforms' the traditional approaches, Therefore, I mainly focused on the 2D model. The 1D model use raw audio file as data, whereas 2D model preprocesses data with log-mel spectograms. This is why it employs 2D convolution. LFLB (local feature learning block) that consists convolutional layer, batch normalization layer, exponential linear unit layer, and max pooling layer, was used to capture local features. The LSTM was used to learn global features, followed by a dense layer at the end of the model.

&nbsp;&nbsp;My goal was to apply this model to Korean speech dataset sourced from AIHUB, specifically the 감정 분류를 위한 대화 음성 데이터셋. The data was from [here][link].

[1dgithub]: https://github.com/buaaaaang/1D-CNN-LSTM
[link]: https://aihub.or.kr/keti_data_board/language_intelligence

&nbsp;&nbsp;Preprocessing was conducted using librosa library. The preprocessing stage resulted in data with 251 frames and 128 mel frequency bins. The accompanying picture illustrates a sample result of preprocessing step.

![log_mel_sample](https://github.com/buaaaaang/buaaaaang.github.io/assets/69184903/b784417c-898c-4fae-b8f8-5c34386d9a13)

&nbsp;&nbsp;Model was built with Keras, and hyperparamter tuning was performed using KerasTuner. I focused on the accuracy of speaker-dependent experiments. Validation accuracy was 0.73, which is notably similar to the result of [1]. (which was 0.76) After that,Speaker dependent experiment on Korean speech dataset was conducted. 
Validation accuracy was 0.47, which doesn't look usable.

&nbsp;&nbsp;At this point, I pointed out questionable aspect of the paper. The input data, with dimension of (128&times;251&times;1), passes through 4 LFLB layers, and become (1&times;1&times;128). In order to establish a connection with the subsequent LSTM layer, a reshape layer was used to transform the shape into (1&times;128). Here, due to the absence of time information in this reshaped representation, recurrent neural network is invoked only once. This makes utilization of the LSTM layer redundant. To confirm it, I replaced the LSTM layer with two dense layers. Applying changed model to Berlin EmoDB dataset, validation accuracy was 0.76, which is higher than the original model, showing that LSTM layer in the original model is not meaningful.

&nbsp;&nbsp;In order to preserve the time information within the data and make effective use of the LSTM layer, I implemented a modification by adjusting the pooling layer to have a smaller pooling size along the frame-axis. When this modified model was applied to the Berilin EmoDB dataset, the obtained result surpassed those reported in the referenced paper. Validation accuracy increased to 0.80. Indeed, conducting a more detailed hyperparameter tuning process is likely to further enhance the model's performance.

&nbsp;&nbsp;I applied modified model to Korean speech dataset. Validation accuracy was 0.49, which is slightly better, but still not useful.

&nbsp;&nbsp;The Korean speech dataset I came across proved to be challenging. This is due to the ambiguity of certain emotions, making it difficult to distinguish them accurately even for human. However, a positive aspect of this dataset is that it provides not only the primary emotion label but also secondary, tertiary, quaternary, and quinary emotion labels. This additional information opens up possibilities for characterizing emotions within a dimensional emotional model, where multiple dimensions of emotion can be considered simultaneously.

&nbsp;&nbsp;Overall, I successfully implemented the model in the paper, and I could modify it to be better model. There is still more work to do to apply this model for Korean speech dataset.

<br/>
<br/>

[1] Zhao, J., Mao, X., & Chen, L. (2019). Speech emotion recognition using deep 1D & 2D CNN LSTM networks. Biomedical Signal Processing and Control, 47, 312-323.
