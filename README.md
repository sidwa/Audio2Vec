# Audio-embedder
Complete Pytorch code for model with data pipeline

Audio data must be in the format specified in audio_dataset.py (numpy file with extracted MFCC features)
Librosa can be used to extract MFCC features

The model used is a seq2seq autoencoder as described in "Audio word2vec"
https://people.csail.mit.edu/andyyuan/docs/interspeech-16.audio2vec.paper.pdf
