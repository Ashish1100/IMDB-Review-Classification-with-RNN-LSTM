# 🎬 IMDB Review Sentiment Analysis with RNN-LSTM

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Harness the power of deep learning to classify movie reviews! This project showcases an advanced Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers for sentiment analysis on IMDB movie reviews.

## 🚀 Features

- State-of-the-art RNN-LSTM architecture
- Achieves 83% accuracy on test set
- Easy-to-use Jupyter notebook implementation
- Customizable hyperparameters for experimentation

## 📊 Project Overview

Our sentiment analysis journey unfolds through these key steps:

1. 📥 Data loading and preprocessing
2. 🔤 Text tokenization and padding
3. 🏗️ Model architecture design using Keras
4. 🧠 Model training and evaluation
5. 🔮 Prediction on new reviews

## 🛠️ Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn

## 📚 Dataset

We're using the renowned IMDB dataset, featuring 50,000 movie reviews split evenly between positive and negative sentiments. The dataset is divided into 25,000 reviews for training and 25,000 for testing.

## 🧠 Model Architecture

Our RNN-LSTM model boasts the following structure:

1. Embedding layer (input_dim=20000, output_dim=128)
2. LSTM layer (128 units)
3. Output layer (1 unit, sigmoid activation)

## 🚀 Getting Started

1. Clone this repository:
'''bash
git clone https://github.com/yourusername/IMDB-Review-Classification-with-RNN-LSTM.git
'''

## 📈 Results

After training for 5 epochs, our model achieves an impressive accuracy of approximately 83% on the test set.

## 🔧 Customization

Feel free to experiment with various hyperparameters:

- Embedding dimensions
- Number of LSTM units
- Additional LSTM or Dense layers
- Dropout for regularization
- Different optimizers or learning rates

## 📞 Contact

Got questions or feedback? Open an issue in the GitHub repository or reach out.
## 🌟 Star Us!

If you find this project useful, don't forget to give it a star on GitHub! It helps us know you appreciate our work.

## 📣 Spread the Word

Share this project with your fellow data scientists and machine learning enthusiasts:

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fyourusername%2Fimdb-sentiment-analysis-rnn-lstm)](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20IMDB%20Sentiment%20Analysis%20project%20using%20RNN-LSTM!&url=https%3A%2F%2Fgithub.com%2Fyourusername%2Fimdb-sentiment-analysis-rnn-lstm)

## 🎉 Acknowledgements

- [IMDB Dataset](https://www.imdb.com/interfaces/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

Happy coding and may the force of sentiment analysis be with you! 🚀📽️
