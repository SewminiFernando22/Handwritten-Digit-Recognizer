# Handwritten-Digit-Recognizer
A deep learning project to classify handwritten digits using a CNN and Streamlit.
http://localhost:8501/


Digit Wizard: Handwritten Digit Recognizer

Project Overview

The Digit Wizard is an interactive deep learning application that recognizes handwritten digits (0-9) from uploaded images, achieving ~98% accuracy. Built as a showcase of AI and machine learning skills, this project leverages a Convolutional Neural Network (CNN) trained on the MNIST dataset to classify digits with high precision. A sleek Streamlit web app allows users to upload images and view real-time predictions, complete with confidence scores and a probability distribution chart, making it both functional and visually engaging.

Motivation

The goal was to create a portfolio project that demonstrates proficiency in deep learning, computer vision, and web development, essential skills for AI/ML roles. Inspired by real-world applications like optical character recognition (OCR) in document processing and automated form reading, this project combines robust model training with an intuitive user interface to bridge theoretical ML concepts with practical deployment. It was designed to be a beginner-friendly yet impressive entry into the AI job field, showcasing end-to-end development from data preprocessing to model deployment.

Features

Accurate Predictions: Achieves ~98% accuracy on the MNIST test set using a CNN.
Interactive Web App: Built with Streamlit, allowing users to upload PNG/JPG images and see predictions instantly.
Confidence Visualization: Displays prediction confidence as a percentage and a bar chart of probabilities for all digits (0-9).
Polished UI: Custom-styled interface with a sidebar for controls, modern fonts, and a clean layout.
Error Handling: Gracefully handles missing model files and invalid image uploads.

Technologies Used
Python: Core programming language.
TensorFlow/Keras: For building, training, and deploying the CNN model.
Streamlit: For creating the interactive web app.
OpenCV: For image processing and resizing.
Matplotlib: For visualizing prediction probabilities.
NumPy: For numerical operations and data manipulation.
Jupyter Notebook: For model development and experimentation.
Git/GitHub: For version control and project hosting.
MNIST Dataset: A benchmark dataset of 60,000 training and 10,000 test images of handwritten digits.

How It Works

Model Training:
      A CNN is trained on the MNIST dataset, which contains 28x28 grayscale images of handwritten digits.
      The model consists of convolutional layers, max-pooling, and dense layers, optimized with the Adam optimizer and sparse categorical crossentropy loss.
      Training for 5 epochs achieves ~98% accuracy, and the model is saved as digit_recognizer.h5.

Web App:
      The Streamlit app loads the trained model and allows users to upload a handwritten digit image (PNG/JPG).
      The image is preprocessed (converted to grayscale, resized to 28x28, normalized) using OpenCV.
      The model predicts the digit, and the app displays the processed image, predicted digit, confidence score, and a bar chart of probabilities.



Deployment:
      The project is hosted on GitHub, with clear setup instructions for reproducibility.
      Users can clone the repository, set up a virtual environment, and run the app locally.
      Setup Instructions


Clone the repository:
      git clone https://github.com/your-username/Handwritten-Digit-Recognizer.git


Create a virtual environment:
      python -m venv env


Activate the virtual environment:
    env\Scripts\activate

Install dependencies:
    pip install tensorflow numpy pandas matplotlib scikit-learn streamlit opencv-python


Run the Jupyter notebook to train the model:
    jupyter lab
    Open Untitled1.ipynb and run all cells to generate digit_recognizer.h5.

Run the Streamlit app:
    streamlit run app.py
    Open http://localhost:8501 in your browser to use the app.

Results
Accuracy: ~98% on the MNIST test set.
Sample Prediction: [Insert screenshot of app showing a prediction, e.g., images/screenshot1.png]
Probability Chart: [Insert screenshot of bar chart, e.g., images/screenshot2.png]



Future Enhancements
Add webcam support for real-time digit recognition.
Extend to recognize handwritten letters using the EMNIST dataset.
Deploy the app on a cloud platform like Streamlit Sharing or Heroku.

Author

Sewmini Fernando - www.linkedin.com/in/chamika-sewmini-fernando-peter-pulle-520740233

License

This project is licensed under the MIT License - see the LICENSE file for details.
