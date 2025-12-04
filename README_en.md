# BrainScan AI — Automated Brain Tumor Detection

## Project Overview

The goal of this project is to develop an intelligent application capable of:

- Analyzing and classifying brain MRI images to detect tumors.
- Assisting doctors in rapid and reliable interpretation of results.
- Optimizing diagnostic time while reducing human errors.

This initiative combines healthcare, innovation, and AI to improve the quality and speed of medical care in Morocco.

--- 

## Technologies Used

- **Python 3:** development and data analysis
- **NumPy / Pandas:** data manipulation
- **OpenCV:** image processing and resizing
- **Matplotlib / Seaborn:** data visualization
- **TensorFlow / Keras:** CNN model design and training
- **Scikit-learn:** label encoding, train/test splitting, evaluation
- **Streamlit:** interactive user interface

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/anass17/BrainScan
cd BrainScan
```

2. Install dependencies:
```Bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

4. Open the application in your browser:
Streamlit will automatically open a local window; otherwise go to: `http://localhost:8501/`

---

## Feature Stories & Tasks

### User Story 1 : Dataset Loading & Preprocessing

- Import required libraries.
- Load MRI images and verify extensions (jpeg, jpg, bmp, png).
- Explore classes using folder names.
- Shuffle images and labels into two corresponding lists.
- Resize images (e.g., 224×224) with **OpenCV**.
- Convert lists to **NumPy** arrays usable by the CNN.
- Visualize the number of images per class and sample images.
- Check class balance and rebalance if necessary.
- Encode textual labels into numeric format.
- Split data into training and test sets.
- Normalize pixel values to [0, 1].

---

### User Story 2 : CNN Model Design

- Define the architecture (`Conv2D` + `MaxPooling` + `Dropout` + `Dense`).
- Choose optimal activation functions for hidden layers and output layer.
- Compile the model (Adam optimizer, categorical_crossentropy loss).
- Verify the architecture using `model.summary()` and `plot_model()`.
- Determine hyperparameters: learning rate, number of epochs, batch size.
- Measure training time using the time library.

---

### User Story 3 : Training & Evaluation

- Train the model using `model.fit()`.
- Save the best model with ModelCheckpoint.
- Evaluate performance on the test set.
- Visualize training curves (accuracy and loss).
- Generate a confusion matrix and classification report.
- Display examples of correct and incorrect predictions.

---

### User Story 4 : Deployment & Usage

- Create a `predict_image(path)` function to test a single image.
- Save the trained model.
- Develop a **Streamlit** interface to:
    - Input images from the user
    - Display predictions in real time

--- 

### Interface Streamlit
![Streamlit UI](https://github.com/user-attachments/assets/2918d58c-3c19-4e79-83b8-3ef1dc31246e)