# CNN Image Classifier App

This is a custom image classification web application built using **Streamlit**, **PyTorch**, and a **Convolutional Neural Network (CNN)** model. The app allows users to upload a trained CNN model and class labels (in JSON format), and then use the model to classify images. 

The app provides the following functionalities:
- Upload a trained CNN model and its corresponding class labels.
- Specify the image size to resize the uploaded images.
- Upload images for classification.
- Display the prediction result along with the classification probability.

## Features

- **Upload CNN Model:** Users can upload their trained CNN model in `.pt` or `.pth` format.
- **Upload Class Labels:** Users can upload a JSON file containing class labels for the model.
- **Image Classification:** After uploading an image, the app will classify it based on the trained model and show the class label and the prediction probability.

## Requirements

To run the app locally, you need to install the following dependencies:

- Python 3.x
- PyTorch
- Streamlit
- Pillow
- JSON

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/cnn-image-classifier.git
    ```

2. Navigate to the project directory:

    ```bash
    cd cnn-image-classifier
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. The app will open in your web browser at `http://localhost:8501`.

3. **Upload the trained model**:
   - Upload your PyTorch model (`.pt` or `.pth` file).
   - The model should be compatible with the app's architecture.

4. **Upload the class labels JSON**:
   - Upload the JSON file containing the class labels in the format:

     ```json
     {
         "0": "CLASS_NAME_1",
         "1": "CLASS_NAME_2"
     }
     ```

5. **Enter the image size**:
   - Specify the desired image size for classification (default is 224).

6. **Upload images for classification**:
   - Upload one or more images (JPEG, PNG, or JPG).
   - The app will display the uploaded image and show the predicted class along with the probability.

## Example JSON File (class_labels.json)

```json
{
    "0": "NORMAL",
    "1": "PNEUMONIA"
}
```

This JSON file maps class indices to class names. Ensure that the model was trained with the same class labels.

## Code Explanation

- **`load_model()`**: This function loads the user-uploaded model, initializes it, and sets it to evaluation mode.
- **`load_labels()`**: This function loads the class labels from the uploaded JSON file.
- **`transform_image()`**: This function preprocesses the uploaded image to make it compatible with the model input.
- **`predict()`**: This function runs the image through the model and returns the predicted class and the probability.

## Troubleshooting

- **JSON Decode Error**: If you encounter a `JSONDecodeError`, ensure the uploaded JSON file is correctly formatted and contains valid data.
- **KeyError**: Ensure that the class labels JSON file has the correct format and that the model output matches the expected labels.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize this further based on your project needs!