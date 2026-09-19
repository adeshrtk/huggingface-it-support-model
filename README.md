# IT Support Ticket Classifier using Hugging Face Transformers

This Google Colab project demonstrates how to build and deploy a custom IT support ticket classifier using Hugging Face Transformers.

## Project Overview

The goal of this project is to classify IT support requests into predefined categories (e.g., NETWORK, PASSWORD, EMAIL, HARDWARE). We leverage the power of pre-trained Transformer models from the Hugging Face Hub and fine-tune them on a custom dataset.

## Key Learnings

By completing this notebook, you will learn about:

*   **Hugging Face Hub**: Understanding its role as a model registry.
*   **Hugging Face Authentication**: How to authenticate your Colab environment with Hugging Face.
*   **Pretrained Models**: Utilizing models like DistilBERT.
*   **Tokenizers**: The process of converting text into numerical tokens.
*   **Dataset Preparation**: Creating and structuring datasets for Transformer training.
*   **Train/Validation/Test Split**: Best practices for splitting your data.
*   **Transformer Fine-tuning**: Adapting a pre-trained model to a specific task.
*   **`AutoTokenizer` and `AutoModelForSequenceClassification`**: Key classes for loading models and tokenizers.
*   **`TrainingArguments` and `Trainer`**: Configuring and executing the training process.
*   **Evaluation Metrics**: Calculating accuracy, precision, recall, and F1-score.
*   **Confusion Matrix**: Visualizing model performance.
*   **Model Inference**: Making predictions with the fine-tuned model.
*   **Saving and Loading Models**: Storing your trained model and reloading it for future use.
*   **Hugging Face Model Repository**: Pushing your fine-tuned model to the Hugging Face Hub.
*   **API-based Inference**: Performing predictions directly from the Hugging Face Hub via its API.

## Project Architecture

The project follows a common NLP fine-tuning workflow:

1.  **Start with a Pretrained Model**: We begin with a model like DistilBERT from the Hugging Face Hub.
2.  **Fine-tuning with Custom Data**: Your labeled IT support ticket data is used to fine-tune the pretrained model.
3.  **Output a Fine-tuned Model**: The result is a specialized 'IT Support AI' model.
4.  **Deployment Options**: The fine-tuned model can be used locally in a Python application or pushed to the Hugging Face Hub for API-based inference or cloud deployment.

## How to Run This Notebook

1.  **Open in Google Colab**: Click the "Open in Colab" badge (or equivalent) for this notebook.
2.  **Enable GPU Runtime**: Go to `Runtime > Change runtime type` and select `T4 GPU` as the hardware accelerator.
3.  **Install Libraries**: Run the cell that installs necessary libraries (`transformers`, `datasets`, `evaluate`, `accelerate`, `scikit-learn`, `huggingface_hub`).
4.  **Hugging Face Authentication**: Create a Hugging Face account and generate an access token (with 'write' access if you plan to push your model). Use `notebook_login()` to log in from the Colab environment.
5.  **Run All Cells**: Execute all cells in the notebook sequentially. This will:
    *   Load and prepare the dataset.
    *   Tokenize the data.
    *   Load the pre-trained DistilBERT model.
    *   Configure and train the model.
    *   Evaluate the model's performance.
    *   Demonstrate local inference.
    *   Push the fine-tuned model to your Hugging Face repository.
    *   Demonstrate loading the model directly from the Hugging Face Hub.

## Example Usage (after running the notebook)

You can test the deployed model using a simple pipeline:

```python
from transformers import pipeline

# Replace 'YOUR_HF_USERNAME' with your actual Hugging Face username
MODEL_NAME = "YOUR_HF_USERNAME/it-support-model"

remote_classifier = pipeline(
    "text-classification",
    model=MODEL_NAME
)

# Test a new support ticket
result = remote_classifier("My printer is not working")
print(result)
```

This project provides a comprehensive guide to building, evaluating, and deploying a practical text classification model using modern NLP tools.
