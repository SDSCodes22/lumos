import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset

# Function to load the dataset with retries
def load_dataset_with_retries(dataset_name, config_name, split, max_retries=5):
    for attempt in range(max_retries):
        try:
            # Load dataset with trust_remote_code to allow custom code
            return load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)  # Wait before retrying
    raise RuntimeError("Failed to load dataset after several attempts.")

# Function to initialize the model
def initialize_model():
    print("Loading the smaller LLM model...")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-2", revision="smaller")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-2")
    print("Model initialized.")
    return model, processor

# Function to download the dataset
def download_dataset():
    print("Loading VQAv2 dataset...")
    dataset = load_dataset_with_retries('visual_genome', 'question_answers_v1.2.0', split='train')
    print("Dataset loaded.")
    return dataset

# Main execution function
def main():
    # Initialize the model in a separate thread
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_model = executor.submit(initialize_model)
        future_dataset = executor.submit(download_dataset)

        # Wait for both to finish
        model = future_model.result()
        dataset = future_dataset.result()

    # Proceed with your training or evaluation using the model and dataset
    # For example:
    print("Proceeding with training or evaluation...")

if __name__ == "__main__":
    main()
