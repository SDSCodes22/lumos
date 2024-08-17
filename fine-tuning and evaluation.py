from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from datasets import load_metric
from torch.utils.data import DataLoader
import numpy as np

# Load the test set
test_dataset = load_dataset("coco", split="validation[:1%]")

# Preprocess the test set
def preprocess_test(examples):
    pixel_values = feature_extractor(images=examples['image'], return_tensors="pt").pixel_values
    return {"pixel_values": pixel_values}

processed_test_dataset = test_dataset.map(preprocess_test, batched=True, remove_columns=test_dataset.column_names)
test_loader = DataLoader(processed_test_dataset, batch_size=8, shuffle=False)

# Initialize metric
bleu = load_metric("bleu")

# Evaluation loop
model.eval()
predictions = []
references = []

for batch in test_loader:
    with torch.no_grad():
        pixel_values = batch['pixel_values'].squeeze().to(device)
        generated_ids = model.generate(pixel_values)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        predictions.extend(generated_texts)
        references.extend(test_dataset['caption'])

# Calculate BLEU score
bleu_score = bleu.compute(predictions=[pred.split() for pred in predictions],
                          references=[[ref.split()] for ref in references])

print(f"BLEU score: {bleu_score['bleu']:.4f}")

