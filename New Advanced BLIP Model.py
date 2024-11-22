from transformers import BlipModel

#Load the pre-trained BLIP model
from transformers import BlipProcessor, BlipForConditionalGeneration

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")



from transformers import GPT2LMHeadModel

#Initialize the GPT-2 model as the new decoder
gpt2_decoder = GPT2LMHeadModel.from_pretrained('gpt2')


#Replace the BLIP decoder with GPT-2
model.language_model = gpt2_decoder


from datasets import load_dataset

#Load the VQA dataset
from datasets import load_dataset

# If the dataset is available on the Hub
from datasets import load_dataset

# Load the question-answer pairs from Visual Genome
dataset = load_dataset("visual_genome", "question_answers_v1.2.0")

# If you have the dataset locally
# dataset = load_from_disk("/path/to/your/vqa/dataset")



from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=vqa_dataset['train'], # type: ignore
    eval_dataset=vqa_dataset['validation'], # type: ignore
)

trainer.train()


