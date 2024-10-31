import json
from datasets import Dataset, DatasetDict, Features, Value, Sequence

def load_vqa_v2(data_dir):
    # Load questions and answers
    with open(f"{data_dir}/v2_OpenEnded_mscoco_train2014_questions.json", "r") as f:
        train_questions = json.load(f)["questions"]
    with open(f"{data_dir}/v2_mscoco_train2014_annotations.json", "r") as f:
        train_answers = json.load(f)["annotations"]
    with open(f"{data_dir}/v2_OpenEnded_mscoco_val2014_questions.json", "r") as f:
        val_questions = json.load(f)["questions"]
    with open(f"{data_dir}/v2_mscoco_val2014_annotations.json", "r") as f:
        val_answers = json.load(f)["annotations"]

    # Create list of question-answer pairs
    def create_examples(questions, answers):
        examples = []
        for q, a in zip(questions, answers):
            examples.append({
                "question": q["question"],
                "image_id": q["image_id"],
                "answer": a["answers"][0]["answer"]  # Using the first answer
            })
        return examples

    train_examples = create_examples(train_questions, train_answers)
    val_examples = create_examples(val_questions, val_answers)

    # Define features for the dataset
    features = Features({
        "question": Value("string"),
        "image_id": Value("int32"),
        "answer": Value("string"),
    })

    # Create DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_dict(train_examples, features=features),
        "validation": Dataset.from_dict(val_examples, features=features)
    })

    return dataset

# Example usage
data_dir = "/path/to/vqa_data"
dataset = load_vqa_v2(data_dir)
