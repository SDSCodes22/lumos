import torch
from torch import nn
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoModelForCausalLM
from torch.utils.checkpoint import checkpoint

class IntegratedModel(nn.Module):
    def __init__(self, vision_model_name, text_model_name, use_gradient_checkpointing=False):
        super(IntegratedModel, self).__init__()
        
        # Load the vision and text models with gradient checkpointing if specified
        self.vision_encoder_decoder = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            vision_model_name, 
            text_model_name
        )
        
        # Enable gradient checkpointing for memory efficiency if specified
        if use_gradient_checkpointing:
            self.vision_encoder_decoder.encoder.gradient_checkpointing_enable()
            self.vision_encoder_decoder.decoder.gradient_checkpointing_enable()

        # Preload the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
    def forward(self, pixel_values, input_ids, attention_mask):
        def custom_forward(*inputs):
            return self.vision_encoder_decoder(pixel_values=inputs[0], input_ids=inputs[1], attention_mask=inputs[2]).logits

        # Use checkpointing to save memory
        logits = checkpoint(custom_forward, pixel_values, input_ids, attention_mask)
        return logits

    @torch.no_grad()
    def generate(self, pixel_values, max_length=50, num_beams=5, **generate_kwargs):
        # Generate text from image features
        generated_ids = self.vision_encoder_decoder.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            **generate_kwargs
        )
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

# Initialize the integrated model with gradient checkpointing enabled
vision_model_name = "facebook/blip-vision-model"
text_model_name = "gemma2-decoder-model"
model = IntegratedModel(vision_model_name, text_model_name, use_gradient_checkpointing=True)

# Example usage: Assume `pixel_values` and `input_ids` are tensors from your data pipeline
pixel_values = torch.randn((batch_size, 3, 224, 224))  # Example input tensor
input_ids = torch.randint(0, 1000, (batch_size, sequence_length))  # Example input tensor
attention_mask = torch.ones_like(input_ids)  # Example attention mask

# Forward pass
logits = model(pixel_values, input_ids, attention_mask)

# Text generation from image
generated_texts = model.generate(pixel_values)
