from torch.utils.data import DataLoader

# Create a DataLoader
train_loader = DataLoader(processed_dataset, batch_size=8, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        outputs = model(pixel_values, input_ids, attention_mask)
        loss = criterion(outputs.view(-1, model.vision_encoder_decoder.config.decoder.vocab_size), input_ids.view(-1))
        
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

print("Training complete.")

