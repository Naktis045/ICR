from datasets import load_dataset

print("Downloading and loading the Teklia/IAM-line dataset...")

# Load the dataset from Hugging Face
dataset = load_dataset("Teklia/IAM-line")

# Print dataset structure to confirm it loaded correctly
print("\nDataset loaded successfully!")
print(dataset)
