from datasets import load_dataset

# Download from HF
ds = load_dataset("facebook/empathetic_dialogues")

# Save to CSV in your data/ folder
ds["train"].to_csv("data/empathetic_train.csv", index=False)
ds["validation"].to_csv("data/empathetic_val.csv", index=False)
ds["test"].to_csv("data/empathetic_test.csv", index=False)

print("Dataset downloaded and saved to /data/")
