from datasets import load_dataset
import soundfile as sf
import os
import json

# 1. Load the dataset
# Note: Ensure you are logged in via huggingface-cli or pass your token here.
print("Loading dataset...")
dataset = load_dataset(
    "CoRal-project/coral-v3", 
    name="conversation", 
    split="test",
    streaming=True # for efficient disk space handling
    )

# 2. Setup the main output directory
base_output_dir = "data/coral_custom_export"
os.makedirs(base_output_dir, exist_ok=True)
manifest_path = os.path.join(base_output_dir, "manifest.jsonl")

print(f"Extracting files to '{base_output_dir}'...")

# 3. Process each row and build the manifest
with open(manifest_path, "w", encoding="utf-8") as manifest_file:
    for i, item in enumerate(dataset):

         # Keeping regular updates of downloading progress:
        if i % 100 == 0:
            print(f"Processed {i} files...")


        # Safely extract the conversation ID
        conv_id = str(item.get("id_conversation", "unknown_id"))
        
        # Create a dedicated folder for this conversation ID
        conv_folder = os.path.join(base_output_dir, conv_id)
        os.makedirs(conv_folder, exist_ok=True)
        
        # Define the FLAC file path
        flac_filename = f"{conv_id}.flac"
        flac_filepath = os.path.join(conv_folder, flac_filename)
        
        # Extract audio data and save as .flac
        # soundfile automatically infers the FLAC format from the file extension
        audio_data = item["audio"]
        sf.write(flac_filepath, audio_data["array"], audio_data["sampling_rate"])
        
        # Construct the manifest entry with only your requested columns
        # We change the 'audio' column to point to the new local file path
        manifest_entry = {
            "id_conversation": conv_id,
            "audio": os.path.join(conv_id, flac_filename), # Relative path e.g., "ID_123/ID_123.flac"
            "text": item.get("text", ""),
            "overlap": item.get("overlap", None),
            "age": item.get("age", ""),
            "gender": item.get("gender", "")
        }
        
        # Write the JSON object as a single line to the .jsonl file
        manifest_file.write(json.dumps(manifest_entry) + "\n")

print("Successfully exported audio to FLAC and created manifest.jsonl!")