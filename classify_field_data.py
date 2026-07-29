import os
import pandas as pd
from PIL import Image
from transformers import pipeline

# 1. Directory Setup
IMAGE_ROOT = "./images"
OUTPUT_CSV = "./bsf_vision_results.csv"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

if not os.path.exists(IMAGE_ROOT):
    print(f"[!] Error: The folder '{IMAGE_ROOT}' does not exist.")
    exit()

# 2. Load the Google SigLIP Pipeline
print("Loading Google SigLIP model from Hugging Face...")
classifier = pipeline(
    task="zero-shot-image-classification",
    model="google/siglip-base-patch16-224"
)

# 3. Scientific Candidate Labels
candidate_labels = [
    "black soldier fly larvae instar stage",
    "black soldier fly prepupae or pupae",
    "frass and organic waste substrate",
    "mold or fungal contamination in rearing tray"
]

dataset_records = []

print("\n🚀 Crawling Life Cycle Folders & Processing Field Dataset...")
print("=" * 75)

# 4. Recursive Folder Walk
# os.walk automatically deep-dives into 1_Egg, 2_Larva, etc.
for root, dirs, files in os.walk(IMAGE_ROOT):
    # Filter for valid image formats
    image_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]
    
    if not image_files:
        continue
        
    # Extract the current folder stage name (e.g., '2_Larva' or '4_Pupa')
    current_stage = os.path.basename(root)
    print(f"\n📂 Processing Stage Folder: [{current_stage}] ({len(image_files)} images found)")
    print("-" * 75)

    for file_name in image_files:
        image_path = os.path.join(root, file_name)
        print(f" -> Analyzing: {file_name}")
        
        try:
            img = Image.open(image_path).convert("RGB")
            results = classifier(img, candidate_labels=candidate_labels)
            
            # Save filename and the actual source folder stage
            record = {
                "image_name": file_name,
                "field_logged_stage": current_stage
            }
            
            # Map probabilities for each evaluation category
            for res in results:
                label_clean = res['label'].replace(" ", "_")
                record[label_clean] = round(res['score'] * 100, 2)
                
            # Log model determination vs your field tag
            record["top_prediction"] = results[0]['label']
            record["confidence_pct"] = round(results[0]['score'] * 100, 2)
            
            dataset_records.append(record)
            
        except Exception as e:
            print(f"   [!] Failed to process {file_name}: {e}")

# 5. Export Compiled Matrix to CSV Spreadsheet
print("=" * 75)
if dataset_records:
    df = pd.DataFrame(dataset_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Success! Deep data log built across all stages: {OUTPUT_CSV}")
else:
    print("\n[!] Error: No image files were found deep inside the subfolders.")