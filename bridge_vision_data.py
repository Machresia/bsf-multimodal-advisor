import os
import pandas as pd

CSV_PATH = "./bsf_vision_results.csv"
OUTPUT_TEXT_PATH = "./data/vision_summary_manifest.txt"

if not os.path.exists(CSV_PATH):
    print(f"[!] Error: Cannot find {CSV_PATH}. Run classify_field_data.py first!")
    exit()

# Load the field dataset spreadsheet
df = pd.read_csv(CSV_PATH)

# Open a text manifest file inside your LlamaIndex 'data' directory
with open(OUTPUT_TEXT_PATH, "w", encoding="utf-8") as f:
    f.write("=== FIELD RESEARCH BSF LIFE CYCLE DATA MANIFEST ===\n\n")
    f.write("This file contains automated SigLIP computer vision determinations matching field image layers.\n")
    f.write("Cross-reference these records with academic papers during natural language query loops.\n\n")
    
    for _, row in df.iterrows():
        summary = (
            f"Image Record: {row['image_name']}\n"
            f"  - Field Logged Biological Stage: {row['field_logged_stage']}\n"
            f"  - AI Vision Ultimate Prediction: {row['top_prediction']} (Confidence: {row['confidence_pct']}%)\n"
            f"  - Instar Stage Score: {row['black_soldier_fly_larvae_instar_stage']}%\n"
            f"  - Prepupae/Pupae Score: {row['black_soldier_fly_prepupae_or_pupae']}%\n"
            f"  - Substrate Biomass Score: {row['frass_and_organic_waste_substrate']}%\n"
            f"  - Contamination/Mold Score: {row['mold_or_fungal_contamination_in_rearing_tray']}%\n"
            f"---------------------------------------------------------------------------\n"
        )
        f.write(summary)

print(f"[+] Successfully generated text manifest at: {OUTPUT_TEXT_PATH}")
print("[*] Next step: Rebuild your LlamaIndex cache vector storage.")