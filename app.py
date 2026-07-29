import os
import sys
from PIL import Image
from transformers import pipeline

# Import LlamaIndex components
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# ==========================================
# 1. SETUP RAG ENGINE (LlamaIndex + Ollama)
# ==========================================
print("🔄 Initializing local RAG pipeline...")

# Configure embedding model and Ollama LLM
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(
    model="gemma3:4b", 
    request_timeout=300.0,
    additional_kwargs={"num_ctx": 2048}  # Limits RAM usage drastically
)

# Load existing index from local storage folder
STORAGE_DIR = "./storage"

if not os.path.exists(STORAGE_DIR):
    print(f"❌ Error: Storage directory '{STORAGE_DIR}' not found. Ensure your index folder exists.")
    sys.exit(1)

storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
index = load_index_from_storage(storage_context)

# Define custom QA prompt template for practical, general troubleshooting
qa_prompt_tmpl = (
    "You are an expert Black Soldier Fly (BSF) farming consultant.\n"
    "Context information from knowledge base:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Instructions:\n"
    "1. Use the provided context above to directly answer the query.\n"
    "2. If the context gives direct facts, summarize the root cause briefly.\n"
    "3. Always provide clear, bulleted, step-by-step practical actions the farmer can take.\n"
    "4. If the context lacks specific fixes, apply standard BSF farming best practices.\n\n"
    "User Query: {query_str}\n"
    "Answer:"
)
qa_template = PromptTemplate(qa_prompt_tmpl)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_template
)

# ==========================================
# 2. SETUP FINE-TUNED SIGLIP VISION MODEL
# ==========================================
MODEL_PATH = "./my_bsf_siglip_model"

print(f"🔄 Loading fine-tuned SigLIP vision classifier from '{MODEL_PATH}'...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model directory '{MODEL_PATH}' not found. Ensure your unzipped folder is placed here.")
    sys.exit(1)

# Fine-tuned classification pipeline
vision_classifier = pipeline("image-classification", model=MODEL_PATH)

print("✅ All systems initialized successfully!\n" + "="*50)

# ==========================================
# 3. INTERACTIVE BSF ADVISOR LOOP
# ==========================================
def main():
    while True:
        print("\n--- BSF Multimodal Advisor ---")
        image_input = input("Enter path to tray photo (or press Enter to skip vision analysis, 'q' to quit): ").strip()
        
        # Clean quotes if dragged and dropped into terminal
        image_input = image_input.strip("'\"")

        if image_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting BSF Advisor. Goodbye!")
            break

        detected_stage = None
        confidence = 0.0

        # Step A: Process Image Input
        if image_input and os.path.exists(image_input):
            try:
                print(f"\n📸 Analyzing tray image: {image_input}...")
                results = vision_classifier(image_input)
                
                # Extract top prediction label & confidence score
                top_prediction = results[0]
                detected_stage = top_prediction['label']
                confidence = top_prediction['score']

                print(f"🔎 Visual Analysis Result: Detected Stage/Condition = '{detected_stage}' ({confidence:.1%} confidence)")
            except Exception as e:
                print(f"⚠️ Failed to process image: {e}")
        elif image_input:
            print(f"⚠️ Image path '{image_input}' not found. Proceeding with text query only.")

        # Step B: Get User Question
        user_question = input("\nEnter your query/observation: ").strip()
        
        # Fallback if question is skipped but image was provided
        if not user_question and detected_stage:
            user_question = f"What are the best management practices and potential issues for {detected_stage}?"
        elif not user_question and not detected_stage:
            print("⚠️ Please provide either a valid image or a text question.")
            continue

        # Step C: Combine Visual Context + User Query for RAG
        if detected_stage:
            full_query = (
                f"Visual Observation: The tray image shows '{detected_stage}' with {confidence:.1%} confidence.\n"
                f"User Question: {user_question}"
            )
        else:
            full_query = user_question

        # Step D: Query Knowledge Base
        print("\n🤖 Retrieving knowledge base advice...")
        try:
            response = query_engine.query(full_query)
            print("\n💡 --- BSF ADVISOR RESPONSE ---")
            print(response)
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error querying RAG engine: {e}")

if __name__ == "__main__":
    main()