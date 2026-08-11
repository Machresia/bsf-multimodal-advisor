# Location: ./prompt_builder.py

def format_gemma_messages(vision_stage: str = None, transcript: str = None, text_query: str = None):
    """
    Constructs structured messages for the Gemma 3 AutoProcessor.
    """
    user_content = []
    
    # 1. Attach context from voice note and classifier
    context_text = "Diagnostic Context:\n"
    if vision_stage:
        context_text += f"- Detected Life Stage: {vision_stage}\n"
    if transcript:
        context_text += f"- Transcribed Voice Note: \"{transcript}\"\n"
    if text_query:
        context_text += f"- User Note: {text_query}\n"

    user_content.append({"type": "text", "text": context_text})
    
    # Return formatted message payload
    return [
        {
            "role": "user",
            "content": user_content
        }
    ]