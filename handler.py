import os
import json
import logging
import torch
import runpod
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Increase timeout to 5 minutes (300 seconds)
os.environ["HF_HUB_READ_TIMEOUT"] = "300" 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Cleans up your logs

# -------------------- Globals & Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")
GEMMA_MODEL = "google/translategemma-4b-it"

# -------------------- Model Mapping --------------------

def get_model_name(culture):
    prefix = "tts001/"
    mapping = {
        "bg": "translation-ft-de-bg",
        "cs": "translation-ft-de-cz",
        "da": "translation-ft-de-da",
        "en": "translation-ft-de-en",
        "es": "translation-ft-de-es",
        "fr": "translation-ft-de-fr",
        "it": "translation-ft-de-it",
        "nl": "translation-ft-de-nl",
        "pl": "translation-ft-de-pl",
        "lt": "translation-ft-de-lt",
    }
    lang = culture[:2].lower()
    suffix = mapping.get(lang)
    return f"{prefix}{suffix}" if suffix else None

# -------------------- Model Loading --------------------

def load_model(model_name):
    if model_name in MODELS:
        return

    logger.info(f"Loading model: {model_name}")
    if model_name == GEMMA_MODEL:
        MODELS[model_name] = pipeline(
            "image-text-to-text", # Per your specific Gemma config
            model=model_name,
            device=DEVICE,
            dtype=torch.bfloat16,
            token=HF_TOKEN,
        )
    else:
        logger.info(f"Loading MarianMT via AutoModel for {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN).to(DEVICE)
        
        # Store as a dict since we need both model and tokenizer
        MODELS[model_name] = {
            "model": model,
            "tokenizer": tokenizer
        }

# -------------------- Inference Helpers --------------------

def translate_marian(text, model_name):
    # Retrieve the model and tokenizer from our cache
    data = MODELS[model_name]
    model = data["model"]
    tokenizer = data["tokenizer"]
    
    # Perform manual inference
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        generated_tokens = model.generate(**inputs)
    
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return result[0]


def translate_gemma(text, source_culture, target_culture):
    pipe = MODELS[GEMMA_MODEL]
    src_lang = source_culture[:2].lower()
    tgt_lang = target_culture[:2].lower()

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": text,
            "source_lang_code": src_lang,
            "target_lang_code": tgt_lang
        }]
    }]

    result = pipe(text=messages, max_new_tokens=512, generate_kwargs={"do_sample": False})
    
    try:
        return result[0]["generated_text"][-1]["content"].strip()
    except (KeyError, IndexError):
        return "Error: Unexpected output format from Gemma pipeline."

# -------------------- RunPod Handler --------------------

def handler(event):
    """
    RunPod handler that receives the event dictionary.
    Expected input: { "input": { "text": "...", "culture": "en-US", "model": "marian" } }
    """
    try:
        # RunPod sends data inside the 'input' key
        job_input = event.get("input", {})
        
        text = job_input.get("text")
        target_culture = job_input.get("culture")
        source_culture = job_input.get("source_culture", "de-DE")
        model_type = job_input.get("model", "marian").lower()

        # Validation
        if not text:
            return {"error": "Missing 'text' in input"}
        if not target_culture:
            return {"error": "Missing 'culture' (target language) in input"}

        # Logic Selection
        if model_type == "gemma":
            model_name = GEMMA_MODEL
        else:
            model_name = get_model_name(target_culture)
            if not model_name:
                return {"error": f"Culture '{target_culture}' not supported for Marian models."}

        # Lazy load model if not in cache
        load_model(model_name)

        # Run Inference
        if model_name == GEMMA_MODEL:
            translated = translate_gemma(text, source_culture, target_culture)
        else:
            translated = translate_marian(text, model_name)

        return {"translation": translated}

    except Exception as e:
        logger.exception("Inference failed")
        return {"error": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})