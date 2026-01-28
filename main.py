import os
os.makedirs('/workspace/huggingface/tmp', exist_ok=True)
os.environ['HF_HOME'] = '/workspace/huggingface'
os.environ['HF_HUB_CACHE'] = '/workspace/huggingface/hub'
os.environ['TMPDIR'] = '/workspace/huggingface/tmp'

import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# -------------------- Configuration & Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
os.environ["HF_HUB_READ_TIMEOUT"] = "300"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
HF_TOKEN = os.getenv("HF_TOKEN")

# Global State
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEMMA_MODEL = "google/translategemma-4b-it"

app = FastAPI(title="Translation API")

# -------------------- Schemas --------------------

class TranslationRequest(BaseModel):
    text: str
    culture: str  # Target culture (e.g., 'en-US')
    source_culture: str = "de-DE"
    model: str = "marian" # 'marian' or 'gemma'

# -------------------- Model Logic --------------------

def get_model_name(culture: str):
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

def load_model(model_name: str):
    if model_name in MODELS:
        return

    logger.info(f"Loading model: {model_name}")
    if model_name == GEMMA_MODEL:
        MODELS[model_name] = pipeline(
            "image-text-to-text", 
            model=model_name,
            device=DEVICE,
            dtype=torch.bfloat16,
            token=HF_TOKEN,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN).to(DEVICE)
        MODELS[model_name] = {"model": model, "tokenizer": tokenizer}

def translate_marian(text, model_name):
    data = MODELS[model_name]
    inputs = data["tokenizer"](text, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        generated_tokens = data["model"].generate(**inputs)
    return data["tokenizer"].batch_decode(generated_tokens, skip_special_tokens=True)[0]

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
        return "Error: Unexpected output format from Gemma."

# -------------------- API Endpoints --------------------

@app.get("/ping")
async def health_check():
    return {"status": "healthy", "device": str(DEVICE)}

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Determine Model Name
        if request.model.lower() == "gemma":
            model_name = GEMMA_MODEL
        else:
            model_name = get_model_name(request.culture)
            if not model_name:
                raise HTTPException(status_code=400, detail=f"Culture '{request.culture}' not supported for Marian.")

        # Lazy Load
        load_model(model_name)

        # Inference
        if model_name == GEMMA_MODEL:
            translated = translate_gemma(request.text, request.source_culture, request.culture)
        else:
            translated = translate_marian(request.text, model_name)

        return {"translation": translated, "model_used": model_name}

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


