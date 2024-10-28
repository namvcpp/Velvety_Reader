# Velvety_Reader/apps/reader/src/api/translate.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7127"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
device_en2vi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_en2vi.to(device_en2vi)

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslationRequest):
    input_ids = tokenizer_en2vi(request.text, return_tensors="pt").input_ids.to(device_en2vi)
    output_ids = model_en2vi.generate(
        input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    return {"translated_text": " ".join(vi_text)}