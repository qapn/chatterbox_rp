FROM madiator2011/better-pytorch:cuda12.4-torch2.6.0

RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install chatterbox-tts --no-cache-dir --no-deps

RUN pip install s3tokenizer --no-cache-dir --no-deps

RUN pip install "numpy>=1.24.0,<1.26.0" librosa==0.11.0 torchaudio==2.6.0 \
    "transformers==4.46.3" "diffusers==0.29.0" "resemble-perth==1.0.1" \
    "conformer==0.3.2" "safetensors==0.5.3" spacy-pkuseg "pykakasi==2.3.0" \
    pyloudnorm omegaconf onnx runpod --no-cache-dir

RUN pip install resemble-enhance --no-cache-dir --no-deps

RUN pip install scipy resampy --no-cache-dir

RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('ResembleAI/chatterbox-turbo', \
allow_patterns=['*.safetensors','*.json','*.txt','*.pt','*.model'])"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
