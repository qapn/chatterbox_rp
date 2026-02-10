import runpod
import traceback

model = None
INIT_ERROR = None


def load_model():
    global model

    print("[init] Importing chatterbox...", flush=True)
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("[init] Resolving cached model path...", flush=True)
    from huggingface_hub import snapshot_download
    local_path = snapshot_download('ResembleAI/chatterbox-turbo', local_files_only=True)

    print("[init] Loading model...", flush=True)
    model = ChatterboxTurboTTS.from_local(local_path, device='cuda')

    print("[init] Model ready.", flush=True)


try:
    load_model()
except Exception:
    INIT_ERROR = traceback.format_exc()
    print(f"[init] FAILED:\n{INIT_ERROR}", flush=True)


def handler(job):
    if INIT_ERROR:
        return {'error': f'Model failed to load:\n{INIT_ERROR}'}

    import base64
    import io
    import os
    import tempfile

    import torch
    import torchaudio

    inp = job['input']

    text = inp.get('text', '')
    if not text:
        return {'error': 'text is required'}

    ref_audio_b64 = inp.get('reference_audio_base64')
    audio_prompt_path = None
    if ref_audio_b64:
        raw = base64.b64decode(ref_audio_b64)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(raw)
        tmp.close()
        audio_prompt_path = tmp.name

    temperature = float(inp.get('temperature', 0.8))
    exaggeration = float(inp.get('exaggeration', 0.0))
    cfg_weight = float(inp.get('cfg_weight', 0.0))
    repetition_penalty = float(inp.get('repetition_penalty', 1.2))
    top_p = float(inp.get('top_p', 0.95))
    top_k = int(inp.get('top_k', 1000))
    min_p = float(inp.get('min_p', 0.0))
    seed = int(inp.get('seed', 0))

    try:
        if seed != 0:
            torch.manual_seed(seed)

        audio = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        buf = io.BytesIO()
        torchaudio.save(buf, audio.cpu(), model.sr, format='wav')
        buf.seek(0)
        audio_base64 = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        return {'error': str(e)}
    finally:
        if audio_prompt_path:
            os.unlink(audio_prompt_path)

    return {
        'audio_base64': audio_base64,
        'sample_rate': model.sr,
        'format': 'wav',
    }


runpod.serverless.start({'handler': handler})
