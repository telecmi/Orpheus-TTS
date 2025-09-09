from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import os
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(snac_device)

# Local cache to avoid repeated parsing of the same token strings
_token_id_cache = {}
_MAX_CACHE_SIZE = 25000
_CUSTOM_TOKEN_PREFIX = "<custom_token_"

def warm_up_decoder():
    """Warm up the SNAC decoder to reduce latency on first inference."""
    if snac_device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Warm up the model with a dummy inference
        dummy_codes = [
            torch.randint(0, 4096, (1, 1), dtype=torch.int32, device=snac_device),
            torch.randint(0, 4096, (1, 2), dtype=torch.int32, device=snac_device),
            torch.randint(0, 4096, (1, 4), dtype=torch.int32, device=snac_device)
        ]
        
        with torch.inference_mode():
            for i in range(3):
                logger.info(f"Warm-up iteration {i+1}/3")
                _ = model.decode(dummy_codes)
def turn_token_into_id(token_string, index):
    """Convert a custom token string to its numeric ID with caching.

    Args:
        token_string (str): The literal token text coming from the model.
        index (int): Absolute token position (used for offset calculation).

    Returns:
        Optional[int]: Numeric token ID or ``None`` if the token is invalid.
    """
    token_string = token_string.strip()
    mod = index % 7  # Offset cycles every 7 tokens
    cache_key = (token_string, mod)

    if cache_key in _token_id_cache:
        return _token_id_cache[cache_key]

    # Locate the last occurrence of the custom token pattern (mirrors original logic)
    last_idx = token_string.rfind(_CUSTOM_TOKEN_PREFIX)
    if last_idx == -1:
        if len(_token_id_cache) < _MAX_CACHE_SIZE:
            _token_id_cache[cache_key] = None
        return None

    token_substr = token_string[last_idx:]  # from prefix to end

    if not token_substr.startswith(_CUSTOM_TOKEN_PREFIX) or not token_substr.endswith(">"):
        if len(_token_id_cache) < _MAX_CACHE_SIZE:
            _token_id_cache[cache_key] = None
        return None

    digits = token_substr[len(_CUSTOM_TOKEN_PREFIX):-1]
    if not digits.isdigit():
        if len(_token_id_cache) < _MAX_CACHE_SIZE:
            _token_id_cache[cache_key] = None
        return None

    token_id = int(digits) - 10 - (mod * 4096)
    if token_id<0 or token_id>4096:
        return None

    if len(_token_id_cache) < _MAX_CACHE_SIZE:
        _token_id_cache[cache_key] = token_id

    return token_id

def convert_to_audio(multiframe, count):
    """
    Maximum speed version - removes all validation and error checking.
    Use only if you're confident in your input data quality.
    """
    num_frames = len(multiframe) // 7
    
    # Pre-allocate tensors
    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    # Direct tensor assignment (fastest)
    multiframe_tensor = torch.tensor(multiframe, dtype=torch.int32, device=snac_device)
    
    for i in range(num_frames):
        base_idx = i * 7
        codes_0[0, i] = multiframe_tensor[base_idx]
        codes_1[0, i*2:i*2+2] = multiframe_tensor[[base_idx + 1, base_idx + 4]]
        codes_2[0, i*4:i*4+4] = multiframe_tensor[[base_idx + 2, base_idx + 3, base_idx + 5, base_idx + 6]]
    
    with torch.inference_mode():
        audio_hat = model.decode([codes_0, codes_1, codes_2])
        # audio_slice = audio_hat[:, :, 2048:4096]
        audio_slice = audio_hat[:, :, -6144:]  # Use all generated audio        
        return (audio_slice * 32767.0).round().to(torch.int16).cpu().numpy().tobytes()


async def tokens_decoder(token_gen):
    """Decode tokens into audio chunks with reduced latency.

    The first audio chunk is emitted as soon as **one** frame (7 tokens) is
    available, drastically reducing time-to-first-byte. Subsequent chunks are
    processed every 7 tokens using a sliding window of the last 4 frames (28
    tokens) mirroring the original behaviour.
    """
    buffer = []
    count = 0
    first_chunk_sent = False
    MIN_FRAMES_FIRST = 7      # 1 frame for ultra-low latency
    MIN_FRAMES_SUBSEQ = 42    # 4 frames
    PROCESS_EVERY = 21         # process at every full frame boundary

    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is None or token <= 0:
            continue

        buffer.append(token)
        count += 1

        if not first_chunk_sent and count >= MIN_FRAMES_FIRST:
            audio = convert_to_audio(buffer[-MIN_FRAMES_FIRST:], count)
            if audio is not None:
                first_chunk_sent = True
                yield audio
        elif first_chunk_sent and count % PROCESS_EVERY == 0:
            audio = convert_to_audio(buffer[-MIN_FRAMES_SUBSEQ:], count)
            if audio is not None:
                yield audio


def tokens_decoder_sync(syn_token_gen):

    audio_queue = queue.Queue()

    # Convert the synchronous token generator into an async generator.
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        # tokens_decoder.tokens_decoder is assumed to be an async generator that processes tokens.
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()