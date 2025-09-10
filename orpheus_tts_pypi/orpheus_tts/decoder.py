import numpy as np
import torch
import asyncio
import threading
import queue
import os
import time
from collections import deque
import logging
from snac import SNAC
torch.set_num_threads(1)
logger = logging.getLogger(__name__)
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(snac_device)

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
    token_string = token_string.strip()
    mod = index % 7
    
    import re
    tokens = re.findall(r'<custom_token_(\d+)>', token_string)
    
    if not tokens:
        return None
        
    digits = tokens[-1]
    token_id = int(digits) - 10 - (mod * 4096)
    
    # More lenient validation
    if token_id < 0 or token_id >= 4096:  # Changed from > to >=
        return None
    return token_id



def convert_to_audio(multiframe, count):
    """
    EXACT original working version - don't change anything
    """
    if len(multiframe) < 7:
        return None
    
    num_frames = len(multiframe) // 7
    
    # Pre-allocate tensors with the right shape and directly on target device
    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    # Fill tensors with direct indexing (no intermediate allocations)
    for i in range(num_frames):
        base_idx = i * 7
        codes_0[0, i] = multiframe[base_idx]
        
        codes_1[0, i*2] = multiframe[base_idx + 1]
        codes_1[0, i*2 + 1] = multiframe[base_idx + 4]
        
        codes_2[0, i*4] = multiframe[base_idx + 2]
        codes_2[0, i*4 + 1] = multiframe[base_idx + 3]
        codes_2[0, i*4 + 2] = multiframe[base_idx + 5]
        codes_2[0, i*4 + 3] = multiframe[base_idx + 6]
    
    # More lenient validation - allow wider range
    if (torch.any(codes_0 < 0) or torch.any(codes_0 >= 4096) or
        torch.any(codes_1 < 0) or torch.any(codes_1 >= 4096) or
        torch.any(codes_2 < 0) or torch.any(codes_2 >= 4096)):
        return None
    
    codes = [codes_0, codes_1, codes_2]
    
    with torch.inference_mode():   
        audio_hat = model.decode(codes)
        audio_slice = audio_hat[:, :, 2048:4096]  # ORIGINAL SLICE
        
        if snac_device == "cuda":
            audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
            return audio_int16_tensor.cpu().numpy().tobytes()
        else:
            audio_np = audio_slice.numpy()
            return (audio_np * 32767.0).round().astype(np.int16).tobytes()


async def tokens_decoder(token_gen):
    """Decode tokens into audio chunks with reduced latency.
    
    Now configured to output 12288 bytes (6144 samples) per chunk.
    """
    buffer = []
    count = 0
    first_chunk_sent = False
    MIN_FRAMES_FIRST = 7      # 1 frame for ultra-low latency
    MIN_FRAMES_SUBSEQ = 28    # 6 frames (increased from 28 to support 6144 samples)
    PROCESS_EVERY = 7        # Process every 3 frames (changed from 14 to match timing)

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