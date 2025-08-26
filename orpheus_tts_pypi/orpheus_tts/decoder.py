from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import logging 



logger = logging.getLogger(__name__)

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda"
model = model.to(snac_device)

#Warm up the decoder 
if snac_device == "cuda":
   torch.backends.cudnn.benchmark = True 
   torch.backends.cudnn.deterministic = False 

   dummy_codes = [
      #codes 0
      torch.randint(0, 4096, size=(1, 1), dtype=torch.int32, device=snac_device),
      #codes_1 
      torch.randint(0, 4096, size=(1, 1), dtype=torch.int32, device = snac_device),
      #codes 2 
      torch.randint(0, 4096, size=(1, 1), dtype=torch.int32, device = snac_device)
   ]

   with torch.inference_mode():
     output_audio_sample = model.decode(dummy_codes)


CUSTOM_TOKEN_PREFIX = "<custom_token_"
CUSTOM_TOKEN_SUFFIX = ">"
MIN_FRAMES_FIRST = 7
MIN_FRAMES_SUBSEQ = 28 
PROCESS_EVERY = 7

def convert_to_audio(multiframe, count):

  """
    Highly optimized version of convert_to_audio that eliminates inefficient 
    tensor operations and reduces CPU-GPU transfers for much faster inference
    on high-end GPUs. Optimized for concurrent requests.
    """

  #return audio bytes

  frames = []
  if len(multiframe) < 7:
    return None
  
  num_frames = len(multiframe) // 7

  # Pre-allocate tensors with the right shape and directly on target device
  codes_0 = torch.tensor([1, num_frames], device=snac_device, dtype=torch.int32)
  codes_1 = torch.tensor([1, num_frames], device=snac_device, dtype=torch.int32)
  codes_2 = torch.tensor([1, num_frames], device=snac_device, dtype=torch.int32)

  # Fill tensors with direct indexing
  for i in range(num_frames):
      base_idx = i * 7
      codes_0[0, i] = multiframe[base_idx]
      
      codes_1[0, i*2] = multiframe[base_idx + 1]
      codes_1[0, i*2 + 1] = multiframe[base_idx + 4]
      
      codes_2[0, i*4] = multiframe[base_idx + 2]
      codes_2[0, i*4 + 1] = multiframe[base_idx + 3]
      codes_2[0, i*4 + 2] = multiframe[base_idx + 5]
      codes_2[0, i*4 + 3] = multiframe[base_idx + 6]
    
  # validation for range check
  if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or
      torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or
      torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
      return None
  
  codes = [codes_0, codes_1, codes_2]
    
  with torch.inference_mode():   
      audio_hat = model.decode(codes)
      audio_slice = audio_hat[:, :, 2048:4096]
      
      if snac_device == "cuda":
          audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
          return audio_int16_tensor.cpu().numpy().tobytes()
      else:
          audio_np = audio_slice.numpy()
          return (audio_np * 32767.0).round().astype(np.int16).tobytes()

def turn_token_into_id(token_string, index):
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        print("No token found in the string")
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None
  
    


def fade_in(audio_bytes, fade_ms=15, sr=24000):
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    fade_len = int(sr * fade_ms / 1000)
    window = np.linspace(0, 1, fade_len)
    audio[:fade_len] *= window
    return audio.astype(np.int16).tobytes()

# async def tokens_decoder(token_gen):
#     buffer = []
#     count = 0
#     consecutive_nones = 0
#     first_chunk_sent = False
    
#     async for token_sim in token_gen:       
#         token = turn_token_into_id(token_sim, count)
        
#         # Handle None tokens
#         # if token == 49158:
#         #    print("EOS reached, stopping stream")
#         #    break
#         if token is None:
#             consecutive_nones += 1
#             if consecutive_nones >= 2:
#                 break
#             continue
#         else:
#             consecutive_nones = 0
            
#         if token > 0:
#             buffer.append(token)
#             count += 1
            
#             # Fast first chunk (with padding if needed)
#             if not first_chunk_sent and len(buffer) >= 14:
#                 # Pad to 28 tokens for first chunk
#                 # padded = buffer + [buffer[-1]] * (28 - len(buffer))
#                 buffer_to_proc = buffer[-14:]
#                 audio_samples = convert_to_audio(buffer_to_proc, count) #audio_bytes
#                 audio_samples = fade_in(audio_samples)
#                 if audio_samples is not None:
#                     yield audio_samples
#                     first_chunk_sent = True
            
#             # Regular processing after first chunk
#             elif first_chunk_sent and count % 7 == 0 and len(buffer) >= 28:
#                 buffer_to_proc = buffer[-28:]
#                 audio_samples = convert_to_audio(buffer_to_proc, count)
#                 if audio_samples is not None:
#                     yield audio_samples


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
                audio = fade_in(audio)
                yield audio
        elif first_chunk_sent and count % PROCESS_EVERY == 0:
            audio = convert_to_audio(buffer[-MIN_FRAMES_SUBSEQ:], count)
            audio = fade_in(audio)
            if audio is not None:
                yield audio


# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
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