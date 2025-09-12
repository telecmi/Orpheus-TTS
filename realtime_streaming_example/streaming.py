import torch
import asyncio
from orpheus_tts import OrpheusModel
import wave
import time
async def test_streaming_with_timestamps():
   # model = OrpheusModel(model_name="canopylabs/3b-hi-ft-research_release")
   model = OrpheusModel(model_name="./OrpheusTTS/checkpoints",  #PATH TO THE CHECKPOINTS
                        # dtype = "half" #torch.float16
                        )
   await model.ensure_warm() #warmup here
    
   '''
   Prompt is the text field  
   '''
   prompt ='''Delhi की एक retail chain ने हमारे solutions से अपनी sales में 
   30 percent तक increase देखी है, 
   <hmm..> उनका feedback बहुत encouraging रहा है|
   '''
    
   start_time = time.monotonic()
   first_chunk_time = None
   
   with wave.open("output_hindi.wav", "wb") as wf:
      wf.setnchannels(1)
      wf.setsampwidth(2)
      wf.setframerate(24000)
      
      chunk_counter = 0
      
      async for audio_chunk in model.generate_speech_async(
         prompt=prompt,
         voice=None
      ):
         chunk_counter += 1
         current_time = time.monotonic() - start_time
         
         # Record time to first chunk (TTFB)
         if first_chunk_time is None:
               first_chunk_time = current_time
               print(f"⚡ First chunk received at: {first_chunk_time:.3f}s")
         
         chunk_size = len(audio_chunk)
         print(f"Chunk {chunk_counter}: {chunk_size} bytes at {current_time:.3f}s")
         wf.writeframes(audio_chunk)
      
      total_time = time.monotonic() - start_time
      print(f"\n📊 Time to first byte: {first_chunk_time:.3f}s")
      print(f"📊 Total generation time: {total_time:.3f}s")

asyncio.run(test_streaming_with_timestamps())

from orpheus_tts import OrpheusModel
import wave
import time

model = OrpheusModel(model_name ="SachinTelecmi/Orpheus-tts-hi", max_model_len=1024)
prompt = '''बिल्कुल! हमारा CRM software businesses को leads manage करने में मदद करता है। क्या आप जानना चाहेंगे कि ये software आपके business के लिए कैसे फायदेमंद हो सकता है?'''

start_time = time.monotonic()
syn_tokens = model.generate_speech_audio(
   prompt=prompt,
   ref_text = "नमस्ते, मैं राकेश बोल रहा हूँ <happy> How can I assist you today?",
   voice_path = "/home/user/voice/Orpheus-TTS/finetune/hf_cache/datasets--telecmiusa--tts-hi-data/snapshots/d564239b4542d4e25ee213660bf0104e700858ac/SPEECHRIV/1_AGENT.wav",
   voice=None,
   )

with wave.open("output_hi_23.wav", "wb") as wf:
         wf.setnchannels(1)
         wf.setsampwidth(2)
         wf.setframerate(24000)
         
         chunk_counter = 0
         
         async for audio_chunk in model.generate_speech_async(
            prompt=prompt,
         ):
            chunk_counter += 1
            current_time = time.monotonic() - start_time
            
            # Record time to first chunk (TTFB)
            if first_chunk_time is None:
                  first_chunk_time = current_time
                  print(f"⚡ First chunk received at: {first_chunk_time:.3f}s")
            
            chunk_size = len(audio_chunk)
            print(f"Chunk {chunk_counter}: {chunk_size} bytes at {current_time:.3f}s")
            wf.writeframes(audio_chunk)
         
         total_time = time.monotonic() - start_time
         print(f"\n📊 Time to first byte: {first_chunk_time:.3f}s")
         print(f"📊 Total generation time: {total_time:.3f}s")




# CLONING
# import torch
# import asyncio
# from orpheus_tts import OrpheusModel
# import wave
# import time
# async def test_streaming_with_timestamps():
#    # model = OrpheusModel(model_name="canopylabs/3b-hi-ft-research_release")
#    model = OrpheusModel(model_name ="SachinTelecmi/Orpheus-tts-hi", max_model_len=1024)
#    prompt = '''बिल्कुल! हमारा CRM software businesses को leads manage करने में मदद करता है। क्या आप जानना चाहेंगे कि ये software आपके business के लिए कैसे फायदेमंद हो सकता है?'''

#    # await model.ensure_warm() #warmup here
    
#    # '''
#    # Prompt is the text field  
#    # '''
#    # prompt ='''Delhi की एक retail chain ने हमारे solutions से अपनी sales में, 30 percent तक increase देखी है, <hmm..> उनका feedback बहुत encouraging रहा है|
#    # '''
    
#    start_time = time.monotonic()
#    first_chunk_time = None
   
#    with wave.open("output_hi.wav", "wb") as wf:
#       wf.setnchannels(1)
#       wf.setsampwidth(2)
#       wf.setframerate(24000)
      
#       chunk_counter = 0
      
#       async for audio_chunk in model.generate_speech_async(
#          prompt=prompt,
#          ref_text = "नमस्ते, मैं राकेश बोल रहा हूँ <happy> How can I assist you today?",
#          voice_path = "/home/user/voice/Orpheus-TTS/finetune/hf_cache/datasets--telecmiusa--tts-hi-data/snapshots/d564239b4542d4e25ee213660bf0104e700858ac/SPEECHRIV/1_AGENT.wav",
#          voice=None,

#       ):
#          chunk_counter += 1
#          current_time = time.monotonic() - start_time
         
#          # Record time to first chunk (TTFB)
#          if first_chunk_time is None:
#                first_chunk_time = current_time
#                print(f"⚡ First chunk received at: {first_chunk_time:.3f}s")
         
#          chunk_size = len(audio_chunk)
#          print(f"Chunk {chunk_counter}: {chunk_size} bytes at {current_time:.3f}s")
#          wf.writeframes(audio_chunk)
      
#       total_time = time.monotonic() - start_time
#       print(f"\n📊 Time to first byte: {first_chunk_time:.3f}s")
#       print(f"📊 Total generation time: {total_time:.3f}s")

# asyncio.run(test_streaming_with_timestamps())
