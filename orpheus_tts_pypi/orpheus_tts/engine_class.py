import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import threading
import queue
from .decoder import tokens_decoder_sync, tokens_decoder
import uuid
class OrpheusModel:
    def __init__(self, model_name, dtype=torch.bfloat16):
        self.model_name = model_name #self._map_model_params(model_name)
        self.dtype = dtype
        self.engine = self._setup_engine()
        self.available_voices = ["zoe", "zac","jess", "leo", "mia", "julia", "leah", "ऋतिका"]
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self._warmed_up = False

        # Pre-warm the model
        # asyncio.run(self._prewarm())
    # def _map_model_params(self, model_name):
    #     model_map = {
    #         # "nano-150m":{
    #         #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
    #         # }, 
    #         # "micro-400m":{
    #         #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
    #         # }, 
    #         # "small-1b":{
    #         #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
    #         # },
    #         "medium-3b":{
    #             # "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
    #             # "repo_id":  "canopylabs/3b-hi-ft-research_release"
    #             "repo_id":"SachinTelecmi/Orpheus-tts-hi"
    #         },
    #     }
    #     unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
    #     if (model_name  in unsupported_models):
    #         raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported, small, micro and nano models will be released very soon")
    #     elif model_name in model_map:
    #         return model_name[model_name]["repo_id"]
    #     else:
    #         return model_name
     
    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype= "auto", #self.dtype,
            quantization="bitsandbytes",  # Try this first
            load_format="bitsandbytes",  
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=2048,  # Add this
            enable_prefix_caching=True,
            enforce_eager=False,
            max_num_seqs = 1
        )
        return AsyncLLMEngine.from_engine_args(engine_args)
    
    def validate_voice(self, voice):
        if voice:
            if voice not in self.engine.available_voices:
                raise ValueError(f"Voice {voice} is not available for model {self.model_name}")
    
    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokeniser(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokeniser(prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string


    def generate_tokens_sync(self, prompt, voice=None, 
                             request_id=None ,
                             temperature=0.4, top_p=0.8, 
                             max_tokens=1200, 
                             stop_token_ids = [49158], 
                             repetition_penalty=1.1
                            ):
        if request_id is None:
            request_id = f"req-{uuid.uuid4()}",
        prompt_string = self._format_prompt(prompt, voice)
        # print(prompt)
        sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,  # Adjust max_tokens as needed.
        stop_token_ids = stop_token_ids, 
        repetition_penalty=repetition_penalty, 
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
                # Place each token text into the queue.
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)  # Sentinel to indicate completion.

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()
    
    def generate_speech(self, **kwargs):
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))
    
    async def generate_tokens_async(
        self, prompt, voice=None, request_id=None,
        temperature=0.5, top_p=0.5, max_tokens=1024,
        stop_token_ids=[49158], repetition_penalty=1.05
    ):
        if request_id is None:
            request_id = f"req-{uuid.uuid4()}"
        prompt_string = self._format_prompt(prompt, voice)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        prev = ""
        async for result in self.engine.generate(
            prompt=prompt_string,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            full = result.outputs[0].text  # cumulative
            delta = full[len(prev):]
            if delta:
                yield delta
                prev = full

    async def generate_speech_async(self, **kwargs):
        # tokens_decoder is YOUR async decoder from the snippet
        async for audio_bytes in tokens_decoder(self.generate_tokens_async(**kwargs)):
            yield audio_bytes  # PCM16 bytes from SNAC


    async def ensure_warm(self):
        """Call this before first use"""
        if not self._warmed_up:
            await self._prewarm()
            self._warmed_up = True


    async def _prewarm(self):
        """Pre-warm the model with a dummy request"""
        dummy_prompt = ["नमस्ते","Delhi की एक retail chain","<hmm..> उनका feedback बहुत encouraging रहा है ।"]
        for dummy  in dummy_prompt:
            async for _ in self.generate_tokens_async(
                dummy, 
                voice=None,
                max_tokens=50,
                temperature=0.1
            ):
                pass  # Just generate first token to warm up


