import os
from pathlib import Path
from typing import List, Tuple
import torch
import librosa
import numpy as np
from scipy.io.wavfile import write
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC



class OrpheusTTSVoiceClone:
    """Voice cloning TTS engine using reference audio for voice synthesis."""
    
    CODES_PER_GROUP = 7
    SAMPLE_RATE = 24000
    
    def __init__(
        self,
        model_name: str = "SachinTelecmi/Orpheus-tts-hi", #https://huggingface.co/SachinTelecmi/Orpheus-tts-hi
        snac_model_name: str = "hubertsiuzdak/snac_24khz",
        device: str = "cuda"
    ):
        """Initialize the voice cloning TTS engine.
        
        Args:
            config: Configuration dictionary containing token constants
            config_path: Path to YAML config file (alternative to config dict)
            model_name: HuggingFace model identifier for the TTS model
            snac_model_name: HuggingFace model identifier for SNAC audio decoder
            device: Device to run models on
        """
        
        
        # Set token constants from config
        self._setup_token_constants()
        
        # Initialize models
        self.device = device
        self.model_name = model_name
        self.snac_model_name = snac_model_name
        
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.snac_model = self._load_snac_model()
    
    def _setup_token_constants(self):
        """Setup token constants from configuration."""
        self.TOKENIZER_LENGTH = 128256
        self.START_OF_TEXT = 128000
        self.END_OF_TEXT = 128009
        self.START_OF_SPEECH = 128257
        self.END_OF_SPEECH = 128258
        self.START_OF_HUMAN = 128259
        self.END_OF_HUMAN = 128260
        self.START_OF_AI = 128261
        self.END_OF_AI = 128262
        self.PAD_TOKEN = 128263
        self.AUDIO_TOKENS_START = 128266
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer."""
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load the language model."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
        )
        model.to(self.device)
        return model
    
    def _load_snac_model(self) -> SNAC:
        """Load the SNAC audio model."""
        return SNAC.from_pretrained(self.snac_model_name).to(self.device)
    
    def encode_reference_audio(self, audio_file_path: str) -> List[int]:
        """Encode reference audio file to audio tokens.
        
        Args:
            audio_file_path: Path to the reference audio file
            
        Returns:
            List of audio token IDs
        """
        # Load audio at 24kHz sample rate
        audio_array, _ = librosa.load(audio_file_path, sr=self.SAMPLE_RATE)
        waveform = torch.from_numpy(audio_array).unsqueeze(0).to(dtype=torch.float32)
        waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        # Encode with SNAC
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        # Convert to interleaved token sequence
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
            all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

        return all_codes
    
    def prepare_voice_clone_inputs(
        self,
        reference_audio_path: str,
        reference_transcript: str,
        target_texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare inputs for voice cloning generation.
        
        Args:
            reference_audio_path: Path to reference audio file
            reference_transcript: Transcript of the reference audio
            target_texts: List of texts to synthesize in the reference voice
            
        Returns:
            Tuple of (input_ids, attention_mask) tensors
        """
        # Encode reference audio
        audio_tokens = self.encode_reference_audio(reference_audio_path)
        
        # Special tokens
        start_token = torch.tensor([[self.START_OF_HUMAN]], dtype=torch.int64)
        mid_tokens = torch.tensor([[self.END_OF_TEXT, self.END_OF_HUMAN, self.START_OF_AI]], dtype=torch.int64)
        final_tokens = torch.tensor([[self.END_OF_SPEECH, self.END_OF_AI]], dtype=torch.int64)
        
        # Tokenize reference transcript
        transcript_tokens = self.tokenizer(reference_transcript, return_tensors="pt")
        
        # Create reference prompt with audio tokens
        ref_input_ids = transcript_tokens['input_ids']
        reference_prompt = torch.cat([
            start_token, 
            ref_input_ids, 
            mid_tokens,
            torch.tensor([audio_tokens], dtype=torch.int64), 
            final_tokens
        ], dim=1)
        
        # Prepare target text prompts
        all_input_ids = []
        for text in target_texts:
            text_tokens = self.tokenizer(text, return_tensors="pt").input_ids
            full_input = torch.cat([
                reference_prompt,
                start_token,
                text_tokens,
                mid_tokens
            ], dim=1)
            all_input_ids.append(full_input)
        
        # Pad sequences to same length
        max_length = max(ids.shape[1] for ids in all_input_ids)
        
        padded_inputs = []
        attention_masks = []
        
        for input_ids in all_input_ids:
            padding_length = max_length - input_ids.shape[1]
            
            # Pad with PAD_TOKEN
            padded_input = torch.cat([
                torch.full((1, padding_length), self.PAD_TOKEN, dtype=torch.int64),
                input_ids
            ], dim=1)
            
            # Create attention mask
            attention_mask = torch.cat([
                torch.zeros((1, padding_length), dtype=torch.int64),
                torch.ones((1, input_ids.shape[1]), dtype=torch.int64)
            ], dim=1)
            
            padded_inputs.append(padded_input)
            attention_masks.append(attention_mask)
        
        # Stack all inputs
        input_ids = torch.cat(padded_inputs, dim=0).to(self.device)
        attention_mask = torch.cat(attention_masks, dim=0).to(self.device)
        
        return input_ids, attention_mask
    
    def generate_speech(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 990,
        temperature: float = 0.5,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> torch.Tensor:
        """Generate speech tokens using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            
        Returns:
            Generated token IDs
        """
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.END_OF_SPEECH,
                pad_token_id=self.PAD_TOKEN
            )
        
        return generated_ids
    
    def decode_audio_tokens(self, generated_ids: torch.Tensor) -> List[torch.Tensor]:
        """Convert generated token IDs to audio waveforms.
        
        Args:
            generated_ids: Generated token IDs from the model
            
        Returns:
            List of decoded audio tensors
        """
        # Find start of audio tokens
        token_to_find = 128257
        token_to_remove = 128258
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
        else:
            cropped_tensor = generated_ids

        _mask = cropped_tensor != token_to_remove
        processed_rows = []
        for row in cropped_tensor:
            # Apply the mask to each row
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []
        for row in processed_rows:
            # row is a 1D tensor with its own length
            row_length = row.size(0)
            new_length = (row_length // 7) * 7  # largest multiple of 7 that fits in this row
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        my_samples = []
        for code_list in code_lists:
            samples = self._redistribute_and_decode(code_list)
            my_samples.append(samples)

        return my_samples

    
    def _redistribute_and_decode(self, code_list: List[int]) -> torch.Tensor:
        """Redistribute codes into SNAC layers and decode to audio.
        
        Args:
            code_list: List of audio codes
            
        Returns:
            Decoded audio tensor
        """
        num_groups = len(code_list) // self.CODES_PER_GROUP
        
        layer_1, layer_2, layer_3 = [], [], []
        
        for i in range(num_groups):
            base_idx = self.CODES_PER_GROUP * i
            
            layer_1.append(code_list[base_idx])
            layer_2.extend([
                code_list[base_idx + 1] - 4096,
                code_list[base_idx + 4] - (4 * 4096)
            ])
            layer_3.extend([
                code_list[base_idx + 2] - (2 * 4096),
                code_list[base_idx + 3] - (3 * 4096),
                code_list[base_idx + 5] - (5 * 4096),
                code_list[base_idx + 6] - (6 * 4096)
            ])
        
        # Create code tensors
        codes = [
            torch.tensor(layer_1, device=self.device).unsqueeze(0),
            torch.tensor(layer_2, device=self.device).unsqueeze(0),
            torch.tensor(layer_3, device=self.device).unsqueeze(0)
        ]
        
        return self.snac_model.decode(codes)

    
    def clone_voice(
        self,
        reference_audio_path: str,
        reference_transcript: str,
        target_texts: List[str]
    ) -> List[np.ndarray]:
        """Clone a voice using reference audio and generate speech for target texts.
        
        Args:
            reference_audio_path: Path to reference audio file
            reference_transcript: Transcript of the reference audio
            target_texts: List of texts to synthesize
            
        Returns:
            List of audio waveforms as numpy arrays
        """
        # Prepare inputs
        input_ids, attention_mask = self.prepare_voice_clone_inputs(
            reference_audio_path, reference_transcript, target_texts
        )
        
        # Generate speech tokens
        generated_ids = self.generate_speech(input_ids, attention_mask)
        
        # Decode to audio
        audio_tensors = self.decode_audio_tokens(generated_ids)
        
        # Convert to numpy arrays
        audio_arrays = []
        for tensor in audio_tensors:
            if isinstance(tensor, torch.Tensor):
                audio_array = tensor.detach().squeeze().cpu().numpy()
            else:
                audio_array = np.squeeze(tensor)
            audio_arrays.append(audio_array)
        
        return audio_arrays
    
    def save_audio(
        self,
        audio_arrays: List[np.ndarray],
        output_paths: List[str],
        sample_rate: int = None
    ):
        """Save audio arrays to files.
        
        Args:
            audio_arrays: List of audio arrays to save
            output_paths: List of output file paths
            sample_rate: Sample rate for saving (defaults to class default)
        """
        sample_rate = sample_rate or self.SAMPLE_RATE
        
        for audio_array, output_path in zip(audio_arrays, output_paths):
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as WAV file
            write(output_path, sample_rate, audio_array.astype(np.float32))
            print(f"Saved audio to: {output_path}")


def main():
    """Example usage of the voice cloning system."""
    # Initialize voice cloning engine
    voice_cloner = OrpheusTTSVoiceClone(device="cuda")
    
    # Text to synthesize
    target_texts = [
        "Hi IIT madras is currently doing great for indian research and its proud to be associated with it."
    ]
    
    reference_pairs = [(".voice_clone/input_reference.wav", 
                        "Delhi की एक retail chain ने हमारे solutions से अपनी sales में 30% तक वृद्धि देखी है। <hmm..> उनका feedback बहुत encouraging रहा है ।")]
    # Process each reference
    for audio_path, transcript in reference_pairs:
        print(f"Processing reference: {audio_path} - {transcript}")
        
        # Clone voice
        cloned_audio = voice_cloner.clone_voice(audio_path, transcript, target_texts)
        
        # Prepare output paths
        audio_stem = Path(audio_path).stem
        output_dir = Path(audio_path).parent / "inference"
        output_paths = [
            str(output_dir / f"{audio_stem}_{i}.wav") 
            for i in range(len(target_texts))
        ]
        
        # Save cloned audio
        voice_cloner.save_audio(cloned_audio, output_paths)


if __name__ == "__main__":
    main()