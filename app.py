import numpy as np
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
from typing import Optional
import base64
import io
import wave
from pydantic import BaseModel, Field
import inferless
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '1'

@inferless.request
class RequestObjects(BaseModel):
    text: str = Field(default="Hey there! How are you? I had the craziest day today.")
    voice: Optional[str] = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    cfg_coef: Optional[float] = 2.0
    padding_between: Optional[int] = 1

@inferless.response
class ResponseObjects(BaseModel):
    audio_base64: str = Field(default="Test Output")

class InferlessPythonModel:
    def initialize(self):
        # Load the checkpoint info and TTS model
        checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        self.tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, 
            n_q=32, 
            temp=0.6, 
            device=torch.device("cuda")
        )


    def _on_frame(self, frame, pcms):
        """Frame callback function for audio generation"""
        if (frame != -1).all():
            pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcms.append(np.clip(pcm[0, 0], -1, 1))

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        # Prepare the script
        entries = self.tts_model.prepare_script([inputs.text], padding_between=inputs.padding_between)
        
        # Get voice path
        voice_path = self.tts_model.get_voice_path(inputs.voice)
        
        # Make condition attributes
        condition_attributes = self.tts_model.make_condition_attributes(
            [voice_path], cfg_coef=inputs.cfg_coef
        )
        
        # Generate audio
        pcms = []
        
        # Create a lambda function that captures the pcms list
        frame_callback = lambda frame: self._on_frame(frame, pcms)

        all_entries = [entries]
        all_condition_attributes = [condition_attributes]
        
        with self.tts_model.mimi.streaming(len(all_entries)):
            result = self.tts_model.generate(all_entries, all_condition_attributes, on_frame=frame_callback)
        
        # Concatenate audio frames
        audio = np.concatenate(pcms, axis=-1)
        
        # Convert to base64
        audio_base64 = self._audio_to_base64(audio, sample_rate=24000)
        
        return ResponseObjects(
            audio_base64=audio_base64
        )

    def _audio_to_base64(self, audio_array, sample_rate=24000):
        # Ensure audio is in the right format
        audio_array = np.clip(audio_array, -1, 1)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Encode to base64
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode()
        return encoded

    def finalize(self):
        self.tts_model = None
