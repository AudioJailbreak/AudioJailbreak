
import torch
import os
import whisper

from omni_speech.conversation import conv_templates
from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init
from omni_speech.datasets.preprocess import tokenizer_speech_token

from torchaudio.functional import resample
import torchaudio

class LLaMA_Omni():


    def __init__(self, model_path='ICTNLP/Llama-3.1-8B-Omni', model_base=None, is_lora=False, s2s=False):
        # Model
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        self.tokenizer, self.model, self.context_len = load_pretrained_model(model_path, model_base, is_lora=is_lora, s2s=s2s)
        self.fs = 16_000
        self.input_type = "mel"
        self.mel_size = 128
        self.qs = "<speech>\nPlease directly answer the questions in the user's speech."
        self.s2s = s2s
        self.temperature = 0
        self.top_p = None
        self.num_beams = 1
        self.max_new_tokens = 256
        self.conv_mode = 'llama_3'

        self.random = False

    def get_logits_indice(self, audio_input, expected_response, sr=16_000):

        assert torch.is_tensor(audio_input)
        if len(audio_input.shape) == 1:
            audio_input = audio_input.unsqueeze(0)
        elif len(audio_input.shape) == 3:
            assert audio_input.shape[0] == 1
            assert audio_input.shape[1] == 1
            audio_input = audio_input.squeeze(0)
        elif len(audio_input.shape) == 2:
            assert audio_input.shape[0] == 1
        else:
            raise NotImplementedError('Not Supported Shape')
        
        if sr != self.fs:
            audio_input = resample(audio_input, sr, self.fs)
        
        speech = audio_input.squeeze(0)

        assert self.input_type == "mel"
        speech = whisper.pad_or_trim(speech)
        speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.qs)
        conv.append_message(conv.roles[1], expected_response)
        prompt = conv.get_prompt()

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')[:-1] ## skip final <|eot_id|> in "Hate you more than a thousands times<|eot_id|>"

        input_ids_2 = tokenizer_speech_token(expected_response, self.tokenizer, return_tensors='pt')

        speech_length = torch.LongTensor([speech.shape[0]])

        input_ids = input_ids.unsqueeze(0)
        speech_tensor = speech_tensor.unsqueeze(0)
        speech_length = speech_length.unsqueeze(0)

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)

        assert not self.s2s
        logits = self.model(
            input_ids,
            speech=speech_tensor,
            speech_lengths=speech_length,
            use_cache=True,
        ).logits


        return logits, logits.shape[1] - len(input_ids_2)
        

    def encode(self, string):

        target_label = tokenizer_speech_token(string, self.tokenizer, return_tensors='pt').cuda()
        return target_label


    def get_rsp(self, audio_input, sr=16_000, history=None):

        assert torch.is_tensor(audio_input)
        if len(audio_input.shape) == 1:
            audio_input = audio_input.unsqueeze(0)
        elif len(audio_input.shape) == 3:
            assert audio_input.shape[0] == 1
            assert audio_input.shape[1] == 1
            audio_input = audio_input.squeeze(0)
        elif len(audio_input.shape) == 2:
            assert audio_input.shape[0] == 1
        else:
            raise NotImplementedError('Not Supported Shape')
        
        if sr != self.fs:
            audio_input = resample(audio_input, sr, self.fs)
        
        speech = audio_input.squeeze(0)

        assert self.input_type == "mel"
        speech = whisper.pad_or_trim(speech)
        speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0).unsqueeze(0)

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0)
        speech_length = torch.LongTensor([speech.shape[0]])

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)

        assert not self.s2s
        outputs = self.model.generate(
            input_ids,
            speech=speech_tensor,
            speech_lengths=speech_length,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            pad_token_id=128004,
        )
        output_ids = outputs
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs
    

    def get_rsp_path(self, audio_input):

        assert isinstance(audio_input, str)
        audio_input, sr = torchaudio.load(audio_input)

        audio_input = audio_input.cuda()
        return self.get_rsp(audio_input, sr, history=None)