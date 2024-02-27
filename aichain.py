
import warnings
import gradio as gr
import openai
from openai import OpenAI
import subprocess as sp
from pathlib import Path
import typing as tp

from tempfile import NamedTemporaryFile
import os
import sys
import time

from einops import rearrange
import torch
from concurrent.futures import ProcessPoolExecutor
from torch import nn

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen, MultiBandDiffusion


openai.api_key = os.getenv("OPENAI_API_KEY")

class GPT:
    """Instantiate GPT model for a multi-step conversation."""

    def __init__(self, **params):
        """Setup model parameters and system prompt."""
        self.client = OpenAI()
        self.params = {'model': 'gpt-3.5-turbo', **params}
        self.messages = []

    def chat(self, user_prompt, chat_history, system_prompt, *params):
        """Generate a response with a given parameters."""

        # update GPT parameters
        temperature, top_p, frequency_penalty, presence_penalty = params
        self.params['temperature'] = temperature
        self.params['top_p'] = top_p
        self.params['frequency_penalty'] = frequency_penalty
        self.params['presence_penalty'] = presence_penalty

        # update the message buffer
        self.messages = [{'role': 'system', 'content': f'{system_prompt}'}]
        for prompt, response in chat_history:
            self.messages.append({'role': 'user', 'content': f'{prompt}'})
            self.messages.append({'role': 'assistant', 'content': f'{response}'})
        self.messages.append({'role': 'user', 'content': f'{user_prompt}'})

        # generate a response
        completion = self.client.chat.completions.create(messages=self.messages, **self.params)
        response = completion.choices[0].message.content
        self.messages.append({'role': 'assistant', 'content': response})
        chat_history.append((user_prompt, response))

        return '', chat_history

#music generate model    
device = torch.device("cuda")
MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
IS_BATCHED = "facebook/MusicGen" in SPACE_ID or 'musicgen-internal/musicgen_dev' in SPACE_ID
print(IS_BATCHED)
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break
                
file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='facebook/musicgen-melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        del MODEL
        torch.mps.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = MusicGen.get_pretrained(version)


def load_diffusion():
    global MBD
    if MBD is None:
        print("loading MBD")
        MBD = MultiBandDiffusion.get_mbd_musicgen()


def _do_predictions(texts, melodies, duration, progress=False, gradio_progress=None, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    try:
        if any(m is not None for m in processed_melodies):
            outputs = MODEL.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=USE_DIFFUSION
            )
        else:
            outputs = MODEL.generate(texts, progress=progress, return_tokens=USE_DIFFUSION)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    if USE_DIFFUSION:
        if gradio_progress is not None:
            gradio_progress(1, desc='Running MultiBandDiffusion...')
        tokens = outputs[1]
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            left, right = MODEL.compression_model.get_left_right_codes(tokens)
            tokens = torch.cat([left, right])
        outputs_diffusion = MBD.tokens_to_wav(tokens)
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            assert outputs_diffusion.shape[1] == 1  # output is mono
            outputs_diffusion = rearrange(outputs_diffusion, '(s b) c t -> b (s c) t', s=2)
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            pending_videos.append(pool.submit(make_waveform, file.name))
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    out_videos = [pending_video.result() for pending_video in pending_videos]
    for video in out_videos:
        file_cleaner.add(video)
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_videos, out_wavs


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('facebook/musicgen-stereo-melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return res


def predict_full(model, model_path, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    model_path = model_path.strip()
    if model_path:
        if not Path(model_path).exists():
            raise gr.Error(f"Model path {model_path} doesn't exist.")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path {model_path} must be a folder containing "
                           "state_dict.bin and compression_state_dict_.bin.")
        model = model_path
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        progress(0, desc="Loading diffusion model...")
        load_diffusion()
    else:
        USE_DIFFUSION = False
    load_model(model)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    videos, wavs = _do_predictions(
        [text], [melody], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef,
        gradio_progress=progress)
    if USE_DIFFUSION:
        return videos[0], wavs[0], videos[1], wavs[1]
    return videos[0], wavs[0], None, None


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2


def run_chatbot(gpt):
    """Configure and launch the chatbot interface."""
    with gr.Blocks() as interface:
        with gr.Blocks(title='ChatGPT') as chatbot:

            # chatbot interface
            chat_history = gr.Chatbot(height=500, layout='bubble', label='ChatGPT')
            with gr.Row():
                user_prompt = gr.Textbox(placeholder='Message ChatGPT...', container=False, min_width=500, scale=9)
                submit_button = gr.Button('Submit')

            # parameters accordion
            with gr.Accordion(label='GPT Parameters', open=False):
                info = gr.Markdown('For parameter documentation see [OpenAI Chat API Reference]' \
                                + '(https://platform.openai.com/docs/api-reference/chat)')
                system_prompt = gr.Textbox('You are a helpful assistant.', label='system prompt')
                with gr.Row():
                    temperature = gr.Slider(0., 2., value=1., step=.1, min_width=200, label='temperature')
                    top_p = gr.Slider(0., 1., value=1., step=.01, min_width=200, label='top_p')
                    frequency_penalty = gr.Slider(-2., 2., value=0, step=.1, min_width=200, label='frequency_penalty')
                    presence_penalty = gr.Slider(-2., 2., value=0, step=.1, min_width=200, label='presence_penalty')

            # submit user prompt
            inputs = [user_prompt, chat_history, system_prompt,
                    temperature, top_p, frequency_penalty, presence_penalty]
            outputs = [user_prompt, chat_history]
            submit_button.click(gpt.chat, inputs=inputs, outputs=outputs)
            user_prompt.submit(gpt.chat, inputs=inputs, outputs=outputs)
        with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Paste your music description",interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file","mic"],value="file",
                                        label="Condition on a melody(optional File or Mic)")
                        melody = gr.Audio(sources=["upload"], type="numpy", label="File",
                                            interactive=True, elem_id="melody-input")
                with gr.Row():
                    generate = gr.Button("Generate")
                    _ = gr.Button("Interrupt").click(fn=interrupt,queue=False)
                with gr.Row():
                    model = gr.Radio(["facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                                        "facebook/musicgen-large", "facebook/musicgen-melody-large",
                                        "facebook/musicgen-stereo-small", "facebook/musicgen-stereo-medium",
                                        "facebook/musicgen-stereo-melody", "facebook/musicgen-stereo-large",
                                        "facebook/musicgen-stereo-melody-large"],
                                        label="Model", value="facebook/musicgen-stereo-melody", interactive=True)
                    model_path = gr.Text(label="Model Path (custom models)")
                    
                with gr.Row():
                        decoder = gr.Radio(["Default", "MultiBand_Diffusion"],
                                        label="Decoder", value="Default", interactive=True)
                with gr.Row():    
                        duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
                with gr.Row():
                        topk = gr.Number(label="Top-k", value=250, interactive=True)
                        topp = gr.Number(label="Top-p", value=0, interactive=True)
                        temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                        cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
                with gr.Column():
                    output = gr.Video(label="Generated Music")
                    audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
                    diffusion_output = gr.Video(label="MultiBand Diffusion Decoder")
                    audio_diffusion = gr.Audio(label="MultiBand Diffusion Decoder (wav)", type='filepath')
                generate.click(toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False,
                        show_progress=False).then(predict_full, inputs=[model, model_path, decoder, text, melody, duration, topk, topp,
                                                                        temperature, cfg_coef],
                                                outputs=[output, audio_output, diffusion_output, audio_diffusion])
                radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)
            
        # instantiate the chatbot
        gr.close_all()
        interface.queue().launch(share=True)
        chatbot.queue().launch(share=True)


if __name__ == '__main__':
    gpt = GPT()
    run_chatbot(gpt)
