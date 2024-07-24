import gradio as gr
from diffusers import DiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
import os
import zipfile
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from moviepy.editor import *


pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
pipe = pipe.to("cpu")

params = SpectrogramParams()
converter = SpectrogramImageConverter(params)

max_video_clips = 9
video_clips = []
model_output = []
clips = []
number_of_clips = 1

#This function is needed because specification of model width parameter,
#otherwise you will get an error if width is not divisible by 8.
def make_divisible_by_8(number):
    remainder = number % 8

    if remainder == 0:
        return number
    
    adjustment = 8 - remainder
    adjusted_number = number + adjustment
    return adjusted_number

def predict(prompt, clip_number):
    clip_number = int(clip_number)
    clip = video_clips[clip_number - 1]
    clip = f"clip_{clip_number}.mp4"
    video = VideoFileClip(clip)
    duration = video.duration
    width = int(make_divisible_by_8(round(duration) * 100))
    spec = pipe(
        prompt,
        negative_prompt='',
        num_inference_steps=20,
        width=width,
    ).images[0]
    
    wav = converter.audio_from_spectrogram_image(image=spec)
    wav.export('output.wav', format='wav')
    return 'output.wav'

def divide_video(video_path, num_clips):
    video = VideoFileClip(video_path)
    duration = video.duration
    clip_duration = duration / num_clips
   
    for i in range(num_clips):
        start_time = i * clip_duration
        end_time = (i + 1) * clip_duration
        clip = video.subclip(start_time, end_time)
        clip_filename = f"clip_{i+1}.mp4"
        clip.write_videofile(clip_filename, codec='libx264')
        clips.append(clip_filename)

    return clips

def display_clips(video, num_clips):
    global number_of_clips 
    number_of_clips = int(num_clips)
    outputs = [None] * max_video_clips
    clips = divide_video(video, num_clips)
    for i in range(max_video_clips):
        if( i < num_clips):
            print(i)
            print(clips[i])
            outputs[i] = gr.Video(value=clips[i])
    
    return outputs

def display_clips_with_replaced_audio(video, num_clips, clip_number, audio_output):
    global number_of_clips 
    number_of_clips = int(num_clips)
    outputs = [None] * max_video_clips
    clips = divide_video(video, num_clips)
    for i in range(max_video_clips):
        if( i < num_clips):
            if i == clip_number - 1:
                video_clip = VideoFileClip(clips[i])
                audio = AudioFileClip("output.wav")
                video_clip = video_clip.set_audio(audio)
                video_clip.write_videofile("replaced_audio.mp4", codec='libx264', audio_codec='aac')
                outputs[i] = gr.Video(value="replaced_audio.mp4")
            else: 
                outputs[i] = gr.Video(value=clips[i])
        
    return outputs


def variable_outputs(k):
    k = int(k)
    return [gr.Video(visible=True)]*k + [gr.Video(visible=False)]*(max_video_clips-k)


def zip_video_clips(*video_clips):
    zip_path = "video_clips_archive.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for idx, clip in enumerate(video_clips):
            if clip is not None:
                    video_path = clip
                    zipf.write(video_path, os.path.basename(video_path))
    return zip_path
                                                  

with gr.Blocks() as combined_interface:
    with gr.Blocks() as video_interface:
            gr.Markdown("# Video Clip Divider")
            gr.Markdown("Upload a video file and specify the number of clips to divide it into. The clips will be displayed in rows and columns dynamically.")
        
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload your video file")
                    s = gr.Slider(1, max_video_clips, value=1, step=1, label="How many video clips to show:")
                    
                    for i in range(3):  
                        with gr.Row():
                            for j in range(3):  
                                video_clips.append(gr.Video(label=f"clip {i*3 + j + 1}", visible=False))   
            s.change(variable_outputs, s, video_clips)

            submit_button = gr.Button("Process Video")

            submit_button.click(
            display_clips,
            inputs=[video_input, s],
            outputs=video_clips
            )


    with gr.Blocks() as iface:
        gr.Markdown("# Riffusion Music Generator")
        gr.Markdown("Generate music from text prompts using the Riffusion model")

        with gr.Row():
            with gr.Column():
                text_prompt = gr.Textbox(label="Text Prompt")
                clip_number = gr.Number(minimum=1, maximum=9, label="Select clip")
                generate_button = gr.Button("Generate")
                audio_output = gr.Audio(type='filepath', label="Generated Audio")
            
            generate_button.click(predict, inputs=[text_prompt, clip_number], outputs=audio_output).then(
    display_clips_with_replaced_audio, inputs=[video_input, s, clip_number, audio_output], outputs=model_output
)


   
    with gr.Blocks() as result: 
        with gr.Row():
                    with gr.Column():
                        for i in range(3):  
                            with gr.Row():
                                for j in range(3):  
                                    model_output.append(gr.Video(label=f"clip {i*3 + j + 1}", value=video_clips[j].label,  visible=False))  
        s.change(variable_outputs, s, model_output) 

        download_button = gr.Button("Download All Clips")
        download_button.click(zip_video_clips, inputs=model_output, outputs=gr.File())

        
if __name__ == "__main__":
    combined_interface.launch()
