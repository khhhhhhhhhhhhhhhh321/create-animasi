import torch
import gc
import scipy.io.wavfile
import streamlit as st
from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler, AudioLDMPipeline
from diffusers.utils import export_to_video
from moviepy.editor import VideoFileClip, AudioFileClip

st.set_page_config(page_title="Studio Sutradara AI", page_icon="🎬", layout="centered")

@st.cache_resource
def load_pipelines():
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipe_vid = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe_vid.scheduler = DPMSolverMultistepScheduler.from_config(pipe_vid.scheduler.config)
    pipe_vid.enable_vae_slicing()
    pipe_vid.enable_model_cpu_offload()
    pipe_vid.enable_xformers_memory_efficient_attention()
    
    pipe_aud = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=torch.float16)
    pipe_aud.enable_model_cpu_offload()
    return pipe_vid, pipe_aud

with st.spinner("Sedang memuat mesin AI ke server... Mohon tunggu sebentar."):
    pipe_vid, pipe_aud = load_pipelines()

st.title("🎬 Studio Sutradara AI Ultimate")
st.write("Buat video animasi lengkap dengan suara otomatis secara permanen.")

rasio_video = {
    "TikTok / FB Reels / FB Story / YT Shorts (9:16)": (512, 768),
    "Facebook Postingan Portrait (4:5)": (512, 640),
    "Facebook Postingan Persegi (1:1)": (512, 512),
    "YouTube Video Standar (16:9)": (768, 512)
}

style_visual = {
    "Cinematic Realism": "photorealistic, 8k, ultra-detailed, cinematic lighting, hyperrealistic, dramatic, 35mm lens, sharp focus",
    "Anime / Studio Ghibli": "masterpiece, best quality, 2d, colorful anime style, studio ghibli animation, incredibly detailed",
    "3D Pixar / Disney": "3d render, pixar style, disney style, cute, vibrant colors, unreal engine 5, highly detailed",
    "Edukasi / Sci-Fi / Medis": "highly detailed 3d illustration, informative, macro photography, national geographic style",
    "Bebas (Sesuai Ketikan)": "best quality, masterpiece, ultra high res, 4k"
}

gerakan_kamera = {
    "Diam (Statis)": "",
    "Zoom In (Mendekat)": "camera zooming in, moving forward, closing in,",
    "Zoom Out (Menjauh)": "camera zooming out, moving backward, pulling back,",
    "Pan Right (Geser Kanan)": "camera panning right, moving right,",
    "Pan Left (Geser Kiri)": "camera panning left, moving left,"
}

style_audio = {
    "Epic Cinematic BGM": "epic cinematic background music, orchestral, dramatic, high quality",
    "Lo-Fi / Chill BGM": "lo-fi hip hop background music, chill, relaxing, calm, aesthetic",
    "Real Sound Effects (SFX)": "realistic ambient sound effects, high fidelity, clear, raw audio",
    "Misterius / Horror": "creepy ambient background music, suspenseful, dark, scary"
}

karakter = st.text_input("👤 Karakter Utama", value="a cute robot teaching math")
scene = st.text_area("🏞️ Scene / Kejadian", value="standing in front of a glowing digital blackboard")

col1, col2 = st.columns(2)
with col1:
    kamera = st.selectbox("🎥 Gerak Kamera", list(gerakan_kamera.keys()))
    rasio = st.selectbox("📱 Format Platform", list(rasio_video.keys()))
with col2:
    visual = st.selectbox("🎨 Filter Visual", list(style_visual.keys()))
    audio = st.selectbox("🎵 Musik & Suara", list(style_audio.keys()))

with st.expander("⚙️ Pengaturan Lanjutan"):
    steps = st.slider("Tingkat Kejernihan (Steps)", 10, 30, 15)
    fps = st.slider("Kecepatan Gerak (FPS)", 6, 16, 8)
    seed_num = st.number_input("Kode Seed (-1 untuk Random)", value=-1, step=1)

if st.button("🚀 RENDER KARYA SEKARANG!", type="primary", use_container_width=True):
    with st.status("Sedang memproses video dan audio...", expanded=True) as status:
        try:
            torch.cuda.empty_cache()
            gc.collect()

            st.write("🎬 Sutradara AI sedang merender video...")
            lebar, tinggi = rasio_video[rasio]
            prompt_vid = f"{gerakan_kamera[kamera]} {karakter}, {scene}, {style_visual[visual]}"
            prompt_neg = "bad quality, worst quality, deformed, mutated, text, watermark, blurry, low resolution, ugly"
            
            generator_seed = torch.Generator("cpu").manual_seed(int(seed_num)) if seed_num != -1 else torch.Generator("cpu").manual_seed(torch.randint(0, 1000000, (1,)).item())

            out_vid = pipe_vid(
                prompt=prompt_vid,
                negative_prompt=prompt_neg,
                num_frames=16,
                guidance_scale=7.5,
                num_inference_steps=steps,
                width=lebar,
                height=tinggi,
                generator=generator_seed,
            )
            
            temp_video = "temp_video.mp4"
            export_to_video(out_vid.frames[0], temp_video, fps=fps)

            st.write("🎵 Penata Suara AI sedang merender audio...")
            prompt_aud = f"{style_audio[audio]}, sound of {scene}, {karakter}"
            out_aud = pipe_aud(prompt_aud, num_inference_steps=15, audio_length_in_s=3.0).audios[0]
            
            temp_audio = "temp_audio.wav"
            scipy.io.wavfile.write(temp_audio, rate=16000, data=out_aud)

            st.write("🎞️ Menggabungkan video & audio...")
            video_clip = VideoFileClip(temp_video)
            audio_clip = AudioFileClip(temp_audio).subclip(0, video_clip.duration)
            final_clip = video_clip.set_audio(audio_clip)
            
            final_video = "Sutradara_AI_Ultimate.mp4"
            final_clip.write_videofile(final_video, codec="libx264", audio_codec="aac", fps=fps, verbose=False, logger=None)
            
            video_clip.close()
            audio_clip.close()
            
            torch.cuda.empty_cache()
            gc.collect()

            status.update(label="✅ Selesai!", state="complete", expanded=False)
            st.success("Video berhasil dibuat!")
            st.video(final_video)
            
        except Exception as e:
            torch.cuda.empty_cache()
            st.error(f"Terjadi Kendala: {str(e)}")
