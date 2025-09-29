from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
import torch

print("diffusers =", __import__("diffusers").__version__)
print("transformers =", __import__("transformers").__version__)

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "zai-org/CogVideoX-5b-I2V", dtype=torch.float32
)
pipe.to("cpu")

img = Image.new("RGB", (720, 480), (0, 0, 0))
out = pipe(
    prompt="abstract techno lights, neon beams, cinematic, no text",
    image=img,
    num_frames=49,                 # 固定 49 帧 ≈6s
    num_inference_steps=6,         # CPU 先低步数冒烟
    guidance_scale=4.5,
    height=480, width=720,
    generator=torch.Generator(device="cpu").manual_seed(42),
).frames[0]

export_to_video(out, "/tmp/cvx_smoke.mp4", fps=8)
print("OK -> /tmp/cvx_smoke.mp4")
