from gradio_client import Client, handle_file
import base64
import requests

# 連到該 Space 的 URL
client = Client("https://yuanshi-ominicontrol-art.hf.space/")

# 上傳你自己的圖片檔
result = client.predict(
    style="Studio Ghibli",
    original_image=handle_file("me.webp"),
    inference_mode="High Quality",
    image_guidance=1.5,
    image_ratio="Auto",
    use_random_seed=True,
    seed=42,
    steps=20,
    api_name="/infer"
)

print(result)
