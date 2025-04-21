import os
import io
import time
import shutil
import requests
from rembg import remove
from PIL import Image
from gradio_client import Client, handle_file

# 計時開始
start_time = time.time()

# 建立 outputs 資料夾（如果不存在）
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 去背流程
input_path = "me.webp"
nobg_path = os.path.join(output_dir, "me_nobg.png")

with open(input_path, "rb") as f:
    input_bytes = f.read()
    output_bytes = remove(input_bytes)

    image = Image.open(io.BytesIO(output_bytes))
    image.save(nobg_path)
    print(f"✅ 已完成去背，儲存為：{nobg_path}")

# 呼叫 Hugging Face 上的風格化模型
client = Client("https://yuanshi-ominicontrol-art.hf.space/", hf_token="REMOVED_TOKEN")

result = client.predict(
    style="Studio Ghibli",
    original_image=handle_file(nobg_path),
    inference_mode="High Quality",
    image_guidance=1.5,
    image_ratio="Auto",
    use_random_seed=True,
    seed=42,
    steps=20,
    api_name="/infer"
)

# 下載風格化圖片並存到本地
print("🎯 result =", result)

stylized_local_path = result[0]
stylized_path = os.path.join(output_dir, "stylized.png")

if os.path.exists(stylized_local_path):
    shutil.copy(stylized_local_path, stylized_path)
    print(f"🎨 已複製風格化圖片，儲存為：{stylized_path}")
else:
    raise FileNotFoundError(f"⚠️ 找不到圖片檔案：{stylized_local_path}")

# 計時結束
end_time = time.time()
elapsed_time = end_time - start_time
print(f"⏱️ 任務總耗時：約 {elapsed_time:.2f} 秒")
