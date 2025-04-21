import os
import io
import shutil
import time
from flask import Flask, request, send_file, jsonify
from rembg import remove
from PIL import Image
from gradio_client import Client, handle_file

app = Flask(__name__)
client = Client("https://yuanshi-ominicontrol-art.hf.space/", hf_token="hf_你的token")
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

@app.route("/stylize", methods=["POST"])
def stylize():
    if "image" not in request.files:
        return jsonify({"error": "請上傳圖片"}), 400

    uploaded_file = request.files["image"]
    style = request.form.get("style", "Studio Ghibli")

    start_time = time.time()

    # 去背
    input_bytes = uploaded_file.read()
    output_bytes = remove(input_bytes)
    nobg_image = Image.open(io.BytesIO(output_bytes))
    nobg_path = os.path.join(output_dir, "temp_nobg.png")
    nobg_image.save(nobg_path)

    # 呼叫風格化 API
    result = client.predict(
        style=style,
        original_image=handle_file(nobg_path),
        inference_mode="High Quality",
        image_guidance=1.5,
        image_ratio="Auto",
        use_random_seed=True,
        seed=42,
        steps=20,
        api_name="/infer"
    )

    stylized_local_path = result[0]
    if not os.path.exists(stylized_local_path):
        return jsonify({"error": "圖片生成失敗"}), 500

    stylized_path = os.path.join(output_dir, "stylized.png")
    shutil.copy(stylized_local_path, stylized_path)

    elapsed_time = time.time() - start_time
    print(f"✅ 完成風格化處理，耗時 {elapsed_time:.2f} 秒")

    return send_file(stylized_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
