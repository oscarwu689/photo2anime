import os
import io
import time
import shutil
from flask import Flask, request, send_file, jsonify
from rembg import remove
from PIL import Image
from gradio_client import Client, handle_file
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
hf_token = os.getenv("hf_token")

if not hf_token:
    raise RuntimeError("❌ 環境變數 hf_token 未設定！請在 Render 上加上環境變數")

client = Client("https://yuanshi-ominicontrol-art.hf.space/", hf_token=hf_token)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"}), 200



@app.route("/stylize", methods=["POST"])
def stylize():
    if "image" not in request.files:
        return jsonify({"error": "請上傳圖片"}), 400

    uploaded_file = request.files["image"]
    style = request.form.get("style", "Irasutoya Illustration")

    start_time = time.time()

    # === Step 1: 去背原圖 ===
    input_bytes = uploaded_file.read()
    output_bytes = remove(input_bytes)
    nobg_image = Image.open(io.BytesIO(output_bytes))
    nobg_path = os.path.join(output_dir, "temp_nobg.png")
    nobg_image.save(nobg_path)

    # === Step 2: 呼叫風格化 API ===
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

    # === Step 3: 複製風格化圖 ===
    stylized_path = os.path.join(output_dir, f"stylized_{int(time.time())}.png")
    shutil.copy(stylized_local_path, stylized_path)

    # === Step 4: 再次去背風格化圖 ===
    with open(stylized_path, "rb") as f:
        styled_bytes = f.read()
        final_output = remove(styled_bytes)

    final_image = Image.open(io.BytesIO(final_output))
    final_path = stylized_path.replace(".png", "_nobg.png")
    final_image.save(final_path)

    elapsed_time = time.time() - start_time
    print(f"✅ 完成風格化 + 去背，耗時 {elapsed_time:.2f} 秒")

    return send_file(final_path, mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)

