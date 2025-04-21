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
client = Client("https://yuanshi-ominicontrol-art.hf.space/", hf_token=hf_token)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

@app.route("/")
def upload_form():
    return '''
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>圖片風格化 Demo</title>
    </head>
    <body>
        <h1>🖼️ 上傳圖片進行風格化</h1>
        <form method="POST" action="/stylize" enctype="multipart/form-data">
            <label>選擇圖片：</label>
            <input type="file" name="image" accept="image/*" required><br><br>

            <label>風格：</label>
            <input type="text" name="style" value="Studio Ghibli"><br><br>

            <input type="submit" value="開始轉換">
        </form>

        <br>
        <p>圖片會直接顯示在新頁面中</p>
    </body>
    </html>
    '''


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
     app.run(debug=True, host="0.0.0.0", port=5000)
