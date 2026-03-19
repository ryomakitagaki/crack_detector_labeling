# logic.py
import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image
import io

def get_gemini_traced_image(api_key, image_bytes, prompt, model_id):
    """Gemini APIを呼び出して、ひび割れに赤線を引いた画像データを取得する"""
    client = genai.Client(api_key=api_key)
    
    # MIMEタイプは一旦image/jpeg固定（または拡張子から判別）
    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        ),
    )

    # 画像パートを探して返す
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    return None

def process_yolo_segmentation(traced_bytes, original_width, original_height, min_area_px=10):
    """赤線画像からYOLOセグメンテーション用テキストと可視化画像を生成する"""
    # バイト列をOpenCV形式に変換
    nparr = np.frombuffer(traced_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 赤色領域の抽出（BGR）
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([50, 50, 255])
    mask = cv2.inRange(img, lower_red, upper_red)

    # モルフォロジー処理で点を繋ぐ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        
        # ポリゴン近似（データ軽量化）
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue

        # 正規化座標 (0.0 - 1.0) に変換
        coords = []
        for pt in approx:
            nx = max(0.0, min(1.0, pt[0][0] / original_width))
            ny = max(0.0, min(1.0, pt[0][1] / original_height))
            coords.append(f"{nx:.6f} {ny:.6f}")
        
        yolo_lines.append(f"0 {' '.join(coords)}")

    return "\n".join(yolo_lines), img