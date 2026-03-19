# app.py
import streamlit as st
import io
import zipfile
from PIL import Image
from pathlib import Path

# 自作ロジックの読み込み
import logic

# --- UI設定 ---
st.set_page_config(page_title="Crack Analyzer", layout="wide")
st.title("🧱 ひび割れ解析 & YOLOデータセット作成")

# --- サイドバー (設定エリア) ---
with st.sidebar:
    st.header("🔑 認証設定")
    api_key = st.text_input("Gemini API Key", type="password", help="Google AI Studioで取得したキーを入力してください")
    
    st.header("⚙️ 解析設定")
    model_id = st.selectbox("モデル", ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"])
    
    prompt_type = st.radio("検出感度", ["ひび割れ", "欠損・はがれ", "その他"])
    min_area = st.number_input("最小ポリゴン面積(px)", value=10, min_value=0)

# プロンプト定義
# 内容
# ひび割れ認識
PROMPT_FOR_NANOBANANA_V1 = """
写真にうつる建築物の表面を解析し，ひび割れを特定してください。
直線的なタイルやブロックの目地，建材の稜線，塗料の剥がれ部，異種材料の境界部分はひび割れではありません。
建材表面の幾何学的な模様や陰影はひび割れではありません。
特定したひび割れの上に、鮮明な赤色(透過率80%)の線を，ひび割れの太さに応じて描画した画像を生成してください。
元の画像と赤色(透過率80%)の線のみで構成された画像を返してください。
"""
# 欠損・はがれ
PROMPT_FOR_NANOBANANA_V2 = """
写真にうつる建築物の表面を解析し，建材の欠損部や剥離部を特定してください。
欠損部や剥離部は寸法の縦横のアスペクト比0.5～2.0の範囲です．
アスペクト比が0.5未満の細長いものや、2.0を超える細長いものはひび割れの可能性が高いですが、今回は対象外としてください。
直線的なタイルやブロックの目地，建材の稜線，異種材料の境界部分は欠損部や剥離部ではありません。
建材表面の幾何学的な模様や陰影は欠損部や剥離部ではありません。
特定した欠損部や剥離部の上に、鮮明な赤色(透過率80%)を描画した画像を生成してください。
元の画像と赤色(透過率80%)の描画領域のみで構成された画像を返してください。

"""
# 
PROMPT_FOR_NANOBANANA_V3 = """
写真にうつる建築物の表面を解析し，ひび割れを特定してください。
直線的なタイルやブロックの目地，建材の稜線，塗料の剥がれ部，異種材料の境界部分はひび割れではありません。
建材表面の幾何学的な模様や陰影はひび割れではありません。
特定したひび割れの上に、鮮明な赤色(透過率80%)の線を，ひび割れの太さに応じて描画した画像を生成してください。
元の画像と赤色(透過率80%)の線のみで構成された画像を返してください。
"""
 

# プロンプト定義
PROMPT_MAP = {
    "ひび割れ": PROMPT_FOR_NANOBANANA_V1,
    "欠損・はがれ": PROMPT_FOR_NANOBANANA_V2,
    "その他": PROMPT_FOR_NANOBANANA_V3
}

# --- メインエリア ---
uploaded_files = st.file_uploader("解析する画像をアップロード（複数可）", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # 解析開始ボタン
    if st.button("🚀 解析を開始する", use_container_width=True):
        
        # ⚠️ APIキー未入力チェック
        if not api_key:
            st.error("❌ APIキーが入力されていません。サイドバーから入力してください。")
            st.info("APIキーをお持ちでない場合は、[Google AI Studio](https://aistudio.google.com/) で無料で取得できます。")
            st.stop() # ここで処理を中断

        # ZIPファイル作成用のバッファ
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 処理中: {uploaded_file.name}", expanded=True):
                    try:
                        # 画像の読み込みとサイズ取得
                        img_bytes = uploaded_file.read()
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        w, h = pil_img.size

                        # Step 1: Gemini API呼び出し (logic.pyより)
                        with st.spinner("画像を解析中..."):
                            traced_data = logic.get_gemini_traced_image(
                                api_key, img_bytes, PROMPT_MAP[prompt_type], model_id
                            )

                        if traced_data:
                            # Step 2: YOLO変換 (logic.pyより)
                            yolo_text, vis_img = logic.process_yolo_segmentation(traced_data, w, h, min_area)

                            # 結果表示
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(pil_img, caption="元の画像")
                            with col2:
                                st.image(vis_img, channels="BGR", caption="解析済み(赤線検出)")

                            # ZIPへの書き込み
                            zf.writestr(f"labels/{Path(uploaded_file.name).stem}.txt", yolo_text)
                            zf.writestr(f"images/{uploaded_file.name}", img_bytes)
                            st.success("✅ 正常に処理されました")
                        else:
                            st.error("解析画像の取得に失敗しました。")

                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")

        # 全件終了後のダウンロードボタン
        st.divider()
        st.success("✨ すべての画像の処理が完了しました！")
        st.download_button(
            label="📁 YOLOデータセット(ZIP)をダウンロード",
            data=zip_buffer.getvalue(),
            file_name="yolo_dataset.zip",
            mime="application/zip",
            use_container_width=True
        )