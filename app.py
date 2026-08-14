import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageFilter


st.set_page_config(page_title="Anime Face Tool", layout="wide")


# -----------------------------
# MediaPipe setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

LEFT_EYE_IDS = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_IDS = [362, 263, 387, 386, 385, 373, 374, 380]

NOSE_TIP_ID = 1
CHIN_ID = 152


@st.cache_resource
def get_face_mesh():
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )


@dataclass
class FaceInfo:
    face_center: tuple
    bbox: tuple
    roll_deg: float
    yaw_norm: float
    pitch_norm: float
    eye_distance: float


# -----------------------------
# Utility
# -----------------------------
def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def np_to_pil_rgba(arr: np.ndarray) -> Image.Image:
    if arr.shape[2] == 3:
        return Image.fromarray(arr).convert("RGBA")
    return Image.fromarray(arr)


def average_point(landmarks_px, ids):
    pts = np.array([landmarks_px[i] for i in ids], dtype=np.float32)
    return pts.mean(axis=0)


def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


# -----------------------------
# Face analysis
# -----------------------------
def detect_face_info(image_pil: Image.Image) -> FaceInfo | None:
    face_mesh = get_face_mesh()
    rgb = pil_to_rgb_np(image_pil)
    h, w = rgb.shape[:2]

    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    landmarks_px = []
    for lm in face_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        landmarks_px.append((x, y))
    landmarks_px = np.array(landmarks_px, dtype=np.float32)

    oval_pts = landmarks_px[FACE_OVAL]
    x_min = float(np.min(oval_pts[:, 0]))
    x_max = float(np.max(oval_pts[:, 0]))
    y_min = float(np.min(oval_pts[:, 1]))
    y_max = float(np.max(oval_pts[:, 1]))

    left_eye = average_point(landmarks_px, LEFT_EYE_IDS)
    right_eye = average_point(landmarks_px, RIGHT_EYE_IDS)
    nose = landmarks_px[NOSE_TIP_ID]
    chin = landmarks_px[CHIN_ID]

    face_center = oval_pts.mean(axis=0)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    eye_distance = float(np.hypot(dx, dy))

    # 顔の回転角（左右の傾き）
    roll_deg = math.degrees(math.atan2(dy, dx))

    # Yaw の簡易推定
    eye_mid = (left_eye + right_eye) / 2.0
    half_eye_dist = max(eye_distance / 2.0, 1e-6)
    yaw_norm = float((nose[0] - eye_mid[0]) / half_eye_dist)
    yaw_norm = clamp(yaw_norm, -1.0, 1.0)

    # Pitch の簡易推定
    upper = max(nose[1] - eye_mid[1], 1.0)
    lower = max(chin[1] - nose[1], 1.0)
    ratio = lower / upper
    # ざっくり基準 1.55 からのずれ
    pitch_norm = float((ratio - 1.55) / 0.75)
    pitch_norm = clamp(pitch_norm, -1.0, 1.0)

    return FaceInfo(
        face_center=(float(face_center[0]), float(face_center[1])),
        bbox=(x_min, y_min, x_max, y_max),
        roll_deg=roll_deg,
        yaw_norm=yaw_norm,
        pitch_norm=pitch_norm,
        eye_distance=eye_distance
    )


# -----------------------------
# Anime face prep
# -----------------------------
def remove_near_white_bg(img_rgba: Image.Image, threshold=245) -> Image.Image:
    arr = np.array(img_rgba.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    near_white = np.all(rgb >= threshold, axis=2)
    alpha[near_white] = 0
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def crop_to_alpha(img_rgba: Image.Image) -> Image.Image:
    arr = np.array(img_rgba.convert("RGBA"))
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        return img_rgba
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return img_rgba.crop((x1, y1, x2 + 1, y2 + 1))


def prepare_anime_face(
    anime_pil: Image.Image,
    auto_remove_bg: bool,
    white_threshold: int
) -> Image.Image:
    anime_rgba = anime_pil.convert("RGBA")

    # アルファが全部 255 で、背景が白っぽい場合に簡易透過
    alpha = np.array(anime_rgba)[:, :, 3]
    all_opaque = np.all(alpha == 255)

    if auto_remove_bg and all_opaque:
        anime_rgba = remove_near_white_bg(anime_rgba, threshold=white_threshold)

    anime_rgba = crop_to_alpha(anime_rgba)
    return anime_rgba


# -----------------------------
# Overlay logic
# -----------------------------
def soften_alpha(img_rgba: Image.Image, blur_radius: float) -> Image.Image:
    if blur_radius <= 0:
        return img_rgba
    r, g, b, a = img_rgba.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.merge("RGBA", (r, g, b, a))


def overlay_anime_face(
    base_pil: Image.Image,
    anime_face_pil: Image.Image,
    face_info: FaceInfo,
    width_scale: float,
    height_scale: float,
    blur_radius: float
) -> Image.Image:
    base = base_pil.convert("RGBA")
    anime = anime_face_pil.convert("RGBA")

    x_min, y_min, x_max, y_max = face_info.bbox
    face_w = x_max - x_min
    face_h = y_max - y_min
    cx, cy = face_info.face_center

    # 簡易 yaw / pitch 補正
    yaw = face_info.yaw_norm
    pitch = face_info.pitch_norm

    # 横向きのときは横幅を少しだけ詰める
    yaw_width_factor = 1.0 - abs(yaw) * 0.10
    yaw_width_factor = clamp(yaw_width_factor, 0.88, 1.05)

    # 上下向きのときの縦補正
    pitch_height_factor = 1.0 + pitch * 0.06
    pitch_height_factor = clamp(pitch_height_factor, 0.92, 1.10)

    target_w = int(face_w * width_scale * yaw_width_factor)
    target_h = int(face_h * height_scale * pitch_height_factor)

    target_w = max(target_w, 10)
    target_h = max(target_h, 10)

    anime_resized = anime.resize((target_w, target_h), Image.LANCZOS)

    # 回転
    anime_rotated = anime_resized.rotate(
        face_info.roll_deg,
        resample=Image.BICUBIC,
        expand=True
    )

    anime_rotated = soften_alpha(anime_rotated, blur_radius)

    # yaw/pitch に応じて中心位置を少しずらす
    shift_x = yaw * face_w * 0.06
    shift_y = pitch * face_h * 0.04 - face_h * 0.03

    paste_x = int(cx - anime_rotated.width / 2 + shift_x)
    paste_y = int(cy - anime_rotated.height / 2 + shift_y)

    out = base.copy()
    out.alpha_composite(anime_rotated, dest=(paste_x, paste_y))
    return out


# -----------------------------
# ZIP helper
# -----------------------------
def build_zip(file_items):
    """
    file_items: list of (filename, bytes)
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in file_items:
            zf.writestr(filename, data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# -----------------------------
# Main app
# -----------------------------
def main():
    st.title("アニメ顔 自動合成ツール")
    st.caption("V1: 元写真の顔位置・サイズ・角度を自動検出して、アニメ顔を複数画像へまとめて合成します。")

    with st.sidebar:
        st.header("設定")
        auto_remove_bg = st.checkbox("アニメ顔の白背景を自動で透過（簡易）", value=True)
        white_threshold = st.slider("白背景判定しきい値", 220, 255, 245)

        st.subheader("自動合成の微調整")
        width_scale = st.slider("顔の横幅スケール", 0.70, 1.60, 1.05, 0.01)
        height_scale = st.slider("顔の縦幅スケール", 0.70, 1.80, 1.18, 0.01)
        blur_radius = st.slider("境界ぼかし", 0.0, 6.0, 1.5, 0.1)

    st.info("アニメ顔は **透過PNG** だとかなり綺麗です。JPG でも使えますが、白背景がある場合は簡易透過になります。")

    anime_file = st.file_uploader(
        "① アニメ顔画像をアップロード",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False
    )

    photo_files = st.file_uploader(
        "② 元写真を複数アップロード",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    run_btn = st.button("③ 自動合成を実行", type="primary")

    if anime_file is not None:
        anime_preview = Image.open(anime_file)
        st.subheader("アニメ顔プレビュー")
        st.image(anime_preview, width=250)

    if run_btn:
        if anime_file is None:
            st.error("アニメ顔画像をアップロードしてください。")
            return
        if not photo_files:
            st.error("元写真を1枚以上アップロードしてください。")
            return

        anime_pil = Image.open(anime_file).convert("RGBA")
        anime_prepared = prepare_anime_face(
            anime_pil=anime_pil,
            auto_remove_bg=auto_remove_bg,
            white_threshold=white_threshold
        )

        st.subheader("処理結果")
        progress = st.progress(0)
        results_for_zip = []
        success_count = 0
        fail_count = 0

        for idx, photo_file in enumerate(photo_files, start=1):
            progress.progress(idx / len(photo_files))

            try:
                base_pil = Image.open(photo_file).convert("RGBA")
                face_info = detect_face_info(base_pil)

                with st.container():
                    st.markdown(f"### {photo_file.name}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.caption("元写真")
                        st.image(base_pil, use_container_width=True)

                    if face_info is None:
                        with col2:
                            st.error("顔を検出できませんでした。")
                        fail_count += 1
                        continue

                    out_img = overlay_anime_face(
                        base_pil=base_pil,
                        anime_face_pil=anime_prepared,
                        face_info=face_info,
                        width_scale=width_scale,
                        height_scale=height_scale,
                        blur_radius=blur_radius
                    )

                    with col2:
                        st.caption("合成後")
                        st.image(out_img, use_container_width=True)

                    img_bytes = io.BytesIO()
                    out_img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)

                    out_name = f"{photo_file.name.rsplit('.', 1)[0]}_anime.png"
                    st.download_button(
                        label=f"{out_name} をダウンロード",
                        data=img_bytes.getvalue(),
                        file_name=out_name,
                        mime="image/png",
                        key=f"dl_{idx}"
                    )

                    results_for_zip.append((out_name, img_bytes.getvalue()))
                    success_count += 1

            except Exception as e:
                st.error(f"{photo_file.name} の処理中にエラー: {e}")
                fail_count += 1

        st.success(f"処理完了: 成功 {success_count} / 失敗 {fail_count}")

        if results_for_zip:
            zip_bytes = build_zip(results_for_zip)
            st.download_button(
                label="全部まとめて ZIP ダウンロード",
                data=zip_bytes,
                file_name="anime_face_results.zip",
                mime="application/zip"
            )


if __name__ == "__main__":
    main()
