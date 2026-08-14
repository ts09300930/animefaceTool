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
from PIL import Image, ImageDraw, ImageFilter


st.set_page_config(page_title="Anime Face Tool", layout="wide")


# -----------------------------
# MediaPipe Tasks setup
# -----------------------------
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = Path("/tmp/face_landmarker.task")

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
def ensure_model():
    if not MODEL_PATH.exists():
        urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


@st.cache_resource
def get_face_landmarker():
    model_path = ensure_model()

    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    return vision.FaceLandmarker.create_from_options(options)


@dataclass
class FaceInfo:
    face_center: tuple
    bbox: tuple
    roll_deg: float
    yaw_norm: float
    pitch_norm: float
    eye_distance: float
    left_eye: tuple
    right_eye: tuple
    eye_mid: tuple
    nose: tuple
    oval_pts: np.ndarray


# -----------------------------
# Utility
# -----------------------------
def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def average_point(landmarks_px, ids):
    pts = np.array([landmarks_px[i] for i in ids], dtype=np.float32)
    return pts.mean(axis=0)


def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


# -----------------------------
# Face analysis
# -----------------------------
def detect_face_info(image_pil: Image.Image) -> FaceInfo | None:
    landmarker = get_face_landmarker()

    rgb = pil_to_rgb_np(image_pil)
    h, w = rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face_landmarks = result.face_landmarks[0]

    landmarks_px = []
    for lm in face_landmarks:
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
    roll_deg = math.degrees(math.atan2(dy, dx))

    eye_mid = (left_eye + right_eye) / 2.0
    half_eye_dist = max(eye_distance / 2.0, 1e-6)

    yaw_norm = float((nose[0] - eye_mid[0]) / half_eye_dist)
    yaw_norm = clamp(yaw_norm, -1.0, 1.0)

    upper = max(nose[1] - eye_mid[1], 1.0)
    lower = max(chin[1] - nose[1], 1.0)
    ratio = lower / upper

    pitch_norm = float((ratio - 1.55) / 0.75)
    pitch_norm = clamp(pitch_norm, -1.0, 1.0)

    return FaceInfo(
        face_center=(float(face_center[0]), float(face_center[1])),
        bbox=(x_min, y_min, x_max, y_max),
        roll_deg=roll_deg,
        yaw_norm=yaw_norm,
        pitch_norm=pitch_norm,
        eye_distance=eye_distance,
        left_eye=(float(left_eye[0]), float(left_eye[1])),
        right_eye=(float(right_eye[0]), float(right_eye[1])),
        eye_mid=(float(eye_mid[0]), float(eye_mid[1])),
        nose=(float(nose[0]), float(nose[1])),
        oval_pts=oval_pts
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

    alpha = np.array(anime_rgba)[:, :, 3]
    all_opaque = np.all(alpha == 255)

    if auto_remove_bg and all_opaque:
        anime_rgba = remove_near_white_bg(anime_rgba, threshold=white_threshold)

    anime_rgba = crop_to_alpha(anime_rgba)
    return anime_rgba


# -----------------------------
# Mask helpers
# -----------------------------
def create_face_mask(
    base_size: tuple,
    face_info: FaceInfo,
    expand_x: float,
    expand_y: float,
    forehead_ratio: float,
    blur_radius: float
) -> Image.Image:
    w, h = base_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    pts = np.array(face_info.oval_pts, dtype=np.float32)
    cx, cy = face_info.face_center
    face_w = face_info.bbox[2] - face_info.bbox[0]
    face_h = face_info.bbox[3] - face_info.bbox[1]

    # 顔輪郭を少し拡張
    pts[:, 0] = (pts[:, 0] - cx) * expand_x + cx
    pts[:, 1] = (pts[:, 1] - cy) * expand_y + cy

    # おでこ方向へ少しだけ上に持ち上げる
    pts[:, 1] -= face_h * forehead_ratio

    polygon = [tuple(p) for p in pts]
    draw.polygon(polygon, fill=255)

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return mask


def soften_alpha(img_rgba: Image.Image, blur_radius: float) -> Image.Image:
    if blur_radius <= 0:
        return img_rgba

    r, g, b, a = img_rgba.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.merge("RGBA", (r, g, b, a))


# -----------------------------
# Overlay logic
# -----------------------------
def overlay_anime_face(
    base_pil: Image.Image,
    anime_face_pil: Image.Image,
    face_info: FaceInfo,
    face_size_scale: float,
    face_mask_expand_x: float,
    face_mask_expand_y: float,
    face_mask_forehead_ratio: float,
    face_mask_blur: float,
    edge_blur: float,
    vertical_offset_ratio: float
) -> Image.Image:
    base = base_pil.convert("RGBA")
    anime = anime_face_pil.convert("RGBA")

    x_min, y_min, x_max, y_max = face_info.bbox
    face_w = x_max - x_min
    face_h = y_max - y_min

    eye_mid_x, eye_mid_y = face_info.eye_mid
    eye_distance = face_info.eye_distance

    # 横幅・縦幅を別々にいじらない
    # 目と目の距離基準で "全体サイズ" のみ決定
    target_w = int(eye_distance * 2.90 * face_size_scale)

    anime_ratio = anime.height / max(anime.width, 1)
    target_h = int(target_w * anime_ratio)

    target_w = max(target_w, 10)
    target_h = max(target_h, 10)

    anime_resized = anime.resize((target_w, target_h), Image.LANCZOS)

    anime_rotated = anime_resized.rotate(
        face_info.roll_deg,
        resample=Image.BICUBIC,
        expand=True
    )

    anime_rotated = soften_alpha(anime_rotated, edge_blur)

    # アニメ画像内の「目の位置」をざっくり仮定
    # 四角貼り感を減らすため、顔上部を目基準で合わせる
    anime_eye_x = anime_rotated.width * 0.50
    anime_eye_y = anime_rotated.height * 0.38

    shift_x = face_info.yaw_norm * face_w * 0.02
    shift_y = face_info.pitch_norm * face_h * 0.02 + face_h * vertical_offset_ratio

    paste_x = int(eye_mid_x - anime_eye_x + shift_x)
    paste_y = int(eye_mid_y - anime_eye_y + shift_y)

    # まず透明キャンバス上にアニメ顔を配置
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.alpha_composite(anime_rotated, dest=(paste_x, paste_y))

    # 人物顔の輪郭に沿ったマスクを作って、四角い貼り付け感を消す
    face_mask = create_face_mask(
        base_size=base.size,
        face_info=face_info,
        expand_x=face_mask_expand_x,
        expand_y=face_mask_expand_y,
        forehead_ratio=face_mask_forehead_ratio,
        blur_radius=face_mask_blur
    )

    overlay_arr = np.array(overlay)
    mask_arr = np.array(face_mask)

    overlay_alpha = overlay_arr[:, :, 3].astype(np.float32)
    masked_alpha = overlay_alpha * (mask_arr.astype(np.float32) / 255.0)
    overlay_arr[:, :, 3] = masked_alpha.astype(np.uint8)

    overlay_masked = Image.fromarray(overlay_arr, mode="RGBA")

    out = base.copy()
    out.alpha_composite(overlay_masked)

    return out


# -----------------------------
# ZIP helper
# -----------------------------
def build_zip(file_items):
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
    st.caption("V2: 顔検出 + 顔輪郭マスクで、四角い貼り付け感を減らした版")
    with st.sidebar:
        st.header("設定")

        auto_remove_bg = st.checkbox(
            "アニメ顔の白背景を自動で透過（簡易）",
            value=True
        )

        white_threshold = st.slider(
            "白背景判定しきい値",
            220, 255, 245
        )

        st.subheader("自動合成の微調整")

        face_size_scale = st.slider(
            "アニメ顔サイズ",
            0.70, 1.50, 1.10, 0.01
        )

        vertical_offset_ratio = st.slider(
            "顔の上下位置補正",
            -0.20, 0.20, 0.00, 0.01
        )

        face_mask_expand_x = st.slider(
            "顔マスク横方向の広がり",
            0.90, 1.40, 1.06, 0.01
        )

        face_mask_expand_y = st.slider(
            "顔マスク縦方向の広がり",
            0.90, 1.50, 1.24, 0.01
        )

        face_mask_forehead_ratio = st.slider(
            "おでこ方向の拡張",
            0.00, 0.20, 0.09, 0.01
        )

        face_mask_blur = st.slider(
            "顔マスクのぼかし",
            0.0, 25.0, 8.0, 0.5
        )

        edge_blur = st.slider(
            "アニメ顔の境界ぼかし",
            0.0, 8.0, 1.5, 0.1
        )

    st.info(
        "この版では、アニメ顔の縦横比を固定したまま合成します。"
        " 四角い貼り付け感を減らすため、人物の顔輪郭マスクで切り抜いています。"
        " 透過PNGのアニメ顔だとより自然です。"
    )

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
                        face_size_scale=face_size_scale,
                        face_mask_expand_x=face_mask_expand_x,
                        face_mask_expand_y=face_mask_expand_y,
                        face_mask_forehead_ratio=face_mask_forehead_ratio,
                        face_mask_blur=face_mask_blur,
                        edge_blur=edge_blur,
                        vertical_offset_ratio=vertical_offset_ratio
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
