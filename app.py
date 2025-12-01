import streamlit as st
import pydicom
import io
import base64
from PIL import Image
import requests
import json


# -------------------------
# Page config
# -------------------------

st.set_page_config(page_title="MazAI", layout="centered")

st.logo("assets/logo.png", size="large")
st.title("MazAI")
st.write("Загрузите изображение рентгенографии питомца в формате PNG, JPG или DICOM, и MazAI вернет его клиническое описание.")


# -------------------------
# Helper Functions
# -------------------------

def read_image(file):
    """Read DICOM, PNG, or JPEG and return a PIL.Image."""
    if file.name.lower().endswith(".dcm"):
        dcm = pydicom.dcmread(file)
        image = Image.fromarray(dcm.pixel_array)
    else:
        image = Image.open(file)
    return image.convert("RGB")


def encode_image_to_base64(image: Image.Image):
    """Convert PIL image to Base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_openrouter_vlm(image: Image.Image) -> str:
    """Send the image to Qwen-VL (OpenRouter API) and get the description."""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }

    base64_image = encode_image_to_base64(image)
    data_url = f"data:image/jpeg;base64,{base64_image}"

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — русскоязычный ИИ-ассистент ветеринара-рентгенолога. "
                "Твоя задача — описывать рентгеновские снимки животных "
                "так, как это делает ветеринар-рентгенолог. "
                "Описывай профессионально, но понятно. "
                "Если качество изображения низкое — упомяни это. "
                "Избегай выдумывания данных, которых нет на снимке. "
                "Избегай комментариев касательно метаданных (дат, имён) отображающиеся на изображении, описывай только анатомические признаки."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Опиши этот снимок."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]

    payload = {
        "model": "qwen/qwen3-vl-235b-a22b-instruct",
        "messages": messages,
        "stream": True
    }

    buffer = b""
    with requests.post(url, json=payload, headers=headers, stream=True) as r:
        for chunk in r.iter_content(chunk_size=1024):
            if not chunk:
                continue

            buffer += chunk
            while True:
                try:
                    # Find the next complete SSE line
                    line_end = buffer.find(b"\n")
                    if line_end == -1:
                        break

                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1:]

                    if not line.startswith(b"data: "):
                        continue

                    data = line[6:]

                    if data == b"[DONE]":
                        break

                    try:
                        data_obj = json.loads(data.decode("utf-8"))
                        content = data_obj["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass
                except Exception:
                    break


# -------------------------
# UI
# -------------------------

uploaded_file = st.file_uploader(
    "Загрузите изображение", 
    type=["png", "jpg", "jpeg", "dcm"]
)

if uploaded_file:
    image = read_image(uploaded_file)

    st.image(image, caption="Загруженное изображение", use_container_width=True)

    if st.button("Анализ с помощью ИИ"):
        st.write_stream(call_openrouter_vlm(image))
