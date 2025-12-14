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


def call_openrouter_vlm(messages: list[dict]) -> str:
    """Send the image to Qwen-VL (OpenRouter API) and get the description."""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }
    prepared_messages = [
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
        }
    ]
    for msg in messages:
        prepared_msg = {
            "role" : msg["role"],
            "content" : []
        }
        image = msg.get("image", None)
        prepared_msg["content"].append(
            {
                "type" : "text",
                "text" : msg["content"]
            }
        )
        if image:
            base64_image = encode_image_to_base64(image)
            data_url = f"data:image/jpeg;base64,{base64_image}"
            prepared_msg["content"].append(
                {
                    "type" : "image_url",
                    "image_url" : {
                        "url" : data_url
                    }
                }
            )
        prepared_messages.append(prepared_msg)

    payload = {
        "model": "qwen/qwen3-vl-235b-a22b-instruct",
        "messages": prepared_messages,
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


if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "clear_input" in st.session_state and st.session_state.clear_input:
    st.session_state.user_input = ""
    st.session_state.uploader_key += 1 
    st.session_state.clear_input = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["content"].strip():
            st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"], width=300)

uploaded_file = st.file_uploader(
    "📎 Прикрепите изображение (опционально)",
    type=["png", "jpg", "jpeg"],
    key=f"uploader_{st.session_state.uploader_key}"
)
user_text = st.text_input("💬 Ваш запрос", key="user_input", placeholder="Введите запрос или прикрепите изображение")

if st.button("📤 Отправить"):
    if user_text.strip() or uploaded_file:
        msg = {"role": "user", "content": user_text}

    if uploaded_file:
        msg["image"] = read_image(uploaded_file)

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"], width=300)
    st.session_state.messages.append(msg)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in call_openrouter_vlm(st.session_state.messages):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

    st.session_state.clear_input = True
    st.rerun()