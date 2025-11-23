import streamlit as st
import pydicom
from PIL import Image

st.set_page_config(page_title="MazAI", layout="centered")

st.logo("assets/logo.png", size="large")
st.title("MazAI")
st.write("Загрузите изображение рентгенографии питомца в формате PNG, JPG или DICOM, и MazAI вернет его клиническое описание.")

uploaded_file = st.file_uploader(
    "Загрузите изображение", 
    type=["png", "jpg", "jpeg", "dcm"]
)

def load_dicom(file):
    """Convert DICOM to a PIL image."""
    ds = pydicom.dcmread(file)
    arr = ds.pixel_array
    img = Image.fromarray(arr)
    return img

if uploaded_file:
    file_type = uploaded_file.name.lower()

    try:
        if file_type.endswith(".dcm"):
            image = load_dicom(uploaded_file)
        else:
            image = Image.open(uploaded_file)

        st.image(image, caption="Загруженное изображение", use_container_width=True)

        st.subheader("Описание")
        st.write("""
        Это временное описание.
        В дальнейшем оно будет сгенерировано искусственным интеллектом.
        
        **Пример описания:**
        - Рентген показывает фронтальный вид грудной клетки.
        - Явных переломов не обнаружено.
        - Поля легких в основном чистые.
        - Силуэт сердца выглядит нормально.
        """)

    except Exception as e:
        st.error(f"Error processing file: {e}")
