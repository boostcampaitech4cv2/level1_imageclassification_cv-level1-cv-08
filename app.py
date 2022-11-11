import io
from PIL import Image
from typing import Tuple
import streamlit as st
import torch
from model.model import TimmModelMulti
from utils.util import get_age, get_gender, get_mask
from data_loader.transform import streamlit_transform


@st.cache
def load_model() -> TimmModelMulti:
    with open("streamlit_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimmModelMulti(model_name="efficientnetv2_rw_s").to(device)
    model.load_state_dict(torch.load(config["model_path"])["state_dict"])

    return model


def get_prediction(
    model: TimmModelMulti, image_bytes: bytes
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = streamlit_transform(image_bytes=image_bytes).to(device)
    outputs = model.forward(tensor)
    pred = get_mask(outputs) + get_gender(outputs) + get_age(outputs, device)
    return tensor, pred


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


def main():
    st.title("Mask Classification AI")
    st.header("Did you wear a mask?")
    with open("streamlit_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = load_model()
    model.eval()

    if uploaded_file := st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"]
    ):
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption="Uploaded Image")
        st.write("Loading...")
        _, y_hat = get_prediction(model, image_bytes)
        label = config["classes"][y_hat.item()][0]

        st.write(f"{label}이신가요?")


main()
