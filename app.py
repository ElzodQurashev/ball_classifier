import streamlit as st
from fastai.vision.all import *
import plotly.express as px


# PosixPath
import platform
import pathlib
plat = platform.system()
# if plat == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title("To'p turini klassifikatsiya qiliuvchi dastur")
st.text("To'p (koptok) bilan o'ynaladigan sport o'yinlaridagi to'p turini klassifikatsiya qiluvchi dastur")

# rasmni joylash
file  = st.file_uploader("Rasm yuklash", type=['jpg', 'png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    # PIL convert
    img = PILImage.create(file)

    # Model
    model = load_learner('ball.pkl')

    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
