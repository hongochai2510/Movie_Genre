import streamlit as st
import joblib
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO
from PIL import Image
from xgboost import XGBClassifier
import glob
import os

meta = joblib.load("movie_genre_meta.pkl")

state_dict = meta["backbone_state_dict"]
scaler = meta["scaler"]
svd = meta["svd"]
mlb = meta["mlb"]
thresholds = meta["thresholds"]

backbone = models.densenet169(weights="IMAGENET1K_V1")
backbone.classifier = torch.nn.Identity()
backbone.load_state_dict(state_dict)
backbone.eval()
device = torch.device("cpu")
backbone = backbone.to(device)

json_files = sorted(glob.glob("xgb_model/class_*.json"))

estimators = []
for f in json_files:
    clf = XGBClassifier()
    clf.load_model(f)
    estimators.append(clf)

# custom OneVsRest wrapper
class OVR_XGB:
    def __init__(self, estimators):
        self.estimators = estimators
    
    def predict_proba(self, X):
        all_probs = []
        for clf in self.estimators:
            p = clf.predict_proba(X)[:, 1]  
            all_probs.append(p)
        return np.vstack(all_probs).T  # shape (N, C)

xgb = OVR_XGB(estimators)

yolo = YOLO("yolov8s.pt")
n_obj = len(yolo.names)

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])

def extract_yolo_features(img_pil):
    r = yolo(img_pil, verbose=False)[0]
    v = np.zeros(n_obj, dtype=np.float32)

    if r.boxes is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for c, cf in zip(classes, confs):
            v[c] += float(cf)

    return np.clip(v, 0, 1)

st.title("Movie Genre Classification - XGBoost Demo")
st.write("Tải lên poster phim để dự đoán thể loại.")

uploaded_img = st.file_uploader("Chọn poster phim", type=["jpg","jpeg","png"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Poster đã tải lên", width=300)

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        dense_feat = backbone(x).numpy()

    yolo_feat = extract_yolo_features(img).reshape(1, -1)

    feat = np.hstack([dense_feat, yolo_feat])

    feat_scaled = scaler.transform(feat)

    feat_svd = svd.transform(feat_scaled)

    probs = xgb.predict_proba(feat_svd)[0]

    binary_pred = (probs >= thresholds).astype(int)
    binary_pred = binary_pred.reshape(1, -1)

    predicted_genres = mlb.inverse_transform(binary_pred)[0]

    st.subheader("Kết quả dự đoán:")
    st.write(predicted_genres)

    st.subheader("Confidence cho từng thể loại:")
    for label, p in zip(mlb.classes_, probs):
        st.write(f"**{label}**: {p:.3f}")
