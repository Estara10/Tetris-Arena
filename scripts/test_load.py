import os
from src.ai.dqn_model import load_inference_model

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models/next_state/best.pt")
print(f"Loading from: {model_path}")
print(f"Exists: {os.path.exists(model_path)}")
try:
    model, err = load_inference_model(model_path)
    print("Success:", model is not None)
    print("Error:", err)
except Exception as e:
    print("Exception:", e)
