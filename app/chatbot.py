from .emotion_model import load_emotion_pipeline

# init once
ID2LABEL = {
    0: "anger",
    1: "sadness",
    2: "joy",
    3: "neutral",
}

emotion_pipeline = load_emotion_pipeline(
    checkpoint_path="models/emotion_lstm.pt",
    vocab_path="models/vocab.pkl",
    id2label=ID2LABEL,
)

# inside generate_response(...)
emotion, confidence = emotion_pipeline.predict(user_message)
