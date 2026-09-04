from src.ser.emotion_classifier import EmotionClassifier

def test_ser_initialization():
    ser = EmotionClassifier()
    assert ser.model_path is not None

def test_ser_extract_features_empty():
    ser = EmotionClassifier()
    features = ser.extract_features(b"")
    assert isinstance(features, dict)

def test_ser_classify_empty():
    ser = EmotionClassifier()
    result = ser.classify(b"")
    assert "label" in result
    assert "confidence" in result

