from transformers import pipeline
classifier = pipeline('sentiment-analysis')

def predict(data):
    return classifier(data)