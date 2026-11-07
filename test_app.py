from app import add
import os

def test_add():
    assert add(2, 3) == 5

def test_key():
    api_key=os.getenv("GEMINI_API_KEY")
    assert api_key=="chintu"