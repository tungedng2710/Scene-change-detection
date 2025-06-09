import os

USE_VLM = True
PROVIDER = "gemini" # ollama or gemini
OLLAMA_URL = "http://27.66.108.135:7860/api/generate"
OLLAMA_MODEL_NAME = "gemma3:27b"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY = 'AIzaSyBv6EN6aibqHtIAH1p2agOzC66hF_rcbrM'

DEFAULT_INSTRUCTION = """You are the supervisor to monitor the change of scene,
The first image is the reference,
Describe the major changes of scene in the second image only.
"""
UPGRADE_INSTRUCTION = """Unlock the full power of TonAI with Premium Access! 🚀
With TonAI Premium, you can harness advanced LLM capabilities to analyze and describe differences between two images — perfect for spotting subtle changes, scene shifts, or detailed visual comparisons.
👉 Upgrade now to experience AI that sees and explains like a human.
"""