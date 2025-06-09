import argparse
from ollama_api import  ollama_stream_inference

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_img", type=str, default="1.png")
    parser.add_argument("--test_img", type=str, default="rgb_image.png")
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    # prompt = """
    # You are the supervisor to monitor the change of scene,
    # The first image is the reference,
    # Describe the major changes of scene in the second image only.
    # """
    prompt = """
    Hãy mô tả ngắn gọn sự thay đổi cảnh quan giữa 2 bức ảnh
    """
    full_response = ollama_stream_inference(
        image_path=[args.ref_img, args.test_img],
        prompt = prompt,
        model = "gemma3:27b",
        url = "http://27.66.108.30:7860/api/generate"
    )
    with open("temp.txt", "w", encoding="utf-8") as f:
        f.write(full_response)