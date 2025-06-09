import requests
import json
import base64
from io import BytesIO

def ollama_stream_inference(
    prompt: str = "Hi, this is a dummy prompt",
    model: str = "deepseek-r1:14b",
    url: str = None,
    image_path: str = ""
):
    """
    Send a streaming request to Ollama using the given prompt and model,
    and print out the response text in real time.
    """
    assert url is not None
    assert model is not None
    
    payload = {
        "model": model
    }
    task = url.split('api/')[-1]
    if task == "embed":
        payload["input"] = prompt
    else:
        payload["prompt"] = prompt

    if image_path is not None and image_path != "":
        if type(image_path) == list:
            payload["images"] = [encode_image_to_base64(img) for img in image_path]
        else:
            payload["images"] = [encode_image_to_base64(image_path)]

    # We’ll store the entire response in this list as we stream chunks
    all_chunks = []

    # Use 'stream=True' for streaming responses
    with requests.post(url, json=payload, stream=True) as resp:
        # Raise an error if the request is not 200 OK
        resp.raise_for_status()
        
        if task == "embed":
            return resp.json()["embeddings"]
        
        # Iterate over each line that Ollama sends back
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                # If there's a blank line (keep-alive), just skip
                continue

            try:
                # Each line is a separate JSON object
                data = json.loads(line)
            except json.JSONDecodeError:
                # If you get partial or malformed data, handle/log it
                continue

            # Extract the chunk of text
            text_chunk = data.get("response", "")
            # Print directly to terminal (no extra newline, flush so it appears in real time)
            print(text_chunk, end="", flush=True)

            # Append chunk to our list so we can reconstruct later if we want
            all_chunks.append(text_chunk)

            # If "done" is True, the server indicates it's done streaming
            if data.get("done", False):
                break

    # Combine all chunks if you want the comprehensive string
    full_response = "".join(all_chunks)
    # print("\n\n---\nComplete response:\n", full_response)
    return full_response


def encode_image_to_base64(image_path: str) -> str:
    """
    Convert an image file to a base64 encoded string.

    Args:
        image_path (str): The file path of the image to encode.

    Returns:
        str: The base64 encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ollama_chat_stream(
    messages: list = [], # List of message dictionaries
    model: str = "deepseek-r1:14b",
    url: str = None,
    image_path: str = "" # Chat API might handle images differently or not at all depending on model/version
):
    """
    Send a streaming chat request to Ollama using the /api/chat endpoint.
    Maintains conversation history via the 'messages' argument.
    """
    assert url is not None and "/api/chat" in url, "URL must be for /api/chat"
    assert model is not None
    assert isinstance(messages, list) and len(messages) > 0

    payload = {
        "model": model,
        "messages": messages,
        "stream": True # Ensure streaming is enabled
    }

    # Note: Handling images with the chat endpoint might require specific formatting
    # within the 'messages' structure if the model supports it.
    # The simple 'images' key like in /api/generate might not work directly here.
    # Check Ollama documentation for the specific model if you need multimodal chat.
    if image_path is not None and image_path != "":
        # This part needs verification based on Ollama's chat API spec for images
        # It might need to be part of the last user message content
        print("Warning: Image handling in chat API needs verification.")
        # payload["images"] = [encode_image_to_base64(image_path)] # Might not work

    all_chunks = []
    assistant_response = ""

    # Use 'stream=True' for streaming responses
    with requests.post(url, json=payload, stream=True) as resp:
        # Raise an error if the request is not 200 OK
        resp.raise_for_status()

        # Iterate over each line that Ollama sends back
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                # If there's a blank line (keep-alive), just skip
                continue

            try:
                # Each line is a separate JSON object
                data = json.loads(line)
            except json.JSONDecodeError:
                # If you get partial or malformed data, handle/log it
                print(f"\nError decoding JSON line: {line}\n")
                continue

            # For the chat endpoint, the text chunk is inside response['message']['content']
            message_chunk = data.get("message", {})
            text_chunk = message_chunk.get("content", "")

            if text_chunk:
                # Print directly to terminal (no extra newline, flush so it appears in real time)
                print(text_chunk, end="", flush=True)
                # Append chunk to reconstruct the full response
                all_chunks.append(text_chunk)

            # Check if the stream is done for this request
            # The 'done' field indicates the end of the *entire* streaming response,
            # not just one message chunk.
            if data.get("done", False):
                # print(f"\n[DEBUG] Received 'done': True. Full data: {data}") # Optional debug
                break # Exit the loop once the stream is done

    # Combine all chunks to get the complete assistant response for this turn
    assistant_response = "".join(all_chunks)
    print() # Add a newline after the assistant finishes speaking
    return assistant_response


if __name__ == "__main__":
    # For generative models
    full_response = ollama_stream_inference(
        prompt ="Why is the sky blue?",
        model = "gemma3:27b",
        url = "http://0.0.0.0:7860/api/generate"
    )
    
    # # For embedding models
    # full_response = ollama_stream_inference(
    #     prompt ="Why is the sky blue?",
    #     model = "nomic-embed-text",
    #     url = "http://0.0.0.0:7860/api/embed"
    # )
    
    # print(full_response)