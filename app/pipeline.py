import time
from google import genai
from google.genai import types
from ollama_api import  ollama_stream_inference
import PIL.Image
from segment_any_change import infer
import llm_config
import asyncio

def describe_change(provider: str = "ollama",
                    api_key: str = None,
                    url: str = None, 
                    model: str = None,
                    ref_path: str = None,
                    test_path: str = None):
    response = ""
    if provider=="ollama":
        response = ollama_stream_inference(
                image_path=[ref_path, test_path],
                prompt = llm_config.DEFAULT_INSTRUCTION,
                model = model or llm_config.OLLAMA_MODEL_NAME,
                url = url or llm_config.OLLAMA_URL
            )
    
    elif provider=="gemini":
        current_api_key = api_key or llm_config.GEMINI_API_KEY
        if not current_api_key:
            return llm_config.UPGRADE_INSTRUCTION
        client = genai.Client(api_key=current_api_key)
        response = client.models.generate_content(
            model=model or llm_config.GEMINI_MODEL_NAME,
            contents=[llm_config.DEFAULT_INSTRUCTION, PIL.Image.open(ref_path), PIL.Image.open(test_path)]
        )
        response = response.text
    
    else:
        response = f"{provider} is not supported yet"
    return response

def run_change_detection(ref_path: str = None,
                         test_path: str = None,
                         output_mask_path: str = None,
                         describe_result: bool = False):
    percent_content = "0.0%"
    try:   
        percent_content = infer(ref_path, test_path, overlay_path=output_mask_path)
        print(percent_content)
        if describe_result:
            # Using gemini by default as an example, can be made configurable
            describe = describe_change(provider=llm_config.PROVIDER,
                                       api_key=llm_config.GEMINI_API_KEY,
                                       url=llm_config.OLLAMA_URL, 
                                       ref_path=ref_path,
                                       test_path=test_path)
        else:
            describe = llm_config.UPGRADE_INSTRUCTION
        
        return describe, output_mask_path, percent_content
    except Exception as e:
        return f"Error occurred: {e}", None, percent_content


async def run_change_detection_async(ref_path=None, test_path=None,
                                     output_mask_path=None, describe_result=False):
    loop = asyncio.get_running_loop()

    # Kick infer() off-thread so it doesn't block the event loop
    # output_mask_path = f"outputs/{time.time()}_mask_overlay.png"
    infer_task = loop.run_in_executor(
        None, infer, ref_path, test_path, output_mask_path
    )

    # Kick description either off-thread or as native async if you refactor later
    if describe_result:
        desc_task = loop.run_in_executor(
            None,
            describe_change,
            llm_config.PROVIDER, llm_config.GEMINI_API_KEY,
            llm_config.OLLAMA_URL, ref_path, test_path
        )
    else:
        desc_task = asyncio.sleep(0, result=llm_config.UPGRADE_INSTRUCTION)

    # Gather results concurrently
    percent_content, describe = await asyncio.gather(infer_task, desc_task)
    return describe, output_mask_path, percent_content

    
if __name__ == "__main__":
    current_time = time.time()
    mask_path = f"outputs/{current_time}_mask_overlay.png"
    run_change_detection("test_samples/test1_1.png", 
                         "test_samples/test1_2.png",
                         output_mask_path=mask_path, 
                         describe_result=False)
