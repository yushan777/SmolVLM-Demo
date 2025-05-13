import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from colored_print import color, style
import os
import time
import sys
import json
import argparse
from threading import Thread
from transformers.generation.streamers import TextIteratorStreamer
from huggingface_hub import snapshot_download
import xxhash
from smolvlm.verify_download_model import hash_file, validate_model_files, check_model_files, download_model_from_HF

# Enable MPS fallback to CPU for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def check_image_exists(image_path):
    """Check if input image file exists"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}", color.RED)
        print("Please check the path and ensure the file extension is included (e.g., .jpg, .png, .webp)", color.YELLOW)
        sys.exit(1)

# ====================================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


# ====================================================================
def load_model(model_path):
    device = get_device()
    print(f"Using {device} device", color.GREEN)
    

    # FIRST CHECK IF MODEL EXISTS IN LOCAL DIRECTORY (./models/)

    print(f"MODEL_PATH = {model_path}", color.CYAN)

    processor = AutoProcessor.from_pretrained(model_path)
    
    # Attention fallback order 
    attention_fallback = [
        "flash_attention_2",  # Best performance if available
        "sdpa",              # Good default in PyTorch 2.0+
        "xformers",          # Good alternative, memory efficient
        "eager",             # Reliable fallback
        None                 # Absolute fallback
    ]
    
    # Try each attention implementation
    for impl in attention_fallback:
        try:
            if impl is not None:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    _attn_implementation=impl
                ).to(device)
                print(f"✓ Loaded with {impl} attention", color.GREEN)
            else:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                ).to(device)
                print("✓ Loaded with no attention specified", color.GREEN)

            return processor, model, device
        
        except ImportError as e:
            if impl == "flash_attention_2" and "flash_attn" in str(e):
                print(f"  flash_attention_2 not available (package not installed)", color.YELLOW)
            else:
                print(f"  Failed with {impl}: {e}", color.RED)
            continue
        except Exception as e:
            print(f"  Failed with {impl}: {e}", color.RED)
            continue
    
    # If we get here, all attempts failed
    raise Exception("Failed to load model with any attention implementation")


# ===============================================================
def main():
    # Parse command line arguments for streaming option
    parser = argparse.ArgumentParser(description="Run SmolVLM with optional streaming")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode for text generation")
    parser.add_argument("--image", type=str, default="input/dog.webp", help="Input image path")
    parser.add_argument("--prompt", type=str, default="Caption this image - brief but detailed.", help="Text prompt")
    parser.add_argument("--model", 
                    choices=["SmolVLM-Instruct", "SmolVLM-500M-Instruct", "SmolVLM-256M-Instruct"],
                    default="SmolVLM-Instruct", 
                    help="Model to use (default: SmolVLM-Instruct)")
    args = parser.parse_args()
    
    # MODEL SELECTION AND PATH
    MODEL_PATH = f"models/{args.model}"

    
    print(f"MODEL_PATH = {MODEL_PATH}", color.CYAN)

    # ========================================================
    # Check if image file exists
    check_image_exists(args.image)
    


    start_time = time.time()
    

    # checks and validates the model files 
    filesokay = check_model_files(MODEL_PATH)

    # if not exist or incomplete then
    if not filesokay:
        # Use the new download function
        download_model_from_HF(MODEL_PATH)

    print(f"MODEL_PATH = {MODEL_PATH}", color.ORANGE)

    # LOAD MODEL
    processor, model, DEVICE = load_model(MODEL_PATH)

    
    
    # Load image with error handling
    try:
        image1 = load_image(args.image)
        print(f"Successfully loaded image: {args.image}", color.GREEN)
    except Exception as e:
        print(f"Error loading image: {e}", color.RED)
        print("Please ensure the file path is correct and includes the file extension.", color.YELLOW)
        sys.exit(1)
    
    # Construct multi-modal input msg
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt}
            ]
        },
    ]
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image1], return_tensors="pt")
    inputs = inputs.to(DEVICE)
    
    print(f"Running in {'streaming' if args.stream else 'non-streaming'} mode...\n", color.BLUE)
    
    if args.stream:
        # Streaming version
        print("Generated Caption (streaming):", end="", flush=True)
        
        # Setup streamer
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        
        # Prepare generation arguments
        generation_args = {
            **inputs, 
            "streamer": streamer,
            "max_new_tokens": 156,
            "do_sample": True,
            "temperature": 0.4,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }
        
        # Run generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_args)
        thread.start()
        
        # Stream generated text as it comes in
        generated_text_so_far = ""
        is_first_chunk = True
        
        for new_text_chunk in streamer:
            # Strip leading space from the first chunk only
            if is_first_chunk:
                new_text_chunk = new_text_chunk.lstrip()
                is_first_chunk = False
            
            generated_text_so_far += new_text_chunk
            # Print the new chunk and flush immediately
            print(new_text_chunk, end="", flush=True)
        
        thread.join()
        print("\n")  # New line after streaming is complete
        
        # Process the full response
        if "Assistant:" in generated_text_so_far:
            response_only = generated_text_so_far.split("Assistant: ")[-1].strip()
        else:
            response_only = generated_text_so_far.strip()
        
        print(f"\n>>>>> Final Response\n{response_only}", color.ORANGE)
        
    else:
        # Non-streaming version (original code)
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=156,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        # Since we only passed in one image, we just need the first response
        full_output = generated_texts[0]
        print(f">>>>> Full Output\n{full_output}", color.BRIGHT_BLUE)
        
        # Split based on the assistant role
        if "Assistant:" in full_output:
            response_only = full_output.split("Assistant: ")[-1].strip()
        else:
            response_only = full_output.strip()
        
        print(f">>>>> Assistant Response Only\n{response_only}", color.ORANGE)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.4f} seconds")


# ==========================================================
if __name__ == "__main__":
    main()
