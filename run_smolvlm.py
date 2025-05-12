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

def check_image_exists(image_path):
    """Check if input image file exists"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}", color.RED)
        print("Please check the path and ensure the file extension is included (e.g., .jpg, .png, .webp)", color.YELLOW)
        sys.exit(1)

def download_model_from_HF(model_path):
    
    # Download model from HF.

    REPO_NAME = f"yushan777/{os.path.basename(model_path)}"

    try:
        print(f"⬇️ Downloading model from HuggingFace repo: {REPO_NAME}", color.ORANGE)
        
        # Download the repository to the specified path
        snapshot_download(
            repo_id=REPO_NAME,
            local_dir=model_path,
            local_dir_use_symlinks=False,  
        )
        
        print(f"✅ Model downloaded successfully", color.GREEN)
        
        # Verify the downloaded files
        if validate_model_files(model_path):
            print(f"✅ Downloaded files validated", color.GREEN)
            return True
        else:
            print(f"❌ Downloaded files validation failed", color.RED)
            return False
            
    except Exception as e:
        print(f"❌ Failed to download model: {str(e)}", color.RED)
        return False


def check_model_files(model_path):
    print(f"Checking Model Path: {model_path}", color.ORANGE)

    # validate model files first
    if validate_model_files(model_path):
        print(f"✅ All model files are valid", color.GREEN)
        return True
    
    # If we get here, either directory doesn't exist or files are invalid
    print(f"⚠️ Model files are missing or corrupted - attempting to download", color.YELLOW)
    return False 

            
# ==============================================================
def hash_file(filepath, chunk_size=1024 * 1024):  # default 1MB
    h = xxhash.xxh3_64()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

# ==========================================================================
def validate_model_files(model_path, chunk_size=1024 * 1024):
    # files hashed with xxhash.xxh3_64()
    
    if "SmolVLM-256M-Instruct" in model_path:
        required_files = [
            {"name": "added_tokens.json","hash": "966a479d6d5d5128"},
            {"name": "chat_template.json","hash": "23bf0f409ddc6e30"},
            {"name": "config.json","hash": "6489279a8c3c5ae7"},
            {"name": "generation_config.json","hash": "6e99ea1697338d6d"},
            {"name": "merges.txt","hash": "4d16a8257a0470ad"},
            {"name": "model.safetensors","hash": "804a944c3ae77765"},
            {"name": "preprocessor_config.json","hash": "2bdb8382f60bdb98"},
            {"name": "processor_config.json","hash": "1db78eee2f186fd5"},
            {"name": "special_tokens_map.json","hash": "5969276611f60ff1"},
            {"name": "tokenizer.json","hash": "7c81a296f87a3d25"},
            {"name": "tokenizer_config.json","hash": "1179e1f25d5b3e19"},
            {"name": "vocab.json","hash": "e3790d332807f48a"},
        ]
    elif "SmolVLM-500M-Instruct" in model_path:
        required_files = [
            {"name": "added_tokens.json", "hash": "966a479d6d5d5128"},
            {"name": "chat_template.json", "hash": "23bf0f409ddc6e30"},
            {"name": "config.json", "hash": "32fbfed32a41d912"},
            {"name": "generation_config.json", "hash": "6e99ea1697338d6d"},
            {"name": "merges.txt", "hash": "4d16a8257a0470ad"},
            {"name": "model.safetensors", "hash": "6db68c3544f56c2f"},
            {"name": "preprocessor_config.json", "hash": "2bdb8382f60bdb98"},
            {"name": "processor_config.json", "hash": "1db78eee2f186fd5"},
            {"name": "special_tokens_map.json", "hash": "5969276611f60ff1"},
            {"name": "tokenizer.json", "hash": "7c81a296f87a3d25"},
            {"name": "tokenizer_config.json", "hash": "1179e1f25d5b3e19"},
            {"name": "vocab.json", "hash": "e3790d332807f48a"},
        ]
    elif "SmolVLM-Instruct" in model_path:
        required_files = [
            {"name": "added_tokens.json", "hash": "a66c8cf27a9b91f9"},
            {"name": "chat_template.json", "hash": "23bf0f409ddc6e30"},
            {"name": "config.json", "hash": "33ccb05bfe1f09a9"},
            {"name": "generation_config.json", "hash": "a3ed37c06f67d572"},
            {"name": "merges.txt", "hash": "4d16a8257a0470ad"},
            {"name": "model.safetensors", "hash": "4531c8bb61db480b"},
            {"name": "preprocessor_config.json", "hash": "ff86f770d8bb049f"},
            {"name": "processor_config.json", "hash": "6d5856bccc2944b5"},
            {"name": "special_tokens_map.json", "hash": "5969276611f60ff1"},
            {"name": "tokenizer.json", "hash": "7995f46b407a54ce"},
            {"name": "tokenizer_config.json", "hash": "f626bc19dde956c7"},
            {"name": "vocab.json", "hash": "e3790d332807f48a"},
        ]
    else:
        print(f"❌ Unknown model type: {model_path}", color.RED)
        return False        

    valid = True  # assume everything is fine until proven otherwise

    for file_info in required_files:
        file_path = os.path.join(model_path, file_info["name"])
        
        if not os.path.isfile(file_path):
            print(f"❌ Missing file: {file_info['name']}", color.RED)
            valid = False
            continue

        try:
            file_hash = hash_file(file_path, chunk_size=chunk_size)

            if file_hash == file_info["hash"]:    
                # print(f' - {file_info["name"]}: {file_hash}: OKAY', color.BRIGHT_GREEN)
                pass
            else:
                print(f' - {file_info["name"]}: {file_hash}: MISMATCH. Expected {file_info["hash"]}', color.BRIGHT_RED)
                valid = False

        except Exception as e:
            print(f'❌ Error checking hash for {file_info["name"]}: {str(e)}', color.RED)
            valid = False

    return valid




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
    MODEL_PATH = f"model/{args.model}"

    # ========================================================
    # Check if image file exists
    check_image_exists(args.image)
    
    # Enable MPS fallback to CPU for operations not supported on MPS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
    # Determine device
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print("Using CUDA device")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
        print("Using MPS device")
    else:
        DEVICE = "cpu"
        print("No GPU available, using CPU")
    
    start_time = time.time()
    

    # checks and validates the model files 
    filesokay = check_model_files(MODEL_PATH)

    # if not exist or incomplete then
    if not filesokay:
        # Use the new download function
        download_model_from_HF(MODEL_PATH)






    # ========================================================
    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    model.to(DEVICE)

    
    
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


if __name__ == "__main__":
    main()