import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from colored_print import color, style
import os
import time
import sys
import argparse
from threading import Thread
from transformers.generation.streamers import TextIteratorStreamer
from huggingface_hub import snapshot_download

def check_image_exists(image_path):
    """Check if input image file exists"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}", color.RED)
        print("Please check the path and ensure the file extension is included (e.g., .jpg, .png, .webp)", color.YELLOW)
        sys.exit(1)

def check_model_exists_otherwise_download(model_path):
    print(f"Checking Model Path: {model_path}", color.ORANGE)

    # Step 1: Check if directory exists
    if os.path.exists(model_path):
        print(f"✅ Directory exists", color.ORANGE)

        # Step 1b: Check if all required files exist and have correct hashes
        if validate_model_files(model_path):
            print(f"✅ All model files are valid", color.GREEN)
            return True
        else:
            print(f"⚠️  Model files are missing or corrupted", color.YELLOW)
            # Will fall through to download section        

    else:
        print(f"❌ Directory not found: attempting to download.", color.ORANGE)

        REPO_NAME = f"yushan777/{os.path.basename(model_path)}"

        # Download section
        try:
            print(f"⬇️  Downloading model from HuggingFace repo: {REPO_NAME}", color.ORANGE)
            
            # Download the repository to the specified path
            snapshot_download(
                repo_id=REPO_NAME,
                local_dir=model_path,
                local_dir_use_symlinks=False,  # Download actual files, not symlinks
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
            



# ==========================================================================
def validate_model_files(model_path):

    # This is where you'd implement your hash checking logic
    # For example:
    
    if "SmolVLM-256M-Instruct" in model_path:
        required_files = [
            "added_tokens.json",
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "processor_config.json",
            "README.md",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
        ]
    elif "SmolVLM-500M-Instruct" in model_path:
        required_files = [
            "added_tokens.json",
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "processor_config.json",
            "README.md",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
        ]
    elif "SmolVLM-Instruct" in model_path:
        required_files = [
            "added_tokens.json",
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "LICENSE",
            "merges.txt",
            "mixture_the_cauldron.png",
            "model.safetensors",
            "preprocessor_config.json",
            "processor_config.json",
            "README.md",
            "smolvlm-data.pdf",
            "SmolVLM.png",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
        ]
    else:
        # Handle the case where none of the conditions match
        print(f"❌ Unknown model type: {model_path}", color.RED)
        return False        
    
    # Check if all required files exist
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.isfile(file_path):
            print(f"❌ Missing file: {file}", color.RED)
            return False
    

    # # TODO: Add hash checking for each file
    # # For now, just return True if all files exist
    # print("📋 All required files found (hash check pending)", color.YELLOW)
    return True

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
    

    check_model_exists_otherwise_download(MODEL_PATH)





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