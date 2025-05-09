import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import gradio as gr
from colored_print import color, style
import os
import time

# Enable MPS fallback to CPU for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ====================================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

# ====================================================================
def load_model():
    device = get_device()
    print(f"Using {device} device")
    
    processor = AutoProcessor.from_pretrained("model/SmolVLM-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "model/SmolVLM-Instruct",
        torch_dtype=torch.float16,  # Keep half precision for efficiency
    ).to(device)
    
    return processor, model, device

# ====================================================================
# Load model and processor at startup
processor, model, DEVICE = load_model()

# ====================================================================
def generate_caption(
    image, 
    caption_style,
    max_new_tokens=156,
    do_sample=True,
    temperature=0.4,
    top_p=0.9,
    repetition_penalty=1.1
):
    

    # debug_info = f"""
    # Image Format: {image.format}
    # Image Mode: {image.mode}
    # Image Size: {image.size}
    # """

    # print(debug_info, color.YELLOW)

    start_time = time.time()
    
    # Define caption style prompts
    style_prompts = {
        "Short and concise": "Caption this image - short and concise.",
        "Brief detailed": "Caption this image - brief but detailed.",
        "Moderately detailed": "Caption this image - moderately detailed, moderately descriptive.",
        "Highly detailed": "Caption this image - highly detailed, highly descriptive."
    }
    
    prompt_text = style_prompts.get(caption_style, "Caption this image.")
    
    # construct multi-modal input msg
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # Get only the assistant's response
    full_output = generated_texts[0]
    
    if "Assistant:" in full_output:
        response_only = full_output.split("Assistant: ")[-1].strip()
    else:
        response_only = full_output.strip()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return response_only, f"Generated in {execution_time:.2f} seconds"

# ====================================================================
# GRADIO UI
# ====================================================================
# Create Gradio interface
with gr.Blocks(title="Image Captioner") as demo:
    gr.Markdown("# Image Captioner using SmolVLM-Instruct")
    gr.Markdown("Upload an image and adjust the settings to generate a caption")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            
            caption_style = gr.Dropdown(
                choices=["Short and concise", "Brief detailed", "Moderately detailed", "Highly detailed"],
                value="Brief detailed",
                label="Caption Style"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                max_tokens = gr.Slider(minimum=50, maximum=500, value=156, step=1, label="Max New Tokens")
                do_sample_checkbox = gr.Checkbox(value=True, label="Do Sample")
                temperature_slider = gr.Slider(minimum=0.1, maximum=1.5, value=0.4, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top P")
                rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, label="Repetition Penalty")
            
            submit_btn = gr.Button("Generate Caption", variant="primary")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Generated Caption", lines=5)
            info_text = gr.Textbox(label="Info", lines=1)
    
    submit_btn.click(
        fn=generate_caption,
        inputs=[
            input_image,
            caption_style,
            max_tokens,
            do_sample_checkbox,
            temperature_slider,
            top_p_slider,
            rep_penalty
        ],
        outputs=[output_text, info_text]
    )

    gr.Markdown("""
    
    ### Parameters:
    - **Max New Tokens**: Controls the maximum length of the generated caption
    - **Do Sample**: When enabled, uses sampling for more diverse outputs
    - **Temperature**: Higher values (>1.0) make output more random, lower values make it more deterministic
    - **Top P**: Controls diversity via nucleus sampling
    - **Repetition Penalty**: Higher values discourage repetition in the text
    """)

# Launch the Gradio app
if __name__ == "__main__":
    print(f"Model loaded on {DEVICE}")
    demo.launch()