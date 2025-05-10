import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import gradio as gr
from colored_print import color, style
import os
import time

# Enable MPS fallback to CPU for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Define caption style prompts
STYLE_PROMPTS = {
    "Short and concise": "Caption this image, stick to the facts but make it short and concise.",
    "Brief but detailed": "Caption this image, stick to the facts but make it a bit longer than short, but stil detailed.",
    "Moderately detailed": "Caption this image, stick to the facts and make it moderately detailed and moderately descriptive.",
    "Highly detailed": "Caption this image, stick to the facts and make it highly detailed and highly descriptive."
}

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

    # Check if image is provided, if not, quit and show msg
    if image is None:
        msg = "Please upload an image first to generate a caption."
        return msg, msg
    
        
    start_time = time.time()
        
    prompt_text = STYLE_PROMPTS.get(caption_style, "Caption this image.")
    
    print(f"prompt_text = {prompt_text}", color.ORANGE)

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
    
    # Create editable text     
    additional_text = response_only
    
    return response_only, additional_text

# ====================================================================
def process_caption(additional_text):
    print("process_caption() called")

# ====================================================================
# GRADIO UI
# ====================================================================
# Create Gradio interface
with gr.Blocks(title="Image Captioner", 
               css="""           
                    /* outermost wrapper of the entire Gradio app */         
                    .gradio-container {
                        max-width: 100% !important;
                        margin: 0 auto !important;
                    }
                    /* main content area within the Gradio container */
                    .main {
                        max-width: 1200px !important;
                        margin: 0 auto !important;
                    }
                    /* Individual columns */
                    .fixed-width-column {
                        width: 600px !important;
                        flex: none !important;
                    }
                    /* Custom color for the editable text box */
                    #additional_text_box textarea {
                        /*  color: #2563eb !important;  text color */
                        font-family: 'Courier New', monospace !important; /* Optional: Change font */                        
                    }                                
                """) as demo:   
    
    gr.Markdown("# Image Captioner : SmolVLM-Instruct")
    gr.Markdown("Upload an image and adjust the settings to generate a caption")
    
    with gr.Row():
        with gr.Column(elem_classes=["fixed-width-column"]):
            input_image = gr.Image(type="pil", label="Input Image", height=512)
                        
            caption_style = gr.Dropdown(
                choices=["Short and concise", "Brief but detailed", "Moderately detailed", "Highly detailed"],
                value="Brief but detailed",
                label="Caption Style"
            )
            
            submit_btn = gr.Button("Generate Caption", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                max_tokens = gr.Slider(minimum=50, maximum=500, value=128, step=1, label="Max New Tokens")
                do_sample_checkbox = gr.Checkbox(value=True, label="Do Sample")
                temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top P")
                rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, label="Repetition Penalty")

                gr.Markdown("""    
                            ### Parameters:
                            - **Max New Tokens**: Controls the maximum length of the generated caption
                            - **Do Sample**: When enabled, uses sampling for more diverse outputs
                            - **Temperature**: Higher values (>1.0) = output more random, lower values = more deterministic
                            - **Top P**: Controls diversity via nucleus sampling
                            - **Repetition Penalty**: Higher values discourage repetition in the text
                            """)

            
                    
        with gr.Column(elem_classes=["fixed-width-column"]):
            output_text = gr.Textbox(label="Generated Caption", lines=5)
            # Add the new text box here
            additional_text_box = gr.Textbox(label="Caption (Editable)", lines=4, interactive=True, elem_id="additional_text_box", info="you can edit the caption here before proceeding")
    
            # Add the Process button under the second column
            process_btn = gr.Button("Process", variant="primary")

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
        outputs=[output_text, additional_text_box]
    )

    # Add the click handler for the Process button
    process_btn.click(
        fn=process_caption,
        inputs=[additional_text_box]
    )    


# Launch the Gradio app
if __name__ == "__main__":
    print(f"Model loaded on {DEVICE}")
    demo.launch()