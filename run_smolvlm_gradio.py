import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
import gradio as gr
from colored_print import color, style
import os
import time

# Enable MPS fallback to CPU for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# MODEL PATHS ====================================

MODEL_PATH = "model/SmolVLM-Instruct"
# MODEL_PATH = "model/SmolVLM-500M-Instruct"
# MODEL_PATH = "model/SmolVLM-256M-Instruct"

# ================================================
# DEFAULT VALS
MAX_NEW_TOKENS = 512
REP_PENALTY = 1.2

# TOP P SAMPLING VALUES
DO_SAMPLING = False
TOP_P = 0.8
TEMP = 0.4


# Define caption style prompts
STYLE_PROMPTS = {
    "Brief and concise": "Caption this image with a brief and concise description.",
    "Moderately detailed": "Caption this image with a moderately detailed description.",
    "Highly detailed": "Caption this image with a highly detailed and comprehensive description."
}
# list of just the keys for gradio dropdown
CAPTION_STYLE_OPTIONS = list(STYLE_PROMPTS.keys())

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
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH,torch_dtype=torch.float16,_attn_implementation="sdpa").to(device)



    return processor, model, device

# ====================================================================
# Load model and processor at startup
processor, model, DEVICE = load_model()

print(f"Model {os.path.basename(MODEL_PATH)} loaded on {DEVICE}", color.ORANGE)

# ====================================================================
def generate_caption(
    image, 
    caption_style,
    max_new_tokens=MAX_NEW_TOKENS,
    repetition_penalty=REP_PENALTY,
    do_sample=DO_SAMPLING,
    temperature=TEMP,
    top_p=TOP_P
    
):

    # Check if image is provided, if not, quit and show msg
    if image is None:
        msg = "Please upload an image first to generate a caption."
        return msg, msg
    
        
    start_time = time.time()
        
    prompt_text = STYLE_PROMPTS.get(caption_style, "Caption this image.")
    
    # print(f"prompt_text = {prompt_text}", color.ORANGE)

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

    # Generate args
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
    }
    # only include temp and top p if do sample
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    generated_ids = model.generate(
        **inputs,
        **generation_kwargs
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
    
    print(f"execution_time = {execution_time:.2f} seconds.", color.BRIGHT_BLUE)
    
    return response_only

# ====================================================================
def process_edited_caption(additional_text):
    print(additional_text)

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
                    #text_box textarea {
                        /*  color: #2563eb !important;  text color */
                        font-family: 'monospace', monospace !important; 
                        font-size: 12px !important; 
                    }                                
                """) as demo:   
    
    gr.Markdown("# Image Captioner : SmolVLM-Instruct")
    gr.Markdown("Upload an image and adjust the settings to generate a caption")
    
    with gr.Row():
        # ================================================
        # COL 1
        with gr.Column(elem_classes=["fixed-width-column"]):
            input_image = gr.Image(type="pil", label="Input Image", height=512)
                                    
            submit_btn = gr.Button("Generate Caption", variant="primary")



            
        # ================================================
        # COL 2                    
        with gr.Column(elem_classes=["fixed-width-column"]):

            caption_style = gr.Dropdown(
                choices=CAPTION_STYLE_OPTIONS,
                value=CAPTION_STYLE_OPTIONS[1] if len(CAPTION_STYLE_OPTIONS) > 1 else CAPTION_STYLE_OPTIONS[0] if CAPTION_STYLE_OPTIONS else "Moderately detailed",
                label="Caption Style"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    max_tokens = gr.Slider(minimum=50, maximum=1024, value=MAX_NEW_TOKENS, step=1, label="Max New Tokens")
                    rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=REP_PENALTY, step=0.1, label="Repetition Penalty")

                # Group the sampling-related controls together
                with gr.Group():
                    do_sample_checkbox = gr.Checkbox(value=DO_SAMPLING, label="Do Sample")
                    with gr.Row():
                        temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, value=TEMP, step=0.1, label="Temperature")
                        top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=TOP_P, step=0.1, label="Top P")
                
                

                gr.Markdown("""    
                            ### Parameters:
                            - **Max New Tokens**: Controls the maximum length of the generated caption
                            - **Repetition Penalty**: Higher values discourage repetition in the text
                            - **Do Sample**: Enabled: uses Top P sampling for more diverse outputs. Disabled: use greedy mode (deterministic)
                            - **Temperature**: Higher values (>1.0) = output more random, lower values = more deterministic
                            - **Top P**: Higher values (0.8-0.95): More variability, more diverse outputs, Lower values (0.1-0.5): Less variability, more consistent outputs
                            
                            """)
                
    with gr.Row():
        with gr.Column():
            output_text = gr.Textbox(label="Generated Caption", lines=5, interactive=True, elem_id="text_box", info="you can edit the caption here before proceeding")
    
            # Add the Process button under the second column
            process_btn = gr.Button("Continue", variant="primary")

    submit_btn.click(
        fn=generate_caption,
        inputs=[
            input_image,
            caption_style,
            max_tokens,
            rep_penalty,
            do_sample_checkbox,
            temperature_slider,
            top_p_slider
            
        ],
        outputs=[output_text]
    )

    # Add the click handler for the Process button
    process_btn.click(
        fn=process_edited_caption,
        inputs=[output_text]
    )    


# Launch the Gradio app
if __name__ == "__main__":
    
    demo.launch()
