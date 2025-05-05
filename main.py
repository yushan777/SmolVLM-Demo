import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import os
import time

# Enable MPS fallback to CPU for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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

# Load image
image1 = load_image("input/woman-cafe.jpg")

# Initialize processor and model with proper dtype
processor = AutoProcessor.from_pretrained("model/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "model/SmolVLM-Instruct",
    torch_dtype=torch.float16,  # Keep half precision for efficiency
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "describe this image keep the description concise."}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs with optimized parameters for Apple Silicon
generated_ids = model.generate(
    **inputs, 
    max_new_tokens=77,
    do_sample=True,
    temperature=0.7,       # Balance between creativity and determinism
    top_p=0.9,             # Control diversity
    repetition_penalty=1.1, # Discourage repetition
)

generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(">>>>>>>>>>>")
print(generated_texts[0])

end_time = time.time()
execution_time = end_time - start_time
print(f"Function executed in {execution_time:.4f} seconds")