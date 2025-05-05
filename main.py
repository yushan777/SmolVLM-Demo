import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load image
image1 = load_image("input/woman-cafe.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("model/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "model/SmolVLM-Instruct",
    torch_dtype=torch.float16,
    _attn_implementation="sdpa" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe the image?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
# generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_ids = model.generate(
    **inputs, 
    max_new_tokens=500,
    do_sample=True, # required for temp and top_p
    temperature=0.7,       # Adjust for creativity vs determinism
    top_p=0.9,             # Adjust for diversity
    repetition_penalty=1.1 # Discourage repetition
)


generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(">>>>>>>>>>>")
print(generated_texts[0])

"""
## Generation Parameters for SmolVLM

### 1. max_new_tokens
This parameter controls the maximum number of tokens the model will generate in response, regardless of the input prompt length. For example, if you set this to 500 as in your code, the model will generate up to 500 tokens in its response.

### 2. temperature
Temperature regulates the randomness of the generated text. Lower values (closer to 0) make responses more deterministic and predictable, while higher values increase creativity and variability. 

- **Low temperature (0.1-0.3)**: More focused, factual, and deterministic responses
- **Medium temperature (0.5-0.7)**: Balanced between predictability and creativity
- **High temperature (0.8-1.5)**: More creative, diverse, and sometimes unexpected outputs

### 3. top_p (nucleus sampling)
This parameter sets a threshold probability for token inclusion during generation. A lower value produces more factual and precise responses, while higher values enable more diverse outputs.

- **Low top_p (0.1-0.3)**: More deterministic and focused output
- **Medium top_p (0.5-0.7)**: Moderate diversity
- **High top_p (0.8-0.9)**: More creative and varied responses

### 4. top_k
This integer controls the number of top tokens to consider during generation. Set to -1 to consider all tokens. For example, if top_k=50, the model only samples from the 50 most likely next tokens.

### 5. repetition_penalty
A float value that penalizes new tokens based on whether they appear in the prompt and generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage repetition.

### 6. presence_penalty
Controls how much the model avoids repeating the same tokens, regardless of how many times they've appeared. This encourages more diverse vocabulary usage.

### Additional Parameters

- **seed**: When fixed to a specific value, the model attempts to provide the same response for repeated requests, though deterministic output isn't guaranteed.
- **min_tokens**: Minimum number of tokens to generate before allowing the model to stop
- **stop_token_ids**: List of token IDs that will stop the generation when produced

## Best Practices

1. It's generally recommended to adjust either temperature OR top_p, but not both simultaneously, to avoid conflicting behaviors.

2. For factual, consistent outputs (like image descriptions):
   - Use lower temperature (0.1-0.3)
   - Use lower top_p (0.1-0.5)
   - Higher repetition_penalty (1.1-1.5)

3. For creative outputs:
   - Use higher temperature (0.7-1.0)
   - Use higher top_p (0.8-0.95)
   - Moderate repetition_penalty

4. Testing small adjustments is key: a temperature of 0.5 might work for factual Q&A, while 1.0 could suit creative storytelling.

You can add these parameters to your code in the `model.generate()` call:

```python
generated_ids = model.generate(
    **inputs, 
    max_new_tokens=500,
    temperature=0.7,       # Adjust for creativity vs determinism
    top_p=0.9,             # Adjust for diversity
    repetition_penalty=1.1 # Discourage repetition
)
```
"""
