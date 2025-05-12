source venv/bin/activate

python3 run_smolvlm.py \
--image 'input/woman-cafe.jpg' \
--prompt 'caption this image in detail' \
--model 'SmolVLM-Instruct' \
--stream 
