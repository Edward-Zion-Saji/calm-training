import re
import os

with open('/datadrive/edward/VAE_TRAINING_GUIDE.md', 'r') as f:
    content = f.read()

files_to_extract = [
    'configs/speech_vae.yaml',
    'data/speech_dataset.py',
    'models/discriminators.py',
    'models/wavlm_distill.py',
    'losses/vae_losses.py',
    'train_vae.py',
    'utils/normalise.py'
]

for fname in files_to_extract:
    ext = fname.split('.')[-1]
    lang = 'yaml' if ext == 'yaml' else 'python'
    pattern = rf"### {fname}\s*```(?:{lang})?\n(.*?)```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        code = match.group(1)
        full_path = os.path.join('/datadrive/edward', fname)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as out:
            out.write(code)
    else:
        print(f"Warning: Could not find block for {fname}")

vae_path = '/datadrive/edward/models/vae.py'
os.makedirs(os.path.dirname(vae_path), exist_ok=True)
with open(vae_path, 'w') as out:
    match1 = re.search(r"### models/vae.py\s*```python\n(.*?)```", content, re.DOTALL)
    if match1: out.write(match1.group(1) + "\n\n")
    else: print("Warning: Could not find first block for models/vae.py")
    
    match2 = re.search(r"### Bottleneck code[^\n]*\n\s*```python\n(.*?)```", content, re.DOTALL)
    if match2: out.write(match2.group(1) + "\n")
    else: print("Warning: Could not find bottleneck block for models/vae.py")

eval_path = '/datadrive/edward/evaluate.py'
match_eval = re.search(r"### Evaluation script\s*```python\n(.*?)```", content, re.DOTALL)
if match_eval:
    with open(eval_path, 'w') as out:
         out.write(match_eval.group(1) + "\n")
else: print("Warning: Could not find evaluate script")

for d in ['configs', 'data', 'models', 'losses', 'utils']:
    init_path = f"/datadrive/edward/{d}/__init__.py"
    if not os.path.exists(init_path):
        open(init_path, 'w').close()
        
print("Extraction complete.")
