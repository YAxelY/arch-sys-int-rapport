import json
import base64
import os

notebook_path = '/home/axel/Downloads/udsbooks/M2/semester1/Architecture/arch-sys-int-rapport/diabetes_pipeline_final(2)(1).ipynb'
output_dir = '/home/axel/Downloads/udsbooks/M2/semester1/Architecture/arch-sys-int-rapport/Chapters/images/extracted'

os.makedirs(output_dir, exist_ok=True)

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

image_count = 0
for cell_idx, cell in enumerate(notebook.get('cells', [])):
    if 'outputs' in cell:
        for output_idx, output in enumerate(cell['outputs']):
            if 'data' in output and 'image/png' in output['data']:
                image_data = output['data']['image/png']
                image_bytes = base64.b64decode(image_data)
                
                # Try to get a meaningful name if possible or just use indices
                filename = f"extracted_image_{cell_idx}_{output_idx}.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as img_file:
                    img_file.write(image_bytes)
                image_count += 1
                
print(f"Extracted {image_count} images to {output_dir}")
