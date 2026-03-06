import json
import os

notebook_path = '/home/axel/Downloads/udsbooks/M2/semester1/Architecture/arch-sys-int-rapport/diabetes_pipeline_final(2)(1).ipynb'
output_path = '/home/axel/Downloads/udsbooks/M2/semester1/Architecture/arch-sys-int-rapport/notebook_content.md'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

with open(output_path, 'w', encoding='utf-8') as out:
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        if cell['cell_type'] == 'markdown':
            out.write(''.join(cell.get('source', [])))
            out.write('\n\n')
        elif cell['cell_type'] == 'code':
            out.write('```python\n')
            out.write(''.join(cell.get('source', [])))
            out.write('\n```\n')
            
            # Print output text if any
            for output in cell.get('outputs', []):
                if 'text' in output:
                    out.write('```output\n')
                    out.write(''.join(output['text']))
                    out.write('\n```\n')
                elif 'data' in output and 'text/plain' in output['data']:
                    out.write('```output\n')
                    out.write(''.join(output['data']['text/plain']))
                    out.write('\n```\n')
                if 'data' in output and 'image/png' in output['data']:
                    out.write(f"![Image {cell_idx}](Chapters/images/extracted/extracted_image_{cell_idx}_{cell['outputs'].index(output)}.png)\n")
            out.write('\n')

print(f"Notebook converted to {output_path}")
