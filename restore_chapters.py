import re

def restore_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find where the old text starts. It's marked by "% COMMENTAIRES DE L'ANCIEN TEXTE"
    # or we can just look for the commented out chapters.
    
    start_idx = -1
    for i, line in enumerate(lines):
        if "% COMMENTAIRES DE L'ANCIEN TEXTE" in line:
            start_idx = i
            break
            
    if start_idx != -1:
        old_lines = lines[start_idx:]
        
        # Remove the commenting "%" from the beginning of these lines
        restored_lines = []
        for line in old_lines:
            if line.startswith("% \\iffalse") or line.startswith("% [Ancien texte ignoré]") or line.startswith("% \\fi") or line.startswith("% COMMENTAIRES"):
                continue
            if line.startswith("%"):
                restored_lines.append(line[1:])
            else:
                restored_lines.append(line)
        
        with open(filepath, 'w') as f:
            f.writelines(restored_lines)
        print(f"Restored {filepath}")
    else:
        print(f"Could not find old text marker in {filepath}")

restore_file('/home/axel/Downloads/udsbooks/M2/semester1/Architecture/arch-sys-int-rapport/Chapters/chapter5.tex')
restore_file('/home/axel/Downloads/udsbooks/M2/semester1/Architecture/arch-sys-int-rapport/Chapters/chapter6.tex')
