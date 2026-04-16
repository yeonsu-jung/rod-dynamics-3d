import os
import glob
import re

base_dirs = [
    "/Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/6,7,8",
    "/Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/37,178,56",
    "/Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/919,461,568"
]

for base in base_dirs:
    if not os.path.exists(base):
        print(f"Base dir not found: {base}")
        continue
        
    print(f"Processing {base}...")
    # Find all x_relaxed.txt files
    files = glob.glob(os.path.join(base, "**/x_relaxed.txt"), recursive=True)
    for fpath in files:
        dir_path = os.path.dirname(fpath)
        log_path = os.path.join(dir_path, "log.txt")
        
        radius = None
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    if "rod radius:" in line:
                        try:
                            radius = float(line.split(":")[1].strip())
                            break
                        except:
                            pass
        
        if radius is None:
            # Fallback to directory name parsing if log.txt is missing or malformed
            dirname = os.path.basename(dir_path)
            # Try to find AR in dirname
            match = re.search(r'AR(\d+)', dirname)
            if match:
                ar = int(match.group(1))
                radius = 0.5 / ar
        
        if radius is not None:
            with open(fpath, 'r') as f:
                content = f.read()
            
            metadata = f"# rod_radius={radius}\n# rod_length=1.0\n"
            
            # Remove any existing commented lines from the top
            lines = content.splitlines()
            data_start = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith("#"):
                    data_start = i
                    break
            else:
                data_start = len(lines)
            
            data_content = "\n".join(lines[data_start:])
            
            print(f"Updating {fpath} with radius={radius}")
            with open(fpath, 'w') as f:
                f.write(metadata + data_content + "\n")
        else:
            print(f"Warning: Could not determine radius for {fpath}")
