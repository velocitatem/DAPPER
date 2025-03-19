import zipfile
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# get from args
import sys
zip_file_path = sys.argv[1]

widths, heights = [], []

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for img_name in image_files:
        with zip_ref.open(img_name) as img_file:
            try:
                img = Image.open(BytesIO(img_file.read()))
                widths.append(img.width)
                heights.append(img.height)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")

# Convert to numpy arrays for analysis
widths, heights = np.array(widths), np.array(heights)

# Print statistics
print(f"Total images analyzed: {len(widths)}")
print(f"Width: min={widths.min()}, max={widths.max()}, mean={widths.mean():.2f}, median={np.median(widths)}")
print(f"Height: min={heights.min()}, max={heights.max()}, mean={heights.mean():.2f}, median={np.median(heights)}")

# Plot distribution for visual insight
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(widths, bins=20, color='skyblue')
plt.title('Width Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Number of Images')

plt.subplot(1, 2, 2)
plt.hist(heights, bins=20, color='salmon')
plt.title('Height Distribution')
plt.xlabel('Height (pixels)')
plt.ylabel('Number of Images')

plt.tight_layout()
plt.show()

# Suggest CNN input size (close to median or rounded mean)
suggested_width = int(np.median(widths) // 32 * 32)
suggested_height = int(np.median(heights) // 32 * 32)

print(f"Suggested CNN input dimension: {suggested_width}x{suggested_height}")
