import h5py
import numpy as np
import matplotlib.pyplot as plt

# ✅ Use raw string for Windows path
file_path = r"C:\Users\Sravani\Downloads\archive\BraTS2020_training_data\content\data\volume_1_slice_75.h5"

print("Opening:", file_path)

with h5py.File(file_path, "r") as f:
    print("Keys inside file:", list(f.keys()))
    
    for key in f.keys():
        print(f"\nDataset: {key}")
        print("Shape:", f[key].shape)
        print("Data type:", f[key].dtype)

    # Try common keys
    image = None
    mask = None

    for key in f.keys():
        if "image" in key.lower():
            image = f[key][:]
        if "mask" in key.lower() or "label" in key.lower():
            mask = f[key][:]

if image is None or mask is None:
    print("⚠ Could not automatically detect image/mask keys.")
    exit()

print("\nImage shape:", image.shape)
print("Mask shape:", mask.shape)
print("Unique mask values:", np.unique(mask))

# Visualize
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
if image.ndim == 3:
    plt.imshow(image[:,:,0], cmap="gray")
else:
    plt.imshow(image, cmap="gray")
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="jet")
plt.title("Mask")

plt.show()