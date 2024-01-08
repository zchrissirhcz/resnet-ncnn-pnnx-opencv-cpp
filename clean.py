# remove files and directories according to .gitignore

import os
import shutil

required_files = [
    "resnet50.ncnn.param",
    "resnet50.ncnn.bin",

    "ResNet-50-model-sim.param",
    "ResNet-50-model-sim.bin",
]

# remove *.param in root directory
for f in os.listdir('.'):
    if f in required_files:
        continue
    if f.endswith('.param'):
        os.remove(f)

# remove *.bin in root directory
for f in os.listdir('.'):
    if f in required_files:
        continue
    if f.endswith('.bin'):
        os.remove(f)

# remove *.onnx in root directory
for f in os.listdir('.'):
    if f in required_files:
        continue
    if f.endswith('.onnx'):
        os.remove(f)

# remove *.pt in root directory
for f in os.listdir('.'):
    if f in required_files:
        continue
    if f.endswith('.pt'):
        os.remove(f)