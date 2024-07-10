import os
import shutil

"""
data only test : https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset/code

checkpoint : https://www.kaggle.com/datasets/marquis03/hr-viton
"""




# Install the torchgeometry library
os.system("pip install torchgeometry")

# Clone the HR-VITON repository from GitHub
os.system("git clone https://github.com/sangyun884/HR-VITON.git")

# Move pretrained weights from the input directory to the output directory in Kaggle
os.makedirs("./HR-VITON/eval_models/weights/v0.1",
            exist_ok=True)
shutil.copytree(
    "/kaggle/input/hr-viton/", "./HR-VITON/eval_models/weights/v0.1/")

# Move test data from the input directory to the working directory in Kaggle
os.makedirs("/kaggle/working/data/test", exist_ok=True)
shutil.copytree("/kaggle/input/high-resolution-viton-zalando-dataset/test/",
                "/kaggle/working/data/test/")
