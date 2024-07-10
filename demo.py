import os
import sys
import time
import glob
import warnings
import subprocess
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from IPython.display import clear_output


class dif3D_Viton():
    def __init__(self, model_images_path, cloth_images_path):
        self.model_images_path = model_images_path
        self.cloth_images_path = cloth_images_path
        warnings.filterwarnings('ignore')
        tqdm.pandas()

    def viton_sample_generator(self):
        image = ["00017_00.jpg", "13891_00.jpg",
                 "02372_00.jpg", "00055_00.jpg",
                 "00057_00.jpg", "00084_00.jpg",
                 "00190_00.jpg", "00627_00.jpg",
                 "00884_00.jpg", "00782_00.jpg",
                 "02007_00.jpg", "02299_00.jpg",
                 "02273_00.jpg", "07874_00.jpg",
                 "02254_00.jpg", "05338_00.jpg",
                 "10404_00.jpg", "05997_00.jpg",
                 "08340_00.jpg", "02660_00.jpg",
                 "08321_00.jpg", "08079_00.jpg",
                 "03797_00.jpg", "03802_00.jpg",
                 "04096_00.jpg"]  # , "00017_00.jpg"]  # to be completed

        clothes = ["04131_00.jpg", "06778_00.jpg",
                   "07874_00.jpg", "02372_00.jpg",
                   "00828_00.jpg", "05338_00.jpg",
                   "04700_00.jpg", "00828_00.jpg",
                   "00736_00.jpg", "02007_00.jpg",
                   "05956_00.jpg", "02007_00.jpg",
                   "12365_00.jpg", "03797_00.jpg",
                   "05876_00.jpg", "08348_00.jpg",
                   "01248_00.jpg", "05830_00.jpg",
                   "08512_00.jpg", "13004_00.jpg",
                   "08079_00.jpg", "08321_00.jpg",
                   "03745_00.jpg", "04096_00.jpg",
                   "03922_00.jpg"]  # , "02660_00.jpg"]  # to be completed

        df = pd.DataFrame({"image": image, "clothes": clothes})
        df.to_csv(
            "HR-VITON/data/pairs1.txt",
            index=False, header=False, sep=" ")

        print(f"list of images and clothes:\n {df}\n")
        self.HR_viton_run()
        clear_output()
        for idx in tqdm(range(len(image)), total=len(image),
                        desc="Processing images"):
            self.img_ori = plt.imread(
                f"./data/test/image/{image[idx]}")
            self.img_clo = plt.imread(
                f"./data/test/cloth/{clothes[idx]}")
            img_new = f"./data/output/{image[idx][:-4]}_{clothes[idx][:-4]}.png"
            self.Viton_res(out_path=img_new)
        _ = input()
        sys.exit(0)

    def file_number(path):
        num_files = len(glob.glob(path + '/*'))
        print(f"There are {num_files} files in the directory.")

    def image_showcase(self, path, pg):
        # Get a list of all image files in the directory
        img_files = [f for f in os.listdir(path) if f.endswith(('.jpg'))]

        # Calculate the number of rows and columns for a 10x10 grid
        num_rows = 10
        num_cols = 10

        # Create a figure and axis object with
        # the calculated grid configuration
        fig, ax = plt.subplots(
            nrows=num_rows, ncols=num_cols, figsize=(15, 15))

        end = 100*pg
        start = end-100
        # Iterate over the image files and display them in the grid
        for i, img_file in enumerate(img_files[start:end]):
            img_path = os.path.join(path, img_file)
            img = plt.imread(img_path)
            ax[i // num_cols, i % num_cols].imshow(img)
            ax[i // num_cols, i % num_cols].set_title(img_file)
            ax[i // num_cols, i % num_cols].axis('off')
        # Show the plot
        plt.show()

    def M_C_selector(self):
        clear_output()
        print("you have selected, these images: \n")
        # show images
        self.img_ori = plt.imread(
            f"HR-VITON/data/test/image/{self.m_selector}")
        self.img_clo = plt.imread(
            f"HR-VITON/data/test/cloth/{self.c_selector}")
        # Create a figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Display the images
        ax[0].imshow(self.img_ori)
        ax[1].imshow(self.img_clo)

        # Remove axis labels and titles
        ax[0].axis('off')
        ax[1].axis('off')

        # Show the plot
        plt.show()

        # setting images to be proceed
        images = list()
        clothes = list()
        images.append(self.m_selector)
        clothes.append(self.c_selector)
        df = pd.DataFrame({"image": images, "clothes": clothes})
        df.to_csv(
            "HR-VITON/data/pairs1.txt", index=False, header=False, sep=" ")
        print(f"selected set is \n{df}")

    def menu(self):
        self.m_selector = self.selector(self.model_images_path)
        self.c_selector = self.selector(self.cloth_images_path)
        self.M_C_selector()
        print("starting HR_Viton...")
        self.HR_viton_run()
        output = f"./data/output/{self.m_selector[:-4]}_{self.c_selector[:-4]}.png"
        self.Viton_res(output)

    def selector(self, path):
        clear_output()
        i = 1
        selector = None
        while ((selector is None) and (1 <= i <= 20)):
            print(f"""based on the following you can select a model image from the below images and from 1-20 pages:
                ("n"or"N" for next page, "p"or"P" for previous page, "s"or "S" to select a model/cloth)
                ("x" or "X" to exit and "g" or "G" for to create viton sample)
                {i} of 20 pages""")
            self.image_showcase(path, pg=i)
            selector = input("\n>>>")
            if selector.capitalize() == "N":
                i += 1
                selector = None
            elif selector.capitalize() == "P":
                i -= 1
                selector = None
            elif selector.capitalize() == "G":
                clear_output()
                self.viton_sample_generator()  
                # generates samples onlyfor viton
            elif selector.capitalize() == "X":
                sys.exit()  # Exit
            elif selector.capitalize() == "S":
                selector = input("\n>>>")
                if selector.lower().endswith(('.jpg', '.jpeg')):
                    return selector
                else:
                    print("please enter a valid image file name")
                    selector = None
            else:
                clear_output()
                time.sleep(3)  # pause for 5 seconds
                selector = None
                print("invalid input!!!")
            clear_output()
        else:
            pass
            # ************************************

    def HR_viton_run(self):

        # Define the command and arguments
        cmd = ["python3", "test_generator.py"]
        args = [
            "--occlusion",
            "--cuda", "True",
            "--gpu_ids", "0",
            "--dataroot", "./data/",
            "--data_list", "pairs1.txt",
            "--output_dir", "./data/output/"
        ]

        # Change the working directory to ./HR-VITON
        os.chdir("./HR-VITON")

        # Run the command with arguments
        subprocess.run(cmd + args)

    def Viton_res(self, out_path):
        img_new = plt.imread(out_path, 0)
        plt.figure(figsize=(10, 8))
        grid = plt.GridSpec(2, 3, wspace=0, hspace=0.2)

        plt.subplot(grid[0, 0])
        plt.imshow(self.img_ori)
        plt.axis("off")
        plt.title("Original Image")

        plt.subplot(grid[1, 0])
        plt.imshow(self.img_clo)
        plt.axis("off")
        plt.title("Cloth Image")

        plt.subplot(grid[:, 1:])
        plt.imshow(img_new)
        plt.axis("off")
        plt.title("Generated Image")

        plt.show()
