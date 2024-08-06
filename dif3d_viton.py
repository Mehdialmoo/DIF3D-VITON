# Import necessary libraries
import os
import sys
import time
import glob
import torch
import logging
import pathlib
import warnings
import subprocess
import pandas as pd
import matplotlib.pyplot as plt


# import functions
from tqdm import tqdm
from tsr.system import TSR

from tsr.utils import save_video
from tsr.gaussianutil import Gaussian
from IPython.display import clear_output


# Define a Timer class to measure execution time
class Timer:
    def __init__(self) -> None:
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        # Start timing a task
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        # End timing a task and log the execution time
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


class Runtime():
    def __init__(self):
        # Initialize the timer
        self.timer = Timer()
        # Initialize the pre-model
        self.premodel = Gaussian()
        # Initialize the TSR model (None for now)
        # based on pretrainde or to be trained
        self.model = None
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

    def set_variables(
            self,
            input_path="input/",
            output_path="output/",
            pretrained_model="stabilityai/TripoSR",
            chunk_size=8192,
            padding=16,
            foreground_ratio=0.85,
            mc_resolution=256,
            model_save_format="obj"
    ):
        # Set the input image path
        self.image_path = input_path
        # Set the output path
        self.out_path = output_path
        # Set the pre-trained model name
        self.pretrained = pretrained_model
        # Set the chunk size for rendering
        self.chunk_size = chunk_size
        # Set the resolution for mesh extraction
        self.mc_resolution = mc_resolution
        # Set the format for saving the 3D mesh
        self.format = model_save_format

        # Set variables for the pre-model
        self.premodel.set_variables(
            in_path=input_path,
            out_path=output_path,
            foreground_ratio=foreground_ratio,
            padding=padding)
        # Check and create the necessary output directories
        self.output_address_chk()
        # Check and set the processor either GPU (CUDA) or CPU
        self.processor_check()

    def output_address_chk(self):
        # creates directory for output results
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(f"{self.out_path}images/", exist_ok=True)
        os.makedirs(f"{self.out_path}renderfiles/", exist_ok=True)
        os.makedirs(f"{self.out_path}3dfiles/", exist_ok=True)

    def processor_check(self):
        # Set the device to use (GPU or CPU)
        # If a CUDA-compatible device is available,
        # use it; otherwise, use the CPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def initilize(self):
        # Initialize the model
        # This will load the pre-trained model and prepare it for use
        self.timer.start("Initializing pre-model(Gaussian model)")
        self.premodel.gassuin_load()
        self.timer.end("Initializing pre-model(Gaussian model)")
        self.timer.start("Initializing TSR (diffsusion 3D) model")
        self.model = TSR.from_pretrained(
            self.pretrained,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(self.chunk_size)
        self.model.to(self.device)
        self.timer.end("Initializing TSR (diffsusion 3D) model")

    def img_process(self):
        # Start a timer for the image processing step
        self.timer.start("Processing images")
        # Load and preprocess the input image using
        # the pre_process method of the premodel object
        self.image = self.premodel.pre_process()
        # Estimate the depth of the input image using the
        # depth_estimation method of the premodel object
        self.depth_image = self.premodel.depth_estimation()
        # Perform a depth prediction comparison visualization
        # using the dp_comparison_visual method of the premodel object
        self.premodel.dp_comparison_visual()
        # End the timer for the image processing step
        self.timer.end("Processing images")

    def modelRun(self):
        # Log a message to inform the user that
        # the process might take a few minutes
        logging.info("please wait this process might take a few minutes...")
        # Start a timer for the model running step
        self.timer.start("Running model")
        # Generate a point cloud
        self.premodel.pointcloud()
        # clean the pointcloud and create a forfront mesh
        # for next model hologan and zero1-to-3 to create mesh
        self.premodel.post_process()
        # Run the model on the input image with no gradient computation
        with torch.no_grad():
            self.scene_codes = self.model([self.image], device=self.device)
        # End the timer for the model running step
        self.timer.end("Running model")

    def render(self):
        # Start a timer to track the rendering time
        self.timer.start("Rendering")
        # Render images using the model, with 30 views, and return PIL images
        render_images = self.model.render(
            self.scene_codes, n_views=30, return_type="pil")
        # Iterate over the rendered images
        for ri, render_image in enumerate(render_images[0]):
            # Save each image to a file
            # with a numbered filename (e.g. render_001.png)
            render_image.save(os.path.join(
                f"{self.out_path}renderfiles/", f"render_{ri:03d}.png"))
            # Save a video using all the rendered images
            save_video(render_images[0], os.path.join(
                f"{self.out_path}renderfiles/", "render.mp4"), fps=30)
            # End the timer (this should be done after the loop, not inside it)
            self.timer.end("Rendering")

    def export_mesh(self):
        # Start a timer to track the mesh export time
        self.timer.start("Exporting mesh")
        # Extract the mesh using the model, with the specified resolution
        meshes = self.model.extract_mesh(
            self.scene_codes, resolution=self.mc_resolution)
        # Export the first mesh to a file with the specified format
        meshes[0].export(
            os.path.join(
                f"{self.out_path}3dfiles/",
                f"mesh.{self.format}"))
        # End the timer
        self.timer.end("Exporting mesh")


class dif3D_Viton():
    def __init__(self):
        self.root = pathlib.Path().resolve()
        os.chdir(self.root)
        self.VTN_pth = "./viton"
        self.DATA_pth = "./viton/data"
        warnings.filterwarnings('ignore')
        tqdm.pandas()

    def viton_sample_generator(self):
        image = [
            "00017_00.jpg", "00017_00.jpg",
            "13891_00.jpg", "02372_00.jpg",
            "00055_00.jpg", "00057_00.jpg",
            "00084_00.jpg", "00190_00.jpg",
            "00627_00.jpg", "00884_00.jpg",
            "00782_00.jpg", "02007_00.jpg",
            "02299_00.jpg", "02273_00.jpg",
            "07874_00.jpg", "02254_00.jpg",
            "05338_00.jpg", "10404_00.jpg",
            "05997_00.jpg", "08340_00.jpg",
            "02660_00.jpg", "08321_00.jpg",
            "08079_00.jpg", "03797_00.jpg",
            "03802_00.jpg", "04096_00.jpg",]

        clothes = clothes = [
            "04131_00.jpg", "02660_00.jpg",
            "06778_00.jpg", "07874_00.jpg",
            "02372_00.jpg", "00828_00.jpg",
            "05338_00.jpg", "04700_00.jpg",
            "00828_00.jpg", "00736_00.jpg",
            "02007_00.jpg", "05956_00.jpg",
            "02007_00.jpg", "12365_00.jpg",
            "03797_00.jpg", "05876_00.jpg",
            "08348_00.jpg", "01248_00.jpg",
            "05830_00.jpg", "08512_00.jpg",
            "13004_00.jpg", "08079_00.jpg",
            "08321_00.jpg", "03745_00.jpg",
            "04096_00.jpg", "03922_00.jpg"]

        df = pd.DataFrame({"image": image, "clothes": clothes})
        df.to_csv(
            f"{self.DATA_pth}/pairs.txt",
            index=False, header=False, sep=" ")

        print(f"list of images and clothes:\n {df}\n")
        print("starting HR_Viton...")
        self.HR_viton_run()
        print("shutingdown HR_Viton...")
        os.chdir(self.root)
        for idx in tqdm(range(len(image)), total=len(image),
                        desc="Processing images"):
            self.img_ori = plt.imread(
                f"{self.DATA_pth}/test/image/{image[idx]}")
            self.img_clo = plt.imread(
                f"{self.DATA_pth}/test/cloth/{clothes[idx]}")
            img_new = f"{self.DATA_pth}/output/{image[idx][:-4]}_{clothes[idx][:-4]}.png"
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
            nrows=num_rows, ncols=num_cols, figsize=(25, 25))
        # adjust the vertical spacing between subplots
        fig.subplots_adjust(hspace=0.5)

        end = 100*pg
        start = end-100
        # Iterate over the image files and display them in the grid
        for i, img_file in enumerate(img_files[start:end]):

            img_path = os.path.join(path, img_file)
            # Read the image from the file path
            img = plt.imread(img_path)

            # Calculate the row and column indices for the subplot
            row_idx = i // num_cols
            col_idx = i % num_cols

            # Display the image on the subplot
            ax[row_idx, col_idx].imshow(img, aspect='auto')

            # Turn off the axis labels and ticks
            ax[row_idx, col_idx].axis('off')

            # Add a text label below the image
            ax[row_idx, col_idx].text(
                0.5, -0.1, img_file, ha='center', va='top',
                transform=ax[row_idx, col_idx].transAxes)
        # Show the plot
        plt.show()

    def M_C_selector(self):
        clear_output()
        print("you have selected, these images:\n")
        # show images
        self.img_ori = plt.imread(
            f"{self.DATA_pth}/test/image/{self.m}")
        self.img_clo = plt.imread(
            f"{self.DATA_pth}/test/cloth/{self.c}")
        # Create a figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Display the images
        ax[0].imshow(self.img_ori)
        ax[1].imshow(self.img_clo)

        # Remove axis labels and titles
        ax[0].axis('off')
        ax[1].axis('off')

        ax[0].title.set_text('Selected Image')
        ax[1].title.set_text('Selected Coth')

        # Show the plot
        plt.show()

        # setting images to be proceed
        images = list()
        clothes = list()
        images.append(self.m)
        clothes.append(self.c)
        df = pd.DataFrame({"image": images, "clothes": clothes})
        df.to_csv(
            f"{self.DATA_pth}/pairs.txt",
            index=False, header=False, sep=" ")
        print(f"selected set is \n{df}")

    def menu(self):
        image_path = f"{self.DATA_pth}/test/image/"
        cloth_path = f"{self.DATA_pth}/test/cloth/"
        self.m = self.selector(image_path)
        self.c = self.selector(cloth_path)
        self.M_C_selector()
        print("starting HR_Viton...")
        self.HR_viton_run()
        print("shutingdown HR_Viton...")
        os.chdir(self.root)
        self.output = f"{self.DATA_pth}/output/{self.m[:-4]}_{self.c[:-4]}.png"
        self.Viton_res(self.output)

    def selector(self, path):
        clear_output()
        i = 1
        selector = None
        while True:
            while ((selector is None) and (1 <= i <= 20)):
                print(
                    f"""based on the following you can select a model image,
                    from the below images and from 1-20 pages:
                    ("n"or"N" for next page, "p"or"P" for previous page,
                    "s"or"S" to select a model/cloth)
                    ("x"or"X" to exit and "g"or"G" for to create viton sample)
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
                    print("invalid input!!!")
                    time.sleep(3)  # pause for 5 seconds
                    selector = None
                clear_output()
            else:
                i = 1
                selector = None

    def HR_viton_run(self):

        # Define the command and arguments
        cmd = ["python", "test_generator.py"]
        args = [
            "--occlusion",
            "--cuda", "True",
            "--gpu_ids", "0",
            "--dataroot", "./data/",
            "--data_list", "pairs.txt",
            "--output_dir", "./data/output/"
        ]

        # Change the working directory to ./HR-VITON
        os.chdir("./viton")

        # Run the command with arguments
        subprocess.run(cmd + args)

    def Viton_res(self, out_path):
        # ===================================================================
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
        # =======================================================================================================
        plt.imshow(img_new)
        plt.axis("off")
        plt.title("Generated Image")

        plt.show()

    def dif3d_viton_run(self,
                        render=False,
                        chunk_size=8192,
                        padding=16,
                        foreground_ratio=0.85,
                        mc_resolution=256):
        # runing the menu
        self.menu()
        # creating an instance from the runtime to enable us
        # to use the functions and models
        B3D_fusiion = Runtime()
        # setvariables is about giving attributes that
        # we need to fine tune or adjust the model
        # only input files address is mandetory
        B3D_fusiion.set_variables(
            input_path=self.output,
            chunk_size=chunk_size,
            padding=padding,
            foreground_ratio=foreground_ratio,
            mc_resolution=mc_resolution)

        # loading the models
        B3D_fusiion.initilize()

        # preprocessing the images
        # (resize/remove background/create depth map/ etc.)

        B3D_fusiion.img_process()

        # give the processed image and depth maps and cloud point and mesh
        B3D_fusiion.modelRun()

        # this step renders the hologan model to generate
        # the different angels based on Zero-1-to-3++ model
        if (render):
            B3D_fusiion.render()
        # saves a video and 30 frames from different angels

        # after rendering the model now can export a full mesh with uv map
        B3D_fusiion.export_mesh()
