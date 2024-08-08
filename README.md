# DIF3D_VITON: 3D Model Generation from Images

![](./Figures/Slide4.png)

Table of Contents:
- [Repository map](#repositorymap)
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Achivements and contributions](#achivements-and-contributions)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Methodology](#methodology)
- [Conclusion](#conclusion)

## Repositorymap
```
    DIF3D-VITON
    │
    ├── output/
    │   ├── 3dfiles   
    │   ├── images
    │   └── renderfiles
    │
    ├── viton/
    │   ├── Data   
    │   ├── eval
    │   └── * other files
    |
    ├── tsr/ [*for more information you can visit DIF3D @*]
    │    
    ├── .gitignore
    ├── README.md
    ├── LICENSE
    ├── dif3d_viton_ENV.yml
    ├── dif3d_viton.py
    └── demo.ipynb
```
## Introduction
The DIF3D-VITON system operates entirely within the digital realm, revolutionising the way humans interact with clothing. Beginning with a user-provided photograph a snapshot of the body. This image serves as the foundation for constructing a digital mannequin. The next step, selecting a clothing piece, dress, shirt, jacket, and more which will fold on the mannequin where size, fit, and style are all considered to create a three-dimensional representation of your body shape and proportions. With DIF3D-VITON, users can explore countless outfits virtually. Within simple clicks, followed by a couple of minutes of loading, the user can have a glimpse of their 3D representation wearing the selected clothing unit. 
In summary, DIF3D-VITON invites users to dress up with pixels, transforming people’s perception of fashion and enhancing the online shopping experience. The simplified pipeline which leads to 3D mesh creation using DIF3D-VITON is shown in Figure 3, down below. 


## Motivation
The motivation behind the DIF3D-VITON study stems from the growing demand for enhanced online shopping experiences in the fashion industry, where traditional methods of virtual try-on often fall short of providing realistic and personalized visualizations of clothing. As online retail continues to expand, consumers face challenges such as sizing inaccuracies and the inability to visualize how garments will fit and look on their unique body shapes. The study aims to address these issues by developing a novel 3D virtual try-on system that leverages deep learning and computer vision techniques to create high-resolution, detailed 3D models from just a single image of the user and a garment. This innovative approach not only seeks to improve user satisfaction and engagement but also aims to reduce return rates associated with online purchases, contributing to a more efficient and effective online shopping experience. By tackling the technical limitations and complexities of existing methods, the study aspires to revolutionize the way consumers interact with fashion in a digital environment.

## Achivements and contributions

#### Innovative Use of Diffusion Models:
DIF3D-VITON integrates diffusion models to enhance the accuracy of 3D garment fitting. This approach allows for the generation of high-fidelity 3D representations of clothing that align closely with the user's body shape and pose.

##### Addressing Common Challenges:
The system effectively tackles prevalent challenges in virtual try-on technology, such as misalignment and occlusion. By ensuring accurate garment fitting and realistic visualizations, DIF3D-VITON improves user satisfaction.

#### Realistic 3D Reconstruction:
The system achieves fully textured 3D meshes through refined shape field-based reconstruction. This method extracts shape features from 2D images, significantly improving the realism of the virtual try-on experience.

#### Enhanced Visual Quality:
The system demonstrates superior visual quality in virtual try-on results compared to existing methods. The use of advanced algorithms results in more realistic and appealing garment representations.


#### Contribution to Research and Development:
The research conducted in developing DIF3D-VITON contributes to the broader field of artificial intelligence and computer vision, providing insights and methodologies that can be applied to other areas of virtual reality and augmented reality.

#### Seamless User Experience:
By providing a streamlined process for users to visualize clothing on their digital avatars, DIF3D-VITON enhances the online shopping experience. Users can easily upload their images and see how different garments will look on them without the need for physical fitting.

#### Personalization in Fashion:
DIF3D-VITON allows for highly personalized shopping experiences by enabling users to visualize how specific clothing items will fit their unique body shapes and styles. This personalization can lead to better purchasing decisions.

In summary, DIF3D-VITON represents a significant advancement in 3D virtual try-on technology, contributing to improved user experiences, enhanced garment visualization, and potential benefits for the fashion retail industry. Its achievements highlight the effectiveness of integrating advanced AI techniques in practical applications.

## Setup
To be able to use the code firstly, run the following command :
```bash
git clone git@github.com:Mehdialmoo/DIF3D-VITON.git
```
To install the required packages, and set up the environment, you will need to have conda installed on your system.
Once you have conda installed, you can create a new environment with the following command:
```bash
conda env create -f dif3d_viton_ENV.yml
```
This will create a new conda environment named dif3d_viton_ENV with all the required dependencies.
Once the environment is set up, you can activate it with the following command:

```bash
conda activate dif3d_viton_ENV
```
You can then run the DIF3D-VITON project within this environment. now you need to download the required files from [Kaggle](https://www.kaggle.com/datasets/mehdialmousavie/dif3d-viton-files-data-checkpoints).After downloading the required files you need to unzip the files the copy the required files into the following address:

* ### ./viton/

by compeleting these steps, the system is now ready to use!

## Usage
To use the runtime environment, follow these steps:

create a python file or jupyter notebook file, first cell you only need to call one class from Dif3d_viton.py as show below:
```python
from dif3d_viton import dif3D_Viton
``` 
After this step, you need  to create an instance from Runtime class to be able to use the modules. as follows:
```python
test = dif3D_Viton()
```
### Configuration
now, we recommend to leave values as defult but if you have the intreset you can change the values for better textured mesh, for a better output we recommend enabling rendering, but this will add extra time towards resulting an output.

```
defualt vales are :
    render: bool = False
    chunk_size: int = 8192, #check your VRAM
    padding: int = 16, #for CNN model padding
    foreground_ratio: float = 0.85, 
    #for depth estimation of foreground and background
    mc_resolution: int = 256, #final mesh resloution
```
all of these variables are reacheable using the dif3d_viton_run method. like the code below:
```python
test.dif3d_viton_run()
```
```
test.dif3d_viton_run(render=True,  mc_resolution: int = 512)
```

also you cam use [demo.ipynb](./demo.ipynb) file that is already contained in the repository.
after setting the variables, you can run the cell/ python file. after running the code there should appear a menu of models and a menu of garments follow the instructions below for reaching out the output.

until now if you have followed you will notice the program created a out put folder with three sub folders of different outputs what will be produced during the work.
## Runtime


## Results
Runing the code completley will show case models and garments for selection, after selecting and the system goes through preprocessing, VTO deforming and fitting then for 3D generation, finally results and saves the final results in a folder named output as defualt (but you can change that also)The runtime environment generates the following output:

+ ####  Processed images in the output/images/ directory.
+ #### Rendered images in the output/renderfiles/ directory.
+ #### A video of the rendered images in the output/renderfiles/ directory.
+ #### A 3D mesh in the output/3dfiles/ directory.

now let us look into different experiments that have been done:
![](./Figures/Slide3.png)
here are the examples as  rendered videos, created with enabling render option:

https://github.com/user-attachments/assets/9722abb3-683b-4eb0-93b6-bef0ae492f3f

https://github.com/user-attachments/assets/040d020e-45ce-4510-8503-5c11533740d6  

https://github.com/user-attachments/assets/f720418b-c707-419f-bb33-193b3d9a0d64

## Methodology
The methodology of the DIF3D-VITON study is designed to create an advanced 3D virtual try-on (VTO) system that enhances the online shopping experience by providing realistic garment fitting. The approach is structured into several key components:

### Data Collection:
The study utilizes a sub-dataset from VITON-HD, which includes high-resolution images of various clothing items and corresponding user images. This dataset is crucial for training the model to recognize and accurately render different garments on diverse body types.
The dataset is pre-processed to ensure consistency in image quality and to facilitate effective training of the neural networks.

### Model Architecture:
The DIF3D-VITON system employs a novel architecture that integrates diffusion models for 3D reconstruction. This approach allows for the generation of detailed 3D meshes from 2D images.
The architecture consists of two main phases: the Image Generation Phase and the 3D Generation Phase. Each phase is further divided into sub-stages:
Image Generation Phase: This includes a try-on condition generator that deforms the clothing image to fit the user's body, producing a segmentation map that aligns the garment with the user's pose.
3D Generation Phase: This phase involves extracting depth information, generating point clouds, and ultimately creating a detailed 3D mesh of the garment on the user.

### Image Processing Techniques:
The methodology incorporates advanced image processing techniques, including human pose estimation to accurately align clothing with the user's body. This ensures that the garment fits realistically based on the user's posture.
The system also utilizes appearance flow methods to warp the clothing based on the original garment's texture and features, enhancing the realism of the virtual try-on experience.

### System
we conducted experiments on a personal computer running Windows 11. The hardware configuration included an Nvidia 4080 graphics card with 8GB VRAM and an Intel Gen 13th CPU with 16GB RAM
## Conclusion
DIF3D-VITON presents a groundbreaking advancement in 3D virtual try-on technology, integrating sophisticated deep learning algorithms with state-of-the-art computer vision techniques. This innovative system generates high-resolution, textured 3D models from a single image of both the user and the garment. 
Integrating diffusion models to correctly fit the selected garment with the user’s body, enables the system to effectively address prevalent challenges such as misalignment and occlusion in 3D results. Consequently, the virtual try-on process becomes seamless and realistic, paving the way for future innovations in the fashion and retail sectors.

DIF3D-VITON significantly optimises the online shopping experience by providing users with highly personalised and realistic previews of how garments will appear when worn. Furthermore, DIF3D-VITON holds the potential to revolutionise the fashion industry by optimising supply chain efficiency, enhancing personalised shopping journeys, and reducing return rates. By enabling precise visualisation of clothing fit and style, this technology improves the overall customer experience, making online shopping more engaging and satisfying. 
 
