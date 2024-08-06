# DIF3D_VITON: 3D Model Generation from Images

![](./Figures/Slide4)


Table of Contents:
- [Repository map](#repositorymap)
- [Introduction](#introduction)
- [Motivation](#motivation)
- [contribution](#contribution)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Methodology](#methodology)
    + [Dif3d](#dif3d)
    + [Utils](#utils)
    + [Gaussianutils](#gaussianutils)
    + [System](#system)
-[Conclusion](#conclusion)


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


## Motivation


## Contribution


## Setup
To be able to use the code firstly, run the following command :
```bash
git clone 
```
To install the required packages, and set up the environment, you will need to have conda installed on your system.
Once you have conda installed, you can create a new environment with the following command:
```bash
conda env create -f 
```
This will create a new conda environment named dif3d_ENV with all the required dependencies.
Once the environment is set up, you can activate it with the following command:

```bash
conda activate 
```
You can then run the dif3d project within this environment.

## Usage
To use the runtime environment, follow these steps:

create a python file or jupyter notebook file, first cell you only need to call one class from Dif3d.py as show below:
```python

``` 
After this step, you need  to create an instance from Runtime class to be able to use the modules. as follows:
```python

```
### Configuration
now, Set the input image path, we recommend to leave other values as defult but if you have the intreset you can change the addresses or fine tune the TSR and Gausssian models.
other variables are as follows:

```

```
all of these variables are reacheable using the set_variables method. like the code below:
```python

```
after setting the variables, you can initilize the models based on the variables as follows:
```python

```
now, you can run the main blocks of our work to start the image process as follows:
```python

```
until now if you have followed you will notice the program created a out put folder with three sub folders of different outputs what will be produced during the work. We will be discussing this in [Results]() section.
next you need to Initialize the pre-model and TSR model using the initilize method to be able to run and gen 3d result from mesh , point cloud and forfront depth estimations. using below code

```python

```
Render the 3D model using the render method to generate mising point cloud areas.
method.
```python

```
Finally to Export the 3D mesh in the specified format using the export_mesh :
```python

```
Here's an example usage:

```python
```
all the steps are already been coded in workspace.ipynb file as well.

## Results
Runing the code completley will show case models and garments for selection, after selecting and the system goes through preprocessing, VTO deforming and fitting then for 3D generation, finally results and saves the final results in a folder named output as defualt (but you can change that also)The runtime environment generates the following output:

+ ####  Processed images in the output/images/ directory.
+ #### Rendered images in the output/renderfiles/ directory.
+ #### A video of the rendered images in the output/renderfiles/ directory.
+ #### A 3D mesh in the output/3dfiles/ directory.

now let us look into different experiments that have been done:
![](./Figures/Slide3.png)
for more details here are the examples as  rendered videos:

## Methodology

### System

## Conclusion

