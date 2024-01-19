<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="./media/img/kisz_logo.png" width="192" height="69"> 
<h1> Working with embeddings:
<h2>An introductory workshop with applications on Semantic Search
</div>

## Instructions for Environment and Tool Setup <!-- omit in toc -->

**Table of Contents**

- [1. Getting the code](#1-getting-the-code)
- [2. Tool for Python environments](#2-tool-for-python-environments)
- [3. Setting up the environment](#3-setting-up-the-environment)
- [4. Preparing data and models](#4-preparing-data-and-models)
- [5. Finally…](#5-finally)

### 1. Getting the code

For retrieving the code that we will use in the workshop you have two options:

- **without using Git**
  
  > 1. Go to the [GitHub repository page](https://github.com/KISZ-BB/kisz-nlp-embeddings)
  > 2. Download the entire repository from the GitHub web interface as a ZIP file to your computer. You can find the Download ZIP option when you click on the <kbd><> Code</kbd> drop menu.
    > 3. Once the archive is downloaded, extract its contents to your desired directory. 

- **using Git**
  
  > 1. Open your Terminal or Command Prompt. On Windows, you can use Command Prompt or PowerShell. On macOS and Linux, you can use the Terminal
  >
  > 2. Navigate to the directory where you want to clone the repository
  >
  > 3. Clone the repository with the command:
  >
  > &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; `git clone https://github.com/KISZ-BB/kisz-nlp-embeddings`

And that’s it! You now have the code for the workshop on your local machine.

> **Side Note**: If you have problems with any of the following steps, you can open an issue in this repository and tell us about it.

### 2. Tool for Python environments

For avoiding reproducibility and package dependency problems, you will be using a custom Python environment (created with Miniconda) during the workshop. You can find the requirements for the environment in the <kbd>environment.yml</kbd> file in the <kbd>envs</kbd> folder.

We recommend you to set up the environment with Miniconda for several reasons:

1. Conda provides a compact and efficient way to create and manage Python environments.
2. Miniconda provides a minimal installation compared with Anaconda, where the most common packages for data science are automatically installed.
3. Even though installation with venv and pip is also possible, some packages like *faiss* can only be installed with conda.

If you don't have miniconda, you can find the installation instructions [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

### 3. Setting up the environment

Once you have miniconda installed, you can open the Miniconda shell. You can usually find it under the name **Anaconda prompt (miniconda)**. Alternatively, you can also use the Miniconda PowerShell. You should see the tag <kbd>(base)</kbd> leading your prompt.

> **Warning**: The size of the environment could easily reach 10 or 11 Gbs in your hard disc drive, and we will also need space for downloading some additional files, so please make sure you have enough space.

We have prepared everything into a Makefile. To create the conda environment and activate it, try to execute the following command in the folder where you put the code:

> **<kbd>make</kbd>**

If everything went ok, you should see now the tag <kbd>(.embed_env)</kbd>. If it did not work, we will have to do it manually. Follow this steps:

> 1. Check that there is no subfolder in <kbd>envs</kbd>. If you find any subfolder, please delete it
>
> 2. Create the environment with the required packages using the command
>
> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; <kbd>conda env create -f ./envs/environment.yml -p ./envs/.embed_env</kbd>
> 
> 3. Activate the environment with the command
>
> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; <kbd>conda activate ./envs/.embed_env</kbd>
>
> 4. Check that you see the tag <kbd>(.embed_env)</kbd> in your prompt

As sanity check, if you want to be sure that all packages were properly installed in the right versions you can run the command

> <kbd>pytest -m env -v -p no:warnings</kbd>

### 4. Preparing data and models
We will be using some data, models and tools that need to be downloaded before the workshop. Please be aware that the execution of the following instruction can take some minutes.  

We assume you still have the Miniconda Prompt open, so just run the following command

> <kbd>python ./src/setup.py</kbd>

This will install the basic data and models.

> **Side note**: You might be interested in installing other models, tools or datasets that are mentioned in the workshop. In this case, feel free to edit the <kbd>setup.py</kbd> file in the <kbd>src</kbd> folder and uncomment the lines corresponding to the models you want to preload, save the file and run the command above again.

### 5. Finally…
Congratulations! Everything is prepared and you are set up for the workshop.

If something went wrong or did not run as expected, if you found bugs or incomplete documentation, or even if you want to propose new features, topics or ways of improving the workshop or the code, feel free to open an issue in the GitHub repository and we will get back to you as soon as possible.

Thank you for your attention and collaboration, and we hope you enjoy the workshop.