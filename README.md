
# Beat and Tempo Fluctuation Analysis with "beat_it" Toolbox

This repository complements the workshop titled "**How Constant Is Your Beat? Computer-Assisted Analysis of Beat and Tempo Fluctuations from Acousmatic Music to Minimal Techno with 'beat_it' Toolbox**" presented at the **Rhythm under the Microscope** interdisciplinary conference on microrhythm and groove in popular music, held 25-27 September 2024, at the University of Music and Performing Arts Vienna.

The repository contains Jupyter notebooks that demonstrate techniques for analyzing beat and tempo fluctuations using the **beat_it** toolbox. It also includes all necessary audio example files and annotations.

## Repository Structure

- **/examples/**: Audio example files used in the analysis.
- **/csv/**: Corresponding cue points and annotations in CSV format.
- **/sv/**: Sonic Visualizer projects with annotations.

## Prerequisites

The code and notebooks in this repository are optimized for **Python 3.11**. We recommend using a virtual environment to avoid dependency issues.

### Setting Up the Environment

We suggest using [Conda](https://docs.conda.io/en/latest/) to create a new virtual environment for this project. After cloning the repository, follow these steps to set up the environment:

1. Create a new environment:

    ```bash
    conda create -n gfm_eallm python=3.11
    ```

2. Activate the environment:

    ```bash
    conda activate gfm_eallm
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Pitch Detection Dependencies (Optional)

If you want to explore the pitch detection notebooks (e.g., `parm_pitch.ipynb`), make sure to install the following additional dependencies:

```bash
pip install crepe parselmouth tensorflow
```

## Usage

Once you have set up the environment and installed the required packages, you can run the Jupyter notebooks to explore various analyses.

## Colab Notebooks

The Jupyter notebooks are also available for use in Google Colab. You can access and run the notebooks directly from the following Colab folder:

[Colab Folder for Notebooks](https://drive.google.com/drive/folders/1cMQOTImAuaW0he8JIG6GxjjL297TnY-a?usp=sharing)


### Running Jupyter Notebooks

Start Jupyter by running:

```bash
jupyter notebook
```

Then open the desired notebook from the root folder to begin your exploration.

## License

This project is distributed under the [MIT License](LICENSE).

## Acknowledgements

This repository is part of the research and workshop presented at the "Rhythm under the Microscope" interdisciplinary conference, held at the University of Music and Performing Arts Vienna, 2024.
