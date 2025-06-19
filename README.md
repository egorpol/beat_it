# Beat and Tempo Fluctuation Analysis with "beat_it" Toolbox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Disclaimer:** This project is currently in a very early alpha stage of development. A first alpha version is planned for release later this year.

This repository supports the workshop "**How Constant Is Your Beat? Computer-Assisted Analysis of Beat and Tempo Fluctuations from Acousmatic Music to Minimal Techno with 'beat_it' Toolbox**," presented at the [Rhythm under the Microscope](https://www.ipop.at/rhythm/) conference (University of Music and Performing Arts Vienna, 25-27 September 2024). It also accompanies the presentation "**Understanding and Emulating Time: Analyzing and Simulating Musical Microrhythm Timing with the beat_it Toolbox**" at the [inmusic25](https://www.inmusicconference.com/) conference.

The repository provides Jupyter notebooks that demonstrate techniques for analyzing beat and tempo fluctuations using the **beat_it** Python toolbox. It also includes all the necessary audio examples and annotations.

## Repository Structure

-   `**/examples/`**: Contains the audio example files for analysis.
-   `**/csv/`**: Includes corresponding cue points and annotations in CSV format.
-   `**/sv/`**: Provides annotation layers for import into Sonic Visualizer.
-   `**/sv_project_files/`**: Contains complete Sonic Visualizer project files (`.sv`).
    -   *Note:* Audio files may need to be manually relinked from the `/examples/` directory after opening a project.

## Getting Started

### Prerequisites

The code and notebooks in this repository are optimized for **Python 3.11**. We recommend using a virtual environment to manage dependencies.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-link>
    cd <your-repository-name>
    ```

2.  **Set up the environment:**

    We suggest using [Conda](https://docs.conda.io/en/latest/) to create and manage the environment.

    ```bash
    # Create and activate a new conda environment
    conda create -n beat_it python=3.11
    conda activate beat_it
    ```

3.  **Install dependencies:**

    Install the required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Optional Pitch Detection Dependencies:**

    To use the pitch detection notebooks (e.g., `parm_pitch.ipynb`), install these additional libraries:
    ```bash
    pip install crepe parselmouth tensorflow
    ```

## Usage

Once you have set up the environment and installed the required packages, you can run the Jupyter notebooks to explore various analyses.

## Colab Notebooks

The Jupyter notebooks are also available for use in Google Colab. You can access and run the notebooks directly from the following Colab folder:

[Colab Folder for Notebooks](https://drive.google.com/drive/folders/1cMQOTImAuaW0he8JIG6GxjjL297TnY-a?usp=sharing)

## License

This project is distributed under the [MIT License](LICENSE).
