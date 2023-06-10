A Python code for generating polyhedral uncertainty sets for energy problems.
The code is associated with the paper [A data-driven uncertainty modelling and reduction approach for energy optimisation problems](https://arxiv.org/abs/2212.01478).

This folder is composed of:
- the notebook [uncertainty_modeling.ipynb](https://github.com/julien-vaes/uncertainty_modelling_for_energy/blob/master/uncertainty_modeling.ipynb),
which demonstrate how polyhedral uncertainty set can be computed and obtained with a linear inequality.
- the pdf [Implementation: Uncertainty modelling via polyhedral uncertainty sets](https://github.com/julien-vaes/uncertainty_modelling_for_energy/blob/master/pus_implementation.pdf),
which explains step by step how what the notebook [uncertainty_modeling.ipynb](https://github.com/julien-vaes/uncertainty_modelling_for_energy/blob/master/uncertainty_modeling.ipynb) does.
- the notebook [reproducibility_paper_escape.ipynb](https://github.com/julien-vaes/uncertainty_modelling_for_energy/blob/master/reproducibility_paper_escape.ipynb),
which reproduces all the plots in the paper [A data-driven uncertainty modelling and reduction approach for energy optimisation problems](https://arxiv.org/abs/2212.01478).
- the 'txt' file [requirements.txt](https://github.com/julien-vaes/uncertainty_modelling_for_energy/blob/master/requirements.txt),
which describes the environments needed to run the scripts, i.e.
  
  1. Clone or download the project repository that contains the `requirements.txt` file.
  
  2. Open a terminal or command prompt and navigate to the project directory.
  
  3. *(Optional)* It is recommended to create a new virtual environment to isolate this project's dependencies from your system-wide Python installation. You can create a virtual environment using the following command:
  
     `python3 -m venv myenv`
  
     This will create a new directory called `myenv` (you can choose a different name if you prefer) that contains the necessary files for the virtual environment.
  
  4. Activate the virtual environment:
  - In the terminal, run the appropriate command based on your operating system:
    - For Windows:
      ```
      myenv\Scripts\activate
      ```
    - For macOS/Linux:
      ```
      source myenv/bin/activate
      ```
  - You will see `(myenv)` prefix in your terminal, indicating that the virtual environment is active.
  
  5. Install the project dependencies using the `requirements.txt` file:
  
     `pip install -r requirements.txt`
  
     This command will read the `requirements.txt` file and install all the necessary packages and their versions into the current virtual environment.
  6. Before running ```jupyter```, make sure that it corresponds to the one installed in the environment myenv by running:
  - For Windows:
  ```
  where jupyter
  ```
  - For macOS/Linux:
  ```
  which jupyter
  ```
  If the location is still your base environment, activate again the environment (and then verify that the jupyter is now install in myenv):
  - For Windows:
    ```
    myenv\Scripts\activate
    ```
  - For macOS/Linux:
    ```
    source myenv/bin/activate
    ```
  
  7. Once the installation is complete, you can run your code within the activated virtual environment, and it will use the installed packages and versions specified in the `requirements.txt` file.
  
  By following these steps, you will have recreated the environment based on the `requirements.txt` file, ensuring that all the required packages and versions are installed and ready to use.
  Launch jupyter with:
  ```
  jupyter notebook
  ```
