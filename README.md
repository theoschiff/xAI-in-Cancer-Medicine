# CS433 Machine Learning Course Project - xAI in Cancer Medicine Competition

## About 

This project is part of the Kaggle competition [xAI in Cancer Medicine Competition](https://www.kaggle.com/competitions/xai-in-cancer-medicine/overview) hosted by Dr. Arvind Mer aimed at predicting drug responses using cancer genomic data. The objective is twofold: 

1. Develop accurate models for drug response prediction.  
2. Ensure the models provide clear, human-understandable explanations of the biological factors influencing drug responsiveness.

By combining accuracy and interpretability, this project contributes to advancing personalized medicine and improving cancer treatment strategies.

**Project advisor:** Prof. Dr. Jasmina Bogojeska

**Team members:** Jérémy Barghorn, Théo Schifferli & Jérémy Chaverot

## ✨🌟 Finished 2nd in the overall competition🥈, after IEEE SSCI 2025 presentation in Trondheim, Norway 🇳🇴🎉

### Project Structure

```
├── data
│   ├── additional
│   │   └── ...
│   ├── train_augmented.csv
│   |── train_targets_augmented.csv
│   ├── test.csv
│   ├── train.csv
│   ├── train_targets.csv
│   ├── x_test_partition.csv
│   └── y_test_partition.csv
│
├── shap_plots
│   └── All the plots for top 5, median 5 and bottom 5 predictions
│ 
├── explanations
│   └── All the textual interpretations made by the LLM 
|
├── models
│   └── Directory used to store the models  
│
├── src
│   ├── data_loader.py
│   ├── llm.py
│   ├── model.py
│   ├── train.py
│   ├── plots.py
│   └── submission.py
|
├── pdf
│   ├── explanations.pdf 
|   └── project2_description.pdf
|
├── WPFS
│   └── Modified project repository of the WPFS paper
│   
│
├── submissions
│   └── (All the submission files used for Kaggle scores)
│
├── report.pdf
├── README.md
├── requirements.txt
├── ElasticNet.ipynb
├── EncoderDecoder.ipynb
├── LinearRegression.ipynb
├── NeuralNet.ipynb
├── NeuralNetClasses.ipynb
├── SVR.ipynb
├── XGBoost.ipynb
├── data_augmentation.ipynb
├── svr_submission.ipynb
```

### Directory and File Descriptions

#### Directories
- **`data/`**: Stores all dataset files.
  - Original competition data files.
  - Augmented versions of training data and the not augmented test partition.
  - Data used for augmentation.
- **`shap_plots/`**: Contains SHAP plots for top, median, and bottom predictions.
- **`explanations/`**: Textual interpretations generated by the LLM.
- **`src/`**: Contains notebooks for model training and evaluation.
  - **`data_loader.py`**: Data loading and preprocessing functions, allows to load base version or augmented version of the data, scaling, splitting, dimensionality reduction, gaussian noise for robustness and computations of the metrics.
  - **`llm.py`**: LLM model implementation for textual explanations. Contains a Generator class allowing to initialize a model ([Qwen](https://huggingface.co/Qwen/Qwen2-7B-Instruct), [LLaMa](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) or [Gemma](https://huggingface.co/google/gemma-7b)) with easy-to-use functions to generate explanations. Alows control over system prompt, temperature, sampling and CoT.
  - **`model.py`**: Torch definitions of the models implemented. Contains architectures for the Encoder-Decoder, Neural Network and CNNs in different variations.
  - **`train.py`**: Training loops and evaluation functions for the models.
  - **`plots.py`**: Functions to generate SHAP plots for model evaluation.
  - **`submission.py`**: Functions to generate submission files.
- **`pdf/`**: Contains project description and best model output explanations report.
- **`submissions/`**: Folder for Kaggle competition submission files.
- **`WPFS/`**: Modified project repository of the WPFS paper [WPFS](https://github.com/andreimargeloiu/WPFS). Allows to run the implemented models of the paper on our custom dataset and generate predictions. The code was adapted in order to run with more recent versions o f pytorch. The models implemented and used are the WPFS, the [FSNET](https://ieeexplore.ieee.org/document/10191985) and the dietnetworks.
- **`ElasticNet.ipynb`**: Notebook showing an implementation of the ElasticNet model.
- **`EncoderDecoder.ipynb`**: Notebook showing an implementation of the Encoder-Decoder model. This implementation shows an example model for this architecture but a lot of other variations were tested and the code to load different variations of datasets, dimensionality reduction, training and evaluation are available in the `src` folder.
- **`LinearRegression.ipynb`**: Notebook showing an implementation of the Linear Regression model.
- **`NeuralNet.ipynb`**: Notebook showing an implementation of the Neural Network model.
- **`NeuralNetClasses.ipynb`**: Notebook showing an implementation of the Neural Network model.
- **`SVR.ipynb`**: Notebook showing an implementation of the Support Vector Regression model.
- **`XGBoost.ipynb`**: Notebook showing an implementation of the XGBoost model.
- **`data_augmentation.ipynb`**: Notebook showing the data augmentation process.
- **`svr_submission.ipynb`**: Notebook showing the generation of the submission file for the SVR model.

#### Key Files 
- **`report.pdf`**: The written summary of our ML project.
- **`README.md`**: Overview and instructions for the project.  
- **`requirements.txt`**: List of Python libraries required.  

## Quickstart 

Follow these steps to set up and run the project locally. We only support Windows installation at the moment.
The libraries used to run the project are listed in the `requirements.txt` file and can take some time to install due to their diversity (Transformers, PyTorch, Sklearn, Plotting libraries, Quantization, etc...). 

### 1. Clone the Repository

```bash
git clone git@github.com:CS-433/ml-project-2-theunderfitters.git
cd ml-project-2-theunderfitters

# Install the required libraries
pip install -r requirements.txt

# In order to see an example of an implementation of the models, you can check the notebooks in the root directory

# In order to run the WPFS models, you can check the WPFS folder
cd ./WPFS/
# For the WPFS model, on our custom augmented dataset, you can run the following command:
python src/main.py  --model 'wpfs' --dataset 'your_custom_dataset' --use_best_hyperparams  --experiment_name 'WPFS' --max_steps 200

# For the FSNET model, on our custom augmented dataset, you can run the following command:
python src/main.py  --model 'fsnet' --dataset 'your_custom_dataset' --use_best_hyperparams  --experiment_name 'FSNET' --max_steps 200

# For the dietnetworks model, on our custom augmented dataset, you can run the following command:
python src/main.py  --model 'dietnetworks' --dataset 'your_custom_dataset' --use_best_hyperparams  --experiment_name 'dietnetworks' --max_steps 200


```

<!-- ### 2. Set Up the Conda Environment

Install the environment depending on your operating system: _(**TODO:** à voir si les dépendances changent selon l'OS)_

- **For Windows**:
  ```bash
  conda env create -f environments/windows_env.yml
  ```

- **For macOS**:
  ```bash
  conda env create -f environments/mac_env.yml
  ``` -->

<!-- ### 3. Activate the Environment

Once the environment is created, activate it:

```bash
conda activate xAI-in-cancer-medicine
```

You’re all set! -->


## Results

| Model Description                               | Kaggle Spearman's <br> Rank Correlation $\sigma_k$ | Test Spearman's <br> Rank Correlation $\sigma_t$ |
|-------------------------------------------------|---------------------------------------------------|------------------------------------------------|
| Linear Regression                               | 0.26                                              | 0.32                                           |
| Elastic Net                                     | 0.41                                              | 0.39                                           |
| Neural Network                                  | 0.39                                              | **0.61**                                       |
| Neural Network (w/ classes)                     | 0.34                                              | 0.58                                           |
| Encoder-Decoder                                 | 0.16                                              | 0.21                                           |
| WPFS                                            | 0.35                                              | 0.32                                           |
| Linear SVR, $\epsilon=0.18$                     | **0.58**                                          | 0.30                                           |
| Linear SVR $^\star$, $\epsilon=0.18$             | -0.20                                            | 0.19                                           |
| Sigmoid SVR, $\epsilon=0.23$                    | 0.56                                              | 0.30                                           |
| Sigmoid SVR $^\star$, $\epsilon=0.23$            | 0.53                                              | 0.36                                           |
| Linear SVR, $\epsilon=0.28$, $k=500$            | 0.49                                              | 0.42                                           |

**Note:** Models marked with $^\star$ were trained on the augmented dataset.



