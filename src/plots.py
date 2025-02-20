import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

def compute_shap_values(explainer, x_instance):
    """
    Compute SHAP values for a single instance.
    """
    return explainer.shap_values(x_instance)

def model_predict(model, data):
        """Wrap the model prediction to return outputs for SHAP."""
        return model.predict(data)

def plot_shap_summary(model, x_test, features_names, output_dir="shap_plots", filename="shap_summary_plot.png", background_data=None):
    """
    Plot the SHAP summary plot.
    
    Parameters:
    - model: Trained model.
    - x_test: Test dataset (DataFrame or NumPy array).
    - features_names: Names of the features.
    - output_dir: Directory to save SHAP summary plot, defaults to shap_plots.
    - filename: Name of the output file.
    - background_data: Background dataset used by the SHAP kernel explainer for reference values.
   
    """ 
    assert len(features_names) == x_test.shape[1], "The number of feature names should match the number of features in the test set"

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # feature importance
    explainer = shap.TreeExplainer(model) if ((background_data is None) and (model.__class__.__name__ in ["XGBRegressor", "LGBMRegressor"])) else shap.KernelExplainer(model_predict, background_data)

    shap_values = compute_shap_values(explainer, x_test)
    print(f"x_test shape: {x_test.shape}")
    fig = shap.summary_plot(shap_values, features=x_test, feature_names=features_names, show=False)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()



def plot_beeswarm_shap(model, x_test, y_pred, selected_feature_names, output_dir="shap_plots", filename="beeswarm_test.png", background_data=None):
    """
    Plot the SHAP beeswarm plot.
    
    Parameters:
    - model_predict: Function that returns model predictions.
    - x_test: Test dataset (DataFrame or NumPy array).
    - feature_names: Names of the features.
    - output_dir: Directory to save SHAP beeswarm plot, defaults to shap_plots.
    - filename: Name of the output file.
    - background_data: Background dataset used by the SHAP kernel explainer for reference values.
   
    """

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)


    # feature importance
    explainer = shap.TreeExplainer(model) if ((background_data is None) and (model.__class__.__name__ in ["XGBRegressor", "LGBMRegressor"])) else shap.KernelExplainer(model, background_data)

    sorted_indices = np.argsort(y_pred)
    top_indices = sorted_indices[-5:]
    bottom_indices = sorted_indices[:5:]
    median_index = sorted_indices[len(sorted_indices) // 2]
    median_indices = [median_index - 2, median_index - 1, median_index, median_index + 1, median_index + 2]
    list_to_explain = np.concatenate((x_test[top_indices], x_test[median_indices], x_test[bottom_indices]))
    
    shap_values = explainer(list_to_explain)
    shap_values.feature_names = selected_feature_names

    # shap_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=x_test)

    fig = shap.plots.beeswarm(shap_values, color=plt.get_cmap("cool"), show=False)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")

def plot_shap_decision(model, x_test, y_pred, feature_names, output_dir="shap_plots", background_data=None):
    """
    Plot the SHAP decision plots for selected samples.

    Parameters:
    - model: Trained model.
    - X_test: Test dataset (DataFrame or NumPy array).
    - y_pred: Model predictions for the test set.
    - feature_names: Names of the features.
    - output_dir: Directory to save SHAP decision plots.
    - background_data: Background dataset used by the SHAP kernel explainer for reference values.

    """
    assert len(x_test) == len(y_pred), "The number of predictions should match the number of samples in the test set"
    assert len(feature_names) == x_test.shape[1], "The number of feature names should match the number of features in the test set"

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # feature importance
    explainer = shap.TreeExplainer(model) if ((background_data is None) and (model.__class__.__name__ in ["XGBRegressor", "LGBMRegressor"])) else shap.KernelExplainer(model, background_data)

    shap_values = explainer.shap_values(x_test)

    # identify samples for explanations
    sorted_indices = np.argsort(y_pred)
    top_indices = sorted_indices[-5:]
    bottom_indices = sorted_indices[:5:]
    median_index = sorted_indices[len(sorted_indices) // 2]

    # SHAP decision plots for selected samples
    for idx in list(top_indices) + list(bottom_indices) + [median_index]:
        fig = shap.decision_plot(
            explainer.expected_value,
            shap_values[idx],
            features=x_test[idx],
            feature_names=list(feature_names),
            show=False
        )
        plt.savefig(os.path.join(output_dir, f"shap_decision_plot_{feature_names[idx]}.png"))
        # still show it after otherwise bugs
        plt.show()



def plot_waterfall_shap(model, x_test, y_pred, feature_names, output_dir="shap_plots", filename="waterfall.png", background_data=None):
    """
    Plot the SHAP waterfall plot.
    
    Parameters:
    - model: Trained model.
    - x_test: Test dataset (DataFrame or NumPy array).
    - feature_names: Names of the features.
    - output_dir: Directory to save SHAP waterfall plot, defaults to shap_plots.
    - filename: Name of the output file.
    - background_data: Background dataset used by the SHAP kernel explainer for reference values.
   
    """

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    sorted_indices = np.argsort(y_pred)
    top_indices = sorted_indices[-5:]
    bottom_indices = sorted_indices[:5:]
    median_index = sorted_indices[len(sorted_indices) // 2]


    explainer = shap.TreeExplainer(model) if ((background_data is None) and (model.__class__.__name__ in ["XGBRegressor", "LGBMRegressor"])) else shap.KernelExplainer(model, background_data)

    shap_values = explainer.shap_values(x_test)
    shap_values.feature_names = feature_names
    
    # SHAP decision plots for selected samples
    for idx in list(top_indices) + list(bottom_indices) + [median_index]:
        fig =  shap.waterfall_plot(
                            explainer.expected_value, 
                            shap_values[idx], 
                            features=x_test[idx], 
                            feature_names=list(feature_names), 
                            show=False)

    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()