"""Training routine for the SDP pigments model.

This module contains the PCA + regression model fitting logic adapted from
`bioOptix_and_PFTs` and used to fit the coefficient ensembles written under
`models/sdp_pigments/coefficients/`.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd
from typing import Tuple, Dict, Any

def rrsModelTrain(
    RrsD: np.ndarray,
    hplc_i: np.ndarray,
    pft_index: str,
    n_permutations: int,
    max_pcs: int,
    k: int,
    mdl_pick_metric: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Train PCA-based regression model for pigment prediction. One set of coefficients are generated for one pigment at a time.

    Uses a 75/25 train/validation split with k-fold CV within the training set. For each permutation, the mean k-fold coefficients are unstandardized and validated against the held-out 25%.

    Note: max_pcs must be <= 0.75 * (1 - 1/k) * n_samples.

    Args:
        RrsD: 2nd derivative of Rrs residuals, shape (n_samples, n_wavelengths).
        hplc_i: Pigment concentrations for a single pigment (ground truth), shape (n_samples,).
        pft_index: Constraint type. 'pigment' (>= 0), 'EOFs' (unconstrained), or 'compositions' (0-1).
        n_permutations: Number of random 75/25 train/validation splits.
        max_pcs: Maximum number of principal components to evaluate.
        k: Number of folds for cross-validation within each training split.
        mdl_pick_metric: Metric for selecting the optimal number of PCs. One of 'R2', 'RMSE', 'avg', 'med', or 'MAE' (McKinna et al. 2021).

    Returns:
        coefficients: Unstandardized regression coefficients, shape (n_wavelengths, n_permutations).
        intercepts: Unstandardized intercepts, shape (n_permutations,).
        summary_gofs: DataFrame with mean/std of R2, RMSE, percent error, bias, and MAE across permutations.
        all_gofs: Dict of per-permutation goodness-of-fit arrays.
    """
    
    # Cannot contain NaNs in training data for either HPLC or 2nd derivative of Rrs residuals
    if np.isnan(hplc_i).any():
        raise ValueError('hplc_i contains NaN values')
    if np.isnan(RrsD).any():
        raise ValueError('RrsD contains NaN values')

    # RrsD and hplc_i must have 1:1 row correspondence (not necessarily cols)
    if RrsD.shape[0] != hplc_i.shape[0]:
        raise ValueError(
            f'RrsD and hplc_i row count mismatch: {RrsD.shape[0]} vs {hplc_i.shape[0]}'
        )
    
    # Set random number generator seed for reproducibility
    np.random.seed(100)

    # Create coefficients array: pigment = RrsD @ betas + alpha
    # pigment is (n_samples) -> n_samples is flattened (lon, lat, time)
    # RrsD is (n_samples, n_wavelengths)
    mean_betas_nonstd = np.zeros((RrsD.shape[1], n_permutations)) # (n_wavelengths, n_permutations)
    mean_alphas_nonstd = np.zeros(n_permutations)

    # preallocate statistics/metrics arrays
    R2s_final = np.zeros(n_permutations)
    RMSEs_final = np.zeros(n_permutations)
    pct_bias = np.zeros(n_permutations)
    pct_errors = np.zeros((n_permutations,len(hplc_i)-int(len(hplc_i) * 0.75))) # 25% of the data for validation
    med_pct_error = np.zeros(n_permutations)
    avg_pct_error = np.zeros(n_permutations)
    CI_pct_error = np.zeros(n_permutations)
    std_pct_error = np.zeros(n_permutations)
    mae_final = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Create broad training data (75%) and validation data (25%)

        training_indices = np.random.permutation(len(hplc_i))[:int(len(hplc_i) * 0.75)]

        pigs_training = hplc_i[training_indices]
        RrsD_training = RrsD[training_indices,:]

        # validation data
        pigs_validate = hplc_i
        pigs_validate = np.delete(pigs_validate,training_indices)
        RrsD_validate = RrsD
        RrsD_validate = np.delete(RrsD_validate, training_indices, axis=0)

        # get set up for k-fold cross validation

        pig_len = len(pigs_training)

        rand_ns = np.random.permutation(pig_len)

        CV_indices = np.full((k, int(np.ceil(len(pigs_training) / k))), np.nan)
        n_leftovers = pig_len % k
        counter_start = n_leftovers
        counter_end = n_leftovers + pig_len // k
        for j in range(k):
            CV_indices[j, :(pig_len // k)] = rand_ns[counter_start:counter_end]
            counter_start += pig_len // k
            counter_end += pig_len // k

        # add the leftovers to the CV_indices array, and put NaN's for the sets
        # where there's not enough leftovers to go around
        leftovers = rand_ns[:n_leftovers]
        na_array = np.full((k - len(leftovers)), np.nan)
        leftovers = np.concatenate([leftovers, na_array])

        CV_indices[:, CV_indices.shape[1]-1] = leftovers

        # preallocate arrays
        n_modes_to_use = np.zeros(k, dtype=int)
        betas = np.zeros((RrsD_training.shape[1], k))
        alpha = np.zeros(k)
        CV_R2s = np.zeros(k)
        CV_RMSEs = np.zeros(k)

        for j in range(k):
            # Split up CV data sets
            these_CV_indices = CV_indices[j, :]
            these_CV_indices = these_CV_indices[~np.isnan(these_CV_indices)].astype(int)
            CV_valid_pigs = pigs_training[these_CV_indices]
            CV_valid_spec = RrsD_training[these_CV_indices, :]
            CV_train_pigs = np.delete(pigs_training, these_CV_indices, axis=0)
            CV_train_spec = np.delete(RrsD_training, these_CV_indices, axis=0)
            
            # standardize spectra for PCs
            CV_train_spec = (CV_train_spec - np.mean(CV_train_spec, axis=0)) / np.std(CV_train_spec, axis=0)
            CV_valid_spec = (CV_valid_spec - np.mean(CV_valid_spec, axis=0)) / np.std(CV_valid_spec, axis=0)

            # Manual PCA without centering using SVD
            U, S, VT = np.linalg.svd(CV_train_spec, full_matrices=False)

            CV_EOFs_train = VT[:max_pcs].T
            CV_AFs_train = U[:, :max_pcs] * S[:max_pcs]

            # Preallocate arrays to hold evaluation metrics
            n_val = len(CV_valid_pigs)
            percent_errors = np.zeros((n_val, CV_AFs_train.shape[1]))
            all_bias = np.zeros((n_val, CV_AFs_train.shape[1]))
            mean_percent_error = np.zeros(CV_AFs_train.shape[1])
            median_percent_error = np.zeros(CV_AFs_train.shape[1])
            bias = np.zeros(CV_AFs_train.shape[1])
            MAE = np.zeros(CV_AFs_train.shape[1])
            R2s = np.zeros(CV_AFs_train.shape[1])
            RMSEs = np.zeros(CV_AFs_train.shape[1])
            ensemble = np.zeros(CV_AFs_train.shape[1])
            pearson = np.zeros(CV_AFs_train.shape[1])

            # Loop over number of components used in model
            for l in range(1, CV_AFs_train.shape[1]+1):
                # Multiple linear regression (MLR) using first l amplitude functions
                lin_model = LinearRegression()
                lin_model.fit(CV_AFs_train[:, :l], CV_train_pigs)

                # Intercept and coefficients
                this_alpha = lin_model.intercept_
                these_betas = lin_model.coef_

                # Map AF coefficients back to spectral domain (EOFs * weights)
                spec_betas = CV_EOFs_train[:, :l] @ these_betas

                # Apply model to validation spectra
                CV_modeled_pigs = CV_valid_spec @ spec_betas + this_alpha

                # Constrain results based on pft_index type
                if pft_index == 'pigment':
                    CV_modeled_pigs[CV_modeled_pigs < 0] = 0
                elif pft_index == 'compositions':
                    CV_modeled_pigs = np.clip(CV_modeled_pigs, 0, 1)
                elif pft_index == 'EOFs':
                    pass  # No constraints applied

                # Compute percent error
                # Divide by zero happens here if CV_valid_pigs contains zero entries (amount detected is not significant)
                percent_errors[:n_val, l-1] = ((CV_valid_pigs - CV_modeled_pigs) / CV_valid_pigs) * 100 # divide by zero here
                mean_percent_error[l-1] = np.mean(np.abs(percent_errors[:, l-1]))
                median_percent_error[l-1] = np.median(np.abs(percent_errors[:, l-1]))

                # Compute bias
                all_bias[:n_val, l-1] = CV_modeled_pigs - CV_valid_pigs
                bias[l-1] = np.mean(all_bias[:, l-1])
                MAE[l-1] = np.mean(np.abs(all_bias[:, l-1]))

                # Correlation and regression metrics
                reg = LinearRegression()
                reg.fit(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs)

                R2s[l-1] = reg.score(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs)
                RMSEs[l-1] = root_mean_squared_error(CV_modeled_pigs, CV_valid_pigs)
                pearson[l-1] = np.corrcoef(CV_modeled_pigs, CV_valid_pigs)[0, 1]

                # Ensemble score
                ensemble[l-1] = (1 - R2s[l-1] + RMSEs[l-1]) / 100

            # Select the best model based on the chosen metric
            if mdl_pick_metric == 'MAE':
                n_modes_to_use[j] = np.argmin(MAE) + 1 # account for python exclusive indexing
 
            # apply your optimized model and record the g.o.f. statistics for this k-th CV:
            X_train = CV_AFs_train[:, :n_modes_to_use[j]]
            y_train = CV_train_pigs

            lin_mdl = LinearRegression()
            lin_mdl.fit(X_train, y_train)

            alpha[j] = lin_mdl.intercept_  
            these_betas = lin_mdl.coef_

            # now turn model coefficients for AF's into coefficients for the combined
            # derivative spectra:
            betas[:, j] = CV_EOFs_train[:, :n_modes_to_use[j]] @ these_betas

            # 5-CV model validation
            CV_modeled_pigs = CV_valid_spec @ betas[:, j] + alpha[j]

            if pft_index == 'pigment':
                CV_modeled_pigs[CV_modeled_pigs < 0] = 0
            elif pft_index == 'compositions':
                CV_modeled_pigs = np.clip(CV_modeled_pigs, 0, 1)
            elif pft_index == 'EOFs':
                pass # No constraints applied

            # fit linear model to look at modeled vs observed
            CV_reg = LinearRegression()
            CV_reg.fit(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs)
            CV_R2s[j] = CV_reg.score(CV_modeled_pigs.reshape(-1, 1), CV_valid_pigs)
            CV_RMSEs[j] = root_mean_squared_error(CV_modeled_pigs, CV_valid_pigs)

        # so now you have k sets of optimized coefficients. grab the
        # average of them and validate against your original 25% validation
        # data set. 
        
        # Store mean/std of each set of k-fold CV betas 
        # (the model coefficients for the ith run of the n_permutations):
        mean_betas = np.mean(betas, axis=1)
        mean_alphas = np.mean(alpha)
        std_betas = np.std(betas, axis=1)
        std_alphas = np.std(alpha)

        # Compute standard deviation and mean across samples (i.e., along axis 0)
        spec_std = np.std(RrsD_training, axis=0, ddof=0)  # MATLAB default is population std (ddof=0)
        spec_mean = np.mean(RrsD_training, axis=0)

        # Unstandardize beta and alpha for model i
        mean_betas_nonstd[:, i] = mean_betas / spec_std
        mean_alphas_nonstd[i] = mean_alphas - np.sum(mean_betas * (spec_mean / spec_std))

        # Validate on the data you set aside previously for this ith run of the n_permutations using mean betas of this
        # permutation from cross-validation, and store g.o.f stats across permutations:
        modeled_pigs = RrsD_validate @ mean_betas_nonstd[:,i] + mean_alphas_nonstd[i]

        if pft_index == 'pigment':
            modeled_pigs[modeled_pigs < 0] = 0
        elif pft_index == 'compositions':
            modeled_pigs = np.clip(modeled_pigs, 0, 1)
        elif pft_index == 'EOFs':
            pass # No constraints applied

        # Fit linear model
        model = LinearRegression().fit(modeled_pigs.reshape(-1, 1), pigs_validate)

        # Save R² and RMSE
        R2s_final[i] = model.score(modeled_pigs.reshape(-1, 1), pigs_validate)
        RMSEs_final[i] = np.sqrt(root_mean_squared_error(pigs_validate, model.predict(modeled_pigs.reshape(-1, 1))))

        # Avoid division by zero by replacing 0 with 1e-4
        pigs_validate_safe = np.where(pigs_validate == 0, 1e-4, pigs_validate)

        # Percent bias and errors
        pct_bias[i] = np.mean(((modeled_pigs - pigs_validate_safe) / pigs_validate_safe) * 100)
        pct_errors[i, :] = np.abs(((modeled_pigs - pigs_validate_safe) / pigs_validate_safe) * 100)
        med_pct_error[i] = np.median(pct_errors[i, :])
        avg_pct_error[i] = np.mean(pct_errors[i, :])

        # 95th percentile confidence interval
        sort_pct_errors = np.sort(pct_errors[i, :])
        CI_pct_error[i] = sort_pct_errors[int(np.ceil(0.95 * len(sort_pct_errors))) - 1]

        # Standard deviation of percent error
        std_pct_error[i] = np.std(pct_errors[i, :])

        # Mean absolute error
        mae_final[i] = np.mean(np.abs(modeled_pigs - pigs_validate))

        # Print progress
        #print(f"hey dude, im doing good. im on permutation # {i}")

    coefficients = mean_betas_nonstd
    intercepts = mean_alphas_nonstd

    # === Summary stats ===
    summary_gofs = [
        np.mean(R2s_final), np.std(R2s_final),
        np.mean(RMSEs_final), np.std(RMSEs_final),
        np.mean(avg_pct_error), np.std(avg_pct_error),
        np.mean(med_pct_error), np.std(med_pct_error),
        np.mean(pct_bias), np.std(pct_bias),
        np.mean(mae_final), np.std(mae_final)
    ]

    summary_gofs_df = pd.DataFrame([summary_gofs], columns=[
        'Mean_R2', 'SD_R2',
        'Mean_RMSE', 'SD_RMSE',
        'Mean_mean_pct_error', 'SD_mean_pct_error',
        'Mean_median_pct_error', 'SD_median_pct_error',
        'Mean_pct_bias', 'SD_pct_bias',
        'Mean_MAE', 'SD_MAE'
    ])

    # === All individual stats ===
    all_gofs = {
        'R2s': R2s_final,
        'RMSEs': RMSEs_final,
        'mean_pct_error': avg_pct_error,
        'median_pct_error': med_pct_error,
        'pct_bias': pct_bias,
        'all_pct_errors': pct_errors,
        'all_mae': mae_final
    }

    return coefficients, intercepts, summary_gofs_df, all_gofs
