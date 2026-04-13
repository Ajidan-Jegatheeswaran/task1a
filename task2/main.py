import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # One-hot encode season
    train_df = pd.get_dummies(train_df, columns=["season"], drop_first=False)
    test_df = pd.get_dummies(test_df, columns=["season"], drop_first=False)

    # Ensure same columns in train and test (excluding target)
    # Convert boolean dummies to int
    for col in train_df.columns:
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(int)
    for col in test_df.columns:
        if test_df[col].dtype == bool:
            test_df[col] = test_df[col].astype(int)

    # Separate target
    y_train_full = train_df["price_CHF"].values
    X_train_full = train_df.drop(columns=["price_CHF"])

    # Align columns
    common_cols = sorted(set(X_train_full.columns) & set(test_df.columns))
    X_train_full = X_train_full[common_cols]
    X_test = test_df[common_cols]

    # Impute missing feature values using IterativeImputer on combined data
    imputer = IterativeImputer(max_iter=50, random_state=42, sample_posterior=False)
    X_combined = pd.concat([X_train_full, X_test], axis=0)
    X_combined_imputed = imputer.fit_transform(X_combined)

    n_train = X_train_full.shape[0]
    X_train_imputed = X_combined_imputed[:n_train]
    X_test_imputed = X_combined_imputed[n_train:]

    # Remove rows where target is NaN
    mask = ~np.isnan(y_train_full)
    X_train = X_train_imputed[mask]
    y_train = y_train_full[mask]

    print(f"Training samples (after dropping NaN targets): {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test_imputed.shape[0]}")

    assert (X_train.shape[1] == X_test_imputed.shape[1]) and (X_test_imputed.shape[0] == 100)
    return X_train, y_train, X_test_imputed, common_cols


def select_best_model(X_train, y_train):
    """Try multiple GP kernels and pick the best one via cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    kernels = {
        "RBF": ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        "Matern_1.5": ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1),
        "Matern_2.5": ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
        "RationalQuadratic": ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=0.1),
        "DotProduct": ConstantKernel(1.0) * DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.1),
    }

    best_score = -np.inf
    best_name = None
    best_model = None

    for name, kernel in kernels.items():
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, normalize_y=True)
        scores = cross_val_score(gpr, X_scaled, y_train, cv=5, scoring="r2")
        mean_score = scores.mean()
        print(f"Kernel {name}: R² = {mean_score:.4f} (+/- {scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = gpr

    print(f"\nBest kernel: {best_name} with R² = {best_score:.4f}")
    return best_model, scaler


if __name__ == "__main__":
    X_train, y_train, X_test, cols = load_data()

    print("\n--- Kernel Selection via Cross-Validation ---")
    best_model, scaler = select_best_model(X_train, y_train)

    # Fit best model on full training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    print(f"\nPrediction stats: mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")

    # Save results
    dt = pd.DataFrame(y_pred, columns=["price_CHF"])
    dt.to_csv("results.csv", index=False)
    print("Results file successfully generated!")
