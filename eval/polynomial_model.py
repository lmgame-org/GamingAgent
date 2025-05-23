import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from itertools import product
import json
import argparse
import os
from scipy.stats import rankdata

# Default model order based on comments in the original script
DEFAULT_MODEL_ORDER = [
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219 (thinking)",
    "gemini-2.5-pro-preview-05-06",
    "llama-4-maverick-17b-128e-instruct-fp8",
    "gpt-4o-2024-11-20",
    "o1-2024-12-17",
    "o3-2025-04-16",
    "o4-mini-2025-04-16"
]

MODEL_PERF_RANK_JSON_PATH = "eval/perf/model_perf_rank.json"
OTHER_TASK_RANK_JSON_PATH = "eval/perf/other_task_rank.json"
POLYNOMIAL_MODEL_RESULTS_JSON_PATH = "eval/perf/polynomial_model.json"

# batching helper
def build_batches(rank_matrices):
    batches = []
    valid_matrices = [m for m in rank_matrices if isinstance(m, np.ndarray) and m.shape[0] > 0]
    if len(valid_matrices) != len(rank_matrices):
        print("Warning: Invalid/empty rank matrices for build_batches. Using valid ones.")
        if not valid_matrices: return []
        rank_matrices = valid_matrices
    for row_idxs in product(*(range(m.shape[0]) for m in rank_matrices)):
        batches.append(np.stack([m[i] for m, i in zip(rank_matrices, row_idxs)], axis=0).T)
    return batches

# Predictor
class RankingPredictor:
    """
    Learns   G ≈ Φ( (R / scale) ) · w   with non‑negative w.
    """
    def __init__(self, degree: int, lr: float = 1e-4, epochs: int = 1000, shuffle: bool = True, scale: float = 8.0):
        self.degree, self.lr, self.epochs, self.shuffle, self.scale = degree, lr, epochs, shuffle, scale
        self._phi = PolynomialFeatures(degree=degree, include_bias=True)
        self._w = None
    def _transform(self, R, fit=False): return self._phi.fit_transform(R / self.scale) if fit else self._phi.transform(R / self.scale)
    def fit(self, R_list, G_list):
        if not isinstance(R_list, (list, tuple)): R_list, G_list = [R_list], [G_list]
        X_list = [self._transform(R_list[0], fit=True)] + [self._transform(R) for R in R_list[1:]]
        self._w = np.zeros(X_list[0].shape[1])
        for _ in range(self.epochs):
            order = np.random.permutation(len(X_list)) if self.shuffle else range(len(X_list))
            for i in order:
                X, G = X_list[i], G_list[i]
                self._w -= self.lr * (2.0 * X.T @ (X @ self._w - G))
                self._w = np.maximum(self._w, 0.0)
    def predict(self, R):
        if self._w is None: raise ValueError("Predictor not fitted.")
        return self._transform(R) @ self._w
    def evaluate(self, R, G, norm_type='L2', normalization='mean'):
        pred = self.predict(R)
        if norm_type == 'L1': res = np.sum(np.abs(pred - G))
        elif norm_type == 'Linf': res = np.max(np.abs(pred - G))
        elif norm_type == 'L0.5': res = np.sum(np.abs(pred - G) ** 0.5) ** 2
        else: res = np.linalg.norm(pred - G, ord=2)
        eps = np.finfo(float).eps
        if normalization == 'mean': factor = np.mean(G) + eps
        elif normalization == 'max': factor = np.max(G) + eps
        elif normalization == 'std': factor = np.std(G) + eps
        elif normalization == 'range': factor = (np.max(G) - np.min(G)) + eps
        else: factor = 1
        norm_res = res / factor
        if self.degree == 1:
            pred_std, G_std = np.std(pred) + eps, np.std(G) + eps
            r = 0.0 if pred_std * G_std == 0 else np.clip(np.corrcoef(pred, G)[0, 1], -1.0, 1.0)
            return norm_res, r
        return norm_res

# --- Helper Functions ---
def _load_json(file_path):
    if not os.path.exists(file_path): 
        # print(f"Info: JSON file not found at {file_path}, returning empty dict.")
        return {}
    try: return json.load(open(file_path, 'r'))
    except json.JSONDecodeError as e: print(f"Error decoding JSON from {file_path}: {e}"); return {}

def _save_json(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f: json.dump(data, f, indent=2)
        print(f"Results saved to {file_path}")
    except Exception as e: print(f"Error saving JSON to {file_path}: {e}")

def _scores_to_ranks(scores_list, higher_is_better=True):
    s = np.array(scores_list, dtype=float); m = np.isnan(s)
    s[m] = -np.inf if higher_is_better else np.inf
    return rankdata(-s if higher_is_better else s, method='average').astype(int)

def _get_ranks_for_game(game, models, data, harness="harness_true"):
    scores = []
    for m_name in models:
        try:
            s_list = data.get(m_name, {}).get(harness, {}).get(game, [])
            num_s = [s_val for s_val in s_list if isinstance(s_val, (int, float))]
            scores.append(np.mean(num_s) if num_s else np.nan)
        except Exception: scores.append(np.nan)
    if not any(not np.isnan(s) for s in scores): print(f"Warn: All NaN scores for {game}/{harness}.")
    return _scores_to_ranks(scores, higher_is_better=True)

def _load_other_task_ranks(f_path, models):
    data = _load_json(f_path); num_m = len(models)
    if not data: return {}
    ranks = {}
    for cat, r_lists in data.items():
        if not isinstance(r_lists, list): print(f"Warn: Ranks for '{cat}' not list. Skip."); continue
        valid_v = [v for v in r_lists if isinstance(v, list) and len(v) == num_m]
        if len(valid_v) != len(r_lists): print(f"Warn: Some vectors for '{cat}' bad len/type.")
        if valid_v: ranks[cat] = np.array(valid_v, dtype=int)
        else: print(f"Warn: No valid vectors for '{cat}'.")
    return ranks

def run_polynomial_analysis(
    model_names_str: str | None = None,
    target_game_for_G: str | None = None,
    harness_status_for_G: str = "harness_true",
    poly_degree: int = 2,
    learning_rate: float = 5e-3,
    epochs_multiplier: int = 1,
    force_update_results: bool = False,
    model_perf_rank_json_path: str = MODEL_PERF_RANK_JSON_PATH,
    other_task_rank_json_path: str = OTHER_TASK_RANK_JSON_PATH,
    polynomial_model_results_json_path: str = POLYNOMIAL_MODEL_RESULTS_JSON_PATH,
    target_games_list_for_G: list[str] | None = None
):
    """
    Runs the polynomial ranking prediction analysis.
    If called with no args, uses internal defaults.
    """
    if model_names_str is None:
        current_model_order = DEFAULT_MODEL_ORDER
    else:
        current_model_order = [name.strip() for name in model_names_str.split(',') if name.strip()]
        if not current_model_order:
            current_model_order = DEFAULT_MODEL_ORDER
    print(f"Using models for polynomial analysis: {current_model_order}")

    all_model_perf_data = _load_json(model_perf_rank_json_path)
    all_other_task_ranks = _load_other_task_ranks(other_task_rank_json_path, current_model_order)
    all_polynomial_results = _load_json(polynomial_model_results_json_path)

    if not all_model_perf_data or not all_other_task_ranks:
        print("Error loading critical R/G data for polynomial model. Exiting analysis.")
        return

    target_games_to_process = []
    if target_games_list_for_G:
        target_games_to_process = target_games_list_for_G
        print(f"Using specified game list for polynomial G: {target_games_to_process}")
    elif target_game_for_G:
        target_games_to_process = [target_game_for_G]
        print(f"Using singular target game for polynomial G: {target_games_to_process}")
    else:
        print(f"No specific target game(s) for G, processing all available games for harness '{harness_status_for_G}' in polynomial model...")
        available_games = set()
        for model_data in all_model_perf_data.values():
            if isinstance(model_data.get(harness_status_for_G), dict):
                available_games.update(model_data[harness_status_for_G].keys())
        if not available_games:
            print(f"No games found for harness '{harness_status_for_G}' in polynomial model. Exiting analysis.")
            return
        target_games_to_process = sorted(list(available_games))
        print(f"Found games for polynomial model G: {target_games_to_process}")

    category_combinations = [
        {"name": "Comb1_KnowPuzVisMathCode", "keys": ["knowledge", "puzzle", "visual", "math", "coding"]},
        {"name": "Comb2_KnowVisMathCode", "keys": ["knowledge", "visual", "math", "coding"]},
        {"name": "Comb3_LangPhyVisMathCode", "keys": ["language", "physics", "visual", "math", "coding"]},
        {"name": "Comb4_KnowMathCode", "keys": ["knowledge", "math", "coding"]}
    ]

    for target_game_name in target_games_to_process:
        print(f"\n\n=== Polynomial Model Processing for Target Game (G): {target_game_name} ({harness_status_for_G}) ===")
        G_current = _get_ranks_for_game(target_game_name, current_model_order, all_model_perf_data, harness_status_for_G)
        if G_current is None or len(G_current) != len(current_model_order):
            print(f"Error loading/ranking G for '{target_game_name}' in polynomial model. Skipping."); continue
        print(f"Target Ranks G for '{target_game_name}' (poly model): {G_current}")

        all_polynomial_results.setdefault(target_game_name, {})

        for comb_info in category_combinations:
            comb_name, comb_keys = comb_info["name"], comb_info["keys"]
            print(f"\n--- Poly Model R Combination: {comb_name} (Categories: {comb_keys}) ---")

            if not force_update_results and comb_name in all_polynomial_results[target_game_name]:
                existing_config = all_polynomial_results[target_game_name][comb_name].get("config", {})
                if existing_config.get("target_harness_status") == harness_status_for_G and \
                   existing_config.get("poly_degree") == poly_degree and \
                   existing_config.get("learning_rate") == learning_rate and \
                   existing_config.get("epochs_multiplier") == epochs_multiplier and \
                   existing_config.get("model_order_used") == current_model_order and \
                   set(existing_config.get("R_categories_used", [])) == set(comb_keys):
                    print(f"Results for poly model '{target_game_name}' / '{comb_name}' (Harness: {harness_status_for_G}) with same params exist. Skipping (use --force_polynomial_results_update to overwrite).")
                    continue

            R_sources = [all_other_task_ranks[k] for k in comb_keys if k in all_other_task_ranks]
            if len(R_sources) != len(comb_keys):
                print(f"Warn: Missing R categories for poly model {comb_name}. Skipping."); continue
            
            R_batches = build_batches(R_sources)
            if not R_batches: print(f"No batches for poly model {comb_name}. Skipping."); continue
            
            G_targets = [G_current] * len(R_batches)
            print(f"Poly model data size: {len(R_batches)} batches (shape: {R_batches[0].shape})")
            epochs = epochs_multiplier * (len(R_batches) if R_batches else 100)
            
            predictor = RankingPredictor(poly_degree, learning_rate, epochs)
            predictor.fit(R_batches, G_targets)

            current_run_results = {
                "config": {
                    "poly_degree": poly_degree,
                    "learning_rate": learning_rate,
                    "epochs_multiplier": epochs_multiplier,
                    "model_order_used": current_model_order,
                    "R_categories_used": comb_keys,
                    "target_harness_status": harness_status_for_G
                },
                "evaluation": {},
                "feature_weights": {}
            }

            print(f"-- Poly Eval for {comb_name} (predicting {target_game_name}) --")
            if poly_degree == 1:
                evals = [predictor.evaluate(Rb, G_current) for Rb in R_batches]
                current_run_results["evaluation"]["avg_residual_error"] = np.mean([e[0] for e in evals])
                current_run_results["evaluation"]["avg_pearson_r"] = np.mean([e[1] for e in evals])
                print(f"Poly Avg Pearson's r: {current_run_results['evaluation']['avg_pearson_r']:.4f}, Poly Avg RE: {current_run_results['evaluation']['avg_residual_error']:.4f}")
            else:
                all_err = [predictor.evaluate(Rb, G_current) for Rb in R_batches]
                current_run_results["evaluation"]["avg_residual_error"] = np.mean(all_err)
                print(f"Poly Avg RE: {current_run_results['evaluation']['avg_residual_error']:.4f}")
            
            print(f"-- Poly Params for {comb_name} (predicting {target_game_name}) --")
            w_vals = predictor._w
            try: f_names = predictor._phi.get_feature_names_out(comb_keys)
            except: f_names = [f"feat_{j}" for j in range(len(w_vals))]; print("Warn: Using generic feat names for poly model.")
            
            current_run_results["feature_weights"] = {f_names[i]: w_vals[i] for i in range(len(w_vals))}
            
            s_idx = np.argsort(np.abs(w_vals))[::-1]
            print("Poly Weight ↔ Feature (sorted by abs weight):")
            for i in s_idx: print(f"{f_names[i]:>25} : {w_vals[i]:.6f}")

            all_polynomial_results[target_game_name][comb_name] = current_run_results

    _save_json(all_polynomial_results, polynomial_model_results_json_path)
    print("\n\nPolynomial model script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polynomial ranking predictor for games.")
    parser.add_argument("--model_names", type=str, default=",".join(DEFAULT_MODEL_ORDER),
                        help=f"Comma-separated model names. Default: {len(DEFAULT_MODEL_ORDER)} models.")
    parser.add_argument("--target_game_for_G", type=str, default=None,
                        help="Target game for G. Default: all games.")
    parser.add_argument("--harness_status_for_G", type=str, default="harness_true", choices=["harness_true", "harness_false"],
                        help="Harness status for G. Default: harness_true.")
    parser.add_argument("--poly_degree", type=int, default=2, help="Poly degree. Default: 2.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="LR. Default: 5e-3.")
    parser.add_argument("--epochs_multiplier", type=int, default=1, help="Epochs multiplier. Default: 1.")
    parser.add_argument("--force_update_results", action='store_true', default=False,
                        help="Force update results even if they exist. Default: False.")
    args = parser.parse_args()

    run_polynomial_analysis(
        model_names_str=args.model_names,
        target_game_for_G=args.target_game_for_G,
        harness_status_for_G=args.harness_status_for_G,
        poly_degree=args.poly_degree,
        learning_rate=args.learning_rate,
        epochs_multiplier=args.epochs_multiplier,
        force_update_results=args.force_update_results,
        target_games_list_for_G=None
    )