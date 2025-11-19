"""
Grid Search for Optimal K and HOME_ADV Parameters
--------------------------------------------------

This script performs a grid search to find the optimal values for:
- K: ELO update factor (how quickly ratings change after each game)
- HOME_ADV: Home advantage in ELO points

The script evaluates each combination using both Multinomial Logit
and Ordered Probit models, reporting accuracy for each.
"""

import pandas as pd
import numpy as np
from step1_clean_data import load_and_clean_data
from step2_ELO_ratings import run_step2_ELO
from step3_multinomial_logit import run_step3_multinomial
from step3_ordered_probit import run_step3_ordered_probit
import time


def grid_search(k_values, home_adv_values, model_type='multinomial'):
    """
    Perform grid search over K and HOME_ADV parameters.

    Args:
        k_values: List of K values to test
        home_adv_values: List of HOME_ADV values to test
        model_type: 'multinomial' or 'probit' (which model to optimize for)

    Returns:
        results_df: DataFrame with all results
        best_params: Dictionary with best parameters
    """

    print("="*60)
    print("GRID SEARCH FOR OPTIMAL K AND HOME_ADV")
    print("="*60)

    # Load and clean data once
    print("\nLoading and cleaning data...")
    player_df, games_df = load_and_clean_data()
    print(f"Loaded {len(player_df):,} player records and {len(games_df):,} games")

    # Storage for results
    results = []

    total_combinations = len(k_values) * len(home_adv_values)
    current = 0

    print(f"\nTesting {total_combinations} combinations...")
    print(f"K values: {k_values}")
    print(f"HOME_ADV values: {home_adv_values}")
    print(f"Optimizing for: {model_type.upper()}")
    print("\n" + "-"*60)

    start_time = time.time()

    for K in k_values:
        for HOME_ADV in home_adv_values:
            current += 1

            print(f"\n[{current}/{total_combinations}] Testing K={K}, HOME_ADV={HOME_ADV}")

            try:
                # Run ELO model with these parameters
                delta_df, player_FE = run_step2_ELO(player_df, games_df, K=K, HOME_ADV=HOME_ADV)

                # Run forecasting models
                mlogit_model, mlogit_pred = run_step3_multinomial(delta_df, games_df)
                probit_model, probit_pred = run_step3_ordered_probit(delta_df, games_df)

                # Calculate accuracies
                mlogit_accuracy = (mlogit_pred["pred_class"] == mlogit_pred["actual_class"]).mean()
                probit_accuracy = (probit_pred["pred_class"] == probit_pred["actual_class"]).mean()

                # Store results
                results.append({
                    'K': K,
                    'HOME_ADV': HOME_ADV,
                    'Multinomial_Accuracy': mlogit_accuracy,
                    'Probit_Accuracy': probit_accuracy
                })

                print(f"  Multinomial Logit: {mlogit_accuracy:.4f} ({mlogit_accuracy*100:.2f}%)")
                print(f"  Ordered Probit:    {probit_accuracy:.4f} ({probit_accuracy*100:.2f}%)")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results.append({
                    'K': K,
                    'HOME_ADV': HOME_ADV,
                    'Multinomial_Accuracy': np.nan,
                    'Probit_Accuracy': np.nan
                })

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"Grid search completed in {elapsed:.1f} seconds")
    print("="*60)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Find best parameters
    if model_type == 'multinomial':
        best_idx = results_df['Multinomial_Accuracy'].idxmax()
        accuracy_col = 'Multinomial_Accuracy'
    else:
        best_idx = results_df['Probit_Accuracy'].idxmax()
        accuracy_col = 'Probit_Accuracy'

    best_row = results_df.loc[best_idx]
    best_params = {
        'K': best_row['K'],
        'HOME_ADV': best_row['HOME_ADV'],
        'Multinomial_Accuracy': best_row['Multinomial_Accuracy'],
        'Probit_Accuracy': best_row['Probit_Accuracy']
    }

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nBest parameters (optimizing for {model_type.upper()}):")
    print(f"  K = {best_params['K']}")
    print(f"  HOME_ADV = {best_params['HOME_ADV']}")
    print(f"  Multinomial Logit Accuracy: {best_params['Multinomial_Accuracy']:.4f} ({best_params['Multinomial_Accuracy']*100:.2f}%)")
    print(f"  Ordered Probit Accuracy:    {best_params['Probit_Accuracy']:.4f} ({best_params['Probit_Accuracy']*100:.2f}%)")

    # Show top 5 combinations
    print(f"\nTop 5 combinations by {model_type.upper()} accuracy:")
    top5 = results_df.nlargest(5, accuracy_col)[['K', 'HOME_ADV', 'Multinomial_Accuracy', 'Probit_Accuracy']]
    print(top5.to_string(index=False))

    return results_df, best_params


def main():
    """
    Main function - customize your grid search parameters here
    """

    # Define the grid
    # K: How quickly ELO ratings update (higher = more reactive to recent games)
    k_values = [10, 15, 20, 25, 30, 40]

    # HOME_ADV: Home advantage in ELO points (higher = bigger home advantage)
    home_adv_values = [0, 10, 25, 50, 75, 100]

    # Which model to optimize for ('multinomial' or 'probit')
    model_type = 'multinomial'

    # Run grid search
    results_df, best_params = grid_search(k_values, home_adv_values, model_type)

    # Save results to CSV
    output_file = 'grid_search_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

    return results_df, best_params


if __name__ == "__main__":
    results_df, best_params = main()
