"""Compare all model results and generate summary."""
import pandas as pd
import os
from pathlib import Path

models = ['mlp', 'lstm', 'gru', 'cnn1d', 'patchtst']
results = {}

print('=== 所有模型测试集性能对比 ===\n')

for model in models:
    csv_path = f'reports/tables/{model}_test_metrics.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        results[model] = df
        print(f'{model.upper()}:')
        print(df.to_string())
        print()

# Add ensemble results
ensemble_path = 'runs/ensemble/ensemble_test_metrics.csv'
if os.path.exists(ensemble_path):
    df = pd.read_csv(ensemble_path, index_col=0)
    # Rename index from 1,3,5 to h1,h3,h5 for consistency
    df.index = [f'h{i}' for i in df.index]
    results['ensemble'] = df
    print('ENSEMBLE:')
    print(df.to_string())
    print()

# Create comparison table
if results:
    all_results = []
    for model, df in results.items():
        df_with_model = df.copy()
        df_with_model['Model'] = model.upper()
        all_results.append(df_with_model)
    
    combined = pd.concat(all_results)
    
    # Select available columns
    available_cols = ['Model']
    for col in ['accuracy', 'f1', 'auc', 'brier', 'crps', 'threshold']:
        if col in combined.columns:
            available_cols.append(col)
    
    print('\n=== 模型性能对比汇总表 ===\n')
    print(combined[available_cols].round(4).to_string())
    
    # Save comparison
    output_path = 'reports/tables/all_models_comparison.csv'
    combined.to_csv(output_path)
    print(f'\n已保存对比表到: {output_path}')

