"""
Model Evaluation and Comparison Utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelEvaluator:
    def __init__(self, models_folder="../models"):
        self.models_folder = models_folder
        self.results_df = None
        self.detailed_results = {}  # Store detailed grid search results
        
    def load_all_results(self):
        """Load results from all trained models"""
        results = []
        self.detailed_results = {}
        
        for folder_name in os.listdir(self.models_folder):
            if folder_name == 'scalers':
                continue
                
            results_file = f"{self.models_folder}/{folder_name}/results.json"
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        result = json.load(f)
                    
                    # Flatten the results
                    flat_result = {
                        'config_name': result['config_name'],
                        'model_type': result['model_type'],
                        'train_mae': result['train_metrics']['MAE'],
                        'train_mse': result['train_metrics']['MSE'],
                        'train_rmse': result['train_metrics']['RMSE'],
                        'train_r2': result['train_metrics']['R2'],
                        'test_mae': result['test_metrics']['MAE'],
                        'test_mse': result['test_metrics']['MSE'],
                        'test_rmse': result['test_metrics']['RMSE'],
                        'test_r2': result['test_metrics']['R2'],
                        'trained_at': result['trained_at']
                    }
                    
                    # Add model-specific info
                    if result['model_type'] == 'XGBoost':
                        flat_result['best_cv_score'] = result['best_cv_score']
                        flat_result['feature_count'] = result['feature_count']
                        flat_result['best_params'] = str(result['best_params'])
                        flat_result['cv_combinations'] = result.get('total_cv_combinations', 0)
                        
                        # Load detailed CV results if available
                        cv_file = f"{self.models_folder}/{folder_name}/cv_results_full.csv"
                        if os.path.exists(cv_file):
                            self.detailed_results[f"{folder_name}_cv"] = pd.read_csv(cv_file)
                            
                    else:  # LSTM
                        flat_result['best_val_loss'] = result['best_val_loss']
                        flat_result['total_params'] = result['model_params']['total_params']
                        flat_result['training_epochs'] = result['training_epochs']
                        flat_result['best_params'] = str(result['best_params'])
                        flat_result['grid_combinations'] = result.get('grid_search_combinations', 0)
                        
                        # Load detailed grid search results if available
                        grid_file = f"{self.models_folder}/{folder_name}/grid_search_results_full.csv"
                        if os.path.exists(grid_file):
                            self.detailed_results[f"{folder_name}_grid"] = pd.read_csv(grid_file)
                    
                    results.append(flat_result)
                    
                except Exception as e:
                    print(f"Error loading {results_file}: {e}")
        
        self.results_df = pd.DataFrame(results)
        print(f"Loaded results for {len(results)} models")
        print(f"Detailed results loaded for {len(self.detailed_results)} model searches")
        return self
    
    def create_comparison_table(self):
        """Create comparison table of all models"""
        if self.results_df is None:
            self.load_all_results()
        
        # Key metrics comparison
        comparison_cols = ['config_name', 'model_type', 'test_mae', 'test_rmse', 'test_r2']
        comparison_df = self.results_df[comparison_cols].copy()
        
        # Sort by test RMSE (lower is better)
        comparison_df = comparison_df.sort_values('test_rmse')
        
        print("=== MODEL COMPARISON (sorted by Test RMSE) ===")
        print(comparison_df.to_string(index=False, float_format='%.6f'))
        
        return comparison_df
    
    def plot_performance_comparison(self):
        """Plot performance comparison across models"""
        if self.results_df is None:
            self.load_all_results()
        
        # Prepare data for plotting
        df_plot = self.results_df.copy()
        df_plot['model_config'] = df_plot['config_name'] + '_' + df_plot['model_type']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Test RMSE comparison
        df_sorted = df_plot.sort_values('test_rmse')
        colors = ['red' if mt == 'XGBoost' else 'blue' for mt in df_sorted['model_type']]
        axes[0,0].bar(range(len(df_sorted)), df_sorted['test_rmse'], color=colors, alpha=0.7)
        axes[0,0].set_title('Test RMSE Comparison')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].set_xticks(range(len(df_sorted)))
        axes[0,0].set_xticklabels(df_sorted['model_config'], rotation=45, ha='right')
        
        # Test R² comparison
        df_sorted = df_plot.sort_values('test_r2', ascending=False)
        colors = ['red' if mt == 'XGBoost' else 'blue' for mt in df_sorted['model_type']]
        axes[0,1].bar(range(len(df_sorted)), df_sorted['test_r2'], color=colors, alpha=0.7)
        axes[0,1].set_title('Test R² Comparison')
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('R²')
        axes[0,1].set_xticks(range(len(df_sorted)))
        axes[0,1].set_xticklabels(df_sorted['model_config'], rotation=45, ha='right')
        
        # Train vs Test RMSE
        axes[1,0].scatter(df_plot['train_rmse'], df_plot['test_rmse'], 
                         c=['red' if mt == 'XGBoost' else 'blue' for mt in df_plot['model_type']],
                         alpha=0.7, s=100)
        axes[1,0].plot([df_plot['train_rmse'].min(), df_plot['train_rmse'].max()],
                      [df_plot['train_rmse'].min(), df_plot['train_rmse'].max()], 'k--', alpha=0.5)
        axes[1,0].set_xlabel('Train RMSE')
        axes[1,0].set_ylabel('Test RMSE')
        axes[1,0].set_title('Train vs Test RMSE')
        
        # Model type comparison
        model_comparison = df_plot.groupby('model_type')[['test_mae', 'test_rmse', 'test_r2']].mean()
        x = np.arange(len(model_comparison.columns))
        width = 0.35
        
        axes[1,1].bar(x - width/2, model_comparison.loc['XGBoost'], width, label='XGBoost', alpha=0.7, color='red')
        axes[1,1].bar(x + width/2, model_comparison.loc['LSTM'], width, label='LSTM', alpha=0.7, color='blue')
        axes[1,1].set_xlabel('Metrics')
        axes[1,1].set_ylabel('Average Value')
        axes[1,1].set_title('Average Performance by Model Type')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(model_comparison.columns)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_config_performance(self):
        """Analyze performance by configuration"""
        if self.results_df is None:
            self.load_all_results()
        
        print("\\n=== PERFORMANCE BY CONFIGURATION ===")
        
        # Group by config
        config_stats = self.results_df.groupby('config_name').agg({
            'test_rmse': ['mean', 'std', 'min'],
            'test_r2': ['mean', 'std', 'max']
        }).round(6)
        
        config_stats.columns = ['RMSE_mean', 'RMSE_std', 'RMSE_min', 'R2_mean', 'R2_std', 'R2_max']
        config_stats = config_stats.sort_values('RMSE_mean')
        
        print(config_stats)
        
        # Best model per config
        print("\\n=== BEST MODEL PER CONFIGURATION ===")
        best_per_config = self.results_df.loc[self.results_df.groupby('config_name')['test_rmse'].idxmin()]
        print(best_per_config[['config_name', 'model_type', 'test_rmse', 'test_r2']].to_string(index=False))
        
        return config_stats, best_per_config
    
    def create_interactive_comparison(self):
        """Create interactive comparison plots"""
        if self.results_df is None:
            self.load_all_results()
        
        df = self.results_df.copy()
        df['model_config'] = df['config_name'] + '_' + df['model_type']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Test RMSE by Configuration', 'Test R² by Configuration',
                          'Train vs Test Performance', 'Model Type Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Test RMSE by config
        for model_type in df['model_type'].unique():
            data = df[df['model_type'] == model_type]
            fig.add_trace(
                go.Bar(
                    x=data['config_name'],
                    y=data['test_rmse'],
                    name=f'{model_type} RMSE',
                    text=data['test_rmse'].round(4),
                    textposition='outside'
                ),
                row=1, col=1
            )
        
        # Test R² by config
        for model_type in df['model_type'].unique():
            data = df[df['model_type'] == model_type]
            fig.add_trace(
                go.Bar(
                    x=data['config_name'],
                    y=data['test_r2'],
                    name=f'{model_type} R²',
                    text=data['test_r2'].round(4),
                    textposition='outside'
                ),
                row=1, col=2
            )
        
        # Train vs Test scatter
        for model_type in df['model_type'].unique():
            data = df[df['model_type'] == model_type]
            fig.add_trace(
                go.Scatter(
                    x=data['train_rmse'],
                    y=data['test_rmse'],
                    mode='markers',
                    name=f'{model_type}',
                    text=data['config_name'],
                    marker=dict(size=10)
                ),
                row=2, col=1
            )
        
        # Model type box plot
        for metric in ['test_mae', 'test_rmse']:
            for model_type in df['model_type'].unique():
                data = df[df['model_type'] == model_type]
                fig.add_trace(
                    go.Box(
                        y=data[metric],
                        name=f'{model_type} {metric.upper()}',
                        boxpoints='all'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            title='Model Performance Comparison Dashboard',
            showlegend=True
        )
        
        fig.show()
        
        return fig
    
    def save_comparison_report(self, output_file="model_comparison_report.html"):
        """Save comprehensive comparison report"""
        if self.results_df is None:
            self.load_all_results()
        
        # Create comprehensive report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Water Level Prediction Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h1>Water Level Prediction Model Comparison Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>Total models evaluated: {len(self.results_df)}</p>
            <p>Configurations tested: {self.results_df['config_name'].nunique()}</p>
            <p>Model types: {', '.join(self.results_df['model_type'].unique())}</p>
            
            <h2>Model Performance Comparison</h2>
        """
        
        # Add comparison table
        comparison_df = self.create_comparison_table()
        html_content += comparison_df.to_html(index=False, classes='comparison-table')
        
        # Add best performers
        best_overall = self.results_df.loc[self.results_df['test_rmse'].idxmin()]
        html_content += f"""
            <h2>Best Performing Models</h2>
            <h3>Best Overall (Lowest Test RMSE)</h3>
            <p><strong>{best_overall['config_name']} - {best_overall['model_type']}</strong></p>
            <p>Test RMSE: {best_overall['test_rmse']:.6f}</p>
            <p>Test R²: {best_overall['test_r2']:.6f}</p>
        """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        report_path = f"{self.models_folder}/{output_file}"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Comparison report saved to: {report_path}")
        
        return report_path

    def analyze_detailed_grid_search(self, config_name=None, model_type=None):
        """Analyze detailed grid search results"""
        
        print("\n" + "="*60)
        print("DETAILED GRID SEARCH ANALYSIS")
        print("="*60)
        
        if not self.detailed_results:
            print("No detailed grid search results available.")
            return None
        
        analysis_results = {}
        
        for key, detailed_df in self.detailed_results.items():
            folder_name = key.replace('_cv', '').replace('_grid', '')
            
            # Filter by criteria if specified
            if config_name and config_name not in folder_name:
                continue
            if model_type and model_type.lower() not in folder_name.lower():
                continue
            
            print(f"\n🔍 Analysis for {folder_name}:")
            print(f"   Total combinations tested: {len(detailed_df)}")
            
            if 'cv' in key:  # XGBoost CV results
                score_col = 'mean_test_score'
                if score_col in detailed_df.columns:
                    best_score = detailed_df[score_col].max()
                    worst_score = detailed_df[score_col].min()
                    print(f"   Best CV score: {best_score:.6f}")
                    print(f"   Worst CV score: {worst_score:.6f}")
                    print(f"   Score range: {best_score - worst_score:.6f}")
                    
                    # Top 5 combinations
                    top_5 = detailed_df.nlargest(5, score_col)
                    print(f"   Top 5 parameter combinations:")
                    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                        params = eval(row['params']) if isinstance(row['params'], str) else row['params']
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        print(f"     {idx}. Score: {row[score_col]:.6f} | {param_str}")
                
            elif 'grid' in key:  # LSTM grid results
                score_col = 'val_loss'
                if score_col in detailed_df.columns:
                    best_score = detailed_df[score_col].min()  # Lower is better for loss
                    worst_score = detailed_df[score_col].max()
                    print(f"   Best validation loss: {best_score:.6f}")
                    print(f"   Worst validation loss: {worst_score:.6f}")
                    print(f"   Loss range: {worst_score - best_score:.6f}")
                    
                    # Top 5 combinations
                    top_5 = detailed_df.nsmallest(5, score_col)
                    print(f"   Top 5 parameter combinations:")
                    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                        params = eval(row['params']) if isinstance(row['params'], str) else row['params']
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        print(f"     {idx}. Loss: {row[score_col]:.6f} | {param_str}")
            
            analysis_results[folder_name] = detailed_df
        
        return analysis_results
    
    def save_detailed_analysis_report(self, output_file="detailed_grid_search_analysis.html"):
        """Save detailed grid search analysis report"""
        
        if not self.detailed_results:
            print("No detailed results to analyze")
            return None
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detailed Grid Search Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Detailed Grid Search Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>Total detailed analyses available: {len(self.detailed_results)}</p>
        """
        
        for key, detailed_df in self.detailed_results.items():
            folder_name = key.replace('_cv', '').replace('_grid', '')
            
            html_content += f"""
            <div class="section">
                <h3>{folder_name.upper()} - Grid Search Results</h3>
                <p>Total combinations tested: {len(detailed_df)}</p>
            """
            
            # Add top 10 results table
            if 'cv' in key and 'mean_test_score' in detailed_df.columns:
                top_results = detailed_df.nlargest(10, 'mean_test_score')
                html_content += "<h4>Top 10 CV Results (by mean_test_score)</h4>"
            elif 'grid' in key and 'val_loss' in detailed_df.columns:
                top_results = detailed_df.nsmallest(10, 'val_loss')
                html_content += "<h4>Top 10 Grid Results (by val_loss)</h4>"
            else:
                top_results = detailed_df.head(10)
                html_content += "<h4>First 10 Results</h4>"
            
            html_content += top_results.to_html(index=False, classes='results-table')
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        report_path = f"{self.models_folder}/{output_file}"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Detailed analysis report saved to: {report_path}")
        return report_path

def compare_all_models(models_folder="../models"):
    """
    Complete model comparison pipeline
    """
    evaluator = ModelEvaluator(models_folder)
    
    # Load and compare results
    evaluator.load_all_results()
    
    if evaluator.results_df.empty:
        print("No model results found!")
        return None
    
    # Create comparisons
    print("\\n" + "="*60)
    print("WATER LEVEL PREDICTION MODEL COMPARISON")
    print("="*60)
    
    comparison_df = evaluator.create_comparison_table()
    
    print("\\n" + "="*60)
    config_stats, best_per_config = evaluator.analyze_config_performance()
    
    # Save report
    evaluator.save_comparison_report()
    
    return evaluator

if __name__ == "__main__":
    # Example usage
    evaluator = compare_all_models()