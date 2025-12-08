"""
Visualization and Analysis for Comparison Results
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def visualize_comparison():
    """Create visualizations of comparison results"""
    
    print("📊 Generating visualizations...")
    
    # Load results
    results_csv = Path('comparison_results.csv')
    summary_json = Path('comparison_summary.json')
    
    if not results_csv.exists() or not summary_json.exists():
        print("❌ Results files not found! Run comparison script first.")
        return
    
    # Load data
    results_df = pd.read_csv(results_csv)
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    methods = ['Traditional\nCV+KNN', 'YOLO v8+CNN']
    accuracies = [
        summary['traditional_cv_knn']['accuracy'] * 100,
        summary['yolo_cnn']['accuracy'] * 100
    ]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 100])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Inference Time Comparison
    ax2 = plt.subplot(2, 3, 2)
    times = [
        summary['traditional_cv_knn']['avg_time_ms'],
        summary['yolo_cnn']['avg_time_ms']
    ]
    bars = ax2.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Inference Time per Image', fontsize=13, fontweight='bold')
    
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. Correct Predictions Count
    ax3 = plt.subplot(2, 3, 3)
    correct_counts = [
        summary['traditional_cv_knn']['correct_predictions'],
        summary['yolo_cnn']['correct_predictions']
    ]
    total = summary['total_images_tested']
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, correct_counts, width, label='Correct', 
                    color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, [total - c for c in correct_counts], width, label='Incorrect', 
                    color='#F38181', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Results Distribution', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend(fontsize=10)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # 4. Per-Character Accuracy (if multiple images)
    ax4 = plt.subplot(2, 3, 4)
    results_df['traditional_char_correct'] = (results_df['traditional_pred'] == results_df['ground_truth']).astype(int)
    results_df['cnn_char_correct'] = (results_df['cnn_pred'] == results_df['ground_truth']).astype(int)
    
    ax4.hist([results_df['traditional_char_correct'], results_df['cnn_char_correct']], 
            label=['Traditional CV+KNN', 'YOLO v8+CNN'],
            color=['#FF6B6B', '#4ECDC4'], alpha=0.7, bins=2, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Prediction Result', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Prediction Distribution', fontsize=13, fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Incorrect', 'Correct'])
    ax4.legend(fontsize=10)
    
    # 5. Performance Metrics Table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = [
        ['Metric', 'Traditional CV+KNN', 'YOLO v8+CNN'],
        ['Accuracy', f"{summary['traditional_cv_knn']['accuracy']*100:.2f}%", 
         f"{summary['yolo_cnn']['accuracy']*100:.2f}%"],
        ['Correct/Total', 
         f"{summary['traditional_cv_knn']['correct_predictions']}/{summary['traditional_cv_knn']['total_predictions']}", 
         f"{summary['yolo_cnn']['correct_predictions']}/{summary['yolo_cnn']['total_predictions']}"],
        ['Avg Time', f"{summary['traditional_cv_knn']['avg_time_ms']:.2f}ms", 
         f"{summary['yolo_cnn']['avg_time_ms']:.2f}ms"],
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, 4):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
            table[(i, j)].set_text_props(weight='bold')
    
    ax5.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold', pad=20)
    
    # 6. Improvement Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    improvement = summary['improvement']['accuracy_diff_percent']
    winner = summary['improvement']['winner']
    
    text_content = f"""
    COMPARISON SUMMARY
    {'='*45}
    
    Dataset: LP-characters
    Total Images Tested: {summary['total_images_tested']}
    
    Winner: {winner}
    Improvement: {improvement:+.2f}%
    
    Traditional CV+KNN:
      • Uses HOG features + KNN classifier
      • Fast inference
      • Good baseline for comparison
    
    YOLO v8 + CNN:
      • Uses deep learning approach
      • Better feature extraction
      • More robust to variations
    """
    
    ax6.text(0.05, 0.95, text_content, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8, pad=1),
            fontweight='bold')
    
    plt.suptitle('Traditional CV+KNN vs YOLO v8+CNN Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig('comparison_visualization.png', dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: comparison_visualization.png")
    
    plt.show()


def print_detailed_analysis():
    """Print detailed analysis of results"""
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    results_csv = Path('comparison_results.csv')
    summary_json = Path('comparison_summary.json')
    
    if not results_csv.exists() or not summary_json.exists():
        print("❌ Results files not found!")
        return
    
    results_df = pd.read_csv(results_csv)
    with open(summary_json, 'r') as f:
        summary = json.load(f)
    
    print(f"\n📊 Total Images Tested: {summary['total_images_tested']}")
    
    # Traditional CV+KNN analysis
    print(f"\n🔍 TRADITIONAL CV+KNN:")
    print(f"   • Accuracy: {summary['traditional_cv_knn']['accuracy']*100:.2f}%")
    print(f"   • Correct Predictions: {summary['traditional_cv_knn']['correct_predictions']}/{summary['traditional_cv_knn']['total_predictions']}")
    print(f"   • Average Inference Time: {summary['traditional_cv_knn']['avg_time_ms']:.2f}ms")
    
    # YOLO v8 + CNN analysis
    print(f"\n🔍 YOLO v8 + CNN:")
    print(f"   • Accuracy: {summary['yolo_cnn']['accuracy']*100:.2f}%")
    print(f"   • Correct Predictions: {summary['yolo_cnn']['correct_predictions']}/{summary['yolo_cnn']['total_predictions']}")
    print(f"   • Average Inference Time: {summary['yolo_cnn']['avg_time_ms']:.2f}ms")
    
    # Comparison
    print(f"\n📈 COMPARISON:")
    acc_diff = summary['improvement']['accuracy_diff_percent']
    time_ratio = summary['yolo_cnn']['avg_time_ms'] / summary['traditional_cv_knn']['avg_time_ms']
    
    print(f"   • Accuracy Difference: {acc_diff:+.2f}%")
    print(f"   • Winner: {summary['improvement']['winner']}")
    print(f"   • Speed Ratio (CNN/Traditional): {time_ratio:.2f}x")
    
    if time_ratio > 1:
        print(f"     → Traditional is {time_ratio:.2f}x faster")
    else:
        print(f"     → CNN is {1/time_ratio:.2f}x faster")
    
    # Sample predictions
    print(f"\n📋 SAMPLE PREDICTIONS (first 5 images):")
    print("-" * 80)
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\nImage: {row['image']}")
        print(f"  Ground Truth: {row['ground_truth']}")
        print(f"  Traditional: {row['traditional_pred']} {'✅' if row['traditional_correct'] else '❌'}")
        print(f"  CNN:         {row['cnn_pred']} {'✅' if row['cnn_correct'] else '❌'}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        visualize_comparison()
        print_detailed_analysis()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
