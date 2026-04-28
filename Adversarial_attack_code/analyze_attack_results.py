"""
Analyze adversarial attack results and generate comprehensive visualizations
Parses attack_results_log.txt and creates comparison graphs
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def parse_log_file(log_path="attack_results_log.txt"):
    """Parse the attack results log file"""
    
    results = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Split by "Processing:" to get each attack experiment
    # This keeps the header and results together
    experiments = re.split(r'(?=Processing:)', content)
    
    for experiment in experiments:
        # Look for Processing: lines to get the image name
        proc_match = re.search(r'Processing:\s*(\w+)\s*\(', experiment)
        # Look for Attack: line to get the actual attack method
        attack_match = re.search(r'^Attack:\s*(\w+)', experiment, re.MULTILINE)
        
        if proc_match and attack_match:
            image_name = proc_match.group(1)
            attack_name = attack_match.group(1)
            
            # Extract metrics from this experiment
            result = {
                'image': image_name,
                'attack': attack_name,
                'orig_detections': None,
                'adv_detections': None,
                'orig_confidence': None,
                'adv_confidence': None,
                'source_suppressed': None,
                'target_detected': None,
                'attack_successful': None
            }
            
            # Original detections
            match = re.search(r'Original detections:\s*(\d+)', experiment)
            if match:
                result['orig_detections'] = int(match.group(1))
            
            # Adversarial detections
            match = re.search(r'Adversarial detections:\s*(\d+)', experiment)
            if match:
                result['adv_detections'] = int(match.group(1))
            
            # Original confidence
            match = re.search(r'Original max confidence:\s*([\d.]+)', experiment)
            if match:
                result['orig_confidence'] = float(match.group(1))
            
            # Adversarial confidence (may be missing if confidence is 0)
            match = re.search(r'Adversarial max confidence:\s*([\d.]+)', experiment)
            if match:
                result['adv_confidence'] = float(match.group(1))
            else:
                # If not found, set to 0 (means no detections)
                result['adv_confidence'] = 0.0
            
            # Source suppressed
            match = re.search(r'Source class suppressed:\s*(True|False)', experiment)
            if match:
                result['source_suppressed'] = match.group(1) == 'True'
            
            # Target detected
            match = re.search(r'Target class detected:\s*(True|False)', experiment)
            if match:
                result['target_detected'] = match.group(1) == 'True'
            
            # Attack successful
            match = re.search(r'Attack successful:\s*(True|False)', experiment)
            if match:
                result['attack_successful'] = match.group(1) == 'True'
            
            # Only add if we found actual data
            if result['orig_detections'] is not None:
                results.append(result)
    
    return pd.DataFrame(results)

def plot_attack_success_rates(df, output_dir="analysis_plots"):
    """Plot overall attack success rates"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Figure 1: Success rate by attack method
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Overall success
    success_by_attack = df.groupby('attack')['attack_successful'].mean() * 100
    axes[0].bar(success_by_attack.index, success_by_attack.values, 
                color='steelblue', edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title('Overall Attack Success Rate', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 105])
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(success_by_attack.values):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Source suppression
    source_by_attack = df.groupby('attack')['source_suppressed'].mean() * 100
    axes[1].bar(source_by_attack.index, source_by_attack.values, 
                color='coral', edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Success Rate (%)', fontsize=12)
    axes[1].set_title('Source Class Suppression Rate', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 105])
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(source_by_attack.values):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Target detection
    target_by_attack = df.groupby('attack')['target_detected'].mean() * 100
    axes[2].bar(target_by_attack.index, target_by_attack.values, 
                color='lightgreen', edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Success Rate (%)', fontsize=12)
    axes[2].set_title('Target Class Detection Rate', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 105])
    axes[2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(target_by_attack.values):
        axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attack_success_rates.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/attack_success_rates.png")
    plt.close()

def plot_confidence_reduction(df, output_dir="analysis_plots"):
    """Plot confidence score reduction"""
    
    # Calculate confidence reduction
    df['confidence_reduction'] = df['orig_confidence'] - df['adv_confidence']
    df['confidence_reduction_pct'] = (df['confidence_reduction'] / df['orig_confidence']) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average confidence by attack
    conf_data = df.groupby('attack').agg({
        'orig_confidence': 'mean',
        'adv_confidence': 'mean'
    }).reset_index()
    
    x = np.arange(len(conf_data))
    width = 0.35
    
    axes[0].bar(x - width/2, conf_data['orig_confidence'], width, 
                label='Original', color='darkred', edgecolor='black', linewidth=1.5)
    axes[0].bar(x + width/2, conf_data['adv_confidence'], width, 
                label='Adversarial', color='lightcoral', edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Confidence Score', fontsize=12)
    axes[0].set_title('Source Class Confidence Before/After Attack', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conf_data['attack'], rotation=45)
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    
    # Confidence reduction percentage
    reduction_by_attack = df.groupby('attack')['confidence_reduction_pct'].mean()
    axes[1].bar(reduction_by_attack.index, reduction_by_attack.values, 
                color='purple', edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Confidence Reduction (%)', fontsize=12)
    axes[1].set_title('Average Confidence Reduction', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for i, v in enumerate(reduction_by_attack.values):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confidence_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/confidence_reduction.png")
    plt.close()

def plot_heatmap_by_image(df, output_dir="analysis_plots"):
    """Plot heatmap showing success rate for each attack on each image"""
    
    # Create pivot table for overall success
    pivot_overall = df.pivot_table(
        values='attack_successful', 
        index='image', 
        columns='attack', 
        aggfunc='mean'
    ) * 100
    
    # Create pivot table for source suppression
    pivot_source = df.pivot_table(
        values='source_suppressed', 
        index='image', 
        columns='attack', 
        aggfunc='mean'
    ) * 100
    
    # Create pivot table for target detection
    pivot_target = df.pivot_table(
        values='target_detected', 
        index='image', 
        columns='attack', 
        aggfunc='mean'
    ) * 100
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Overall success heatmap
    sns.heatmap(pivot_overall, annot=True, fmt='.0f', cmap='RdYlGn', 
                vmin=0, vmax=100, ax=axes[0], cbar_kws={'label': 'Success %'},
                linewidths=0.5, linecolor='black')
    axes[0].set_title('Overall Attack Success by Image', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Attack Method', fontsize=12)
    axes[0].set_ylabel('Source Image', fontsize=12)
    
    # Source suppression heatmap
    sns.heatmap(pivot_source, annot=True, fmt='.0f', cmap='Oranges', 
                vmin=0, vmax=100, ax=axes[1], cbar_kws={'label': 'Success %'},
                linewidths=0.5, linecolor='black')
    axes[1].set_title('Source Suppression by Image', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Attack Method', fontsize=12)
    axes[1].set_ylabel('Source Image', fontsize=12)
    
    # Target detection heatmap
    sns.heatmap(pivot_target, annot=True, fmt='.0f', cmap='Greens', 
                vmin=0, vmax=100, ax=axes[2], cbar_kws={'label': 'Success %'},
                linewidths=0.5, linecolor='black')
    axes[2].set_title('Target Detection by Image', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Attack Method', fontsize=12)
    axes[2].set_ylabel('Source Image', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/success_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/success_heatmap.png")
    plt.close()

def plot_detection_count_changes(df, output_dir="analysis_plots"):
    """Plot how detection counts change after attack"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    attacks = df['attack'].unique()
    
    for idx, attack in enumerate(attacks):
        attack_data = df[df['attack'] == attack]
        
        x = np.arange(len(attack_data))
        width = 0.35
        
        axes[idx].bar(x - width/2, attack_data['orig_detections'], width, 
                      label='Original', color='darkblue', edgecolor='black')
        axes[idx].bar(x + width/2, attack_data['adv_detections'], width, 
                      label='Adversarial', color='lightblue', edgecolor='black')
        axes[idx].set_title(f'{attack.upper()} Attack', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Detection Count')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(attack_data['image'], rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].set_ylim([0, max(attack_data['orig_detections'].max(), 
                                    attack_data['adv_detections'].max()) + 1])
    
    plt.suptitle('Source Class Detection Count Changes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detection_count_changes.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/detection_count_changes.png")
    plt.close()

def plot_image_difficulty(df, output_dir="analysis_plots"):
    """Analyze which images are hardest to fool"""
    
    # Average success rate per image
    success_by_image = df.groupby('image').agg({
        'attack_successful': 'mean',
        'source_suppressed': 'mean',
        'target_detected': 'mean'
    }) * 100
    
    success_by_image = success_by_image.sort_values('attack_successful')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(success_by_image))
    width = 0.25
    
    ax.bar(x - width, success_by_image['attack_successful'], width, 
           label='Overall Success', color='steelblue', edgecolor='black')
    ax.bar(x, success_by_image['source_suppressed'], width, 
           label='Source Suppressed', color='coral', edgecolor='black')
    ax.bar(x + width, success_by_image['target_detected'], width, 
           label='Target Detected', color='lightgreen', edgecolor='black')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Attack Success by Source Image (Averaged Across All Attacks)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(success_by_image.index, rotation=45)
    ax.legend()
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/image_difficulty.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/image_difficulty.png")
    plt.close()

def generate_summary_report(df, output_dir="analysis_plots"):
    """Generate a text summary report"""
    
    report_path = f'{output_dir}/summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ADVERSARIAL ATTACK ANALYSIS SUMMARY REPORT\n")
        f.write("Target Class: HORSE\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Number of attacks: {df['attack'].nunique()}\n")
        f.write(f"Number of source images: {df['image'].nunique()}\n")
        f.write(f"Overall success rate: {df['attack_successful'].mean()*100:.1f}%\n")
        f.write(f"Source suppression rate: {df['source_suppressed'].mean()*100:.1f}%\n")
        f.write(f"Target detection rate: {df['target_detected'].mean()*100:.1f}%\n\n")
        
        # Best performing attacks
        f.write("ATTACK PERFORMANCE RANKING\n")
        f.write("-"*70 + "\n")
        attack_perf = df.groupby('attack').agg({
            'attack_successful': 'mean',
            'source_suppressed': 'mean',
            'target_detected': 'mean'
        }).sort_values('attack_successful', ascending=False) * 100
        
        for idx, (attack, row) in enumerate(attack_perf.iterrows(), 1):
            f.write(f"{idx}. {attack.upper()}\n")
            f.write(f"   Overall Success: {row['attack_successful']:.1f}%\n")
            f.write(f"   Source Suppressed: {row['source_suppressed']:.1f}%\n")
            f.write(f"   Target Detected: {row['target_detected']:.1f}%\n")
        f.write("\n")
        
        # Hardest images to fool
        f.write("IMAGE DIFFICULTY RANKING (Hardest to Easiest)\n")
        f.write("-"*70 + "\n")
        image_diff = df.groupby('image')['attack_successful'].mean().sort_values() * 100
        
        for idx, (image, success) in enumerate(image_diff.items(), 1):
            f.write(f"{idx}. {image.upper()}: {success:.1f}% success rate\n")
        f.write("\n")
        
        # Average confidence reduction
        f.write("CONFIDENCE REDUCTION BY ATTACK\n")
        f.write("-"*70 + "\n")
        df['conf_reduction_pct'] = ((df['orig_confidence'] - df['adv_confidence']) / 
                                     df['orig_confidence']) * 100
        conf_reduction = df.groupby('attack')['conf_reduction_pct'].mean().sort_values(ascending=False)
        
        for attack, reduction in conf_reduction.items():
            f.write(f"{attack.upper()}: {reduction:.1f}% average reduction\n")
        f.write("\n")
        
        # Best attack-image combinations
        f.write("TOP 10 MOST SUCCESSFUL ATTACK-IMAGE COMBINATIONS\n")
        f.write("-"*70 + "\n")
        successful = df[df['attack_successful'] == True]
        if len(successful) > 0:
            top_combos = successful.groupby(['attack', 'image']).size().sort_values(ascending=False).head(10)
            for idx, ((attack, image), count) in enumerate(top_combos.items(), 1):
                f.write(f"{idx}. {attack.upper()} on {image.upper()}\n")
        else:
            f.write("No successful attacks found.\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("End of report\n")
    
    print(f"Saved: {report_path}")
    
    # Also print to console
    with open(report_path, 'r') as f:
        print("\n" + f.read())

def main():
    print("Analyzing adversarial attack results...")
    print("="*70)
    
    # Parse log file
    df = parse_log_file()
    
    if len(df) == 0:
        print("ERROR: No data found in attack_results_log.txt")
        print("Make sure to run run_all_targeted_attacks.sh first!")
        return
    
    print(f"Found {len(df)} attack results")
    print(f"Attacks: {', '.join(df['attack'].unique())}")
    print(f"Images: {', '.join(df['image'].unique())}")
    print()
    
    # Create output directory
    output_dir = "analysis_plots"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate all plots
    print("Generating visualizations...")
    plot_attack_success_rates(df, output_dir)
    plot_confidence_reduction(df, output_dir)
    plot_heatmap_by_image(df, output_dir)
    plot_detection_count_changes(df, output_dir)
    plot_image_difficulty(df, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, output_dir)
    
    print("\n" + "="*70)
    print(f"Analysis complete! All plots saved in '{output_dir}/' folder")
    print("="*70)

if __name__ == "__main__":
    main()
