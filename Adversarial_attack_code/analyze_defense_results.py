"""
Analyze defense results and compare with attack results
Generates comprehensive visualizations showing defense effectiveness
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def parse_defense_log(log_path="defense_results_log.txt"):
    """Parse the defense results log file"""
    
    results = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Split by "Processing:" to get each defense experiment
    experiments = re.split(r'(?=Processing:)', content)
    
    for experiment in experiments:
        # Look for the image filename (e.g., "cat_adversarial_fgsm.png")
        file_match = re.search(r'Processing:\s*(\w+)_adversarial_(\w+)\.png', experiment)
        
        if file_match:
            image_name = file_match.group(1)
            attack_name = file_match.group(2)
            
            # Check if detection occurred after defense
            # Look for "Detected N 'classname'" pattern
            detection_match = re.search(r'Detected\s+(\d+)\s+[\'"]?(\w+)', experiment)
            no_detection_match = re.search(r'No\s+[\'"]?(\w+)[\'"]?\s+detected', experiment)
            
            detected_count = 0
            if detection_match:
                detected_count = int(detection_match.group(1))
            elif no_detection_match:
                detected_count = 0
            
            # Extract detection details if available
            detections = []
            confidence_pattern = r'#\d+:\s+confidence=([\d.]+)'
            for conf_match in re.finditer(confidence_pattern, experiment):
                detections.append(float(conf_match.group(1)))
            
            max_confidence = max(detections) if detections else 0.0
            
            result = {
                'image': image_name,
                'attack': attack_name,
                'detected_count': detected_count,
                'max_confidence': max_confidence,
                'defense_succeeded': detected_count > 0  # Defense succeeded if SOURCE object detected again
            }
            
            results.append(result)
    
    return pd.DataFrame(results)


def parse_attack_log(log_path="attack_results_log.txt"):
    """Parse the attack results log file (reuse from analyze_attack_results.py)"""
    
    results = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    experiments = re.split(r'(?=Processing:)', content)
    
    for experiment in experiments:
        proc_match = re.search(r'Processing:\s*(\w+)\s*\(', experiment)
        attack_match = re.search(r'^Attack:\s*(\w+)', experiment, re.MULTILINE)
        
        if proc_match and attack_match:
            image_name = proc_match.group(1)
            attack_name = attack_match.group(1)
            
            result = {
                'image': image_name,
                'attack': attack_name,
                'attack_successful': False
            }
            
            # Attack successful
            match = re.search(r'Attack successful:\s*(True|False)', experiment)
            if match:
                result['attack_successful'] = match.group(1) == 'True'
            
            # Only add if we found actual data
            match = re.search(r'Original detections:\s*(\d+)', experiment)
            if match:
                results.append(result)
    
    return pd.DataFrame(results)


def plot_attack_vs_defense(attack_df, defense_df, output_dir="defense_analysis"):
    """Compare attack success rate vs defense mitigation rate"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Merge dataframes
    merged = pd.merge(
        attack_df[['image', 'attack', 'attack_successful']],
        defense_df[['image', 'attack', 'defense_succeeded']],
        on=['image', 'attack'],
        how='outer'
    )
    
    # Fill NaN values
    merged['attack_successful'] = merged['attack_successful'].fillna(False)
    merged['defense_succeeded'] = merged['defense_succeeded'].fillna(False)
    
    # Calculate metrics
    # Defense is only relevant for successful attacks
    merged['defense_success'] = merged['attack_successful'] & merged['defense_succeeded']
    
    # Overall statistics by attack type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Attack success vs Defense mitigation by attack type
    attack_stats = merged.groupby('attack').agg({
        'attack_successful': 'mean',
        'defense_succeeded': lambda x: x[merged.loc[x.index, 'attack_successful']].mean() if merged.loc[x.index, 'attack_successful'].any() else 0,
        'defense_success': lambda x: x[merged.loc[x.index, 'attack_successful']].mean() if merged.loc[x.index, 'attack_successful'].any() else 0
    }).reset_index()
    
    attack_stats.columns = ['attack', 'attack_success_rate', 'defense_success_rate_temp', 'defense_success_rate']
    
    x = np.arange(len(attack_stats))
    width = 0.35
    
    ax = axes[0, 0]
    bars1 = ax.bar(x - width/2, attack_stats['attack_success_rate'] * 100, width,
                   label='Attack Success', color='darkred', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, attack_stats['defense_success_rate'] * 100, width,
                   label='Defense Success', color='darkgreen', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Attack Success vs Defense Mitigation by Attack Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attack_stats['attack'].str.upper(), rotation=45)
    ax.legend()
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Heatmap of defense effectiveness by image and attack
    ax = axes[0, 1]
    
    # Create pivot table for successful attacks that were mitigated
    pivot = merged[merged['attack_successful']].pivot_table(
        values='defense_success',
        index='image',
        columns='attack',
        aggfunc='mean'
    ) * 100
    
    if not pivot.empty:
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Defense Success %'},
                   linewidths=0.5, ax=ax, vmin=0, vmax=100)
        ax.set_title('Defense Success Rate by Image and Attack\n(Only for successful attacks)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Image')
    else:
        ax.text(0.5, 0.5, 'No successful attacks to defend against',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Defense Success Rate by Image and Attack', fontsize=12, fontweight='bold')
    
    # 3. Defense effectiveness by image
    ax = axes[1, 0]
    
    image_stats = merged[merged['attack_successful']].groupby('image').agg({
        'attack_successful': 'sum',
        'defense_success': 'sum'
    }).reset_index()
    
    image_stats['defense_success_rate'] = (image_stats['defense_success'] / 
                                            image_stats['attack_successful'] * 100)
    image_stats = image_stats.sort_values('defense_success_rate', ascending=True)
    
    bars = ax.barh(image_stats['image'].str.upper(), image_stats['defense_success_rate'],
                   color='steelblue', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Defense Success Rate (%)', fontsize=12)
    ax.set_title('Defense Effectiveness by Image\n(Mitigation of successful attacks)',
                fontsize=12, fontweight='bold')
    ax.set_xlim([0, 105])
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(image_stats.iterrows()):
        ax.text(row['defense_success_rate'] + 2, i,
               f"{row['defense_success_rate']:.0f}% ({int(row['defense_success'])}/{int(row['attack_successful'])})",
               va='center', fontsize=9)
    
    # 4. Overall summary pie chart
    ax = axes[1, 1]
    
    total_attacks = len(merged)
    successful_attacks = merged['attack_successful'].sum()
    failed_attacks = total_attacks - successful_attacks
    mitigated_attacks = merged['defense_success'].sum()
    persistent_attacks = successful_attacks - mitigated_attacks
    
    labels = [
        f'Failed Attacks\n{failed_attacks} ({failed_attacks/total_attacks*100:.1f}%)',
        f'Mitigated by Defense\n{int(mitigated_attacks)} ({mitigated_attacks/total_attacks*100:.1f}%)',
        f'Persistent Attacks\n{int(persistent_attacks)} ({persistent_attacks/total_attacks*100:.1f}%)'
    ]
    sizes = [failed_attacks, mitigated_attacks, persistent_attacks]
    colors = ['lightgray', 'lightgreen', 'darkred']
    explode = (0, 0.1, 0.05)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='', startangle=90, textprops={'fontsize': 10})
    ax.set_title('Overall Attack Outcomes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attack_vs_defense_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/attack_vs_defense_comparison.png")
    plt.close()
    
    return merged, attack_stats


def plot_detailed_analysis(merged_df, output_dir="defense_analysis"):
    """Additional detailed analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Stacked bar chart: attack outcomes
    ax = axes[0, 0]
    
    outcome_by_attack = merged_df.groupby('attack').apply(
        lambda x: pd.Series({
            'Failed': (~x['attack_successful']).sum(),
            'Mitigated': (x['defense_success']).sum(),
            'Persistent': (x['attack_successful'] & ~x['defense_success']).sum()
        })
    )
    
    outcome_by_attack.plot(kind='bar', stacked=True, ax=ax,
                          color=['lightgray', 'lightgreen', 'darkred'],
                          edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Experiments', fontsize=12)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_title('Attack Outcomes by Type', fontsize=14, fontweight='bold')
    ax.set_xticklabels([x.upper() for x in outcome_by_attack.index], rotation=45)
    ax.legend(title='Outcome', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Defense success rate sorted by effectiveness
    ax = axes[0, 1]
    
    defense_rate = merged_df[merged_df['attack_successful']].groupby('attack').agg({
        'defense_success': 'mean'
    }).reset_index()
    defense_rate.columns = ['attack', 'defense_success_rate']
    defense_rate = defense_rate.sort_values('defense_success_rate', ascending=True)
    
    bars = ax.barh(defense_rate['attack'].str.upper(), 
                   defense_rate['defense_success_rate'] * 100,
                   color=plt.cm.RdYlGn(defense_rate['defense_success_rate']),
                   edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Defense Success Rate (%)', fontsize=12)
    ax.set_title('Defense Mitigation Rate by Attack Type\n(When attack was initially successful)',
                fontsize=12, fontweight='bold')
    ax.set_xlim([0, 105])
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(defense_rate.iterrows()):
        ax.text(row['defense_success_rate'] * 100 + 2, i,
               f"{row['defense_success_rate']*100:.1f}%",
               va='center', fontsize=10, fontweight='bold')
    
    # 3. Attack persistence (attacks that survived defense)
    ax = axes[1, 0]
    
    persistent = merged_df[merged_df['attack_successful'] & ~merged_df['defense_success']]
    if len(persistent) > 0:
        persist_counts = persistent.groupby('attack').size().sort_values(ascending=True)
        
        bars = ax.barh(persist_counts.index.str.upper(), persist_counts.values,
                      color='darkred', edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Number of Persistent Attacks', fontsize=12)
        ax.set_title('Attacks That Survived Defense\n(Most resistant to defenses)',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (attack, count) in enumerate(persist_counts.items()):
            ax.text(count + 0.1, i, str(int(count)), va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'All successful attacks were mitigated!',
               ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
        ax.set_title('Attacks That Survived Defense', fontsize=12, fontweight='bold')
    
    # 4. Image vulnerability (total successful + persistent)
    ax = axes[1, 1]
    
    image_vuln = merged_df.groupby('image').apply(
        lambda x: pd.Series({
            'total_attacks': len(x),
            'successful': x['attack_successful'].sum(),
            'persistent': (x['attack_successful'] & ~x['defense_success']).sum()
        })
    ).sort_values('persistent', ascending=False)
    
    x = np.arange(len(image_vuln))
    width = 0.35
    
    ax.bar(x - width/2, image_vuln['successful'], width,
           label='Initial Success', color='orange', edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, image_vuln['persistent'], width,
           label='After Defense', color='darkred', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Number of Successful Attacks', fontsize=12)
    ax.set_title('Attack Success: Before and After Defense', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(image_vuln.index.str.upper(), rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_defense_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/detailed_defense_analysis.png")
    plt.close()


def generate_summary_report(attack_df, defense_df, merged_df, output_dir="defense_analysis"):
    """Generate a text summary report"""
    
    report_path = f"{output_dir}/defense_summary_report.txt"
    
    total_experiments = len(merged_df)
    successful_attacks = merged_df['attack_successful'].sum()
    defense_success = merged_df['defense_success'].sum()
    persistent_attacks = successful_attacks - defense_success
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ADVERSARIAL DEFENSE EVALUATION SUMMARY\n")
        f.write("Defense Techniques: Denoise + Resize + JPEG Compression\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total attack experiments: {total_experiments}\n")
        f.write(f"Initially successful attacks: {int(successful_attacks)} "
                f"({successful_attacks/total_experiments*100:.1f}%)\n")
        f.write(f"Attacks mitigated by defense: {int(defense_success)} "
                f"({defense_success/total_experiments*100:.1f}%)\n")
        f.write(f"Persistent attacks (survived defense): {int(persistent_attacks)} "
                f"({persistent_attacks/total_experiments*100:.1f}%)\n")
        
        if successful_attacks > 0:
            f.write(f"\nDefense mitigation rate: {defense_success/successful_attacks*100:.1f}% "
                   f"(of successful attacks)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("DEFENSE EFFECTIVENESS BY ATTACK TYPE\n")
        f.write("=" * 70 + "\n\n")
        
        for attack in merged_df['attack'].unique():
            attack_data = merged_df[merged_df['attack'] == attack]
            attack_success = attack_data['attack_successful'].sum()
            defense_succ = attack_data['defense_success'].sum()
            
            f.write(f"{attack.upper()}\n")
            f.write(f"  Initially successful: {int(attack_success)}/{len(attack_data)}\n")
            
            if attack_success > 0:
                f.write(f"  Mitigated by defense: {int(defense_succ)}/{int(attack_success)} "
                       f"({defense_succ/attack_success*100:.1f}%)\n")
                f.write(f"  Persistent: {int(attack_success - defense_succ)}\n")
            else:
                f.write(f"  No successful attacks to defend\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("DEFENSE EFFECTIVENESS BY IMAGE\n")
        f.write("=" * 70 + "\n\n")
        
        for image in sorted(merged_df['image'].unique()):
            image_data = merged_df[merged_df['image'] == image]
            attack_success = image_data['attack_successful'].sum()
            defense_succ = image_data['defense_success'].sum()
            
            f.write(f"{image.upper()}\n")
            f.write(f"  Initially successful attacks: {int(attack_success)}/{len(image_data)}\n")
            
            if attack_success > 0:
                f.write(f"  Mitigated by defense: {int(defense_succ)}/{int(attack_success)} "
                       f"({defense_succ/attack_success*100:.1f}%)\n")
                f.write(f"  Persistent attacks: {int(attack_success - defense_succ)}\n")
            else:
                f.write(f"  No successful attacks\n")
            f.write("\n")
        
        # Most resilient attacks (those that survived defense most often)
        f.write("=" * 70 + "\n")
        f.write("MOST RESILIENT ATTACK-IMAGE COMBINATIONS\n")
        f.write("(Attacks that survived defense preprocessing)\n")
        f.write("=" * 70 + "\n\n")
        
        persistent = merged_df[merged_df['attack_successful'] & ~merged_df['defense_success']]
        if len(persistent) > 0:
            for idx, row in persistent.iterrows():
                f.write(f"  {row['attack'].upper()} on {row['image'].upper()}\n")
        else:
            f.write("  None! All successful attacks were mitigated.\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("End of report\n")
        f.write("=" * 70 + "\n")
    
    print(f"Saved: {report_path}")
    
    # Print summary to console
    print("\n" + "=" * 70)
    with open(report_path, 'r') as f:
        print(f.read())


def main():
    print("Analyzing defense results...")
    print("=" * 70)
    
    # Parse both log files
    print("Parsing attack results...")
    attack_df = parse_attack_log("attack_results_log.txt")
    
    print("Parsing defense results...")
    defense_df = parse_defense_log("defense_results_log.txt")
    
    if len(attack_df) == 0:
        print("ERROR: No attack data found in attack_results_log.txt")
        return
    
    if len(defense_df) == 0:
        print("ERROR: No defense data found in defense_results_log.txt")
        print("Make sure to run run_all_defenses.sh first!")
        return
    
    print(f"Found {len(attack_df)} attack results")
    print(f"Found {len(defense_df)} defense results")
    print()
    
    # Create output directory
    output_dir = "defense_analysis"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate comparison plots
    print("Generating comparison visualizations...")
    merged_df, attack_stats = plot_attack_vs_defense(attack_df, defense_df, output_dir)
    plot_detailed_analysis(merged_df, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(attack_df, defense_df, merged_df, output_dir)
    
    print("\n" + "=" * 70)
    print(f"Defense analysis complete! All results saved in '{output_dir}/' folder")
    print("=" * 70)


if __name__ == "__main__":
    main()
