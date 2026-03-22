import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path='metrics.csv'):
    try:
        # 1. Load the data
        df = pd.read_csv(csv_path)
        
        # Clean up column names just in case there are stray spaces
        df.columns = [c.strip() for c in df.columns]
        
        # 2. Setup the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # --- Plot 1: Training & Validation Loss ---
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', marker='o', linewidth=2)
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#d62728', marker='s', linewidth=2)
        
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss Value', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        # --- Plot 2: Validation Accuracy ---
        ax2.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', color='#2ca02c', marker='^', linewidth=2)
        
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        # 3. Final touches
        plt.tight_layout()
        plt.savefig('swinmath_metrics_plot.png', dpi=300)
        print(f"Successfully plotted metrics from {len(df)} epochs.")
        plt.show()

    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Make sure the file is in the same directory.")
    except KeyError as e:
        print(f"Error: Could not find column {e} in CSV. Check your headers!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    plot_metrics('metrics.csv')