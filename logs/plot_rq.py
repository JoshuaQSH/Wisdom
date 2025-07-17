import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import math

color_Zheng = [ '#ECDFFF', '#D4C4EE', '#BDAADE', '#A690CF', '#8E78BF', '#8C75BC', '#6A579C']
color_tradblue = ['#aed0ee', '#88abda', '#6f94cd', '#5976ba', '#2e59a7', '#145ca0']
color_whiteblue = ['#d2dee5', '#bdcbd7', '#b7d3dd', '#99b4cc', '#82a4ca']

def plot_rq1_boxplot(files, file_name="rq1_accuracy_drop.pdf"):
    # Read and combine into one DataFrame
    dfs = []
    for dataset_name, file_path in files.items():
        df = pd.read_csv(file_path)
        df["Dataset"] = dataset_name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.columns = combined_df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Get the unique attribution methods to match with colors
    unique_methods = combined_df["attribution_method"].unique()
    palette = dict(zip(unique_methods, color_Zheng[:len(unique_methods)]))

    # Plot using the custom palette
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=combined_df,
        x="dataset",
        y="accuracy_drop",
        hue="attribution_method",
        palette=palette
    )
    # plt.title("Accuracy Drop Distribution by XAI Method and Dataset")
    plt.ylabel("Accuracy Drop", fontsize=16)
    # plt.xlabel("Dataset", fontsize=16)
    plt.xlabel("")
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(title="XAI Method", fontsize=16, title_fontsize=14)
    plt.tight_layout()
    plt.grid(True, axis='y')

    # plt.show()
    plt.savefig(file_name, dpi=1200, format='pdf', bbox_inches='tight')

def plot_rq1_barplot(files, file_name="rq1_accuracy_drop.pdf"):
    # Create a bar plot for each CSV file individually
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    for i, (dataset_name, file_path) in enumerate(files.items()):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        sns.barplot(
            data=df,
            x="attribution_method",
            hue="attribution_method",
            y="accuracy_drop",
            ax=axs[i],
            capsize=0.3,
            err_kws={'linewidth': 1.5},
            palette=color_Zheng[:df['attribution_method'].nunique()]
        )
        axs[i].set_title(f"Accuracy Drop - {dataset_name}", fontsize=18)
        # axs[i].set_xlabel("Attribution Method", fontsize=14)
        axs[i].set_xlabel("")
        axs[i].set_ylabel("Mean Accuracy Drop", fontsize=16)
        axs[i].tick_params(axis='x', labelsize=14)
        axs[i].tick_params(axis='y', labelsize=14)
        axs[i].grid(True, axis='y')

    plt.tight_layout()
    # plt.show()
    plt.savefig(file_name, dpi=1200, format='pdf', bbox_inches='tight')

def plot_rq3(file_path='RQ_3_Results - LeNet-MNIST.csv', save_path='RQ3_LeNet_MNIST_CoverageChange_Plot.pdf'):
    # Load the CSV file
    df = pd.read_csv(file_path)
    df = df[df['Method'] != 'NBC']
    df = df[df['Method'] != 'SNAC']

    # Inspect unique combinations of 'Attack' and 'Sample_Size'
    unique_combinations = df[['Attack', 'Sample_Size']].drop_duplicates().sort_values(by=['Sample_Size', 'Attack'])
    unique_combinations_list = [f"{row.Attack}_{row.Sample_Size}" for _, row in unique_combinations.iterrows()]
    
    # Create a 3x3 subplot for the first 9 unique (Attack, Sample_Size) combinations
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.3)
    font_properties = {'size': 14}

    # Limit to 9 subplots
    for idx, combination in enumerate(unique_combinations_list[:9]):  
        attack, sample_size = combination.split('_')
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        subset = df[(df['Attack'] == attack) & (df['Sample_Size'] == int(sample_size))]
        
        for label in subset['Method'].unique():
            method_data = subset[subset['Method'] == label]
            if label.lower() == 'wisdom':
                ax.plot(
                    method_data['Adv_Ratio'],
                    method_data['Coverage_Change'],
                    label=label,
                    linewidth=3.0,
                    linestyle='--'
                )
            else:
                ax.plot(
                    method_data['Adv_Ratio'],
                    method_data['Coverage_Change'],
                    label=label,
                    linewidth=1.5
                )

        ax.set_title(f'{attack}_{sample_size}', fontsize=14)
        ax.set_xlabel('Error Ratio', fontsize=12)
        ax.set_ylabel('Coverage Change', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

    # Add legend above all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=12, bbox_to_anchor=(0.5, 0.95))

    # Save the figure as a high-resolution PDF
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------
# Plot overall correlation coefficients per coverage method
# ------------------------------------------------------------------

def plot_rq4_correlation(cor_path, model_data_name):
    cor_df = pd.read_csv(cor_path)

    # Keep only the 'overall' rows
    cor_overall = cor_df[cor_df["test_type"] == "overall"]
    clean_df = cor_df[cor_df["test_type"] == "clean"].dropna(subset=["correlation_coefficient"])
    adv_df = cor_df[cor_df["test_type"] == "adversarial"].dropna(subset=["correlation_coefficient"])

    # One bar plot
    plt.figure()
    plt.bar(cor_overall["coverage_method"], cor_overall["correlation_coefficient"], color=color_Zheng[2])
    plt.xlabel("Coverage Method")
    plt.ylabel("Pearson r (Correlation Coefficient)")
    plt.title("Overall Correlation between Coverage and Output‑Impartiality\n({})".format(model_data_name))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig(f"rq4_overall_correlation_{model_data_name}.pdf", format='pdf', dpi=1200)
    plt.close()
    
    plt.figure(figsize=(10, 4))
    plt.bar(clean_df["coverage_method"], clean_df["correlation_coefficient"], color=color_Zheng[2])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Pearson r (clean)")
    plt.title("Correlation (clean subsets) — Coverage vs. Output‑Impartiality")
    plt.tight_layout()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig(f"rq4_clean_correlation_{model_data_name}.pdf", format='pdf', dpi=1200)
    plt.close()
    
    plt.figure(figsize=(10, 4))
    plt.bar(adv_df["coverage_method"], adv_df["correlation_coefficient"], color=color_Zheng[2])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Pearson r (adversarial)")
    plt.title("Correlation (adversarial subsets) — Coverage vs. Output‑Impartiality")
    plt.tight_layout()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig(f"rq4_adversarial_correlation_{model_data_name}.pdf", format='pdf', dpi=1200)
    plt.close()


# ------------------------------------------------------------------
# Plot coverage & impartiality curves for every coverage method
# ------------------------------------------------------------------
def plot_rq4_coverage_impartiality(cov_imp_path, model_data_name):
    cov_imp_df = pd.read_csv(cov_imp_path)

    df = pd.read_csv(cov_imp_path)

    methods = df["coverage_method"].unique()
    n = len(methods)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, method in zip(axes, methods):
        sub = df[df["coverage_method"] == method].sort_values("sample_size")
        ax.plot(sub["sample_size"], sub["coverage_score"], marker="o", label="Coverage")
        ax.plot(sub["sample_size"], sub["impartiality_score"], marker="x", label="Impartiality")
        ax.set_title(method)
        ax.set_xlabel("Sample size")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", linewidth=0.3)

    # Remove unused axes
    for ax in axes[len(methods):]:
        fig.delaxes(ax)

    # Put a single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Coverage vs. Output‑Impartiality across Methods (single figure)", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig(f"rq4_coverage_impartiality_curves_{model_data_name}.pdf", format='pdf', dpi=1200)
    plt.close()

def plot_rq4_pvalue(cor_path, model_data_name):
    df = pd.read_csv(cor_path)
    
    # Separate clean / adversarial subsets and drop missing p-values
    clean_p = df[df["test_type"] == "clean"].dropna(subset=["p_value"])
    adv_p   = df[df["test_type"] == "adversarial"].dropna(subset=["p_value"])

    # -----------------------------
    # Plot p-values (clean)
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.bar(clean_p["coverage_method"], clean_p["p_value"], color=color_Zheng[1])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("p‑value (clean)")
    plt.title("Statistical Significance — Clean Subsets (p‑values of Pearson r)")
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1, label="0.05 threshold")
    plt.legend()
    plt.tight_layout()
    plt.grid(linestyle='--', linewidth=0.5)    
    plt.savefig(f"rq4_pvalue_clean_{model_data_name}.pdf", format='pdf', dpi=1200)
    plt.close()

    # -----------------------------
    # Plot p-values (adversarial)
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.bar(adv_p["coverage_method"], adv_p["p_value"], color=color_Zheng[1])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("p‑value (adversarial)")
    plt.title("Statistical Significance — Adversarial Subsets (p‑values of Pearson r)")
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1, label="0.05 threshold")
    plt.legend()
    plt.tight_layout()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig(f"rq4_pvalue_adversarial_{model_data_name}.pdf", format='pdf', dpi=1200)
    plt.close()


if __name__ == "__main__":
    # Load all datasets
    rq_1_files = {
        "mnist_lenet": "./rq1_mnist_lenet_acc_drop.csv",
        "cifar10_lenet": "./rq1_cifar10_lenet_acc_drop.csv",
        "cifar10_vgg16": "./rq1_cifar10_vgg16_acc_drop.csv",
        "cifar10_resnet18": "./rq1_cifar10_resnet18_acc_drop.csv",
        "imagenet_resnet18": "./rq1_imagenet_resnet18_acc_drop.csv",
    }
    
    rq_3_list = [
        "rq3_results_mnist_lenet.csv",
        "rq3_results_cifar10_lenet.csv",
    ]
    rq_3_save_list = [
        "rq3_mnist_lenet_coverage_change_plot.pdf",
        "rq3_cifar10_lenet_coverage_change_plot.pdf",
    ]
    
    # rq_3_list = [
    #     'RQ_3_Results - LeNet-MNIST.csv',
    #     'RQ_3_Results - LeNet-CIFAR10.csv',
    #     'RQ_3_Results - VGG-CIFAR10.csv',
    #     'RQ_3_Results - ResNet-CIFAR10.csv',
    # ]
    # rq_3_save_list = [
    #     'RQ3_LeNet_MNIST_CoverageChange_Plot.pdf',
    #     'RQ3_LeNet_CIFAR10_CoverageChange_Plot.pdf',
    #     'RQ3_VGG_CIFAR10_CoverageChange_Plot.pdf',
    #     'RQ3_ResNet_CIFAR10_CoverageChange_Plot.pdf',
    # ]

    
    rq_4_cor_lists = ["rq4_correlation_mnist_lenet.csv", "rq4_correlation_cifar10_lenet.csv", "rq4_correlation_cifar10_vgg16.csv"]
    rq_4_cov_imp_lists = ["rq4_impartiality_mnist_lenet.csv", "rq4_impartiality_cifar10_lenet.csv", "rq4_impartiality_cifar10_vgg16.csv"]
    rq_4_tags = ["LeNet-MNIST", "LeNet-CIFAR10", "VGG16-CIFAR10"]
    
    
    # plot_rq1_barplot(rq_1_files, file_name="rq1_accuracy_drop_bar.pdf")
    plot_rq1_boxplot(rq_1_files, file_name="rq1_accuracy_drop_box.pdf")
    
    for i in range(0, len(rq_3_list)):
        plot_rq3(rq_3_list[i], rq_3_save_list[i])
        
    for i in range(len(rq_4_cor_lists)):
        plot_rq4_correlation(rq_4_cor_lists[i], rq_4_tags[i])
        plot_rq4_coverage_impartiality(rq_4_cov_imp_lists[i], rq_4_tags[i])
        plot_rq4_pvalue(rq_4_cor_lists[i], rq_4_tags[i])