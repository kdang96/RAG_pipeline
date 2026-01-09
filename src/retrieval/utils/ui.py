import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def visualise_metrics(eval_metrics) -> Figure:
    # Create bar chart with matplotlib
    fig, ax = plt.subplots(figsize=(4, 3))
    bar = ax.bar(
        eval_metrics.keys(),
        eval_metrics.values(),
        color=["#4CAF50", "#2196F3", "#FFC107"],
    )
    ax.set_ylim([0, 1])
    ax.bar_label(bar)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics")

    return fig