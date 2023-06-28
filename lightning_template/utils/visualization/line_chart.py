import matplotlib.pyplot as plt

from .generate_color import ncolors


def draw_line_chart(
    data,
    std_data=None,
    upper_data=None,
    lower_data=None,
    title=None,
    x_label=None,
    y_label=None,
    x_ticks=None,
    y_ticks=None,
    save_path=None,
):
    if std_data is not None:
        assert upper_data is None and lower_data is None
        upper_data = data + std_data
        lower_data = data - std_data

    colors = ncolors(len(data))

    for i, name in enumerate(data):
        plt.plot(data.index, data[name], color=colors[i], label=name, linewidth=3.5)
        if upper_data is not None and lower_data is not None:
            plt.fill_between(
                data.index,
                upper_data[name],
                lower_data[name],
                color=colors[i],
                alpha=0.2,
            )

    plt.legend(loc="best")

    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_ticks is not None:
        plt.xticks(x_ticks)
    if y_ticks is not None:
        plt.yticks(y_ticks)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
