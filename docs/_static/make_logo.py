import matplotlib.pyplot as plt

plt.rcParams["hatch.linewidth"] = 4

colors = ['#e4b388', '#2279ab', '#072044']
n = len(colors)

logos = {
    'logo': (2, 2),
    'logo_small' : (1, 1),
}

for name, size in logos.items():
    fig = plt.figure(figsize=size, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    for i in range(n):
        for j in range(i, n):
            plt.rcParams["hatch.color"] = colors[j]
            ax.add_patch(
                plt.Rectangle((j, n-i), 1, 1, fc=colors[i], ec=None, hatch=r'\\')
            )

    ax.set_ylim(1, n+1)
    ax.set_xlim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(f'{name}.png')
plt.show()

