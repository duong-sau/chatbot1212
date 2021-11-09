# Todo
# -> demo với bộ câu trên google group -> pdf
# -> demo online
# -> bên trái bên phải đều có thể T5
# ->
# ->

import matplotlib.pyplot as plt
from threading import Thread


def plot():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    plt.show()


def main():
    thread = Thread(target=plot)
    thread.setDaemon(True)
    thread.start()
    print('Done')


if __name__ == '__main__':
    main()
