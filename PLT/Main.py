import socket
import matplotlib.pyplot as plt

HOST = '127.0.0.1'
PORT = 8000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

line = [0]

x_t5 = [0]
x_bm = [0]
y_t5 = [0]
y_bm = [0]
fig, ax = plt.subplots(2, 1)

line_t5, = ax[0].plot(x_t5, y_t5, 'r-')
ax[0].set_ylim(0, 10)
ax[0].set_xlim(0, 275)
fill_between_col = ax[0].fill_betweenx(y_t5, line[0], line[-1])
line_bm, = ax[1].plot(x_bm, y_bm, 'r-')
ax[1].set_ylim(0, 10)
ax[1].set_xlim(0, 275)

plt.draw()
plt.pause(10)

while True:
    try:
        client, address = s.accept()
        count = 0
        while True:
            data = client.recv(6)
            str_data = data.decode("utf8")
            print(str_data)
            if count >= 50:
                client.close()
            if str_data == "":
                count += 1
            if str_data == "clr-t5":
                x_t5 = [0]
                y_t5 = [0]
            if str_data == "clr-bm":
                x_bm = [0]
                y_bm = [0]
            if str_data == "clr-ln":
                line = [0]
                try:
                    ax.collections.remove(fill_between_col)
                    plt.pause(0.01)
                except:
                    continue
            if str_data == 'ln-fil':
                fill_between_col = ax[0].fill_betweenx(y_t5[int(line[1]):int(line[-1]+1)], line[1], line[-1])
            if str_data.startswith('bm_'):
                try:
                    i = float(str_data[3:])
                except ValueError:
                    i = 0
                x_bm.append(x_bm[-1] + 1)
                y_bm.append(i)
                line_bm.set_data(x_bm, y_bm)
                plt.pause(0.01)
            if str_data.startswith('t5_'):
                try:
                    i = float(str_data[3:])*1.8
                except ValueError:
                    i = 0
                x_t5.append(x_t5[-1] + 1)
                y_t5.append(i)
                line_t5.set_data(x_t5, y_t5)
                plt.pause(0.01)
            if str_data.startswith('l'):
                try:
                    ln = float(str_data[1:])
                except ValueError:
                    ln = 0
                line.append(ln)
                # for l in line:
                #     ax[0].axvline(x=l)
                #     ax[1].axvline(x=l)
                # plt.pause(0.01)
    except (OSError, socket.error) as e:
        print("Error:", e)
        # reset connection
        client.close()
    else:
        client.shutdown(socket.SHUT_RDWR)
        client.close()
