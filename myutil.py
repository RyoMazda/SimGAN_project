# coding: utf-8


# starting time
import time

import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np

class CodeTimeTracker:
    def __init__(self):
        self.start_time = time.time()

    def start(self):
        print("------------------------ code started!")

    def record(self):
        return time.time() - self.start_time

    def end(self):
        end_time = time.time()
        print("Total Time: ", round(end_time - self.start_time), "seconds")
        print("------------------------ code end!")


class FileManamer:
    def __init__(self, dirname=None):
        self.dateinfo = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
        if dirname is None:
            self.out_path = 'output/' + self.dateinfo + '/'
        else:
            self.out_path = 'output/' + dirname + '/' + self.dateinfo + '/'


    def mkdir(self):
        os.makedirs(self.out_path)


    def dump(self, data, destination='./dumped/', filelabel='dump'):
        """
        モデルをdestinationフォルダに保存する。
        :param destination: String. フォルダの相対パス。
        :return: String. pickleバイナリファイルのパス。
        """
        now = datetime.datetime.now()
        unixtime = int(now.timestamp())
        if destination[-1] != "/":
            destination += "/"
        filename = destination + filelabel + str(unixtime) + ".pickle"
        print('I will dump it as:' + filename + ', to:' + destination)
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=4)
        print('and it succeeded')
        return filename



# for plot images in grids out of numpy 1d * 2d vectors
def plot_grid(vecs):
    size = int(np.sqrt(vecs.shape[0]))
    fig = plt.figure(figsize=(size, size))
    gs = gridspec.GridSpec(size, size)
    gs.update(wspace=0.05, hspace=0.05)

    for i, vec in enumerate(vecs):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(vec, cmap='Greys_r')

    return fig



if __name__ == '__main__':
    fm = FileManamer()
    mydata = {'朝': 'humberger', '昼': 'curry', '夜': 'noodle'}
    fm.dump(mydata)


