import os
import random


def speed_augment(srcdir, src_label, tgtdir, tgt_label):
    list_random = [0.85, 0.9, 1.1, 1.2, 1.3, 1.35]
    for filename in os.listdir(srcdir):
        filename = filename.replace(".wav", "")
        if not os.path.isfile(os.path.join(src_label, filename + ".txt")):
            continue
        speed = random.choice(list_random)
        print(filename)
        os.system("sox {} {} tempo {}".format(os.path.join(srcdir, filename + ".wav"),
                                              os.path.join(tgtdir, filename + "-speed.wav"), speed))

        with open(os.path.join(src_label, filename + ".txt"), "r") as f:
            label = f.read().split()

        label = [int(x) for x in label]
        speed_label = []
        for i in range(int(len(label) / speed)):
            speed_label.append(str(label[min(int(i * speed), len(label) - 1)]))

        with open(os.path.join(tgt_label, filename + "-speed.txt"), "w") as f:
            f.write(" ".join(speed_label))


if __name__ == '__main__':
    speed_augment("/home/lehoa/PycharmProjects/cough_classification/Data_Cough/augment/audio",
                  "/home/lehoa/PycharmProjects/cough_classification/Data_Cough/augment/labels",
                  "/home/lehoa/PycharmProjects/cough_classification/Data_Cough/augment/audio",
                  "/home/lehoa/PycharmProjects/cough_classification/Data_Cough/augment/labels")
