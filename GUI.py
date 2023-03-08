import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from datetime import datetime
import math


class Demo:
    def __init__(self, root):
        self.root = root
        self.bs = 64
        self.nc = 10
        self.device = torch.device("cuda:0")
        torch.manual_seed(1)
        self.random = True
        self.epsilon = 0.1099
        self.num_steps = 20
        self.step_size = 0.005495

    def load_data(self):
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_set, batch_size=self.bs, shuffle=False, drop_last=True)

        return test_set, test_loader

    def paint_img(self):# TODO
        pass

    def load_model(self):# TODO
        pass

    def infer(self):# TODO
        pass

    def attack(self):# TODO
        pass

    def get_rate(self):# TODO
        return 70

    def main(self):
        def callback(*args):
            print(f"the variable has changed to '{va.get()}'")

        def paint_model(event):
            print(va.get())
            if va.get() == "AlexNet":
                img1 = tk.PhotoImage(file='1.1.jpg').subsample(3)
            elif va.get() == "6nn":
                img1 = tk.PhotoImage(file='1.jpg').subsample(3)
            canvas.create_image(10, 50, image=img1, anchor="nw")
            canvas.image = img1

        def paint_data(event):
            print(v.get())
            if v.get() == "MNIST":
                img = tk.PhotoImage(file='mnist.jpg').subsample(6)
            elif v.get() == "cifar-10":
                img = tk.PhotoImage(file='cifar.jpg').subsample(3)
            li = tk.Label(image=img)
            li.place(x=290, y=40)
            li.image = img

        def paint_rate(event): # BUG
            rate = self.get_rate()
            tk.Label(self.root, text=str(rate) + '%', font=("arial", 10), fg="black").place(x=340, y=220)


        self.root.geometry('512x512')
        self.root.resizable(True, True)
        self.root.title('Demo')

        # dropdown 1
        canvas = tk.Canvas(self.root)
        canvas.pack(expand=True, fill="both")
        va = tk.StringVar(self.root)
        va.set("None")
        # va.trace('w', callback)
        tk.OptionMenu(self.root, va, "None", "6nn", "AlexNet", command=paint_model).place(x=50, y=10)

        # dropdown 2
        v = tk.StringVar(self.root)
        v.set("None")
        tk.OptionMenu(self.root, v, "None", "MNIST", "cifar-10", command=paint_data).place(x=360, y=10)

        img2 = tk.PhotoImage(file='2.jpg').subsample(3)
        canvas.create_image(10, 250, image=img2, anchor="nw")
        img3 = tk.PhotoImage(file='3.jpg').subsample(3)
        canvas.create_image(110, 250, image=img3, anchor="nw")
        img4 = tk.PhotoImage(file='2.jpg').subsample(3)
        canvas.create_image(10, 400, image=img4, anchor="nw")
        img5 = tk.PhotoImage(file='3.jpg').subsample(3)
        canvas.create_image(110, 400, image=img5, anchor="nw")

        # Texts
        tk.Label(self.root, text="Model:", font=("arial", 10), fg="black").place(x=10, y=10)
        tk.Label(self.root, text="Dataset:", font=("arial", 10), fg="black").place(x=300, y=10)
        tk.Label(self.root, text="Image:", font=("arial", 10), fg="black").place(x=10, y=200)
        tk.Label(self.root, text="XAI mask pred:", font=("arial", 10), fg="black").place(x=100, y=200)
        tk.Label(self.root, text="pgd attack", font=("arial", 10), fg="black").place(x=10, y=350)
        tk.Label(self.root, text="disguised attack:", font=("arial", 10), fg="black").place(x=100, y=350)
        tk.Label(self.root, text="Difference before:", font=("arial", 10), fg="black").place(x=320, y=200)

        tk.Label(self.root, text="Difference after:", font=("arial", 10), fg="black").place(x=320, y=250)
        tk.Label(self.root, text=str(30) + '%', font=("arial", 10), fg="black").place(x=340, y=270)

        # attack button
        btn_att = tk.Button(self.root, text="Virgin Attack", bd=1, command=paint_rate, bg="gray", height="1",
                            font=("arial", 10, "bold")).place(x=210, y=256)
        btn_msk = tk.Button(self.root, text="Generate Mask", bd=1, command=None, bg="gray", height="1",
                            font=("arial", 10, "bold")).place(x=210, y=306)
        btn_att_msk = tk.Button(self.root, text="Masked Attack", bd=1, command=None, bg="gray", height="1",
                            font=("arial", 10, "bold")).place(x=210, y=356)

        d, l = self.load_data()
        print(d,l)
        # mainloop
        self.root.mainloop()


if __name__ == '__main__':
    demo = Demo(tk.Tk())
    demo.main()
