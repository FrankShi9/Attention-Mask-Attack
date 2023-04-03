import tkinter as tk
import warnings

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from datetime import datetime
import math
import torchvision.transforms as T
from PIL import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


warnings.filterwarnings("ignore", category=Warning)

def SLU(x, a=.5):
    return torch.max(torch.zeros_like(x), x) + a * torch.sin(x)

def g_normal_(tensor, in_c, out_c, kernel_size):
    # if 0 in tensor.shape:
    #     warnings.warn("Initializing zero-element tensors is a no-op")
    #     return tensor
    std = 2 / ((in_c + out_c) * kernel_size)
    with torch.no_grad():
        return tensor.normal_(0, std)

class Fnn(nn.Module):
    def __init__(self):
        super(Fnn, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim=0)
        return x


class Block(nn.Module):  # vgg/res inspired
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        g_normal_(self.conv1.weight, in_c, out_c, 3)  # init
        self.bn = nn.BatchNorm2d(out_c)
        # self.lrelu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.gn = nn.GroupNorm(8, out_c)
        self.dropout = nn.Dropout2d(p=.25)

    def forward(self, x):
        # return self.gm(self.lrelu(self.conv2(SLU(self.conv1(x)))))

        y = self.conv1(x)
        identity = y
        y = self.bn(y)
        y = SLU(y)
        y = self.conv2(y)
        y = self.gn(y)
        y = self.dropout(y)

        y += identity
        y = self.elu(y)

        return y


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        # self.pool = nn.MaxPool2d(2)
        self.pool_blks = nn.ModuleList([nn.Conv2d(chs[i + 1], chs[i + 1], 1, 2, 0) for i in range(len(chs) - 1)])

    def forward(self, x, cifar=False):
        if not cifar:
            x = x.reshape(1, 28, 28)
            x = x.repeat(1, 3, 1, 1)

        ftrs = []
        i = 1
        for (idx, block) in enumerate(self.enc_blocks):
            x = block(x)
            ftrs.append(x)
            # x = self.pool(x)
            x = self.pool_blks[idx](x)
            i += 1

        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2, 0) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features, cifar=False):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)

        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256), dec_chs=(256, 128, 64), num_class=1, retain_dim=False,
                 out_sz=(28, 28)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)

        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)

        return out


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
        self.test_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
        self.test_loader = DataLoader(self.test_set, batch_size=self.bs, shuffle=False, drop_last=True)
        self.fnn = Fnn().to(self.device)
        self.fnn.load_state_dict(torch.load('fnn.pt'))
        self.fnn.eval()
        self.unet = UNet().to(self.device)
        self.unet.load_state_dict(torch.load('u.pt'))
        self.unet.eval()

        self.t0 = 0
        self.t1 = 0
    # infer
    # def infer(self, data, lb):
    #     loader = self.test_loader
    #     device = self.device
    #     model = self.fnn()
    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     idx = 0
    #     with torch.no_grad():
    #         for batch_idx, (data, target) in enumerate(loader):
    #             data, target = data.to(device), target.to(device)
    #             data = data.view(data.size(0), 28 * 28)
    #             output = model(data)
    #             test_loss += F.cross_entropy(output, target, size_average=False).item()
    #             pred = output.max(1, keepdim=True)[1]
    #             correct += pred.eq(target.view_as(pred)).sum().item()
    #             mis = self.bs-pred.eq(target.view_as(pred)).sum().item()
    #
    #     test_loss /= len(loader.dataset)
    #     test_accuracy = correct / len(loader.dataset)
    #     return test_loss, test_accuracy, mis

    def attack0(self, X, y, random=True):
        epsilon = self.epsilon
        num_steps = self.num_steps
        step_size = self.step_size
        device = self.device
        model = self.fnn
        X_pgd = Variable(X.data, requires_grad=True)
        st = time.time()
        if random:
            noise = torch.FloatTensor(*X_pgd.shape).uniform_(epsilon).to(device)
            X_pgd = Variable(X_pgd.data + noise, requires_grad=True)
        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = F.cross_entropy(model(X_pgd), y)

            loss.backward()
            eta = torch.clamp(X_pgd.data, -step_size,
                              step_size) * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        et = time.time()
        t = et-st
        self.t0 = t
        return X_pgd

    def attack(self, X, mask, y, random=True):
        epsilon = self.epsilon
        num_steps = self.num_steps
        step_size = self.step_size
        device = self.device
        model = self.fnn
        X_pgd = Variable(X.data, requires_grad=True)
        st = time.time()
        if random:
            noise = torch.clamp(mask, -epsilon, epsilon).to(device)  # XAI mask
            X_pgd = Variable(X_pgd.data + noise, requires_grad=True)
        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = F.cross_entropy(model(X_pgd), y)

            loss.backward()
            eta = torch.clamp(mask, -step_size,
                              step_size) * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        et = time.time()
        t = et-st
        self.t1 = t
        return X_pgd

    def mask(self, data):
        model = self.unet
        mask = model(data.to(self.device))

        return mask

    def get_rate(self, att=False, mask=False):
        features, labels = next(iter(self.test_loader))
        img = features[0].squeeze().view(28*28).to(self.device)
        label = labels[0].to(self.device)

        if att and not mask:
            xa = self.attack0(img, label)
            xa = xa.view(1,28*28)
            img = img.view(1,28*28)
            return float(nn.CosineSimilarity()(xa, img))

        if att and mask:
            mas = self.mask(img).view(28*28)
            xa = self.attack(img, mas, label)
            xa = xa.view(1,28*28)
            img = img.view(1,28*28)
            return float(nn.CosineSimilarity()(xa, img))

    def get_pred(self, input):
        model = self.fnn

        with torch.no_grad():
            output = model(input)
            pred = output.max(0, keepdim=True)[1]

        return int(pred)

    def main(self):

        sb, sa = 0, 0

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

        def paint_img1():
            global sb
            features, labels = next(iter(self.test_loader))
            img = features[0].squeeze().view(28*28).to(self.device)
            label = labels[0].to(self.device)
            xa = self.attack0(img, label).view(28,28)
            transform = T.ToPILImage()
            img = transform(xa)
            fig1 = plt.figure(figsize=(1,1), dpi=110)
            b1 = FigureCanvasTkAgg(fig1, self.root)
            b1.get_tk_widget().place(x=10, y=370)
            plt.imshow(img, cmap='gray')

            sb = self.get_rate(True, False)
            #print(sb)
            tk.Label(self.root, text=str(sb*100)[0:6] + '%', font=("arial", 9), fg="black").place(x=340, y=220)
            tk.Label(self.root, text=str(self.t0)[0:5] + 's', font=("arial", 8), fg="black").place(x=400, y=220)
            tk.Label(self.root, text='Pred: ' + str(self.get_pred(xa.view(28*28))), font=("arial", 7), fg="black").place(x=340, y=240)

        def paint_img2():
            features, labels = next(iter(self.test_loader))
            img = features[0].squeeze().to(self.device)
            mas = self.mask(img).reshape(28,28)
            transform = T.ToPILImage()
            img = transform(mas)
            fig3 = plt.figure(figsize=(1,1), dpi=110)
            b3 = FigureCanvasTkAgg(fig3, self.root)
            b3.get_tk_widget().place(x=100, y=220)
            plt.imshow(img, cmap='gray')

        def paint_img3():
            global sa

            features, labels = next(iter(self.test_loader))
            img = features[0].squeeze().view(28*28).to(self.device)
            label = labels[0].to(self.device)
            mas = self.mask(img)
            xa = self.attack(img, mas.view(28*28), label).view(28,28)
            transform = T.ToPILImage()
            img = transform(xa)
            fig2 = plt.figure(figsize=(1,1), dpi=110)
            b2 = FigureCanvasTkAgg(fig2, self.root)
            b2.get_tk_widget().place(x=100, y=370)
            plt.imshow(img, cmap='gray')

            sa = self.get_rate(True, True)
            #print(sa)
            tk.Label(self.root, text=str(sa*100)[0:6] + '%', font=("arial", 9), fg="black").place(x=340, y=270)
            tk.Label(self.root, text=str(self.t1)[0:5] + 's', font=("arial", 8), fg="black").place(x=400, y=270)
            tk.Label(self.root, text='Pred: ' + str(self.get_pred(xa.view(28*28))), font=("arial", 7), fg="black").place(x=340, y=290)

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
        tk.Label(self.root, text="Similarity before:", font=("arial", 10), fg="black").place(x=320, y=200)
        tk.Label(self.root, text=str(sb) + '%', font=("arial", 10), fg="black").place(x=340, y=220)
        tk.Label(self.root, text="Similarity after:", font=("arial", 10), fg="black").place(x=320, y=250)
        tk.Label(self.root, text=str(sa) + '%', font=("arial", 10), fg="black").place(x=340, y=270)

        # attack button
        tk.Button(self.root, text="Virgin Attack", bd=1, command=paint_img1, bg="gray", height="1",
                            font=("arial", 10, "bold")).place(x=210, y=256)
        tk.Button(self.root, text="Mask", bd=1, command=paint_img2, bg="gray", height="1",
                            font=("arial", 10, "bold")).place(x=210, y=306)
        tk.Button(self.root, text="Masked Attack", bd=1, command=paint_img3, bg="gray", height="1",
                                font=("arial", 10, "bold")).place(x=210, y=356)

        # mainloop
        self.root.mainloop()


if __name__ == '__main__':
    demo = Demo(tk.Tk())
    demo.main()
