import tkinter as tk
from PIL import *


class Demo:
    def __init__(self, root):
        self.root = root

    def paint_model(self):
        pass

    def load_data(self):
        pass

    def paint_img(self):
        pass

    def get_rate(self):
        pass

    def main(self):
        rate = 70 # TODO
        self.root.geometry('512x512')
        self.root.resizable(True, True)
        self.root.title('Demo')

        canvas = tk.Canvas(self.root)
        canvas.pack(expand=True, fill="both")

        img1 = tk.PhotoImage(file='1.jpg').subsample(3)
        canvas.create_image(10,50,image=img1,anchor="nw")
        img2 = tk.PhotoImage(file='2.jpg').subsample(3)
        canvas.create_image(10, 250, image=img2, anchor="nw")
        img3 = tk.PhotoImage(file='3.jpg').subsample(3)
        canvas.create_image(110, 250, image=img3, anchor="nw")
        tk.Label(self.root, text="Model:", font=("arial", 10), fg="black").place(x=10, y=10)
        tk.Label(self.root, text="Dataset:", font=("arial", 10), fg="black").place(x=300, y=10)
        tk.Label(self.root, text="Image:", font=("arial", 10), fg="black").place(x=10, y=200)
        tk.Label(self.root, text="Saliency:", font=("arial", 10), fg="black").place(x=100, y=200)
        btn_att = tk.Button(self.root, text="Attack", bd=1, command=None, bg="gray", height="1", font=("arial", 10, "bold")).place(x=230, y=256)
        tk.Label(self.root, text="Detection rate:", font=("arial", 10), fg="black").place(x=300, y=200)
        tk.Label(self.root, text=str(rate)+'%', font=("arial", 10), fg="black").place(x=320, y=220)
        var = tk.StringVar(self.root)
        var.set("MNIST")
        tk.OptionMenu(self.root, var, "cifar-10", "MNIST").place(x=300, y=30)
        self.root.mainloop()


if __name__ == '__main__':
    demo = Demo(tk.Tk())
    demo.main()
