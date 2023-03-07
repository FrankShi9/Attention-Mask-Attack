import tkinter as tk
from PIL import *


class Demo:
    def __init__(self, root):
        self.root = root

    def load_data(self):
        pass

    def paint_img(self):
        pass

    def load_model(self):
        pass

    def infer(self):
        pass

    def attack(self):
        pass

    def get_rate(self):
        pass

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
            li = tk.Label(self.root, image=img).place(x=290, y=40)
            li.image = img

        rate = 70  # TODO
        self.root.geometry('512x512')
        self.root.resizable(True, True)
        self.root.title('Demo')

        # dropdown 1
        canvas = tk.Canvas(self.root)
        canvas.pack(expand=True, fill="both")
        va = tk.StringVar(self.root)
        va.set("None")
        # va.trace('w', callback)

        # img
        tk.OptionMenu(self.root, va, "None", "6nn", "AlexNet", command=paint_model).place(x=50, y=10)

        img2 = tk.PhotoImage(file='2.jpg').subsample(3)
        canvas.create_image(10, 250, image=img2, anchor="nw")
        img3 = tk.PhotoImage(file='3.jpg').subsample(3)
        canvas.create_image(110, 250, image=img3, anchor="nw")
        img4 = tk.PhotoImage(file='2.jpg').subsample(3)
        canvas.create_image(10, 400, image=img4, anchor="nw")
        img5 = tk.PhotoImage(file='3.jpg').subsample(3)
        canvas.create_image(110, 400, image=img5, anchor="nw")
        # img6 = tk.PhotoImage(file='mnist.jpg').subsample(6)
        # canvas.create_image(290, 40, image=img6, anchor="nw")
        # Texts
        tk.Label(self.root, text="Model:", font=("arial", 10), fg="black").place(x=10, y=10)
        tk.Label(self.root, text="Dataset:", font=("arial", 10), fg="black").place(x=300, y=10)
        tk.Label(self.root, text="Image:", font=("arial", 10), fg="black").place(x=10, y=200)
        tk.Label(self.root, text="XAI mask pred:", font=("arial", 10), fg="black").place(x=100, y=200)
        tk.Label(self.root, text="pgd attack", font=("arial", 10), fg="black").place(x=10, y=350)
        tk.Label(self.root, text="disguised attack:", font=("arial", 10), fg="black").place(x=100, y=350)
        tk.Label(self.root, text="Difference before:", font=("arial", 10), fg="black").place(x=300, y=200)
        tk.Label(self.root, text=str(rate) + '%', font=("arial", 10), fg="black").place(x=320, y=220)
        tk.Label(self.root, text="Difference after:", font=("arial", 10), fg="black").place(x=300, y=250)
        tk.Label(self.root, text=str(30) + '%', font=("arial", 10), fg="black").place(x=320, y=270)

        # attack button
        btn_att = tk.Button(self.root, text="Attack", bd=1, command=None, bg="gray", height="1",
                            font=("arial", 10, "bold")).place(x=230, y=256)

        # dropdown 2
        v = tk.StringVar(self.root)
        v.set("None")
        tk.OptionMenu(self.root, v, "None", "MNIST", "cifar-10", command=paint_data).place(x=360, y=10)

        # mainloop
        self.root.mainloop()


if __name__ == '__main__':
    demo = Demo(tk.Tk())
    demo.main()
