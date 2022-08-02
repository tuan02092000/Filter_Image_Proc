import os
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk
import cv2
import filter
import kernel


def showImage():
    global my_image
    window.filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                                title="Select A File",
                                                filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
    img_path.set(window.filename)
    image = Image.open(window.filename).convert('L').resize((650, 500))
    my_image = ImageTk.PhotoImage(image)
    my_image_label = Label(image=my_image)
    my_image_label.place(x=50, y=100)

def showFilter(string=''):
    image = cv2.imread(img_path.get(), cv2.IMREAD_GRAYSCALE)
    if string == 'mean':
        # image_noise = convolution.add_noise_gauss(image, mean=0, var=100)  # Add noise gauss
        output = filter.mean_filter(image, kernal_size=int(kernel_size.get()))
    elif string == 'gaussian':
        # image_noise = convolution.add_noise_gauss(image, mean=0, var=100)  # Add noise gauss
        output = filter.gaussian_filter(image, kernel_size=int(kernel_size.get()), sigma=float(sigma.get()))
    elif string == 'median':
        # image_noise = convolution.add_noise_salt_peper(image)
        output = filter.median_filter(image, kernel_size=int(kernel_size.get()))
    elif string == 'laplacian':
        output = filter.convolve2D(image, kernel=kernel.LAPLACIAN)
    else:
        output = filter.canny_filter(image, int(min_val.get()), int(max_val.get()), sobel_size=int(kernel_size.get()))
    cv2.imwrite('output/output_{}.jpg'.format(string), output)
    image_filter = Image.open("output/output_{}.jpg".format(string)).resize((650, 500))
    show_image = ImageTk.PhotoImage(image_filter)
    image_label = Label(image=show_image)
    image_label.image = show_image
    image_label.place(x=800, y=100)
    frame_label = Label(window, text=string)
    frame_label.place(x=1100, y=100)

window = Tk()
window.title('Image Filter')
window.geometry('1470x1000')

img_path = StringVar()

frame1 = Frame(window, width=650, height=500)
frame1.place(x=50, y=100)
lb_frame1 = Label(window, text='Original Image')
lb_frame1.place(x=300, y=600)

frame2 = Frame(window, width=650, height=500)
frame2.place(x=800, y=100)
lb_frame2 = Label(window, text='Use filter')
lb_frame2.place(x=1100, y=600)

btn_select = Button(window, text="Select Image", command=showImage)
btn_select.place(x=100, y=800)

btn_mean = Button(window, text="Mean Filter", command=lambda *arg: showFilter('mean'))
btn_mean.place(x=250, y=800)

btn_gaussian = Button(window, text="Gaussian Filter", command=lambda *arg: showFilter('gaussian'))
btn_gaussian.place(x=400, y=800)

btn_median = Button(window, text="Median Filter", command=lambda *arg: showFilter('median'))
btn_median.place(x=550, y=800)

btn_laplacian = Button(window, text="Laplacian Filter", command=lambda *arg: showFilter('laplacian'))
btn_laplacian.place(x=700, y=800)

btn_canny = Button(window, text="Canny Filter", command=lambda *arg: showFilter('canny'))
btn_canny.place(x=850, y=800)

btn_exit = Button(window, text='Exit', command=lambda: exit())
btn_exit.place(x=1000, y=800)

lb_kernel = Label(window, text='Kernel size')
lb_kernel.place(x=100, y=900)
kernel_size = StringVar()
kernel_size.set('3')
entry_kernel = Entry(window, textvariable=kernel_size)
entry_kernel.place(x=200, y=900)

lb_sigma = Label(window, text='Sigma (Only Gaussian)')
lb_sigma.place(x=400, y=900)
sigma = StringVar()
sigma.set('1')
entry_sigma = Entry(window, textvariable=sigma)
entry_sigma.place(x=600, y=900)

lb_min = Label(window, text='Min value (Only Canny)')
lb_min.place(x=100, y=950)
min_val = StringVar()
min_val.set('100')
entry_min_val = Entry(window, textvariable=min_val)
entry_min_val.place(x=250, y=950)

lb_max = Label(window, text='Max value (Only Canny)')
lb_max.place(x=400, y=950)
max_val = StringVar()
max_val.set('200')
entry_max_val = Entry(window, textvariable=max_val)
entry_max_val.place(x=600, y=950)

window.mainloop()

