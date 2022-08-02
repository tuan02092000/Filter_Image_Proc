import cv2
import random
import numpy as np
import kernel

def readImage(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image

def add_noise_gauss(image, mean=0, var=100):
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

def add_noise_salt_peper(image):
    row, col = image.shape
    new_img = np.array(image)
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        x_coord = random.randint(0, row - 1)
        y_coord = random.randint(0, col - 1)
        new_img[x_coord, y_coord] = 255
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        x_coord = random.randint(0, row - 1)
        y_coord = random.randint(0, col - 1)
        new_img[x_coord, y_coord] = 0
    return new_img

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def mean_filter(image, kernal_size, padding=0, strides=1):
    kernel = np.ones((kernal_size, kernal_size)) / (kernal_size * kernal_size)
    image = convolve2D(image, kernel, padding=padding, strides=strides)
    return image

def gaussian_kernel(kernel_size, sigma):
    h = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-(np.square(i - h) + np.square(j - h)) / (2 * np.square(sigma))) / np.sqrt((2 * np.pi * np.square(sigma)))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_filter(image, kernel_size, sigma, padding=0, strides=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    image = convolve2D(image, kernel, padding=padding, strides=strides)
    return image

def median_filter(image, kernel_size):
    temp = []
    indexer = kernel_size // 2
    height, width = image.shape
    data_final = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > height - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > width - 1:
                        temp.append(0)
                    else:
                        for k in range(kernel_size):
                            temp.append(image[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

def scale_to_0_255(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val)  # 0-1
    new_img *= 255
    return new_img

def canny_filter(img, min_val, max_val, sobel_size=3, is_L2_gradient=False):
    # Noise Reduction
    smooth_img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=1, sigmaY=1)

    # Finding Intensity Gradient of the Image
    Gx = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    Gy = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize=sobel_size)

    if is_L2_gradient:
        edge_gradient = np.sqrt(Gx * Gx + Gy * Gy)
    else:
        edge_gradient = np.abs(Gx) + np.abs(Gy)

    angle = np.arctan2(Gy, Gx) * 180 / np.pi

    # Round angle to 4 directions
    angle = np.abs(angle)
    angle[angle <= 22.5] = 0
    angle[angle >= 157.5] = 0
    angle[(angle > 22.5) * (angle < 67.5)] = 45
    angle[(angle >= 67.5) * (angle <= 112.5)] = 90
    angle[(angle > 112.5) * (angle <= 157.5)] = 135

    # Non-maximum Suppression
    keep_mask = np.zeros(smooth_img.shape, np.uint8)
    for y in range(1, edge_gradient.shape[0] - 1):
        for x in range(1, edge_gradient.shape[1] - 1):
            area_grad_intensity = edge_gradient[y - 1:y + 2, x - 1:x + 2]  # 3x3 area
            area_angle = angle[y - 1:y + 2, x - 1:x + 2]  # 3x3 area
            current_angle = area_angle[1, 1]
            current_grad_intensity = area_grad_intensity[1, 1]

            if current_angle == 0:
                if current_grad_intensity > max(area_grad_intensity[1, 0], area_grad_intensity[1, 2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0
            elif current_angle == 45:
                if current_grad_intensity > max(area_grad_intensity[2, 0], area_grad_intensity[0, 2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0
            elif current_angle == 90:
                if current_grad_intensity > max(area_grad_intensity[0, 1], area_grad_intensity[2, 1]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0
            elif current_angle == 135:
                if current_grad_intensity > max(area_grad_intensity[0, 0], area_grad_intensity[2, 2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0

    # Hysteresis Thresholding
    canny_mask = np.zeros(smooth_img.shape, np.uint8)
    canny_mask[(keep_mask > 0) * (edge_gradient > min_val)] = 255
    return scale_to_0_255(canny_mask)

if __name__ == '__main__':
    # Read Image
    image = readImage('image/demo1.jpg')
    cv2.imwrite('output/output_original.jpg', image)

    # Gaussian Filter
    image_gauss = add_noise_gauss(image, mean=0, var=100)  # Add noise gauss
    cv2.imwrite('output/output_with_noise_gauss.jpg', image_gauss)

    output_gauss = gaussian_filter(image_gauss, kernel_size=3, sigma=1)
    output_gauss_cv2 = cv2.GaussianBlur(image_gauss, (3, 3), 1, 1)

    cv2.imwrite('output/output_gaussian.jpg', output_gauss)
    cv2.imwrite('output/output_gaussian_cv2.jpg', output_gauss_cv2)

    # Mean Filter
    output_mean = mean_filter(image_gauss, kernal_size=3)
    output_mean_cv2 = cv2.blur(image_gauss, (3, 3))

    cv2.imwrite('output/output_mean.jpg', output_mean)
    cv2.imwrite('output/output_mean_cv2.jpg', output_mean_cv2)

    # Median Filter
    image_median = add_noise_salt_peper(image)  # Add noise salt and peper
    cv2.imwrite('output/output_with_noise_salt_peper.jpg', image_median)

    output_median = median_filter(image_median, 3)
    output_median_cv2 = cv2.medianBlur(image_median, 3)

    cv2.imwrite('output/output_median.jpg', output_median)
    cv2.imwrite('output/output_median_cv2.jpg', output_median_cv2)

    # Laplacian Filter
    output_laplacian = convolve2D(image, kernel.LAPLACIAN)
    output_laplacian_cv2 = cv2.Laplacian(image, cv2.CV_64F, ksize=3, scale=1)
    cv2.imwrite('output/output_laplacian.jpg', output_laplacian)
    cv2.imwrite('output/output_laplacian_cv2.jpg', output_laplacian_cv2)

    # # SobelX Filter
    # output_sobelX = convolve2D(image, kernel.SOBELX)
    # output_sobelX_cv2 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1)
    # cv2.imwrite('output/output_sobelX.jpg', output_sobelX)
    # cv2.imwrite('output/output_sobelX_cv2.jpg', output_sobelX_cv2)
    #
    # # SobelY Filter
    # output_sobelY = convolve2D(image, kernel.SOBELY)
    # output_sobelY_cv2 = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1)
    # cv2.imwrite('output/output_sobelY.jpg', output_sobelY)
    # cv2.imwrite('output/output_sobelY_cv2.jpg', output_sobelY_cv2)

    # Canny filter
    output_canny = canny_filter(image, min_val=100, max_val=200)
    output_canny_cv2 = cv2.Canny(image, 100, 200)
    cv2.imwrite('output/output_canny.jpg', output_canny)
    cv2.imwrite('output/output_canny_cv2.jpg', output_canny_cv2)

