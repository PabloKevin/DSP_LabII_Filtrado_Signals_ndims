import numpy as np
import matplotlib.pyplot as plt
import cv2
cv2.__version__

# Lectura de imagen en BGR y conversión a RGB para visualización con matplotlib
img = cv2.imread("DSP_LabII_Filtrado_Signals_ndims/images/sun.jpg", 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Conversión manual de imagen RGB a escala de grises utilizando ponderaciones perceptuales
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

gray_img = rgb2gray(cv2.imread("DSP_LabII_Filtrado_Signals_ndims/images/sun.jpg",1))

# Transformada de Fourier 2D y desplazamiento del espectro al centro
IMG = np.fft.fft2(gray_img)
IMG = np.fft.fftshift(IMG)
magnitude_spectrum = 20*np.log(np.abs(IMG))

# Creación de una máscara gaussiana para filtrado en frecuencia
sigma_x, sigma_y = 50, 50
(ncols, nrows) = np.shape(gray_img)
c_x, c_y = nrows/2, ncols/2
x = np.linspace(0, nrows, nrows)
y = np.linspace(0, ncols, ncols)
X, Y = np.meshgrid(x, y)
gaussian_mask = np.exp(-((X-c_x)**2 + (Y-c_y)**2)/(2*sigma_x**2)) / (2*np.pi*sigma_x**2)

# Aplicación del filtro gaussiano sobre el espectro
filtered_IMG = IMG * gaussian_mask

# Transformada inversa para obtener la imagen filtrada en el dominio espacial
ifft_img = np.fft.ifft2(filtered_IMG)

# Visualización de etapas
fig, axes = plt.subplots(3, 2, figsize=(8, 6))  
axes[0][0].imshow(img,cmap='gray')
axes[0][0].set_title('Imagen original RGB')
axes[0][0].axis('off') 

axes[0][1].imshow(gray_img,cmap='gray')
axes[0][1].set_title('Imagen en escala de grises')
axes[0][1].axis('off') 

axes[1][0].imshow(magnitude_spectrum, cmap = 'magma') 
axes[1][0].set_title('Magnitud del especto gray')
axes[1][0].axis('off') 

axes[1][1].imshow(gaussian_mask, cmap = 'gray')
axes[1][1].set_title('Máscara Gaussiana')
axes[1][1].axis('off') 

axes[2][0].imshow(20*np.log(np.abs(filtered_IMG)), cmap = 'magma')
axes[2][0].set_title('Espectro filtrado')
axes[2][0].axis('off') 

axes[2][1].imshow(np.abs(ifft_img), cmap = 'gray')
axes[2][1].set_title('Imagen filtrada')
axes[2][1].axis('off') 

plt.tight_layout() 
plt.show()

# Filtro Gaussiano en el dominio espacial con OpenCV
gauss_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

# Canny para detección de bordes
gray_u8 = gray_img.astype(np.uint8)
bordes = cv2.Canny(gray_u8, 110, 250, 15)

# Sobel para detección de gradientes en X e Y
fig, axes = plt.subplots(3, 1, figsize=(6, 8))  
gray_u8 = gray_img.astype(np.uint8)
sobelx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=15)
sobely = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=15)
sobel = cv2.magnitude(sobelx, sobely)

axes[0].imshow(sobelx, cmap='gray')
axes[0].set_title('Filtro Sobel X')
axes[0].axis('off') 
axes[1].imshow(sobely, cmap='gray')
axes[1].set_title('Filtro Sobel Y')
axes[1].axis('off') 
axes[2].imshow(sobel, cmap='gray')
axes[2].set_title('Filtro Sobel magnitud')
axes[2].axis('off') 
plt.show()

# Comparación Canny vs Sobel
fig, axes = plt.subplots(3, 1, figsize=(6, 8))  
axes[0].imshow(gray_img, cmap='gray')
axes[0].set_title('Imagen original en escala de grises')
axes[0].axis('off') 
axes[1].imshow(bordes, cmap='gray')
axes[1].set_title('Filtro Canny para detección de bordes')
axes[1].axis('off') 
axes[2].imshow(sobel, cmap='gray')
axes[2].set_title('Filtro Sobel magnitud para detección de bordes')
axes[2].axis('off') 
plt.show()

# Detección de círculos con Hough (sobre imagen suavizada)
gauss_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
in_img = gauss_img.astype(np.uint8)
circles = cv2.HoughCircles(in_img, cv2.HOUGH_GRADIENT, 2.5, 500, minRadius=200,maxRadius=320)

print("cantidad de circulos encontrados:", circles.shape[1] if circles is not None else 0)

# Conversión a BGR para superponer círculos y centros
cimg = cv2.cvtColor(in_img, cv2.COLOR_GRAY2BGR)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(155,200,50),12) # círculos
        cv2.circle(cimg,(i[0],i[1]),5,(255,0,0),15)      # centros

o_img = img.astype(np.uint8)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(o_img,(i[0],i[1]),i[2],(0,0,255),12)
        cv2.circle(o_img,(i[0],i[1]),5,(0,0,255),15)

# Visualización de círculos detectados sobre imagen filtrada y original
fig, axes = plt.subplots(2, 1, figsize=(6, 5))
axes[0].imshow(cimg)
axes[0].set_title('Imagen gray con círculos detectados')
axes[0].axis('off')
axes[1].imshow(o_img)
axes[1].set_title('Círuclos detectados plotteados sobre la imagen original')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# Visualización por canales RGB independientes
cmaps = ['Reds', 'Greens', 'Blues']
h, w = in_img.shape
fig, axes = plt.subplots(4, 1, figsize=(6, 8))
axes[0].imshow(img)
axes[0].set_title('Imagen original RGB')
axes[0].axis('off')

for channel in range(3):
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_img[..., channel] = img[...,channel]
    axes[channel+1].imshow(rgb_img, cmap=cmaps[channel])
    axes[channel+1].set_title('Canal '+ cmaps[channel])
    axes[channel+1].axis('off')

plt.tight_layout()
plt.show()

# Enfoque por canal verde para mejorar la detección de círculos
in_img = img[...,1].astype(np.uint8)
in_img = cv2.GaussianBlur(in_img, (5, 5), 0)
circles = cv2.HoughCircles(in_img, cv2.HOUGH_GRADIENT, 2.0, 600, minRadius=200,maxRadius=320)

print("cantidad de circulos encontrados:", circles.shape[1] if circles is not None else 0)

h, w = in_img.shape
rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
rgb_img[..., 1] = in_img  # reconstrucción imagen solo canal verde

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(rgb_img,(i[0],i[1]),i[2],(0,0,255),12)
        cv2.circle(rgb_img,(i[0],i[1]),5,(255,0,0),15)

o_img = img.astype(np.uint8)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(o_img,(i[0],i[1]),i[2],(0,0,255),12)
        cv2.circle(o_img,(i[0],i[1]),5,(0,0,255),15)

fig, axes = plt.subplots(2, 1, figsize=(6, 5))
axes[0].imshow(rgb_img,cmap='Greens')
axes[0].set_title('Canal verdes con círculos detectados')
axes[0].axis('off')
axes[1].imshow(o_img)
axes[1].set_title('Círuclos detectados plotteados sobre la imagen original')
axes[1].axis('off')
plt.tight_layout()
plt.show()
