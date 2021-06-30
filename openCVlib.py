import cv2
import matplotlib.pyplot as plt

# # READING IMAGES
# img = cv2.imread(r'C:\Users\mande\Desktop\pep\dsml\machinelearning\imageTest.PNG')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.imshow(img)
# plt.show()

# # AN IMAGE HAS 3 CHANNELS R-RED, G-GREEN AND B-BLUE (RGB)
# # OPENCV USES BGR (SWAPS RED AND BLUE CHANNELS)

# print(img)

img = cv2.imread(
    r'C:\Users\mande\Desktop\pep\dsml\machinelearning\imageTest.PNG')
gray = cv2.imread(
    r'C:\Users\mande\Desktop\pep\dsml\machinelearning\imageTest.PNG',cv2.IMREAD_GRAYSCALE)    
cv2.imshow('Sheldon', img)
cv2.imshow('Sheldon2', gray)
# REPRESENTS TIME (0 MEANS WAIT TILL INFINITE TIME, SIMILARLY 100 MEANS 100 milliseconds)
cv2.waitKey(0)
cv2.destroyAllWindows()
