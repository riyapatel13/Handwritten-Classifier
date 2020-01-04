import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm  # support vector machine


digits = datasets.load_digits()

print("This is a handwritten digits classifier.")

# setting up classifier
gamma = float(input("Choose a learning rate: "))
clf = svm.SVC(gamma=0.001, C=100)
# gamma = learning rate for gradient descent (0.001)
# C = regularization parameter 

train_num = int(input("How much data would you like to train with? (max 1796) "))
# len(digits.data) = 1797
x,y = digits.data[:(train_num-1)], digits.target[:(train_num-1)]
clf.fit(x,y)

print("Here are the results of the test data:")
test_data = []
for i in range(1,1797-train_num+1):
    test_data.append(("Prediction: %d" % clf.predict(digits.data)[-i],"Actual answer: %d" % digits.target[-i]))
print(test_data)

num = 1797 - train_num
display = input("\nIf you want to see the test data image, type which image you would like to see (where 1 is the first piece of test data, max %d). Otherwise, type \"exit\": " % num)
while display != "exit":
    if display.isnumeric():
        if int(display) > num or int(display) < 1:
            display = input("The number is not in range. Try again or type \"exit\": ")
        else:
            plt.imshow(digits.images[-int(display)], cmap=plt.cm.gray_r, interpolation="nearest")
            plt.show()
            display = input("Choose another image to display, or type \"exit\": ")
    else:
        display = input("The input was not valid. Try again or type \"exit\": ")
print("Exited")
