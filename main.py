from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
from gtts import gTTS
import os

# Load your trained model for handwritten digit recognition
model = keras.models.load_model('mnist_1.h5') #trained mnist dataset using cnn reacher#1


def recognize_digit_from_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))      #img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the pixel values
    img_array = img_array.reshape(1, 28, 28)

    # Make a prediction using the model
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Convert the predicted digit to speech
    text_to_speech = f"The predicted digit is {predicted_digit}"
    tts = gTTS(text_to_speech)

    # Save the speech output as an audio file
    tts.save("predicted_digit_output.mp3")

    # Display the image with the recognized digit
    img_with_text = np.array(img.convert('RGB'))
    cv2.putText(img_with_text, f'Predicted Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow('Handwritten Digit Recognition', img_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Play the audio file
    os.system(" predicted_digit_output.mp3")  # to save the output as audio file


# Usage
image_path = 'sample_5.png'
recognize_digit_from_image(image_path)


































































































































'''import tensorflow as tf
import cv2
import numpy as np
import pyttsx3
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_2.h5')

image = cv2.imread('img_data_mini.jpg')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
preprocessed_digits = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # Cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y + h, x:x + w]

    # Resizing that digit to (18, 18)
    resized_digit = cv2.resize(digit, (18, 18))

    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), 'constant', constant_values=0)

    # Adding the preprocessed digit to the list of preprocessed digits
    preprocessed_digits.append(padded_digit)
print("\n\n\n----------------Contoured Image--------------------")
plt.imshow(image, cmap="gray")
plt.show()

inp = np.array(preprocessed_digits)

##

for digit in preprocessed_digits:
    prediction = model.predict(digit.reshape(1, 28, 28, 1))

    print("\n\n---------------------------------------\n\n")
    print("=========PREDICTION============ \n\n")
    plt.imshow(digit.reshape(28, 28), cmap="gray")
    plt.show()
    digit=(print("\n\nFinal Output: {}".format(np.argmax(prediction))))


    # Generate voice output
engine = pyttsx3.init()
engine.say(f"The recognized digit is {digit}")
engine.runAndWait()

# Load an image
#image_path = 'img_data_mini.jpg'
#image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to (28, 28)
#image = cv2.resize(image, (28, 28))

# Expand dimensions to (28, 28, 1)
#image = np.expand_dims(image, axis=-1)

# Normalize pixel values to be between 0 and 1
#image = image / 255.0

# Make a prediction
#predictions = model.predict(np.expand_dims(image, axis=0))
#digit = np.argmax(predictions[0])'''
