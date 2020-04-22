import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

#MODEL_NAME == Fruit.h5 tai Fruit.v2.h5 jne.
model = load_model('MODEL_NAME')

# Kommentoi pois polut jotka ei ole käytössä
# Fruit.h5 == kaikki hedelmät
# Fruit.v2.h5 == apple, banana, carambola, kiwi, mango, orange, peach, pitaya, plum
# Fruit.v3.h5 == apple, banana, kiwi, orange
# Fruit.v4.h5 == apple, banana, orange

apple = my_data_dir+'Apple/'+'Apple AD'
#peach = my_data_dir+'Peach'
#carambola = my_data_dir+'Carambola'
#kiwi = my_data_dir+'Kiwi/'+'Total Number of Kiwi fruit'
#tomatoes = my_data_dir+'Tomatoes'
#persimmon = my_data_dir+'Persimmon'
#plum = my_data_dir+'Plum'
#guava = my_data_dir+'Guava/'+'guava total final'
#pear = my_data_dir+'Pear'
#mango = my_data_dir+'Mango'
#muskmelon = my_data_dir+'muskmelon'
banana = my_data_dir+'Banana'
#pomegranate = my_data_dir+'Pomegranate'
#pitaya = my_data_dir+'Pitaya'
orange = my_data_dir+'Orange'

# Tarkistetaan mallin suoritus vertaamalla correct muuttujaa result muuttujaan.
for iIndex in range(100):
	# correct muuttuja täytyy muuttaa manuaalisesti vastaamaan odotettua arvoa.
    correct = np.array([[0, 0, 0, 1]])

    fruit = FRUIT_TO_FIND+'/'+os.listdir(FRUIT_TO_FIND)[iIndex]
    print(fruit)
    my_image = image.load_img(fruit,target_size=image_shape)
    plt.imshow(my_image)

    my_image = image.img_to_array(my_image)
    my_image = np.expand_dims(my_image, axis=0)
    result = model.predict(my_image)

    if (np.array_equal(result, correct)):
        correct_answers += 1
    else:
        incorrect_answers += 1

    plt.show()

print(correct_answers)
print(incorrect_answers)