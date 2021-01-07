

<!-- ABOUT THE Computer Vision Assignment -->
## About The Computer Vision Assignment

Image Classification is performed using natural scene imagery. Transfer learning is performed using InceptionV3 as backbone.



<!-- Code -->
## Code


### Imports

  ```sh
  import numpy as np # linear algebra
  import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
  import os
  from tensorflow.keras import layers
  from tensorflow.keras import Model
  from tensorflow.keras.applications import Xception
  from tensorflow.keras.optimizers import RMSprop
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  import os
  import matplotlib.pyplot as plt
  import pandas as pd
  from keras.callbacks import ReduceLROnPlateau
  ```

### Dataset Read (along with augumentation)

   ```sh
   datagen = ImageDataGenerator( rescale = 1.0/255,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          zoom_range=0.2,
                                          vertical_flip=True,
                                          horizontal_flip=True,
                                          fill_mode='nearest')

    #Load data
    train_generator = datagen.flow_from_directory(
        '/content/drive/My Drive/Assgn-2/CV_Assignment3/intel-image-classification/seg_train/seg_train/',
        target_size=(150, 150),
        shuffle=True,
        batch_size=32,
        class_mode="categorical")

    validation_generator = datagen.flow_from_directory(
        '/content/drive/My Drive/Assgn-2/CV_Assignment3/intel-image-classification/seg_train/seg_train/',
        target_size=(150, 150),
        batch_size=32,
        shuffle=True,
        class_mode="categorical")

    test_generator = datagen.flow_from_directory(
        '/content/drive/My Drive/Assgn-2/CV_Assignment3/intel-image-classification/seg_test/seg_test/',
        target_size=(150, 150),
        shuffle=True,
        batch_size=32,
        class_mode="categorical",
    )
   ```
### load the pretrained InceptionV3 model

  ```sh
  from keras.applications.inception_v3 import InceptionV3
  inceptionV3 = InceptionV3(include_top= False, input_shape=(150,150,3))

  for layer in inceptionV3.layers:
    layer.trainable = False

  last_layer = inceptionV3.get_layer('mixed9')

  print('last layer output shape: ', last_layer.output_shape)

  last_output = last_layer.output
  ``` 

### Model Set-Up for Training

  ```sh
  LearningRateScheduler = ReduceLROnPlateau(monitor='val_acc',
                                            patience=0,
                                            verbose=1,
                                            factor=0.20,
                                            min_lr=0.000001)

  x = layers.Flatten()(last_output)
  x = layers.Dense(1024, activation='relu')(x)
  x = layers.Dropout(0.2)(x)                  
  x = layers.Dense(6, activation='softmax')(x)           

  model = Model(inceptionV3.input, x) 

  model.compile(optimizer = RMSprop(lr=0.0001), 
                loss = 'categorical_crossentropy', 
                metrics = ['acc'])

  model.summary()
  history = model.fit(train_generator,
                    epochs = 10,
                    verbose = 1,
                   validation_data = validation_generator,
                   callbacks=[LearningRateScheduler])
  ``` 

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
