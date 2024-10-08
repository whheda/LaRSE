# LaRSE
visual-language reasoning segmentation of function-level building footprint


Function-level building footprint is vital for understanding socio-economic heterogeneity and advancing sustainable development. However, typical researches are restricted to either parcel-level urban function identification or binary building footprint extraction, neither of which can delineate human activity spaces at the footprint unit. Besides, common segmentation networks lack human-like reasoning capabilities, as they fail to incorporate surrounding entities or leverage geo-spatial knowledge. 

To this being, this study proposes a novel function-level building footprint extraction based on visual-language reasoning segmentation framework (LaRSE). It simplifies the task into two stages, i.e., visual model for segmenting building footprint and deriving context embeddings, followed by a language model that utilizes these embeddings to perform precise semantic reasoning for each footprint. In this framework, the visual model acts as the "eye" by perceiving input, while the language model functions as the "brain", interpreting these inputs logically. The code will be public soon.

![图1reasoning segmentation](https://github.com/user-attachments/assets/5f8987ed-a311-43fd-84e5-1aead39b7e1b)

In this study, we developed a solid benchmark dataset for deep learning-based building-footprint-scale function identification, named the Building Footprint Function (BuFF) dataset. After assigning function labels to each building polygon, we converted the building polygon in Shenzhen to raster format, and paired it with 0.5-meter resolution high-resolution Google Earth remote sensing images to construct the Building Footprint Function (BuFF) dataset. we cropped a total of 12,940 images at 512×512 pixels with 0.5-meter resolution, covering 500,000 buildings. The number of buildings for each function type is shown in the table. The dataset was split into a 7:1:2 ratio, with 9,056 images for training, 1,304 for validation, and 2,580 for testing. The dataset will be public soon.

![buff_dataset](https://github.com/user-attachments/assets/50b93e45-c5c0-443f-bf39-f45b7263dda1)
