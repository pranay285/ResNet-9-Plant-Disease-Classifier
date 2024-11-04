# ResNet-9-Plant-Disease-Classifier
My project harnesses deep learning with ResNet-9 to detect plant diseases accurately, surpassing traditional methods. By analyzing subtle visual cues, it promises early detection for healthier crops and sustainable agriculture.

Plant Disease Classification using Deep Learning 
Abstract 
In this project, I have attempted research on "Plant Disease Classification using Deep Learning" The timely identification of plant diseases is essential for mitigating their rapid spread. However, the challenge lies in distinguishing between visually similar manifestations of various plant ailments. To address this, we enhance plant disease detection by focusing on subtle distinctions in visual attributes among these distinct diseases. Employing the distinctive ResNet-9 architecture, our method excels in capturing nuanced visual differences among diverse plant diseases. This uniqueness contributes to its superior accuracy and effectiveness in plant disease identification compared to conventional approaches. To address this issue, I utilized deep learning methods to classify Plant diseases, using data from the publicly available PlantVillage dataset for training and testing the models. 
Introduction 
Plant diseases pose a significant threat, affecting approximately 70% of global plant populations due to diverse factors like weather and pathogens. Timely disease detection is crucial to prevent spread and safeguard crops. Deep learning, an advanced form of machine learning, proves invaluable in this context. It discerns subtle differences in plant visual features, enabling accurate disease classification. Unlike conventional methods, deep learning models like CNNs, VGG, and ResNet offer enhanced accuracy by learning complex patterns from vast datasets. By addressing inter-class similarities and variations, deep learning empowers effective and early disease identification, contributing to sustainable agriculture and food security.. The accuracy rates of deep learning models for Plant diseases classification are high. In this project, we have reviewed the deep learning models employed by the authors of the referenced paper and implemented new models to compare with their previous findings. 
Goal 
We need to build a model, which can classify between healthy and diseased crop leaves and also if the crop has any disease, predict which disease is it. The author in his paper has worked on Models DENSENET121 , MOBILENET-V2 and I used RESNET 9 for this model. So compare the results and predict which model is best by accuracy.
Methods 
Data Collection: 
In our study, we employed the openly accessible PlantVillage dataset , comprising 38 distinct plant disease classes encompassing a total of 19,306 images of both diseased and healthy plant leaves. These images were gathered within controlled environments, ensuring a comprehensive representation of various disease conditions 
. 

Data Pre-processing: 
In the context of data preprocessing, the torchvision.datasets class proves valuable for accessing both commonly used and specialized datasets, including the ability to handle custom data arrangements. Particularly, the subclass torchvision.datasets.ImageFolder is employed in this project for loading image data that adheres to a specific organizational structure. 
After data loading, a crucial step involves transforming the pixel values of images, typically ranging from 0 to 255, into a normalized range of 0 to 1. This normalization is important for neural networks, as it promotes effective convergence during training. The entire pixel value array is converted into torch tensors, and then each value is divided by 255, ensuring standardized input for the network. 
An examination of the image shape reveals a format of (3, 256, 256), with 3 representing the RGB color channels and 256 x 256 denoting the width and height of the image. 
For optimizing the neural network's training process, the concept of batch size becomes relevant. Batch size determines the number of samples processed simultaneously during forward propagation. Larger datasets are handled by the DataLoader subclass from torch.utils.data, facilitating efficient memory utilization. By specifying the batch size, we control the number of samples in each batch. 
To introduce diversity and robustness, setting shuffle=True randomizes the dataset order during training. This prevents batches from looking similar between epochs and enhances the model's adaptability.Moreover, the parameter num_workers determines the number of parallel processes used to generate batches

Modeling: 
ResNet-9's simpler structure and shallower depth make it computationally efficient, less prone to overfitting, and easier to interpret compared to DENSENET121 and MOBILENET-V2. It is well-suited for resource-constrained environments, faster experimentation, and serving as a practical baseline for benchmarking. So after implementing the model we can compare the models. Both DENSENET121 and MOBILENET-V2 have attained high accuracy rates in a variety of computer vision tasks like object recognition and picture segmentation because they were trained on extensive image classification datasets like ImageNet. 
ResNets introduce residual blocks to establish connections between layers, mitigating overfitting and vanishing gradients. This technique fosters direct links between layers, even those separated by a few steps, enhancing the training stability of deep neural networks.


Our ImageClassificationBase class serves as the foundation for essential functions. The training_step method gauges model performance by quantifying training or validation "wrongness" using cross-entropy, overcoming limitations of non-differentiable accuracy metrics. In contrast, the validation_step function assesses accuracy through prediction-label comparisons, utilizing a predefined threshold for comprehensive evaluation. The validation_epoch_end function consolidates validation and training metrics while disabling gradient tracking to ensure accurate assessment. 
The epoch_end method furnishes a comprehensive summary of validation and training progress after each epoch, including informative learning rate adjustments. Additionally, the accuracy function computes batch-wise accuracy, contributing a crucial metric to the fit_one_cycle process. 
Preceding model training, we've developed the evaluate function to manage validation, and the comprehensive fit_one_cycle function for effective training. This encompassing process integrates advanced techniques, such as dynamic Learning Rate Scheduling, Weight Decay for regularization, Gradient Clipping for stable training, and Learning Rate Recording for insights into adaptive learning rate adjustments applied throughout training. 
Results 
In this section we present our findings.Firstly using the results provided by the author and then implementing the updated code for the project, I have plotted the Accuracy vs Epochs graph 
Accuracy vs Epochs


from the figure you can see the accuracy increases as the epochs are increasing from 6. We achieved an accuracy of 98%. Which is nearly the same as what the author has achieved but with a simpler and easier structure which requires less computational power and time and which has a very easy architecture. 
The Results of the other models are 
Model 
Accuracy
DenseNet121 
97.10%
MobileNet-V2 
92.10%
Resnet 9 
97.90%



 
Prediction Results 


To Summarize, this project was to understand and implement what the author of the paper “plant disease classification using deep learning” had implemented. The author in the paper explains the effectiveness of deep learning models which can be used to classify plant diseases.ResNet 9 achieved a commendable accuracy of 98% utilizing the ResNet 9 architecture, comparable to the performance of more complex models like DenseNet121 and MobileNet-V2. ResNet 9 offers a compelling advantage with its simplified yet effective structure, demanding fewer computational resources and time. Its intuitive architecture facilitates ease of implementation and interpretation, making it an attractive choice for practical applications. This outcome underscores the potential of ResNet 9 as a robust and efficient solution for plant disease classification, warranting its consideration in future projects and deployments. 
Residual Networks (ResNets) exhibit notable performance gains in image classification tasks through parameter adjustments and the incorporation of strategies like learning rate scheduling, gradient clipping, and weight decay. Remarkably, the model achieves flawless predictions for all images within the test dataset, demonstrating exceptional accuracy and proficiency.. 
Future work : 
In forthcoming stages, there is an avenue for enhancing our project's dataset by augmenting the diversity of classes and introducing novel disease types. This augmentation offers the opportunity to investigate our model's performance as we broaden the spectrum of diseases and class variations. By implementing cutting-edge deep learning methods, we aim to elevate the accuracy of classification for the expanded dataset. To further enhance plant classification, we can also explore the integration of advanced data augmentation strategies and leverage transfer learning from pre-trained models. The continuous exploration of innovative techniques and leveraging domain-specific insights could potentially lead to more refined and effective plant disease classification. 
