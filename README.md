# Task-3-Neural-Style-Transfer
*COMPANY*: COOTECH IT SOLUTIONS
*NAME*: MAYANK GUSAIN
*INTERN ID*: CITS0D838
*DOMAIN*: ARTIFICIAL INTELLIGENCE
*DURATION*: 4 WEEEKS
*MENTOR*: NEELA SANTOSH KUMAR

*DESCRIPTION*: For Task 3 of my CodTech AI internship, I developed a Neural Style Transfer System using PyTorch and Google Colab to demonstrate how deep learning can creatively blend artistic styles with photographs. Neural Style Transfer is an exciting application of deep learning where the content of one image is combined with the style of another, producing a new image that looks like the original photo painted in the brushstrokes and colors of a chosen artwork.

To build this system, I used Python as the programming language and ran the entire project on Google Colab. Google Colab is an ideal platform for this task because it offers free access to GPUs, which significantly speeds up the training and optimization steps required for style transfer. I also used PyTorch, a popular deep learning framework that provides pre-trained convolutional neural network (CNN) models like VGG19, which are well-suited for extracting features for content and style representation.

The implementation started by preparing two images: a content image, which provides the base structure (for example, a clear photograph), and a style image, which supplies the artistic texture and color patterns (like a painting by Picasso or Van Gogh). Both images were uploaded to Colab and resized to the same dimensions to ensure compatibility during processing.

I loaded the pre-trained VGG19 network from PyTorch’s torchvision.models. This network, originally trained for image classification, serves here as a feature extractor to isolate layers that represent content and style. I defined custom loss functions: one for content loss (which measures how much the generated image differs from the content image) and another for style loss (which measures the difference in texture and color patterns by computing Gram matrices).

The main part of the system is an iterative optimization loop where an initial image (a copy of the content image) is updated step by step. During each iteration, the model calculates the total loss as a weighted sum of content and style losses. The optimizer then adjusts the pixels of the image to minimize this loss, gradually blending the content with the desired artistic style.

I ran the style transfer for about 200–300 iterations to reach a visually appealing result. After the optimization, the output was displayed in the Colab notebook and saved as an image file named output.jpg. To document the results, I also took a clear screenshot showing the original content image, the style image, and the final stylized output. This screenshot, saved as screenshot_task3.png, is included in my GitHub repository for reference and proof of successful implementation.

By completing this task, I gained practical experience in working with deep neural networks for computer vision, understanding how feature extraction works in CNNs, and how artistic patterns can be transferred mathematically. This project illustrates how AI can be used creatively beyond traditional tasks, highlighting the intersection between technology and art. It has strengthened my understanding of advanced PyTorch modules, iterative optimization, and real-world AI applications.

The entire project, including the Colab notebook (task3_neural_style_transfer.ipynb), input images (content.jpg and style.jpg), the generated output, and the proof screenshot, is well-documented and shared in my GitHub repository. This fulfills all the requirements for Task 3 of my CodTech AI internship.

IMAGES USED IN THIS TASK:-




![Image](https://github.com/user-attachments/assets/e1b9d4ab-a2b6-4177-a5a5-d8cda6d9c106)
![Image](https://github.com/user-attachments/assets/b41b0604-4472-4e80-a09c-924c29be854d)


OUTPUT


![Image](https://github.com/user-attachments/assets/c4133f36-9868-41f4-ae2e-5c1881ce9f19)
