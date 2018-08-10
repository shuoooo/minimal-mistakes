---
title: "First Step Towards Deep Learning"
mathjax: true
---

I have just finished the [Deep Learning Specification](deeplearning.ai) on Coursera, taught by Andrew Ng. I found this course truly remarkable. Andrew and his team did put a lot of thoughts into this course, and provide an easily accessible way for us to learn both theories and practices of deep learning.

In this blog, some of the important and highly influential papers of deep learning are listed as follows, as well as some additional materials that may help you. This awesome specification has five courses, and the provided papers are provided following this separation. You may also find some deep learning reading list from

1. [Deep learning reading list](http://deeplearning.net/reading-list/)
2. [Deep learning papers reading roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)
3. [The most cited papers in computer vision and deep learning](https://computervisionblog.wordpress.com/2016/06/19/the-most-cited-papers-in-computer-vision-and-deep-learning/)
4. The popular "[Deep Learning](deeplearningbook.org)" book
5. [fast.ai](fast.ai) - making neural nets uncool again
6. [Neural Networks and Deep Learning free online book](http://neuralnetworksanddeeplearning.com/)
7. Harvard CS281: [Advanced Machine Learning](https://www.seas.harvard.edu/courses/cs281/)
8. Ian Goodfellow's GAN: https://arxiv.org/abs/1406.2661
9. Geoffrey Hinton's Capsule Networks (Article Explanation): [Part1](https://medium.com/ai³-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b), [Part2](https://medium.com/ai³-theory-practice-business/understanding-hintons-capsule-networks-part-ii-how-capsules-work-153b6ade9f66), [Part3](https://medium.com/ai³-theory-practice-business/understanding-hintons-capsule-networks-part-iii-dynamic-routing-between-capsules-349f6d30418)
10. Distill - Latest articles about machine learning: https://distill.pub/
11. Stanford CS230 by Andrew Ng: http://cs230.stanford.edu/

## 1. Neural Networks and Deep Learning

- [The matrix calculus you need for deep learning](https://arxiv.org/abs/1802.01528)

------

## 2. Improving Deep Neural Networks: Hyper-parameter tuning, Regularization, and Optimization

Dropout:

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). [Dropout: A simple way to prevent neural networks from overfitting.](http://jmlr.org/papers/v15/srivastava14a.html) Journal of Machine Learning Research, 15 , 1929–1958.

Initialization:

- He, K., Zhang, X., Ren, S., and Sun, J. (2015). [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.](https://arxiv.org/abs/1502.01852) arXiv preprint arXiv:1502.01852
- Glorot, X. and Bengio, Y. (2010). [Understanding the difficulty of training deep feedforward neural networks.](http://proceedings.mlr.press/v9/glorot10a.html) In AISTATS’2010

Optimization:

- RMSprop: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- Kingma, D. and Ba, J. (2014). [Adam: A method for stochastic optimization.](https://arxiv.org/abs/1412.6980) arXiv preprint arXiv:1412.6980
- Dauphin, Y., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., and Bengio, Y. (2014). [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization.](https://arxiv.org/abs/1406.2572) In NIPS’2014

Hyper-parameter tunning:

- Bergstra, J. and Bengio, Y. (2012). [Random search for hyper-parameter optimization.](http://www.jmlr.org/papers/v13/bergstra12a.html) J. Machine Learning Res., 13 , 281–305.
- Bergstra, J, et. al. [Algorithms for Hyper-Parameter Optimization. ](http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)Advances in Neural Information Processing Systems (pp. 2546-2554).

Batch Normalization:

- Ioffe, S. and Szegedy, C. (2015). [Batch normalization: Accelerating deep network training by reducing internal covariate shift.](https://arxiv.org/abs/1502.03167) arXiv preprint arXiv:1502.03167, 2015.

--------

## 4. Convolutional Neural Networks

Classic Networks

- **LeNet-5:** LeCun et al, [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), Proceedings of the IEEE.
- **AlexNet:** Krizhevsky et al, "ImageNet Classification with Deep Convolutional Neural Networks": <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>
- **VGG-16:** Simonyan et al, "Very Deep Convolutional Networks for Large-Scale Image Recognition": <https://arxiv.org/pdf/1409.1556.pdf>

ResNets

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition": <https://arxiv.org/abs/1512.03385>

Networks in Networks and 1$\times$1 Convolutions

- Min Lin, Qiang Chen, Shuicheng Yan, "Network In Network": <https://arxiv.org/abs/1312.4400>

Inception Networks

- Christian Szegedy, and lots of others, "Going Deeper with Convolutions": <https://arxiv.org/abs/1409.4842>

Convolutional Implementation of Sliding Windows

- Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, Yann LeCun, "OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks": <https://arxiv.org/abs/1312.6229>

Bounding Box Predictions:

- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, "You Only Look Once: Unified, Real-Time Object Detection": <https://arxiv.org/abs/1506.02640>

Region Proposals (R-CNN)

- Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation": <https://arxiv.org/abs/1311.2524>
- Girshik, 2015 - Fast R-CNN
- Ren et al, 2016 - Faster R-CNN: Toward real-time object detection with region proposal networks

YOLO

- Redmon et al, 2016 - YOLO9000: Better, Faster, Stronger

Siamese Network

- Taigman et al, "DeepFace: Closing the Gap to Human-Level Performance in Face Verification": <https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf>

Triplet Loss

- Florian Schroff, Dmitry Kalenichenko, James Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering": <https://arxiv.org/abs/1503.03832>

What are deep ConvNets learning:

- Matthew D Zeiler, Rob Fergus, "Visualizing and Understanding Convolutional Networks": <https://arxiv.org/abs/1311.2901>

Neural Stype

- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, "A Neural Algorithm of Artistic Style": <https://arxiv.org/abs/1508.06576>

--------

## 5. Sequence Models

GRU

- On the Properties of Neural Machine Translation: Encoder-Decoder Approaches
- Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling

LSTM:

- [LSTM Sepp Hochreiter et al](http://www.bioinf.jku.at/publications/older/2604.pdf)

Skip-Grams, Hierarchical Softmax:

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

Negative Sampling:

- Distributed Representations of Words and Phrases and their Compositionality

Glove:

- GloVe: Global Vectors for Word Representation

Debaising Word Embeddings:

- Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings

Sequence to Sequence Model:

- Sequence to Sequence Learning with Neural Networks
- Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

Image Captioning:

- Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)
- Show and Tell: A Neural Image Caption Generator Oriol Vinyals
- Deep Visual-Semantic Alignments for Generating Image Descriptions

Bleu Score:

- BLEU: a Method for Automatic Evaluation of Machine Translation

Attention based intuition:

- Neural Machine Translation by Jointly Learning to Align and Translate
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

Speech Recognition:

- Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
