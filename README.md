# Automated game bot detection in MMORPGs via DL and gameplay TF behavioral features
 This code is used for  Master Thesis "Automated game bot detection in MMORPGs via DL and gameplay TF behavioral features"


These are code files I write for my thesis,
following link to dataset files which I used 


https://github.com/TorchCraft/StarData - DataSet


https://github.com/phoglenix/ScExtractor - Preprocessing Tool 


                                                 **Abstract**

Gaming is among the most common industries and people spending money on them, particularly MMORPG (Multiplayer Online Role-Playing Games). With the continuing growth of the internet services market and the engaging nature of MMORPGs, criminal activities in the online world have grown significantly. Gaming bots are automated programs that help scam users to cause instability and a breakdown of user interests in the gaming environment.  The development and advances in in-game bots exhibiting great dynamicity and variety, which makes it incredibly difficult to detect the game bot in MMORPGs. Indeed this issue arises the need to design a system that can detect the game bot dynamically even when the game bot is advanced its behavior. In this study, we conducted a comprehensive literature review to review the existing techniques and their drawbacks used for game bot detection in MMORPGs. 
The objective of this work is to design a system that can automatically detect the game bots in MMORPG to recognize and distinguish between human and bot players. The proposed identification model can be deployed in real-world MMORPGs that can detect cheat via player behavior sequences during gameplay. To be lightweight and efficiently interpretable to other environments, the model employs a minimal feature set including an amount with an observation period. We used StarCraft: Brood War dataset to conduct the experimentations in MMORPGs. We designed the detector using a deep learning algorithm which is Long-Short Term Memory (LSTM). LSTM can handle the live streams of input data efficiently in a short time.  Experimentation on a real-world dataset shows that the proposed approach has achieved significant performance improvements over existing approaches. We performed the training and testing on the dataset with the detector, also performed the validation of the detector with 10-fold cross-validation. Results showed the greater precision and accuracy of detection rate in terms of percentage than the previous studies. The accuracy detection score on the barred account list was improved to 96.06%.
