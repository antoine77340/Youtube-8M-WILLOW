# Youtube-8M-WILLOW
This is the solution of the kaggle Youtube-8M Large-Scale Video Understanding challenge winners.
For more details about our models,  please read our arXiv paper: ... .

The code is built on top of the Google Youtube-8M starter code (https://github.com/google/youtube-8m)
Please look at their README to see the needed dependencies to run the code.

You will additionally only need to have the pandas python library installed.

Hardware requirement: Each model can be run on a single NVIDIA TITAN X 12 GB GPU. Some of the trained
do not fit with a GPU with at least 10GB of memory. In that case, please do not modify the training batch size 
of these models as it might effect the final results.

Our submitted model is a weighted ensemble of 25 models.
However for the sake of simplicity, this repot will contain a much more simple ensemble of 
7 models that are enough to reach the first place with a significant margin.






Antoine Miech
