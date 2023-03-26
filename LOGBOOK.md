This log book keeps track of the experiments and activities on this project.

## Key
- #nbs - Notebook path location. Unless mentioned otherwise, it is located at: `notebooks/research`.
- #run - Wandb run
- #idea - An idea I want to search later on.
- #report - wandb report of some experiments I ran.
- #demo - Demo people can interact with

## Logs

<hr>

## Sunday 26th Mar 23
- Added a way to deploy model to Hugging Face Hub.
- Updating the HuggingFace Space demo to match the progress so far (binomial instead of genus, 50 classes instead of 10 classes) and returning the top 3 predictions.

## Thursday 23th Mar 23
- Finished up the 480px training. I was able to beat the baseline set before even with the 2nd part of the progressive resizing (224). See report #report [link](https://wandb.ai/jimmiemunyi/the-snake-project-cls/reports/Progressive-Resizing-Experiment-with-50-classes--VmlldzozODY3MzQ5)
- Can we go higher? Also trained it for very big image sizes 800px. It gave the best results but takes a long time (~10 mins per epoch). Also included in the report. This might be a nice final part training. 

## Wednesday 15th Mar 23
- Experimenting with progressive resizing to see if I can beat the baseline set on the previous day. 128 -> 224 -> 384 -> 480.
- Trained until the 384 size. 

## Monday 13th Mar 23
- Expanding to predict the binomial name (both genus and species) instead of just genus. And expanded to predicting 50 classes instead of 10. Things are getting big. Will scale to the full dataset soon.
- Also reporting back top_3 accuracy alongside the normal accuracy and F1 Score(macro).
- The accuracy is lower now than when predicting just the genus. It might be harder to predict both the genus and species. Next task is experimenting on binomial, with previously found practices that worked on genus until I get good accuracy.
- Next, before continuing I will also now train a model to clean the dataset. The authors of the dataset intentionally introduced noisy images (that are not snakes, like humans) to make the dataset more challenging.
- Baseline I am trying to beat - #run [link](https://wandb.ai/jimmiemunyi/the-snake-project-cls/runs/jnslvbtw?workspace=user-jimmiemunyi) 69.6% accuracy. Trained for around 4000 steps, for 50 classes (binomial)
- Also implemented `SaveModelCallback` that monitors the F1Score- Basically, it monitors the training and saves the best model in between, and reloads it after the training. That way, even if the final epochs make the model worse, we are sure we will end up with the best model. The reason we monitor the F1Score is, eventually what we care about is the metrics of the model and that is what we optimize. Domain experts know what accuracy is acceptible in the field. While reducing val_loss is also another good indicator the model is getting better, the loss is mostly for the model than the person training the model.
- The reason for including the top 3 accuracy, is that in my final envisioned inference app for this project, I will be returing the 3 most likely snakes. It can be significantly hard to recognize a snake and returning the 3 most likely snakes might be useful. e.g if 2/3 of the predicted snakes are located in Africa, and you were bitten in Australia, then it is more likely (not foolproof) that the snake that bit you is the third one (slightly flawed argument but it makes sense kinda and might be useful). And if the top 3 accuracy is like 90%, we can confidently say that we are 90% sure the snake you are looking for is among these three snakes, use more criteria (country, behaviour description, look at the images to see if you recognize it etc) to fine tune and pick the one you are looking for

## Sunday 12th Mar 23
- Experimenting with Progressize Resizing and saving model checkpoints and reloading them to continue training or try sth different. Impressive results so far.
- For the presizing, you have to be careful with batch size once you get to higher image sizes. Got a few OOM errors!
- #nbs: `nbs/ProgressiveResizing` The main advantage I am seeing with progressive resizing is time saving. Instead of training your model on big images for 40 epochs, which will take lots of time, you can progressively increase the size and train for 10 epochs and rinse and repeat, which will take significantly reduce the time spent.

## Thursday 23rd Feb 23
- First experiments with MixUp. So far looks to be useful for squeezing some extra performance out of your model. But you have to train for significantly longer to gain some extra results. For example, with a non-mixup model, I got to a final accuracy of 85.13% in 3k epochs while with MixUp I got 87.43% in around 11.5k steps. MixUp also acts as a regularizer. No more overfitting.
- Experimented with 50 classes too, got around 77.46% accuracy in around 23.7k steps with mixup and convnext_tiny. Will train a model and save it and update the demo, then look into self supervised learning and introducing predicting the species + genus together which is the final goal. Currently I am only predicting the genus alone. Looking to finish up this project soon (at least to a point I can now share with the world!) then improve it later on.

## Tuesday 21st Feb 23
- Continuing with augmentations experiments. Augmentations help with overfitting. Without them, the training loss goes down further but the val loss is still poorer than the experiment with the augmentations. From now on I will be training with the simple augmentations.
- Switching my default model from resnet18 to convnext_tiny for experimentations.. It gives better results, and is only slightly slower. Once I am done with experimenting, I will train with a bigger model like convnext_base for production. See #report :https://wandb.ai/jimmiemunyi/the-snake-project-cls/reports/Which-model-for-Experimentations---VmlldzozNjA0NTU2
- Also experimented with `MixUp`.

## Monday 20th Feb 23
- Experimented with augmentations today. The default fastai ones and integrating albumentations too.
- Funny thing happens with the small model, `resnet18` where introducing augmentations and unfreezing the model performs worse than training without both those things. But when using a larger model, `convnext_small` (larger but still considerably smaller), augmentations and fine tuning (unfreezing after some epochs) performs better as expected. I think the former implies underfitting?
- 

## Wednesday 15th Feb 23
- Experimenting the effect of unfreezing the model after some point.
- Training the unfreezed model and then unfreezing and training further seems to be performing well, but I can't get it to beat my previous results without unfreezing. The best I could do is similar performance but I had to train for an extra 10 epochs (so 20 in total) after unfreezing compared to 10 only without unfreezing.
- Will look into weight decay next.
- Probably even use a different model. Currently using `resnet18`

## Tuesday 14th Feb 23
- Resumed working on the project. Experimenting with improving the results using fancy tricks like presizing.
- Experimenting with different image sizes too to see if it makes a different: 64, 128 and 224 before incorporating the presizing trick. Conclusion: Increasing the image size increases the accuracy. #report : https://wandb.ai/jimmiemunyi/the-snake-project-cls/reports/How-does-image-size-affect-accuracy---VmlldzozNTUzNDcz
- Presizing looks good but I have to set up a pipeline for loading the pretrained models.
- Noticed I was training the model without unfreezing, will experiment with that

## Sunday 5th Feb 23
- Implemented model saving logic, learning rate finding.
- Debugging wandb because it takes time to finish (literally hangs). Finally I had to downgrade to a previous version of wandb that was working before (0.13.9)

## Saturday 4th Feb 23
- Started experimenting with Hydra and basic config files and thinking about how they will fit into my worflow. So far it looks like an interesting tool.
- Went through other people's workflow to see what I can learn from them. Links:
	- https://github.com/mit-ll-responsible-ai/hydra-zen
	- https://github.com/lkhphuc/lightning-hydra-template
- For now I will just keep my approach simple enough to do the project then improve on it later on. I don't want it to become a bottleneck to the actual project.
- Repo feels ready for experimentation for the baselines now. Only logic missing is saving the model.
- 

## Thursday 2nd Feb 23
- Resumed working on this project. Took a hiatus with a lot to do from my day job and also had a burn out.
- Finished `dataloader/dataloader.py`
- Experimenting with metaflow and pipelines
- #todo: Fix pylint errors when using MetaFlow.
- Noticed I do not need Metaflow in the meantime, I am going to suspend it until I need it later on.
- Noticed some strange things happening:
	- First, getting some sort of `OSError: image file is truncated` : [Link](https://forums.fast.ai/t/oserror-image-file-is-truncated-38-bytes-not-processed/30806) which i never used to get when I trained in the notebooks
	- Training is slower (8 mins with an even smaller model compared to 2 minutes for one epoch) in my scripts compared to noteboook
	- CPU running hot while training on the GPU.
- The only difference is I am not using the star import `from fastai.vision.all import *` . I am curious if this is what is causing the error (Some things that fastai does differentlys)
- I am going to debug the above errors by recreating the noteboook in script format and see if fastai does anything different.

### debugging
- Realized my problems:
	- I was training with the whole dataframe instead of the sample df of 10 classes.
	- Got the same errors I was getting on the scripts on the original notebook. Looks like I am going to have to find a way to verify the images once I move to the full dataset.
 - The reason the CPU was running hot might be related to using the full dataset (which is large)

## Wednesday 25th Jan 23
- Project restructuring day! Decided to restructure the code from notebooks to python code so as to emulate code in production.
- These are the tools and frameworks that I am going to use:
	- fastai and PyTorch for training the models (thinking of introducing mosaicml composer at some point)
	- wandbai for experiment tracking
	- hydra for configs
	- vscode as my editor with black formatting + ruff linter.
	- notebooks for experimentations and reports
	- metaflow for pipeline orchestration
- I am also seperating the classification code and the future incoming NLP code.
- I am also going to integrate tests into the repos.
- At the end of this, I am going to create a template structure on github for my future projects.

## Tuesday 17th Jan 23
- Working on deploying the first demo on huggingface so as to complete my first iteration. Jeremy advises that after each iteration, you should end up with a working prototype of what you intend to have and then iterate from there. This is because by doing this, you will see the areas that need more working on and improving.
- #demo: https://huggingface.co/spaces/Jimmie/snake-image-classification


## Friday 14th Jan 23
- Continued doing some experiments on a sliced down version of the dataset. I picked the 10 most common `genus` classes and did some experiments and fit models to predict the 10 classes.
- #report: https://wandb.ai/jimmiemunyi/the-snake-project-cls/reports/Baseline-10-Classes--VmlldzozMzI4ODI5
- I will now create a small primitive demo to showcase what I have so far and then iterate on it and improve it.

### Thur 13th Jan 23
- Continued with EDA. Extrapolated these fields from the dataset: `species`. 
- #nbs --> `00_EDA.ipynb
- #idea: Scrap Data from Wikipedia?
- Created a public `wandb` project to track the image classification: https://wandb.ai/jimmiemunyi/the-snake-project-cls
- Started training small models with a fraction of the data to see how models perform. Predicting only a single target: `genus`. Since I picked a fraction of the data, I will randomly select the training and the validation. I will resume using the ones provided once I start training on a significant number of images on the dataset. Nbs --> `01_tiny_baselines.ipynb`
- Created a library with nbdev where I will place commonly used functions and variables called `tsp`. Location same with the github location for the project.



### Wed 12th Jan 23
- Did some initial EDA to see what the data is like. Also started brainstorming on how to structure the classification, i.e. which fields to predict from the images. Current contendars are: `genus`, `species` and `continent`. I will then use that info to predict whether the snake is venomous or not, which is the goal of this project.
- Also started doing some reading around snakes to gain some domain knowledge. Currently reading [The Book of Snakes](https://www.amazon.com/Book-Snakes-Life-Size-Hundred-Species/dp/022645939X) by Mark O'shea.


### Tue 11th Jan 23
- Downloaded the datset from: https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge
- Brainstormed on what the goal of the project should be. Used the [DriveTrain](https://www.oreilly.com/radar/drivetrain-approach-data-products/) approach. Here is the drivetrain info for the classification part of the project:

**Objective**: To determine whether a snake is venomous or not.

**Levers**: We can use image classification to identify the genus, species and country of the snake, then use that information to infer whether the snake is venomous or not. 

**Data**: AI Crowd Species Identification Project.
