# Snake Species Identification

## Description
Snakebite is the most deadly neglected tropical disease (NTD), being responsible for a dramatic humanitarian crisis in global health

Snakebite causes over 100,000 human deaths and 400,000 victims of disability and disfigurement globally every year. It affects poor and rural communities in developing countries, which host the highest venomous snake diversity and the highest-burden of snakebite due to limited medical expertise and access to antivenoms

Antivenoms can be life‐saving when correctly administered but this often depends on the correct taxonomic identification (i.e. family, genus, species) of the biting snake. Snake identification is also important for improving our understanding of snake diversity and distribution in a given area (i.e. snake ecology) as well as the impact of different snakes on human populations (e.g. snakebite epidemiology). But snake identification is challenging due to:

- their high diversity
- the incomplete or misleading information provided by snakebite victims
- the lack of knowledge or resources in herpetology that healthcare professionals have


## DriveTrain Approach

**Objective**: To determine which genus and species a snake belongs to.

**Levers**: We can use image classification to identify the genus and
species (binomial) and then use other factors like country/continent where a snake is most likely to be found.

**Data**: AI Crowd Species Identification Project. Can be found
[here](https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge)

## Current Progress

- Currently, the model has been trained on 50 categories as listed on `categories.txt` on the root folder.
The current model achieves the following metrics


| Metric | Value |
| --- | --- |
| Top-3-accuracy | 0.929 |
| Accuracy | 0.802 |
| F1 (macro) | 0.805 |

## Important Links

- HuggingFace Spaces demo - [link](https://huggingface.co/spaces/Jimmie/snake-species-identification)
- Model - [link](https://huggingface.co/Jimmie/snake-species-identification)
- Training Logs and Runs (Wandb) - [link](https://wandb.ai/jimmiemunyi/the-snake-project-cls)
- Training Reports - [link](https://wandb.ai/jimmiemunyi/the-snake-project-cls/reportlist)
- Training LogBook - `LOGBOOK.md` on the root folder.
- TODO list - `TODO.md` on the root folder.
