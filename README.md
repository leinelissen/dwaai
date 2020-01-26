# THE FUTURE OF AINDHOVEN
This repository details a prototype that is used to determine whether an accent can be classified as being Brabants or not. It was conceived as a part of the [Designing with Advanced AI](https://osiris.tue.nl/osiris_student_tueprd/OnderwijsCatalogusToonCursusDirect.do?cursuscode=DDM150&cursusInfo=J&collegejaar=2019&aanvangsblok=GS3) course at Eindhoven University of Technology's Industrial Design Department.

## Directory Structure
This repository contains multiple moving parts, which are all individually documented. The directory structure is as follows:
### `/ThinkDSP`
Details the splicing pipeline, and any preprocessing that can be done on the audio samples. It is inspired by the [ThinkDSP book](https://github.com/AllenDowney/ThinkDSP) and related code by Allen Downey.

### `/mfcc`
Is a repository for the back-end pipeline that transforms audio files into MFCCs, as well as trains the models. The data on which the models are trained can be found in [/mfcc/test_dataset](/mfcc/test_dataset) and [/mfcc/training_validation_dataset](/mfcc/training_validation_dataset).

### `/webapp`
Is a repository for the front-end application through which the audio is recorded, processed and communicated to the Python pipeline.

## Authors
* @evavdborn
* @Jordy-A
* @Almarvds
* @leinelissen