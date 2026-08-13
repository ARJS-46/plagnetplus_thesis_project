# Logbook
Meeting 1 (Friday, 22 May 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor)

Key points discussed:
Welcome meeting, disussed intial steps for project and a plan for the first week was outlined, plan for week 1 is to begin work on the Project plan with a focus on the literature review and getting the pre-existing code to work including attempting to segment the image data independently in order to aid AS's understanding of the pre-existing code.

Feedback received:
Initial Meeting. No feedback yet.

Work plan before next meeting:
Begin work on the Project plan with a focus this week on the literature review, running the pre-exisitng code and attempting to segment images.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

NOTE ON INACTIVITY ISSUE FROM ~5th JUNE: This inactivity was due to a focus at this time on pure reading and research into the topics surrounding the project. At this time no code had been produced and little to no writing of the project plan had yet been produced so there was little reason to yet commit to the repository.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 2 (Friday, 05 June 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor) 

Key points discussed:
Discussed progress with the project plan, what sort of sources would be most beneficial to focus on (ones which apply SSL to image data even if they're not focusing on volcanology or related fields). Futhermore, discussed initial issues with using the CX3 cluster and arranging to meet with the RCDS in order to remedy this.

Feedback received:
Continue to work on project plan.

Work plan before next meeting:
Continue to work on a draft for the project plan to hand in on Tuesday the 8th ahead of the next scheduled meeting on the 9th.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 3 (Wednesday, 10 June 2026)

Location: London (In-Person)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor) 

Key points discussed:
Spent the session going over a draft of the project plan produced for the meeting.

Feedback received:
Project plan was received well with minor fixes suggested and some ideas on figures to improve the quality of the submission.

Work plan before next meeting:
Continue to work on project plan until the deadline on Friday 12 June (12 Noon) then switch to work on the PlagNet code. Begin with attempting to optimise the segmentation and pre-processing code for the volcanic plagioclase thin section images to improve overall data quality before moving on to training models. Also, make sure that all data can/has been segmented before - in the google drive it appears as though only the segmented plagioclase crystals for Aegina are present.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 4 (Thursday, 18 June 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor) 

Key points discussed:
AS discussed progress with CN so far the prject plan has been completed, revised and submitted, and now preprocessing and segmentation code has been studied to discover areas that may be optimizable. Furthermore, a first attempt at producing a basic SimCLR model to compare against the SimSiam model has started to be developed. AS raised some issues with availability of some data in the google drive.

Feedback received:
CN will check with co-creator of PlagNet about the possibility of missing data in google drive.

Work plan before next meeting:
Continue developing models and start to produce draft section of final report regarding methods.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 5 (Thursday, 25 June 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor) 

Key points discussed:
Discussed the training of the SimCLR model which is nearly complete. THe model has been coded and has trained first on a small batch as a test. 

Feedback received:
Some feedback received on the project plan submission and minor areas were highlighted where things may be improved such as giving a more in-depth explanation of model choices for the project.

Work plan before next meeting:
To continue with the development and training of new models. At a minimum train the SimCLR model sucessfully and begin to develop another model named in the project plan. Additionally, continue to draft sections of the final report that can be drafted at this time.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 6 (Friday, 3 July 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor), Alexis Peng (AP, PlagNet Co-Creator) 

Key points discussed:
AP and As shared their respective work on the project. AP shared a new method for clustering the plagioclase images. AS shared progress with the SimCLR model and preliminary results/comparison to SimSiam. It was agreed SSL models (especially SimCLR) appear to be a good approach for the problem. Research into other suggested models will continue.

Feedback received:
SimCLR results were received well.

Work plan before next meeting:
Continue to work on producing/implementing new models (DeepCluster, SWaV, DINO) whilst building a packaging framework which will speed up and streamline the process of running and evaluating each new model.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

NOTE ON INACTIVITY ISSUES FOR THE PERIOD BETWEEN 4th July - 16 July: A combination of Illness (flu - can provide medical note if needed) and a later issue regarding github login (now resolved) are the reason for the inactivity issues at these times. Work was still done during this time but partially at a reduced rate due to illness. Evidence of this work can be seen in the finished Aegina SimCLR analysis notebook updated just after this period.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Meeting 7 (Monday, 20 July 2026) 

NOTE: The reason there was no meeting between for a long period is due to supervisor CN being away on leave. A plan for work on the project was agreed upon between AS and CN before this break 

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor), Alexis Peng (AP, PlagNet Co-Creator) 

Key points discussed:
AP discussed issues faced with segementation of certain samples with resorption textures and/or where very fine plagioclase is present. From inspecting problem images it seems segmentation may be having issues with certain atreas of extreme low and high brightness. AS discusse progress made on producing a unified, validated dataset for all given samples (Aegina, MtStHelens, Rumania, FCT, Methana) and the issues faced with producing a deepcluster model to compare with SimCLR and SimSiam Legacy. 

Feedback received:
Keep going for now with deepcluster but it prooves toohard move onto the next viable model.

Work plan before next meeting:
continue with work on deepcluster on aegina whilst attempting to make a start on one new model. Finalise combined dataset for training aegina-certified models on.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 8 (Monday, 27 July 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor)

Key points discussed:
discussed ways to move ahead with the 4 completed baseline models. gridsearch will be attempted despite large likely compute time. Clustering results were show to CN and it was suggested that AS investigates which clusters fall into which PCA spaces and that elbow plots are used to find optimal k values for each model. Furthermore the remainder of the project was planned and discussed. AS will also begin planning and drafting the discussion this week.

Feedback received:
see key points section.

Work plan before next meeting:
create elbow plots, explore were samples fall in PCA space, draft discussion, look into literature for optimisation methods. In additon to this I should also look into other models/DINOv2.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 9 (Monday, 3 August 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor)

Key points discussed:
Discussed progress with Gridsearch. Two grdisearches for the best performing models on the unified dataset (simsiam + swav) are being run. However progress is slow as the hpc is extremely busy and the queue is taking days to clear. so far about 1/4 of the gridsearch for these models is complete. The written report was also discussed and the writing and structure for the methodology and discussion were discussed in detail. Finally some model comparison plots showing extreme sample examples in PCA space and elbow plots for each model and the resultant improved clusters were shared.

Feedback received:
The elbow plots and PCA extremes plots were received well. The main focus now is to push on with the project report. 

Work plan before next meeting:
Draft the first two sections of the final report (1.0 and 2.0) this includes introductory sections such as exisiting work and novel components of project and also the methodology. Furthermore, continue monitoring the progress of the gridsearch running on the hpc. Lastly, I will begin planning and building the package to hold all of the work produced so far.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 10 (Monday, 10 August 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor)

Key points discussed:
Discussed delay in gridsearch progress due to large (over 1000 long) queue on HPC. Discussed best ways to format and organise project report and the best plots to include in the report as figures.

Feedback received:
Continue to work on the report writing.

Work plan before next meeting:
Draft a near-complete version of the project report for CN to read and give feedback on early. Look into what other models/gridsearches can be run. Run simsiam and swav models optimised after gridsearch? Run gridsearch for the other two less well-performing models for balance?

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Meeting 11 (Monday, 17 August 2026)

Location: London (Meeting Online)

Present in Meeting:
Alexander Stringer (AS, Student), Chetan Nathwani (CN, Primary Internal Supervisor)

Key points discussed:
-

Feedback received:
-

Work plan before next meeting:
-