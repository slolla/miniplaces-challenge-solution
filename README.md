# miniplaces challenge solution
Solution for the miniplace challenge as shown at https://github.com/CSAILVision/miniplaces

# Documentation
`miniplacesdataset.py` contains a custom Dataset object used to load the train and val datasets from the text files given. <br/>
`miniplaces_testdataset.py` contains a custom Dataset object used to load the test dataset (no text file, only from the folder given). <br/>
`model.py` contains the layers used to create the model. <br/>
`train_model.py` contains the script to create and train a new model, or load an existing checkpoint and continue training. <br/>
`test_model.py` contains the script to test on the validation dataset. <br/>
`evaluate_model.py` contains the script to evaluate the model on the test dataset and save predictions to a file. <br/>
`results_val.txt` contains the results of testing on validation (top one accuracy: 29.8%, top five accuracy: 60%). <br/>
`results_test.txt` contains predictions for the test dataset.<br/>
