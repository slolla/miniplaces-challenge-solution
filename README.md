# miniplaces-urop
Solution for the miniplace challenge as shown at https://github.com/CSAILVision/miniplaces

# Documentation
`miniplacesdataset.py` contains a custom Dataset object used to load the train and val datasets from the text files given.
`miniplaces_testdataset.py` contains a custom Dataset object used to load the test dataset (no text file, only from the folder given).
`model.py` contains the layers used to create the model. 
`train_model.py` contains the script to create and train a new model, or load an existing checkpoint and continue training
`test_model.py` contains the script to test on the validation dataset.
`evaluate_model.py` contains the script to evaluate the model on the test dataset and save predictions to a file.
`results_val.txt` contains the results of testing on validation (top one accuracy: 29.8%, top five accuracy: 60%)
`results_test.txt` contains predictions for the test dataset.
