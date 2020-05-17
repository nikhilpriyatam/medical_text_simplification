Keep all the source and target files in this folder.
Keep all the prediction (hypothesis) files in the postprocessed folder.

Run the evaluation code as follows:
python evaluate.py --path <path_to_evaluation_folder>

This will save a file called metrics.csv in your current working directory.
You can also pass the argument --print_metrics to the evaluate.py code to print the output metrics in a pandas dataframe.
