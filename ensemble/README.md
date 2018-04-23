## Ensemble after training.

`kedan_convert.ipynb`: Converting Kedan's output into submission format.

`result_checker.ipynb`: Checking the correctness of results coming from different group members.

`stacking_logic.ipynb`: Do ensembling among different model outputs.

`make_subm.ipynb`: Combine the final predictions for each label, and concatenate them into a submission file.


- For ensembling purpose, group members run the training code using the same `train_validation_split`.
- After training, we share the predictions for both validation set and testing set.
- We tried two ways of ensembling:
    1. Train a shallow network (only one hidden layer), whose input is the validation predictions, and the output is the actually labels of the validation set.
    2. Use a simple linear combination. The optimal ways is obtained via grid search.


