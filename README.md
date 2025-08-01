2 Layer binary classification model that performs T forward passes during testing to determine the Epistemic Uncertainty. A forward pass is completed duing training to evaluate Aleatoric uncertainty.

Training:
python main.py --mode train --data_dir /path/to/data

Testing:
python main.py --mode test --data_dir /path/to/data --checkpoint model/model.pth
