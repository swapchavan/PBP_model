# Model to predict PBP

conda_env folder : requirments.txt file and requirements.yml file for creating python env

Data folder : Dependent and independent variables

scipts folder : python execute_DNN_multi_task.py RDKIT_2D_descr_248_chem.csv classes_248.csv values_248.csv 10 2 --output_dir="YOUR DESTINATION" --device='cpu'

models folder : Model directory storing all scalers and pytorch models

evaluations folder : python evaluation_excel_input.py "Queries_to_test.xlsx" "YOUR/MODEL/DESTINATION" "YOUR/RESULT_storing/DESTINATION"
