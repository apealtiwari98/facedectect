
import os
import sagemaker
from sagemaker.tensorflow import TensorFlow

hyperparameters = {'num_epochs' : 1, 'data_dir' : '/opt/ml/input/data/training', 'test_dir' : '/opt/ml/input/data/testing'}
bucket = 'sagemaker-studio-xb6jhfgb52g'
inputs = {'training' : 's3://sagemaker-studio-xb6jhfgb52g/data/'}

#role = 'arn:aws:iam::474988965879:role/service-role/AmazonSageMaker-ExecutionRole-20200502T131852'
role = 'arn:aws:iam::474988965879:role/aws-flask'
estimator = TensorFlow(
    entry_point='train.py',
    source_dir = 'src',
    train_instance_count = 1,
    train_instance_type='ml.c5.4xlarge',
    role = role,
    framework_version = '2.1.0',
    py_version='py3',
    script_mode = True,
    output_path = 's3://sagemaker-studio-xb6jhfgb52g/model/',
)
#'ml.c4.xlarge
estimator.fit(inputs, wait=False)
print(estimator.latest_training_job.name)


