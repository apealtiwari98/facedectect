
import sagemaker

BUCKET='sagemaker-studio-xb6jhfgb52g'

sagemaker.Session().download_data('./src',BUCKET,'train.py')
