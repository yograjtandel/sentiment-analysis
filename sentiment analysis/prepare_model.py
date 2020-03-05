from sentiment import CreateModel
import sentiment
from clean_text import CleanData

CreateModel.data_frame['review'] = CreateModel.data_frame['review'].apply(CleanData.clean_text)
CreateModel.data_frame['review'] = CreateModel.data_frame['review'].str.replace('\d+', '')

CreateModel.prepare_text_for_inpute()
CreateModel.create_model()
CreateModel.train_model()
CreateModel.evaluate_model()
