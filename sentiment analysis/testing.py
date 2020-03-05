import pickle
from clean_text import CleanData
from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
from keras.models import load_model

max_len = 250
trunc_type = 'post'
padding_type = 'post'

class test(object):
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
		# new_sentences = 'Total bill for this horrible service ? Over $ 8Gs . These crook actually have the nerve to charge u $ 69 for 3 pill . I checked online the pill can be have for 19 cent EACH ! Avoid Hospital ERs at all cost'
		new_sentences = "I *adore* Travis at the Hard Rock 's new Kelly Cardenas Salon ! I 'm always a fan of a great blowout and no stranger to the chain that offer this service ; however , Travis have take the flawless blowout to a whole new level ! Travis 's greets you with his perfectly green swoosh in his otherwise perfectly style black hair and a Vegas-worthy rockstar outfit . Next come the most relax and incredible shampoo -- where you get a full head message that could cure even the very bad migraine in minute -- - and the scent shampoo room . Travis have freakishly strong finger ( in a good way ) and use the perfect amount of pressure . That be superb ! Then start the glorious blowout ... where not one , not two , but THREE people be involve in do the best round-brush action my hair have ever see . The team of stylist clearly get along extremely well , a it 's evident from the way they talk to and help one another that it 's really genuine and not some corporate requirement . It be so much fun to be there ! Next Travis start with the flat iron . The way he flip his wrist to get volume all around without over-doing it and make me look like a Texas pagent girl be admirable . It 's also worth note that he do n't fry my hair -- something that I 've have happen before with less skilled stylist . At the end of the blowout & style my hair be perfectly bouncey and look terrific . The only thing well ? That this awesome blowout last for day ! Travis , I will see you every single time I 'm out in Vegas . You make me feel beauuuutiful !"
		new_sentences.strip()
		new_sentences = CleanData.clean_text(new_sentences)
		new_sentences  = new_sentences.split()
		new_sentences = ' '.join(re.sub('[0-9]','',i) for i in new_sentences)
		new_sentences  = new_sentences.split()		
		new_sentences = tokenizer.texts_to_sequences(new_sentences)
		sequence1 = []
		
		for i in new_sentences:
			for x in i:
				sequence1.append(x)
		print(sequence1)
		mylist = []
		mylist = [sequence1]
		sequence1 = pad_sequences(mylist, maxlen=max_len,padding = padding_type,truncating = trunc_type)
		print("--------index sequence for sentence--------",sequence1.shape)
		print(sequence1)
		model = load_model('model.h5')
		predicted = model.predict(sequence1)
		labels = ['nagetive' , 'positive' , 'neutral' ]
		print(predicted ," " ,labels[np.argmax(predicted)])
