import streamlit as st 
import streamlit.components.v1 as stc
# Text Cleaning Pkgs
import neattext as nt
import neattext.functions as nfx
from collections import Counter
import pandas as pd

# Text Viz Pkgs
from wordcloud import WordCloud 
from textblob import TextBlob

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Agg')
import altair as alt 
from PIL import Image
import os
from pathlib import Path

from nltk.tag import pos_tag, map_tag
from datalab_nlp import nlphelper


nlp = nlphelper.nlphelper()




def comment_analyser(sentence):
    """ Comment analyser fucntion"""
    
    words = nlp.tokenize(sentence) 
    postag = pos_tag(words)
    
    output = ""
    
    # profanity
    profanity = nlp.checkprofanity(sentence)
    if len(profanity)>0:
        output += "You should not use profanities like '" + ", ".join(profanity) + "' because it is unprofessional and hurtful. \n"
    
    # judgmental
    judgemental = nlp.checkjudgemental(sentence)
    if len(judgemental)>0:
        output += "Avoid judgemental words like '" + ", ".join(judgemental) + "' because this tends to put recipient in defensive mode. \n"
    
    # personality attack
    personality = set(nlp.personalityattack(nlp.tokenize(sentence)))
    if len(personality)>0:
        output += "Personality attacking like calling them, or even yourself, '" + ", ".join(personality) + "' is damaging to the esteem of the recipient, instead focus on calling out behaviours. \n"
    
    # sentiment
    sentiment = nlp.sentimentscore(sentence)
    if sentiment['score'] < -2:
        output += "The overall comment has very negative sentiment, please consider using the sandwich approach to highlight positive behaviours before giving negative criticism followed by positive suggestions. \n"
    
    # focus on behaviour verbs
    [prp, adj, adv, noun, pnoun, verb] = nlp.p_pos(postag)
    if verb < 13:
        output += "You seem to have insuffient words to describe the person's behaviour, try using more action verbs. \n"
        
    
    # too many nouns
    if noun > 30:
        output += "Your comment seems to have a lot of nouns, try using more action verbs and adjectives to describe behaviours. \n"
        
    # understandable
    flesch = nlp.flesch_read(sentence) #100 -> 5th grade
    if flesch >= 100:
        output += "Looks like your language level is a little too simplistic, consider writing more comprehensive comment rather than short ones. Use the situation-behaviour-impact as a guide. \n"
    if flesch < 30:
        output += "You writing style is too technical or complex for normal person to understand. "
    
    themes = nlp.tagtheme(sentence)
    skillsduties = nlp.skillduties(postag)
    
    if (themes == "") and (len(skillsduties[1]) == 0):
        output += "Hm... there appears to be no work related skills or work actions mentioned, please consider appropriateness of your feedback in workplace setting only and provide specific work behaviours examples to give context. \n"

    if (output == "") and (themes != ""):
        themes_nice = [th[0:th.find('[') ]for th in themes.split("];")[:-1] ]
        output += "Well done. Your comment is well written. You have mentioned about these skills " + ", ".join(themes_nice)
        
    
    return output




HTML_BANNER = """
    <div style="background-color:#3872fb;padding:10px;border-radius:10px;border-style:ridge;">
    <h1 style="color:white;text-align:center;">Feedback analyser </h1>
    </div>
    """

def get_most_common_tokens(docx,num=10):
	word_freq = Counter(docx.split())
	most_common_tokens = word_freq.most_common(num)
	return dict(most_common_tokens)


def plot_most_common_tokens(docx,num=10):
	word_freq = Counter(docx.split())
	most_common_tokens = word_freq.most_common(num)
	x,y = zip(*most_common_tokens)
	fig = plt.figure(figsize=(20,10))
	plt.bar(x,y)
	plt.title("Most Common Tokens")
	plt.xticks(rotation=45)
	plt.show()
	st.pyplot(fig)


def plot_wordcloud(docx):
	mywordcloud = WordCloud().generate(docx)
	fig = plt.figure(figsize=(20,10))
	plt.imshow(mywordcloud,interpolation='bilinear')
	plt.axis('off')
	st.pyplot(fig)


def plot_mendelhall_curve(docx):
	word_length = [ len(token) for token in docx.split()]
	word_length_count = Counter(word_length)
	sorted_word_length_count = sorted(dict(word_length_count).items())
	x,y = zip(*sorted_word_length_count)
	fig = plt.figure(figsize=(20,10))
	plt.plot(x,y)
	plt.title("Plot of Word Length Distribution")
	plt.show()
	st.pyplot(fig)





def plot_mendelhall_curve_2(docx):
	word_length = [ len(token) for token in docx.split()]
	word_length_count = Counter(word_length)
	sorted_word_length_count = sorted(dict(word_length_count).items())
	x,y = zip(*sorted_word_length_count)
	mendelhall_df = pd.DataFrame({'tokens':x,'counts':y})
	st.line_chart(mendelhall_df['counts'])



# Functions
def generate_tags_with_spacy(docx):
	docx_with_spacy = nlp(docx)
	tagged_docx = [[[(token.text,token.pos_) for token in sent] for sent in docx_with_spacy.sents]]
	return tagged_docx

def generate_tags(docx):
	tagged_tokens = TextBlob(docx).tags
	return tagged_tokens

def generate_tags_with_textblob(docx):
	tagged_tokens = TextBlob(docx).tags
	tagged_df = pd.DataFrame(tagged_tokens,columns=['token','tags'])
	return tagged_df 

def plot_pos_tags(tagged_docx):
	# Create Visualizaer, Fit ,Score ,Show
	pos_visualizer = PosTagVisualizer()
	pos_visualizer.fit(tagged_docx)
	pos_visualizer.show()
	st.pyplot()



TAGS = {
            'NN'   : 'green',
            'NNS'  : 'green',
            'NNP'  : 'green',
            'NNPS' : 'green',
            'VB'   : 'blue',
            'VBD'  : 'blue',
            'VBG'  : 'blue',
            'VBN'  : 'blue',
            'VBP'  : 'blue',
            'VBZ'  : 'blue',
            'JJ'   : 'red',
            'JJR'  : 'red',
            'JJS'  : 'red',
            'RB'   : 'cyan',
            'RBR'  : 'cyan',
            'RBS'  : 'cyan',
            'IN'   : 'darkwhite',
            'POS'  : 'darkyellow',
            'PRP$' : 'magenta',
            'PRP$' : 'magenta',
            'DET'   : 'black',
            'CC'   : 'black',
            'CD'   : 'black',
            'WDT'  : 'black',
            'WP'   : 'black',
            'WP$'  : 'black',
            'WRB'  : 'black',
            'EX'   : 'yellow',
            'FW'   : 'yellow',
            'LS'   : 'yellow',
            'MD'   : 'yellow',
            'PDT'  : 'yellow',
            'RP'   : 'yellow',
            'SYM'  : 'yellow',
            'TO'   : 'yellow',
            'None' : 'off'
        }



def mytag_visualizer(tagged_docx):
	colored_text = []
	for i in tagged_docx:
		if i[1] in TAGS.keys():
		   token = i[0]
		   print(token)
		   color_for_tag = TAGS.get(i[1])
		   result = '<span style="color:{}">{}</span>'.format(color_for_tag,token)
		   colored_text.append(result)
	result = ' '.join(colored_text)
	print(result)
	return result


def main():
	"""Main program here"""
    
	stc.html(HTML_BANNER)
	menu = ["Home","About"]


	choice = st.sidebar.selectbox("Menu",menu)

	if choice == 'Home':
		st.subheader("Feedback Analysis")
		
		raw_text = st.text_area('Your Feedback Here', height = 10)
		if len(raw_text) > 2:   
			st.subheader('Analysis of your feedback text:')
			st.write(comment_analyser(raw_text))
            
			col1,col2 = st.beta_columns(2)
			process_text = nfx.remove_stopwords(raw_text)
			with col1:
				with st.beta_expander("Preview Tagged Text"):
					tagged_docx = generate_tags(raw_text)
					processed_tag_docx = mytag_visualizer(tagged_docx)
					stc.html(processed_tag_docx,scrolling=True)

				with st.beta_expander("Plot Word Freq"):
					st.info("Plot For Most Common Tokens")
					most_common_tokens = get_most_common_tokens(process_text,20)
					# st.write(most_common_tokens)
					tk_df = pd.DataFrame({'tokens':list(most_common_tokens.keys()),'counts':list(most_common_tokens.values())})
					# tk_df = pd.DataFrame(most_common_tokens.items(),columns=['tokens','counts'])
					# st.dataframe(tk_df)
					# st.bar_chart(tk_df)
					brush = alt.selection(type='interval', encodings=['x'])
					c = alt.Chart(tk_df).mark_bar().encode(
						    x='tokens',
						    y='counts',
						    opacity=alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7)),
						    ).add_selection(brush)
						
					st.altair_chart(c,use_container_width=True)

			with col2:
				with st.beta_expander('Processed Text'):
					
					st.write(process_text)

				with st.beta_expander("Plot Wordcloud"):
					st.info("word Cloud")
					plot_wordcloud(process_text)
	                

		elif len(raw_text) == 1:
			st.warning("Insufficient Text, Minimum must be more than 1")


		




		

	elif choice == "About":
		sentences = ["COO is an asshole and dumb ass. I hate coming to work because of her.",
            "My manager is a bit too emotional. She is a stupid bitch. She should keep her mouth shut.",
             "My manager is an real idiot, likes to micromanage me and tell me what to do as if as I am dumb. I lose interest in putting best effort in work because of him.",
             "I want you to start writing the marketing reports. You should not be waiting for me each time to tell you to do so.",
             """Cannot do anything right, I kept asking her to change the report but she did not. She kept complaining things 
             are not working but I don't see her making any efforts to correct them. I don't know how to help her anymore.""",
             "I like her. You will like her too. She is nice.",
             "He is good. I enjoy working with him.",
             "He is able to write Python codes, SQL script, manage AWS Cloud, and has PMP qualifications. Very good in what she is doing. ",
             "I have heard about the 4 Future Fit themes: Customer centricity, Acting as One, Agility and Accountability",
             "Alice did not show participation in the zoom meetings and was not responsive. She need to work on her communication skills.",
             """Fourier based methods such as the Lomb–Scargle periodogram [35], the fast Fourier 
transform-nonlinear least squares (FFT-NLLS) algorithm [36], harmonic regression [37] 
and the spectrum resampling method [38] can be used to extract further parameters, 
namely acrophase, amplitude and period, that are typically of interest to studies of 
circadian rhythmicity.""",
"""The federal industrial laws about workplace agreements have changed a number of times in recent years. Before the WorkChoices laws came 
into effect in March 2006, workplace agreements were called Certified Agreements (agreements between an employer and a group of employees) 
and Australian Workplace Agreements or AWAs (agreements between an employer and an individual employee).""",
"""I'm really happy with your determination to finish this project. I know it wasn't easy, but I knew you could do it. 
Your helpful attitude makes it clear that you can continue to take on new challenges and grow with the company. Thank you for your extra effort.""",
"""Create greater transparency and understanding for how pay raises and bonuses are applied. 
At one company, during the performance review process each employee was given a packet 
with their feedback and details on their merit increases and bonus; it explained the 
precise formula for how their bonus and merit raise was calculated. Because the formula 
was the same for everyone, it increased the perceived fairness of company-wide compensation.
""",
"""All the training you have done with Rico has been very helpful. 
You're giving him a great start to his internship. I have taken notice of your leadership skills and will keep 
this in mind for future projects.""",
]
            

		output = ""      
		for s in sentences:
			output += "\n--Comment--\n"
			output += "\n" + s + "\n"
			output += "\n--Reply--\n"
			output += "\n" + comment_analyser(s) + "\n"
    
		st.subheader("Text Analysis NLP App")
		st.write("Example: ")
		st.write(output)
		


					

if __name__ == '__main__':
	main()
