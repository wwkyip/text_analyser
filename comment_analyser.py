#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:57:54 2020

@author: winnyyip
"""


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
        output += "You writing style is too technical or complex for normal person to understand. \n"
    
    themes = nlp.tagtheme(sentence)
    skillsduties = nlp.skillduties(postag)
    
    if (themes == "") and (len(skillsduties[1]) == 0):
        output += "Hm... there appears to be no work related skills or work actions mentioned, please consider appropriateness of your feedback in workplace setting only. \n"

    if (output == "") and (themes != ""):
        output += "Well done. Your comment is well written. You have mentioned about these skills " + themes + ", correct?"
        
    
    return output


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
            
            
for s in sentences:
    print("\n--Comment--")
    print(s)
    print("\n--Reply--")
    print(comment_analyser(s))
