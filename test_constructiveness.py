#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:48:50 2023

@author: wai.yip
"""


from nltk.tag import pos_tag, map_tag
from datalab_nlp import nlphelper


nlp = nlphelper.nlphelper()
text_array = ["COO is an asshole and dumb ass. I hate coming to work because of her.",
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

for text in text_array:
    print("\n----")
    print("\n"+text)
    print("\nConstructiveness " + str(nlp.constructiveness(text)))