import os
import openai
import spacy

from spacytextblob.spacytextblob import SpacyTextBlob



def check_engine(engine):
    return openai.Engine.retrieve(engine)


def text_generation(user_prompt, temperature=0.7, top_p=0.9, token_limit=512, max_tokens=128, trunc=True):

    assert token_limit < 2048, "ERROR: Token limit exceeds model limit!"
    assert len(user_prompt.split())/0.75 + max_tokens < token_limit, "ERROR: Request exceeds token limit!"
   
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0.5,
        presence_penalty=0,
        echo=False 
    )

    resp_choice = response['choices'][0]['text']
    
    resp_choice_trunc = ''
    if trunc:
        periods = [pos for pos, char in enumerate(resp_choice) if char == '.']
        if len(periods) != 0:
            cut_pos = periods[-1] + 1 
            resp_choice_trunc = resp_choice[:cut_pos]
    else:
        resp_choice_trunc = resp_choice
        
    return resp_choice_trunc


def science_translator(user_prompt, engine="text-davinci-001", token_limit=512, max_tokens=100, trunc=False):

    assert token_limit < 2048, "ERROR: Token limit exceeds model limit!"
    
    PROMPT = "My second grader asked me what this passage means:\n\"\"\"\n{}\n\"\"\"\nI rephrased it for her, in plain language a second grader can understand:\n\"\"\"\n".format(user_prompt)
    
    EXAMPLE = "My second grader asked me what this passage means:\n\"\"\"\nThis program promotes efficient water use in homes and businesses throughout the country by offering a simple way to make purchasing decisions that conserve water without sacrificing quality or product performance.\n\"\"\"\nI rephrased it for her, in plain language a second grader can understand:\n\"\"\"\nThis program helps homeowners and businesses buy products that use less water without sacrificing quality or performance.\n\"\"\"\n"
    
    new_prompt = EXAMPLE + PROMPT
    
    assert len(new_prompt.split())/0.75 + max_tokens < token_limit, "ERROR: Request exceeds token limit!"
    
    response = openai.Completion.create(
        engine=engine,
        prompt=new_prompt,
        temperature=0.5,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.2,
        presence_penalty=0,
        stop=["\"\"\""]
    )

    resp_choice = response['choices'][0]['text']
    
    if trunc:
        periods = [pos for pos, char in enumerate(resp_choice) if char == '.']
        if len(periods) != 0:
            cut_pos = periods[-1] + 1 
            resp_choice_trunc = resp_choice[:cut_pos]
    else:
        resp_choice_trunc = resp_choice
        
    return resp_choice_trunc, new_prompt


def gpt3_classifier(user_prompt, engine="text-davinci-001", token_limit=512, max_tokens=8):

    assert token_limit < 2048, "ERROR: Token limit exceeds model limit!"

    PROMPT = "This is a tweet sentiment classifier\n\n\nTweet: \"I loved the new Batman movie!\"\nSentiment: Positive\n###\nTweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been great\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\n"
    completion = "Tweet: \"" + user_prompt + "\"\nSentiment:"
    prompt = PROMPT + completion

    assert len(prompt.split())/0.75 + max_tokens < token_limit, "ERROR: Request exceeds token limit!"
   
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.3,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0,
        stop=["###"],
        echo=False
    )
    
    sent_class = completion + response['choices'][0]['text']
    return sent_class, prompt


def spacy_classifier(user_prompt):
    
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('spacytextblob')
    reply_spacy = nlp(user_prompt)
    pol = reply_spacy._.polarity
    sub = reply_spacy._.subjectivity
    
    sentiment = ''
    if pol < -0.3:
        sentiment = 'negative'
    elif pol > -0.3 and pol < 0.3:
        sentiment = 'neutral'
    elif pol > 0.3:
        sentiment = 'positive'
    
    sub_res = ''
    if sub < 0.2:
        sub_res = 'very objective'
    elif sub > 0.2 and sub < 0.5:
        sub_res = 'somewhat objective'
    elif sub > 0.5 and sub < 0.8:
        sub_res = 'somewhat subjective'
    elif sub > 0.8:
        sub_res = 'very subjective'
    
    response = "Text: {}\nSentiment: {}\nSubjectivity: {}".format(user_prompt, sentiment, sub_res)
          
    return response, pol, sub


def chatbot(user_prompt, engine="text-davinci-001", temperature=0.9, top_p=1.0, max_tokens=150):

    FRAMING = "The following is a conversation with an AI assistant. \
The assistant is chatty and charming.\
\n\nYou: Hello, who are you?\nAI: I'm your AI assitant. How  can I help you today?\
\nYou: "

    start_sequence = "\nAI:"
    restart_sequence = "\nYou: "

    prompt = FRAMING + user_prompt + start_sequence

    talk = True
    while talk: 
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=[" You:", " AI:"]
            )
    
        reply_sequence = response['choices'][0]['text']

        print(start_sequence + reply_sequence)
    
        user_prompt = input("You: ")
        prompt = prompt + reply_sequence + restart_sequence + user_prompt + start_sequence
    
        if user_prompt == 'stop': 
            talk = False
        
        if len(prompt.split()) > 1500:
            talk = False
            print('WARNING: Conversation is getting too long--aborted by system!') 
               
    return