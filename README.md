# Simple-ChatBot

How to run:
1. Update the Database credentials in settings.py
2. `python simple_chatbot.py`

This bot will:
1. Take in the path to your document.
2. It will process the document(extract text + clean text + generate embeddings + save into vector DB)
3. It will prompt you with 2 options:
   * Option 1: It will have no memory about your previous conversation. Something like one shot.
   * Option 2: It has memory about the previous conversation and you can base your query based on your previous questions.
  
4. You can input "exit" to exit the prompt.

As easy as that. Thank you :)
Hope you will like it!
