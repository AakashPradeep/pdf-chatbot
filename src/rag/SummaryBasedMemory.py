
import json
import os
"""
This class implements a summary-based memory mechanism for a conversational agent.
It maintains a summary of the conversation and a current session history. 
When the current session history exceeds certain thresholds (either in terms of 
number of interactions or total character size), it generates a new summary using 
the provided language model (LLM) and updates the stored summary. The summary is 
persisted to a JSON file, allowing it to be retained across sessions.
"""

class SummaryBasedMemory :
    filename = ".chat_summary.json"
    current_session = []
    cur_chat_history_size = 0

    def __load__(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)
                self.summary = data.get("summary", "")
        else:
            self.summary = ""
    
    def __init__(self, 
                 llm, 
                 summarization_prompt_template, 
                 max_summary_length=500,
                 max_cur_chat_history_length = 10,
                 max_cur_chat_history_size = 500):
        self.llm = llm
        self.summarization_prompt_template = summarization_prompt_template
        self.max_summary_length = max_summary_length
        self.max_cur_chat_history_length = max_cur_chat_history_length
        self.max_cur_chat_history_size = max_cur_chat_history_size

        self.__load__()

    def add_to_memory(self, question, answer):
        # Combine the existing summary with the new information
        combined_info = {"question": question, "answer": answer}
        
        self.current_session.append(combined_info)
        self.cur_chat_history_size += len(question) + len(answer)
        
        # If the combined information exceeds the maximum summary length, summarize it
        if self.cur_chat_history_size > self.max_cur_chat_history_size or len(self.current_session) > self.max_cur_chat_history_length:
            print(f"Summarizing current session with {len(self.current_session)} interactions and total size {self.cur_chat_history_size} characters.")
            prompt_text = (
                "Return a combined summary in 200–300 characters (max 500 characters). "
                "Summarize the conversation and merge it with the current summary.\n\n"
                f"Conversation history:\n{self.current_session}\n\n"
                f"Current summary:\n{self.summary}\n"
            )

            resp = self.llm.invoke(prompt_text)
            self.summary = resp.content.strip()
            self.__persist__(self.summary)
            self.current_session = []
            self.cur_chat_history_size = 0
            print(f"New summary length: {len(self.summary)} characters.")


    def __persist__(self, summary):
        with open(self.filename, 'w') as f:
            json.dump({"summary": summary}, f)

    def get_summary(self):
        return json.dumps({"history": self.summary, "current_session": str(self.current_session)})