import requests
import json
import numpy as np
import ollama
import logging
logger = logging.getLogger(__name__)

class API:
    def __init__(self, model: str = "llama3.1", options: ollama.Options = []):
        """
        Initializes the SQLiteDB object by creating a connection to the SQLite database.

        :param db_file: Path to the SQLite database file.
        """
        self.model = model
        self.options = options

    def chat(self, messages):
        logger.debug(f"chat: {messages[-1]}")
        response = ollama.chat(model=self.model, messages=messages, options=self.options)
        logger.debug(f"chat log:\nresponse:\n{response['message']['content']}\nrequest:\n{messages}")
        return response

    def generate(self, prompt):
        response = ollama.generate(model=self.model, prompt=prompt, options=self.options)
        print(response)
        return response
    
    def embed(self, input):
        response_data = ollama.embed(model=self.model, input=input, options=self.options)
        if 'embedding' in response_data:
            return np.array(response_data['embedding'], dtype=np.float32)
        elif 'embeddings' in response_data:
            if (response_data['embeddings']):
                logger.debug(f"embedding tail: {response_data['embeddings'][1::]}") if response_data['embeddings'][1::] else None
                return np.array(response_data['embeddings'][0], dtype=np.float32)
            elif str(input) == "":
                return np.array([], dtype=np.float32)
            else:
                raise KeyError(f"No embedding found in API response. Response: {response_data}, Input: {input}")
        else:
            raise KeyError(f"No embedding found in API response. Response: {response_data}, Input: {input}")


def stream_api_call(messages, max_tokens):
    prompt = json.dumps(messages)
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True
    }
    try:
        response = requests.post('http://localhost:11434/api/generate', 
                                 headers={'Content-Type': 'application/json'}, 
                                 data=json.dumps(data),
                                 stream=True)
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if 'response' in chunk:
                    full_response += chunk['response'].replace("'",'')
                    yield chunk['response'].replace("'",'')
        if full_response:
            return json.loads(full_response.replace("'",''))
        else:
            raise ValueError("Empty response from API")
    except Exception as e:
        error_message = f"Failed to generate response. Error: {str(e)}"
        return {"title": "Error", "content": error_message, "next_action": "final_answer"}

def get_embedding(text):
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"model": "llama3.1", "input": text})
    response = requests.post('http://localhost:11434/api/embed', headers=headers, data=data)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    response_data = response.json()
    
    if 'embedding' in response_data:
        return np.array(response_data['embedding'], dtype=np.float32)
    elif 'embeddings' in response_data and response_data['embeddings']:
        return np.array(response_data['embeddings'][0], dtype=np.float32)
    else:
        raise KeyError(f"No embedding found in API response. Response: {response_data}")