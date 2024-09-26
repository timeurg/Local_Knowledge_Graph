@nlp
Feature: Chat API
  Given <model_name> as string
    * optional api_settings as dict.fromkeys(['url', 'token'])
    * optional model_settings as ollama.Options

  Scenario: chat
    Given <messages> as list of dict.fromkeys(['role', 'content'])
  Scenario: local call
    Given <api_settings> is None
    When called
    Then return response from ollama.chat(messages)
  Scenario: API call
    When <api_settings> is not None 
    When called
    Then return response ollama.Client(host=api_settings.url, headers={Authorization="Bearer ${api_settings.token}"}).chat(messages)

  Examples: Local llama3.1
    | model_name | api_settings | messages |
    | llama3.1   |              | [{"role": "system", "content": "You ALWAYS reply \"Ni\""}, {"role": "user", "content": "Hi!"}]|
  Examples: Deepseek
    |model_name|api_settings|messages|
    |llama3.1|{"url": "https://api.deepseek.com", "token": ""}|[{"role": "system", "content": "You ALWAYS reply \"Ni!\""}, {"role": "user", "content": "Hi!"}]|