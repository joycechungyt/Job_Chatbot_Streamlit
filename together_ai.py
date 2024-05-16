import requests


def gemmaResponse(auth_token, prompt):

    endpoint = 'https://api.together.xyz/v1/chat/completions'
    res = requests.post(endpoint, json={
        "model": "google/gemma-7b-it",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "prompt": prompt,
        "repetition_penalty": 1,
        "stop": [
            "<eos>",
            "<end_of_turn>"
        ]
    }, headers={
        "Authorization": f"Bearer {auth_token}",
    })


    response = res.json()['choices'][0]['message']['content']

    return response
