import redis
import json
import requests

r = redis.Redis(host='localhost', port=6379, db=0)
channel_subscribe = "coordinator_output"
channel_publish = "final_answer"

OLLAMA_URL = "http://172.20.240.1:11434/api/generate"
MODEL = "mistral"

def generate_response(concepts):
    if not concepts:
        return "No valid concepts."

    prompt = f"Provide a one-word answer using these concepts: {', '.join(concepts)}."
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120  # ️ Increase timeout to handle model latency
        )
        if response.status_code == 200:
            text = response.json().get("response", "").strip()
            return text if len(text) > 1 else "Unclear"
        else:
            return f"[Ollama Error {response.status_code}] {response.text}"
    except requests.exceptions.ConnectTimeout:
        return "[Timeout] Ollama model took too long to respond."
    except requests.exceptions.ConnectionError as e:
        return f"[Connection Error] {str(e)}"
    except Exception as e:
        return f"[Exception] {str(e)}"

def run_response_agent():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_subscribe)
    print(f"[ResponseAgent]  Subscribed to '{channel_subscribe}'")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            data = json.loads(message["data"].decode())
            print("[ResponseAgent]  Received coordinator_output.")

            top_concepts = data.get("final_ranking", [])
            concepts = [item["concept"] for item in top_concepts if item.get("composite_score", 0) >= 0.5]
            if not concepts:
                concepts = [item["concept"] for item in top_concepts[:2]]

            print("[ResponseAgent]  Concepts for response:", concepts)
            answer = generate_response(concepts)

            payload = {"concepts_used": concepts, "answer": answer}
            r.publish(channel_publish, json.dumps(payload))

            print("[ResponseAgent]  Published answer:", answer)

        except Exception as e:
            print("[ResponseAgent]  Error:", str(e))

if __name__ == "__main__":
    run_response_agent()
