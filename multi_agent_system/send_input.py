import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

text = "Governments hesitate to abandon coal due to economic reliance despite growing interest in renewable energy."

r.publish("user_input", json.dumps({"text": text}))
print("✅ Sent test input to user_input channel.")
