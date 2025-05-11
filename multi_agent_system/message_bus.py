import redis
import json

# Shared Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

def publish(channel: str, message: dict):
    """Publish a message to a Redis channel."""
    r.publish(channel, json.dumps(message))

def subscribe(channels: list):
    """Subscribe to one or more Redis channels."""
    pubsub = r.pubsub()
    pubsub.subscribe(*channels)
    return pubsub
