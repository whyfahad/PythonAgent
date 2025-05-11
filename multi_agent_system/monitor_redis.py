import redis
import sys
sys.stdout.flush()
def monitor_redis():
    r = redis.Redis(host='localhost', port=6379, db=0)
    conn = r.monitor()
    print("🔍 Monitoring Redis activity...\n")
    for command in conn.listen():
        print(command)

if __name__ == "__main__":
    monitor_redis()