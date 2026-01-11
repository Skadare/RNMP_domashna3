import argparse
import json
from kafka import KafkaConsumer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="health_data_predicted")
    parser.add_argument("--group", default="health-debug-consumer")
    parser.add_argument("--from-beginning", action="store_true",
                        help="If set, start reading from earliest offset.")
    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        auto_offset_reset="earliest" if args.from_beginning else "latest",
        enable_auto_commit=True,
        group_id=args.group,
        value_deserializer=lambda v: v.decode("utf-8"),
    )

    print(f"Listening on topic '{args.topic}' @ {args.bootstrap}")
    print(f"Offsets: {'earliest' if args.from_beginning else 'latest'}")
    print("-" * 60)

    for msg in consumer:
        try:
            obj = json.loads(msg.value)
            print(json.dumps(obj, indent=2))
        except json.JSONDecodeError:
            print(msg.value)

        print("-" * 60)


if __name__ == "__main__":
    main()
