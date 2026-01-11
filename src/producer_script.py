import pandas as pd
import time
import argparse
from kafka import KafkaProducer

features = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income"
]

label = "Diabetes_binary"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="offline.csv")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="health_data")
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: v.encode("utf-8")
    )

    df = pd.read_csv(args.file)

    if label in df.columns:
        df = df.drop(columns=[label])

    df = df[features]
    counter = 0
    for _, row in df.iterrows():
        msg = row.to_frame().T.to_json(orient="records")
        payload = msg[1:-1]

        producer.send(args.topic, payload)
        counter += 1

        if counter % 1000 == 0:
            producer.flush()
            print(f"Sent {counter}")

        time.sleep(args.sleep)

    producer.flush()
    print(f"Sent {counter}")

if __name__ == '__main__':
    main()
