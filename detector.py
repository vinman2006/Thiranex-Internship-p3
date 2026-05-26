import joblib

model = joblib.load(
"model.pkl"
)

while True:

    email = input(
        "\nEnter Email:\n"
    )

    if email == "exit":
        break

    result = model.predict(
        [email]
    )[0]

    probability = max(
        model.predict_proba(
            [email]
        )[0]
    )

    print(
        "\nPrediction:",
        result
    )

    print(
        "Confidence:",
        round(
            probability*100,
            2
        ),
        "%"
    )
