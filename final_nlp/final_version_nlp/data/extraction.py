from datasets import load_dataset

# Load dataset (restaurants config)
ds = load_dataset(
    "alexcadillon/SemEval2014Task4",
    "restaurants",
    trust_remote_code=True
)

# Save each split as its own CSV
ds["train"].to_csv("semeval2014_restaurants_train.csv", index=False)
ds["test"].to_csv("semeval2014_restaurants_test.csv", index=False)
ds["trial"].to_csv("semeval2014_restaurants_trial.csv", index=False)
