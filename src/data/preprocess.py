from src.utils.config import load_config


def preprocess_data(config_path):
    config = load_config(config_path)

    print("Starting preprocessing...")
    print(f"Data path: {config['data']['raw_path']}")

    # TODO: carica dati grezzi
    # TODO: costruisci grafi con graph_builder
    # TODO: salva dataset processato

    print("Preprocessing completed.")