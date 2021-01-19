API_TOKENS = ["abcdefgh", "12345678"]
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5002


class ConfigLSTM(object):
    seq_size = 20
    num_delay = 5
    hidden_size = 64
    input_size = 40
    num_layers = 2
    dropout_p = 0.5
    output_size = 2
    sample_rate = 16000
