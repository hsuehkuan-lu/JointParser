class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_features = 4
    n_classes = 125
    n_char = 20
    n_gram = 3
    sent_len = 100
    seq_len = 2 * sent_len
    dropout = 0.33
    embed_size = 50
    hidden_size = 512
    rnn_size = 256
    char_hidden_size = 64
    # batch_size = 2048
    batch_size = 150
    n_epochs = 30
    lr = 1e-3
    
    alpha = 1e-1
    
    l2_alpha = 1e-8

    pad_id = 0

    language = 'english'
    with_punct = True
    unlabeled = False
    lowercase = True
    use_pos = False
    use_dep = False
    use_dep = use_dep and (not unlabeled)
    data_path = '/data/lxk/DependencyParsing'
    train_file = 'train.txt'
    dev_file = 'dev.txt'
    test_file = 'test.txt'
    embedding_file = None
    # embedding_file = './data/en-cw.txt'

    total_features = 1 + use_pos + use_dep