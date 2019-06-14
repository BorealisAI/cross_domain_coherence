def add_bigram_args(parser):
    # System Hyperparameters
    parser.add_argument('--data_name', type=str, default='wsj_bigram',
                        help="data name")
    parser.add_argument('--random_seed', type=int, default=2018,
                        help="random seed")
    parser.add_argument('--test', default=False, action='store_true',
                        help="Test with smaller infersent embeddings")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch_size")
    parser.add_argument('--save', default=False, action='store_true',
                        help="whether to save the model")
    parser.add_argument('--portion', type=float, default=1.0,
                        help="portion of negative samples to use")

    # Model Hyperparameters
    parser.add_argument('--loss', type=str, default='margin',
                        help="training loss")
    parser.add_argument('--input_dropout', type=float, default=0.6,
                        help="input_dropout")
    parser.add_argument('--hidden_state', type=int, default=500,
                        help="hidden_state")
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help="hidden_layers")
    parser.add_argument('--hidden_dropout', type=float, default=0.3,
                        help="hidden_dropout")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="num_epochs")
    parser.add_argument('--margin', type=float, default=5.0,
                        help="margin")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--l2_reg_lambda', type=float, default=0.0,
                        help="l2_reg_lambda")
    parser.add_argument('--use_bn', default=False, action='store_true',
                        help="use_bn")
    parser.add_argument('--embed_dim', type=int, default=100,
                        help="embedi_dim")
    parser.add_argument('--dpout_model', type=float, default=0.0,
                        help="dpout_model")
    parser.add_argument('--sent_encoder', type=str, default='infersent',
                        help="sent_encoder")
    parser.add_argument('--bidirectional', default=False, action='store_true',
                        help="bidirectional")

    parser.add_argument('--note', type=str, default='',
                        help="human readable")
