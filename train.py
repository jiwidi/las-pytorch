from data import AudioDataLoader
from data import AudioDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from solver.solver import Solver
import yaml
import enlighten
import pdb
import torch

def main(params):
    # Construct Solver
    # data
    train_dataset = AudioDataset(params['data'], 'train')
    test_dataset = AudioDataset(params['data'], 'test')
    train_loader = AudioDataLoader(train_dataset).loader
    test_loader = AudioDataLoader(test_dataset).loader

    vocab = train_dataset.unit2idx
    sos_id = train_dataset.unit2idx['<SOS>']
    eos_id = train_dataset.unit2idx['<EOS>']
    vocab_size = len(vocab)
    data = {'tr_loader': train_loader, 'cv_loader': test_loader}
    # model
    encoder = Encoder(input_size=params['data']['num_mel_bins'], hidden_size=params['model']['encoder']['hidden_size'],
                  num_layers=params['model']['encoder']['num_layers'], dropout=params['model']['encoder']['dropout'], bidirectional=params['model']['encoder']['bidirectional']).to(device)

    decoder = Decoder(vocab_size=512, embedding_dim=params['model']['decoder']['embed_size'],
                    sos_id=sos_id, eos_id=eos_id, hidden_size=params['model']['decoder']['hidden_size'],
                    num_layers=params['model']['decoder']['num_layers'], bidirectional_encoder=params['model']['decoder']['bidirectional'])
    model = Seq2Seq(encoder, decoder)
    print(model)
    model.cuda()
    # optimizer
    if params['training']['optimizer'] == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=params['training']['lr'],
                                     momentum=params['training']['momentum'],
                                     weight_decay=params['training']['weight_decay'])
    elif params['training']['optimizer'] == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=params['training']['lr'],
                                      weight_decay=params['training']['weight_decay'])
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, params['training'])
    solver.train()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("config.yaml", 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    print(params)
    main(params)
