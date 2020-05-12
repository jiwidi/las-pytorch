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



# train_dataset = AudioDataset(params['data'], 'train')
# train_loader = AudioDataLoader(train_dataset).loader
# print(len(train_loader))
# # pbar = enlighten.Counter(total=len(train_loader), desc='Basic', unit='ticks')
# # for i, (data) in enumerate(train_loader):
# #     #print(data.shape())
# #     pbar.update()

# vocab = train_dataset.unit2idx
# print(vocab)


# for i, (data) in enumerate(train_loader):
#     utt_ids, feature, label = data
#     sample_id = data[0]
#     t_inputs = data[1]
#     t_targets = data[2]
#     inputs = t_inputs['inputs'].to(device)
#     inputs_length = t_inputs['inputs_length'].to(device)
#     labels = t_targets['targets'].to(device)
#     labels_length = t_targets['targets_length'].to(device)

#     break

def main(params):
    # Construct Solver
    # data
    train_dataset = AudioDataset(params['data'], 'train')

    test_dataset = AudioDataset(params['data'], 'test')
    train_loader = AudioDataLoader(train_dataset).loader
    test_loader = AudioDataLoader(test_dataset).loader
    # load dictionary and generate char_list, sos_id, eos_id
    # char_list, sos_id, eos_id = process_dict(args.dict)
    # vocab_size = len(char_list)
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
                    num_layers=params['model']['decoder']['n_layers'], bidirectional_encoder=params['model']['decoder']['bidirectional'])
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
# encoder_out, encoder_hidden = encoder(inputs,inputs_length)
# pdb.set_trace()
# print('encoder output size: ', encoder_out.size())  # source, batch_size, hidden_dim
# print('encoder hidden size: ', encoder_hidden.size()) # n_layers * num_directions, batch_size, hidden_dim