import torch
import torch.nn as nn
import dgl.nn.pytorch
from parse_args import args
from graph_transformer import GraphTransformer

device = torch.device('cuda')

def multiple_operator(a, b):
    return a * b

def rotate_operator(a, b):
    a_re, a_im = a.chunk(2, dim=-1)
    b_re, b_im = b.chunk(2, dim=-1)
    message_re = a_re * b_re - a_im * b_im
    message_im = a_re * b_im + a_im * b_re
    message = torch.cat([message_re, message_im], dim=-1)
    return message

class MyModel(nn.Module):
    def __init__(self, meta_g, drug_number, disease_number):
        super(MyModel, self).__init__()
        self.drug_number = drug_number
        self.meta_g = meta_g
        self.disease_number = disease_number

        self.drug_linear = nn.Linear(300, args.hgt_out_dim)
        self.protein_linear = nn.Linear(320, args.hgt_out_dim)
        self.disease_linear = nn.Linear(64, args.hgt_out_dim)

        self.gt_drug = GraphTransformer(device, args.gt_layer, self.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_disease = GraphTransformer(device, args.gt_layer, self.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_out_dim, int(args.hgt_out_dim/args.hgt_head), args.hgt_head, len(self.meta_g.nodes()), len(self.meta_g.edges()), args.dropout)

        self.hgt = nn.ModuleList()
        for _ in range(args.hgt_layer):
            self.hgt.append(self.hgt_dgl)
        
        if args.decoder_type == 0:
            self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 2))

        elif args.decoder_type == 1:
            encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
            self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
            self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

            self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 2))

        elif args.decoder_type == 2:
            self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 2))
        

    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample):
        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        drug_feature = self.drug_linear(drug_feature)
        protein_feature = self.protein_linear(protein_feature)
        disease_feature = self.disease_linear(disease_feature)

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h') 

        feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt = hgt_out[:self.drug_number, :]
        di_hgt = hgt_out[self.drug_number:self.disease_number + self.drug_number, :]

        dr = torch.stack((dr_sim, dr_hgt), dim=1)
        di = torch.stack((di_sim, di_hgt), dim=1)
        
        if args.decoder_type == 1:
            dr = self.drug_trans(dr)
            di = self.disease_trans(di)

        dr = dr.view(self.drug_number, 2 * args.gt_out_dim)
        di = di.view(self.disease_number, 2 * args.gt_out_dim)

        if args.decoder_type == 2:
            dr_sample = dr[sample[:, 0]]
            di_sample = di[sample[:, 1]]

            m_result = multiple_operator(dr_sample, di_sample)
            r_result = rotate_operator(dr_sample, di_sample)
            drdi_embedding = torch.cat([dr_sample, di_sample, m_result, r_result], dim = 1)
            
        else:
            drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])

        output = self.mlp(drdi_embedding)

        return output, (dr_sim, dr_hgt), (di_sim, di_hgt)
    
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))