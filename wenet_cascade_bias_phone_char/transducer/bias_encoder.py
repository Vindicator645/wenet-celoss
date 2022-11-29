import numpy as np
import torch
import argparse
from torch import nn

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.decoder import  ConformerEncoder
from wenet.transformer.encoder import TransformerEncoder
# from espnet.nets.pytorch_backend.contrib.components.ConformerComponent import ConformerComponent, ConformerComponentArgs
# from espnet.nets.pytorch_backend.contrib.decoders.decoders_transformer import DecoderTransformer, DecoderTransformerArgs

IGNORE_ID = -1

def ContextBiasEncoderArgs(group):
    """Add Encoder common arguments."""

    group.add_argument(
        "--context-extracter-idim",
        type=int,
        default=212,
        help="input dim of context extracter."
    )
    group.add_argument(
        "--encoder-odim",
        type=int,
        default=256,
        help="output dim of encoder."
    )
    group.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="model unit embedding dim."
    )
    group.add_argument('--context-extracter-type', default='blstm', type=str,
                       choices=['blstm', 'lstm', 'transformer'],
                       help='Type of context extracter network architecture')
    group.add_argument(
        "--context-extracter-block",
        type=int,
        default=1,
        help="number of context extracter blocks."
    )
    group.add_argument(
        "--context-extracter-head",
        type=int,
        default=4,
        help="number of context extracter head, only for transformer."
    )
    group.add_argument(
        "--context-extracter-linear-units",
        type=int,
        default=512,
        help="number of context extracter linear units, only for transformer."
    )
    group.add_argument(
        "--context-encoder-hidden-dim",
        type=int,
        default=256,
        help="context encoder hidden dim."
    )
    group.add_argument(
        "--context-encoder-head",
        type=int,
        default=4,
        help="number of context encoder head(MHA)."
    )
    group.add_argument(
        "--context-encoder-block",
        type=int,
        default=2,
        help="number of context encoder head(MHA)."
    )
    group.add_argument(
        "--context-encoder-linear-units",
        type=int,
        default=512,
        help="number of context encoder FF units."
    )
    group.add_argument('--bias-mix-type', default='concat', type=str,
                       choices=['concat', 'add'],
                       help='Type of context extracter network architecture')
 
    group.add_argument(
        "--mix-context-extracter-idim",
        type=int,
        default=6002,
        help="input dim of mix context extracter."
    )
    group.add_argument(
        "--mix-embedding-dim",
        type=int,
        default=256,
        help="model unit mix embedding dim."
    )
    group.add_argument('--mix-context-extracter-type', default='blstm', type=str,
                       choices=['blstm', 'lstm', 'transformer'],
                       help='Type of mix context extracter network architecture')
    group.add_argument(
        "--mix-context-extracter-block",
        type=int,
        default=1,
        help="number of mix context extracter blocks."
    )
    group.add_argument(
        "--mix-context-extracter-head",
        type=int,
        default=4,
        help="number of mix context extracter head, only for transformer."
    )
    group.add_argument(
        "--mix-context-extracter-linear-units",
        type=int,
        default=512,
        help="number of mix context extracter linear units, only for transformer."
    )
    group.add_argument(
        "--mix-context-encoder-hidden-dim",
        type=int,
        default=256,
        help="mix context encoder hidden dim."
    )
    group.add_argument(
        "--mix-context-encoder-head",
        type=int,
        default=4,
        help="number of mix context encoder head(MHA)."
    )
    group.add_argument(
        "--mix-context-encoder-block",
        type=int,
        default=2,
        help="number of mix context encoder block(MHA)."
    )
    group.add_argument(
        "--mix-context-encoder-linear-units",
        type=int,
        default=512,
        help="number of mix context encoder FF units."
    )
    # Add args of all kinds of components. 
    # If you add a new component, DO NOT forget to add args to add_component_args func.
    #group = add_component_args(group)
    return group

class TransformerComponent(torch.nn.Module):

    def __init__(self, input_dim, output_dim, num_layer, attention_head, linear_unit, dropout):
        super(TransformerComponent, self).__init__()
        parser = argparse.ArgumentParser()
        parser = ConformerComponentArgs(parser)
        args = parser.parse_args([])
        args.conformer_mode = "nonstreaming"
        args.conformer_input_dim = input_dim
        args.conformer_output_dim = output_dim
        args.conformer_num_layers = num_layer
        args.conformer_attention_dim = output_dim
        args.conformer_attention_heads = attention_head
        args.conformer_linear_units = linear_unit
        args.conformer_dropout_rate = dropout
        args.conformer_normalize_before = True
        args.conformer_concat_after = False
        args.conformer_macaron_style = False
        args.conformer_position_embedding_type = "abs_pos"
        args.conformer_use_cnn = False
        self.transformer =  ConformerEncoder(args)
    
    def forward(self, x_in, h_context):
        x_out, _, _, _ = self.transformer(x_in, h_context)
        return x_out


class TransformerDecoderComponent(torch.nn.Module):

    def __init__(self, odim, attention_dim, head_num, linear_units, block_num):
        super(TransformerDecoderComponent, self).__init__()
        parser = argparse.ArgumentParser()
        parser = DecoderTransformerArgs(parser)
        args = parser.parse_args([])
        args.transformer_decoder_attn_dim = attention_dim
        args.transformer_decoder_attn_heads = head_num
        args.transformer_decoder_linear_units = linear_units
        args.transformer_decoder_num_blocks = block_num
        args.transformer_decoder_dropout = 0.1
        args.transformer_decoder_pos_dropout = 0.0
        args.transformer_decoder_attn_dropout = 0.1

        self.transformer = TransformerDecoder(odim, args)
    
    def forward(self, x_in, h_context):
        x_out, _ = self.transformer(x_in, h_context, None)
        return x_out


class LSTM_extracter(torch.nn.Module):

    def __init__(self, context_extracter_idim, embedding_dim, context_extracter_block, bidir = True, dropout = 0.1):
        super(LSTM_extracter, self).__init__()
        self.bidir = bidir
        self.embedding_layer = torch.nn.Embedding(context_extracter_idim, embedding_dim)
        self.lstm_layer = torch.nn.LSTM(embedding_dim, embedding_dim, context_extracter_block, batch_first=True, dropout = dropout, bidirectional = bidir)

    def forward(self, x_in):        # need modify
        "x_in: N * L"
        input_len = []
        for seq in x_in:
            input_len.append(len(seq[seq!= 0]))
        x_emb = self.embedding_layer(x_in)  # N * L * F
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x_emb, input_len, batch_first=True, enforce_sorted=False)
        _, last_state = self.lstm_layer(x_pack)   # N * L * (2 * F), (2 * num_layers) * N * F
        laste_h = last_state[0]
        laste_c = last_state[1]

        if(self.bidir):
            state = torch.cat([laste_h[-1, :, :], laste_h[-2, :, :], laste_c[-1, :, :], laste_c[-2, :, :]], dim=-1)  # N * (4 * F)
        else:
            state = torch.cat([laste_h[-1, :, :], laste_c[-1, :, :]], dim=-1)  # N * (2 * F)
        return state

class Transformer_extracter(torch.nn.Module):

    def __init__(self, context_extracter_idim, embedding_dim, context_extracter_block, extracter_head, extracter_linear_unit, dropout = 0.1):
        super(Transformer_extracter, self).__init__()
        self.embedding_layer = torch.nn.Embedding(context_extracter_idim, embedding_dim)
        self.transformer_layer = TransformerComponent(embedding_dim, embedding_dim, context_extracter_block, extracter_head, extracter_linear_unit, dropout)

    def forward(self, x_in):
        input_len = []
        for seq in x_in:
            input_len.append(len(seq[seq!= 0]))
        #print(x_in.shape)
        x_emb = self.embedding_layer(x_in)
        #print(x_emb.shape)
        x_out = self.transformer_layer(x_emb, input_len)     # N * L * F
        return x_out[:, 0, :]


class ContextBiasEncoder(torch.nn.Module):

    def __init__(self, args):
        super(ContextBiasEncoder, self).__init__()
        self.encoder_output_dim = args.encoder_output_dim
        self.context_extracter_idim = args.context_extracter_idim
        self.embedding_dim = args.embedding_dim
        self.context_extracter_type = args.context_extracter_type
        self.context_extracter_block = args.context_extracter_block
        self.context_extracter_head = args.context_extracter_head
        self.context_extracter_linear_units = args.context_extracter_linear_units
        self.context_encoder_hidden_dim = args.context_encoder_hidden_dim
        self.context_encoder_head = args.context_encoder_head
        self.context_encoder_block = args.context_encoder_block
        self.context_encoder_linear_units = args.context_encoder_linear_units
        
        self.mix_context_extracter_idim = args.mix_context_extracter_idim
        self.mix_embedding_dim = args.mix_embedding_dim
        self.mix_context_extracter_type = args.mix_context_extracter_type
        self.mix_context_extracter_block = args.mix_context_extracter_block
        self.mix_context_extracter_head = args.mix_context_extracter_head
        self.mix_context_extracter_linear_units = args.mix_context_extracter_linear_units
        self.mix_context_encoder_hidden_dim = args.mix_context_encoder_hidden_dim
        self.mix_context_encoder_head = args.mix_context_encoder_head
        self.mix_context_encoder_block = args.mix_context_encoder_block
        self.mix_context_encoder_linear_units = args.mix_context_encoder_linear_units
        self.bias_mix_type = args.bias_mix_type

        self.pred_hw = args.unified_hw_weight != 0.0

        MHSA_dropout = 0.1
        #context_extracter
        if(self.context_extracter_type == "blstm"):
            self.context_extracter = LSTM_extracter(self.context_extracter_idim, self.embedding_dim, self.context_extracter_block)
        elif(self.context_extracter_type == "lstm"):
            self.context_extracter = LSTM_extracter(self.context_extracter_idim, self.embedding_dim, self.context_extracter_block, bidir = False)
        elif(self.context_extracter_type == "transformer"):
            self.context_extracter = Transformer_extracter(self.context_extracter_idim, self.embedding_dim, self.context_extracter_block, self.context_extracter_head, self.context_extracter_linear_units)
        else:
            print("Component {} is not supported now!".format(self.context_extracter_type))
            return NotImplemented

        if(args.mix_mode):
            if(self.mix_context_extracter_type == "blstm"):
                self.mix_context_extracter = LSTM_extracter(self.mix_context_extracter_idim, self.mix_embedding_dim, self.mix_context_extracter_block)
            elif(self.mix_context_extracter_type == "lstm"):
                self.mix_context_extracter = LSTM_extracter(self.mix_context_extracter_idim, self.mix_embedding_dim, self.mix_context_extracter_block, bidir = False)
            elif(self.mix_context_extracter_type == "transformer"):
                self.mix_context_extracter = Transformer_extracter(self.mix_context_extracter_idim, self.mix_embedding_dim, self.mix_context_extracter_block, self.mix_context_extracter_head, self.mix_context_extracter_linear_units)
            else:
                print("Component {} is not supported now!".format(self.context_extracter_type))
                return NotImplemented

        if(args.joint_mode):
            if(args.ns_decoder == 'phone-char'):
                self.context_dec_phone = TransformerDecoderComponent(212, self.embedding_dim, self.context_extracter_head, self.context_extracter_linear_units, self.mix_context_extracter_block)
                self.context_dec_char = TransformerDecoderComponent(6002, self.embedding_dim, self.context_extracter_head, self.context_extracter_linear_units, self.mix_context_extracter_block)
            elif(args.ns_decoder == 'phone-bpe'):
                self.context_dec_phone = TransformerDecoderComponent(347, self.embedding_dim, self.context_extracter_head, self.context_extracter_linear_units, self.mix_context_extracter_block)
                self.context_dec_char = TransformerDecoderComponent(3000, self.embedding_dim, self.context_extracter_head, self.context_extracter_linear_units, self.mix_context_extracter_block)

        if(self.encoder_output_dim != self.context_encoder_hidden_dim):
            self.enc_proj = nn.Linear(self.encoder_output_dim, self.context_encoder_hidden_dim)
        else:
            self.enc_proj = nn.Identity()
        
        if(args.ns_decoder == "transformer"):
            if(args.transformer_decoder_linear_units != self.context_encoder_hidden_dim):
                self.dec_proj = nn.Linear(args.transformer_decoder_linear_units, self.context_encoder_hidden_dim)
            else:
                self.dec_proj = nn.Identity()
        else:
            if(args.rnnt_decoder_hidden_dim != self.context_encoder_hidden_dim):
                self.dec_proj = nn.Linear(args.rnnt_decoder_hidden_dim, self.context_encoder_hidden_dim)
            else:
                self.dec_proj = nn.Identity()

        if(self.context_extracter_type == "blstm"):
            self.context_enc = TransformerComponent(self.embedding_dim * 4, self.context_encoder_hidden_dim, self.context_encoder_block, self.context_encoder_head, self.context_encoder_linear_units, MHSA_dropout)
        elif(self.context_extracter_type == "lstm"):
            self.context_enc = TransformerComponent(self.embedding_dim * 2, self.context_encoder_hidden_dim, self.context_encoder_block, self.context_encoder_head, self.context_encoder_linear_units, MHSA_dropout)
        elif(self.context_extracter_type == "transformer"):
            self.context_enc = TransformerComponent(self.embedding_dim, self.context_encoder_hidden_dim, self.context_encoder_block, self.context_encoder_head, self.context_encoder_linear_units, MHSA_dropout)

        if(args.mix_mode):
            if(self.mix_context_extracter_type == "blstm"):
                self.mix_context_enc = TransformerComponent(self.mix_embedding_dim * 4, self.mix_context_encoder_hidden_dim, self.mix_context_encoder_block, self.mix_context_encoder_head, self.mix_context_encoder_linear_units, MHSA_dropout)
            elif(self.mix_context_extracter_type == "lstm"):
                self.mix_context_enc = TransformerComponent(self.mix_embedding_dim * 2, self.mix_context_encoder_hidden_dim, self.mix_context_encoder_block, self.mix_context_encoder_head, self.mix_context_encoder_linear_units, MHSA_dropout)
            elif(self.mix_context_extracter_type == "transformer"):
                self.mix_context_enc = TransformerComponent(self.mix_embedding_dim, self.mix_context_encoder_hidden_dim, self.mix_context_encoder_block, self.mix_context_encoder_head, self.mix_context_encoder_linear_units, MHSA_dropout)

            self.mix_linear = nn.Linear(self.context_encoder_hidden_dim*2, self.context_encoder_hidden_dim)
        
        self.encoder_bias = MultiHeadedAttention(self.context_encoder_head, self.context_encoder_hidden_dim, MHSA_dropout, -1, -1)
        self.decoder_bias = MultiHeadedAttention(self.context_encoder_head, self.context_encoder_hidden_dim, MHSA_dropout, -1, -1)


        self.encoder_norm = torch.nn.LayerNorm(self.context_encoder_hidden_dim)
        self.encoder_bias_norm = torch.nn.LayerNorm(self.context_encoder_hidden_dim)
        self.decoder_norm = torch.nn.LayerNorm(self.context_encoder_hidden_dim)
        self.decoder_bias_norm = torch.nn.LayerNorm(self.context_encoder_hidden_dim)


        if(self.bias_mix_type == "concat"):
            output_layer_idim = self.context_encoder_hidden_dim * 2
        else:
            output_layer_idim = self.context_encoder_hidden_dim
        
        if(args.ns_decoder == "transformer"):
            dec_output_layer_odim = args.transformer_decoder_linear_units
        else:
            dec_output_layer_odim = args.rnnt_decoder_hidden_dim

        self.encoder_output_layer = nn.Linear(output_layer_idim, self.encoder_output_dim)
        self.decoder_output_layer = nn.Linear(output_layer_idim, dec_output_layer_odim)

        if(self.pred_hw):
            self.hw_output_layer_enc = nn.Linear(self.context_encoder_hidden_dim, args.unified_hw_odim)
            self.hw_output_layer_dec = nn.Linear(self.context_encoder_hidden_dim, args.unified_hw_odim)
            self.hw_bias = MultiHeadedAttention(args.unified_hw_odim, args.unified_hw_odim, MHSA_dropout, -1, -1)
            self.hw_bias_norm = torch.nn.LayerNorm(args.unified_hw_odim)
            self.hw_output_layer = nn.Linear(args.unified_hw_odim, args.unified_hw_odim)
    
    def forward(self, hw_list, h_enc, h_dec):
        context_bias_vector = self.context_extracter(hw_list)
        h_context = self.context_enc(context_bias_vector.unsqueeze(0), [context_bias_vector.shape[0]])
        h_context_enc = h_context.expand(h_enc.shape[0], -1, -1)
        h_context_dec = h_context.expand(h_dec.shape[0], -1, -1)
        
        h_enc_proj = self.enc_proj(h_enc)
        h_enc_bias = self.encoder_bias(h_enc_proj, h_context_enc, h_context_enc)

        h_dec_proj = self.dec_proj(h_dec)
        h_dec_bias = self.decoder_bias(h_dec_proj, h_context_dec, h_context_dec)

        if(self.bias_mix_type == "concat"):
            h_enc_cat = torch.cat([self.encoder_norm(h_enc_proj), self.encoder_bias_norm(h_enc_bias)], axis = -1)
            h_dec_cat = torch.cat([self.decoder_norm(h_dec_proj), self.decoder_bias_norm(h_dec_bias)], axis = -1)

            h_enc_out = self.encoder_output_layer(h_enc_cat) + h_enc
            h_dec_out = self.decoder_output_layer(h_dec_cat) + h_dec
        else:
            h_enc_out = self.encoder_output_layer(self.encoder_bias_norm(h_enc_bias) + self.encoder_norm(h_enc_proj))
            h_dec_out = self.decoder_output_layer(self.decoder_bias_norm(h_dec_bias) + self.decoder_norm(h_dec_proj))

        return h_enc_out, h_dec_out

    def forward_context_enc(self, hw_list):
        context_bias_vector = self.context_extracter(hw_list)
        #print(context_bias_vector.shape)
        h_context = self.context_enc(context_bias_vector.unsqueeze(0), [context_bias_vector.shape[0]])
        #print(h_context.shape)
        return h_context


    def forward_context_joint_dec(self, ys_in_phone_pad, ys_in_char_pad, h_context):        
        context_dec_phone_out = self.context_dec_phone(ys_in_phone_pad, h_context)
        context_dec_char_out = self.context_dec_char(ys_in_char_pad, h_context)

        return context_dec_phone_out, context_dec_char_out


    def forward_enc(self, h_enc, h_context):
        h_context_enc = h_context.expand(h_enc.shape[0], -1, -1)
        h_enc_proj = self.enc_proj(h_enc)
        h_enc_bias = self.encoder_bias(h_enc_proj, h_context_enc, h_context_enc)
        if(self.pred_hw):
            h_enc_hw_output = self.hw_output_layer_enc(h_enc_bias)
        else:
            h_enc_hw_output = None
        if(self.bias_mix_type == "concat"):
            h_enc_cat = torch.cat([self.encoder_norm(h_enc_proj), self.encoder_bias_norm(h_enc_bias)], axis = -1)
            h_enc_out = self.encoder_output_layer(h_enc_cat) + h_enc
        else:
            h_enc_out = self.encoder_output_layer(self.encoder_bias_norm(h_enc_bias) + self.encoder_norm(h_enc_proj))
        return h_enc_out, h_enc_hw_output

    def forward_dec(self, h_dec, h_context):
        h_context_dec = h_context.expand(h_dec.shape[0], -1, -1)
        h_dec_proj = self.dec_proj(h_dec)
        h_dec_bias = self.decoder_bias(h_dec_proj, h_context_dec, h_context_dec)
        if(self.pred_hw):
            h_dec_hw_output = self.hw_output_layer_dec(h_dec_bias)
        else:
            h_dec_hw_output = None
        if(self.bias_mix_type == "concat"):
            h_dec_cat = torch.cat([self.decoder_norm(h_dec_proj), self.decoder_bias_norm(h_dec_bias)], axis = -1)
            h_dec_out = self.decoder_output_layer(h_dec_cat) + h_dec
        else:
            h_dec_out = self.decoder_output_layer(self.decoder_bias_norm(h_dec_bias) + self.decoder_norm(h_dec_proj))
        return h_dec_out, h_dec_hw_output

 

    def forward_hw_pred(self, h_enc_hw_output, h_dec_hw_output):
        h_hw_bias = self.hw_bias(h_dec_hw_output, h_enc_hw_output, h_enc_hw_output)
        h_hw_out = self.hw_output_layer(self.hw_bias_norm(h_hw_bias))
        return h_hw_out



if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser()
    parser = ContextBiasEncoderArgs(parser)
    args = parser.parse_args()

    args.context_extracter_idim = 212
    args.embedding_dim = 256
    args.context_extracter_type = "transformer"
    args.context_extracter_block = 2
    args.context_extracter_head = 4
    args.context_extracter_linear_units = 256
    args.context_encoder_hidden_dim = 256
    args.context_encoder_head = 4
    args.context_encoder_block = 2
    args.context_encoder_linear_units = 512
    args.bias_mix_type = "concat"

    args.encoder_output_dim = 128
    args.transformer_decoder_linear_units = 212
    args.rnnt_decoder_hidden_dim = 212
    args.ns_decoder = "transducer"

    context_bias = ContextBiasEncoder(args)

    hw_list = torch.zeros((4,8), dtype=torch.int64)
    h_enc = torch.zeros((32,100,128))
    h_dec = torch.zeros((10,7,212))

    h_enc_out, h_dec_out = context_bias(hw_list, h_enc, h_dec)
    print(h_enc_out.shape, h_dec_out.shape)
