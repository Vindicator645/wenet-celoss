from typing import List

import torch
from random import random
import numpy as np
def edit_distance(tensor1, tensor2):
    m = len(tensor1)
    n = len(tensor2)
    
    # Create a distance matrix with dimensions (m+1) x (n+1)
    distance = np.zeros((m+1, n+1))
    
    # Initialize the first row and column of the matrix
    for i in range(m+1):
        distance[i][0] = i
    for j in range(n+1):
        distance[0][j] = j
    
    # Fill in the rest of the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            if tensor1[i-1] == tensor2[j-1]:
                cost = 0
            else:
                cost = 1
            
            distance[i][j] = min(distance[i-1][j] + 1,    # deletion
                                  distance[i][j-1] + 1,    # insertion
                                  distance[i-1][j-1] + cost)  # substitution
    
    # Return the bottom-right cell of the matrix as the minimum editing distance
    return distance[m][n]

def basic_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    context_list: torch.Tensor = torch.IntTensor([0]),
    context_lengths: torch.Tensor = torch.IntTensor([0]),
    n_steps: int = 64,
    context_filter_state: str= 'off',
    context_decoder_labels_padded: torch.Tensor = torch.IntTensor([0]),

) -> List[List[int]]:
    # fake padding
    padding = torch.zeros(1, 1).to(encoder_out.device)
    # sos
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1).to(encoder_out.device)
    cache = model.predictor.init_state(1,
                                       method="zero",
                                       device=encoder_out.device)
    new_cache: List[torch.Tensor] = []
    t = 0
    hyps = []
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = n_steps
    per_frame_noblk = 0
    prev_out_hw=1
    go_back_flag=0
    go_back_end=-1
    # print(context_list)
    # quit()
    bias_hidden = model.context_bias.forward_bias_hidden(context_list, context_lengths)
    context_list_empty=torch.zeros((1, 1), dtype=torch.int)
 
    # context_list_empty=context_list_empty.unsqueeze(0)
    context_lengths_empty=context_lengths[0]
    context_lengths_empty=context_lengths_empty.unsqueeze(0)
    bias_hidden_empty = model.context_bias.forward_bias_hidden(context_list_empty, context_lengths_empty)
    encoder_out_unbiased=encoder_out.clone()
    encoder_out , encoder_out_bias= model.context_bias.forward_encoder_bias(bias_hidden, encoder_out)
    encoder_out_empty , encoder_out_bias_empty= model.context_bias.forward_encoder_bias(bias_hidden_empty,encoder_out_unbiased)
    # print(context_lengths_empty)
    # print(context_lengths)
    context_decoder_labels_padded=context_decoder_labels_padded.squeeze(0)

    # print(encoder_out- encoder_out_empty)
    # print(torch.eq(bias_hidden, bias_hidden_empty))

        # print("____________________________________________")
        # quit()
    result=[]
    cache_list=[]
    step_list=[]
    input_list=[]
    last_t=0
    # print("")
    token_id=0
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        encoder_out_step_empty = encoder_out_empty[:, t:t + 1, :]# [1, 1, E]
        if prev_out_nblk:
            step_outs = model.predictor.forward_step(pred_input_step, padding,
                                                     cache)  # [1, 1, P]
            pred_out_step, new_cache = step_outs[0], step_outs[1]
            step_list.append(pred_out_step)
            cache_list.append(cache)
            input_list.append(pred_input_step)
            pred_out_step2, _ = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)

            if context_filter_state=='on':
                hw_output = model.context_bias.forward_hw_pred(bias_hidden,pred_out_step2)
                hw_output=hw_output.squeeze()
                values, indices = hw_output.topk(1)
                # print(pred_out_step)
                # print(pred_out_step.shape)
                # print(hw_output)
                # print(int (indices.item()),end="")
                gate=int (indices.item())
                if go_back_flag==0:
                    if gate==0:
                        result.append(0)
                        last_t=t
                        # prev_out_hw=0
                    else:
                        if len(result)!=0 and result[-1]==0:
                            go_back_end=t
                            t=last_t
                            go_back_flag=1
                            result.pop()
                            hyps.pop()
                            input_list.pop()
                            # prev_out_nblk = True
                            per_frame_noblk = per_frame_noblk - 1
                            step_list.pop()
                            cache_list.pop()
                            pred_out_step=step_list[-1]
                            cache=cache_list[-1]
                            pred_input_step = input_list[-1]
                            continue
                        result.append(1)
                        # prev_out_hw=1
                else:
                    result.append(1)
                    if t>=go_back_end:
                        go_back_flag=0

            else:
                result.append(1)
            if result[-1]==1:
                pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden_empty, pred_out_step)
            else:
                pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)

        if result[-1]==1:
            joint_out_step = model.joint(encoder_out_step,
                                pred_out_step)  # [1,1,v]
        else:
            joint_out_step = model.joint(encoder_out_step_empty,
                                            pred_out_step)  # [1,1,v]
        joint_out_probs = joint_out_step.log_softmax(dim=-1)

        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
        if joint_out_max != model.blank:
            hyps.append(joint_out_max.item())
            # result.append(prev_out_hw)
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            # state_m, state_c =  clstate_out_m, state_out_c
            cache = new_cache
        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == model.blank:
                prev_out_nblk = False
            # TODO(Mddct): make t in chunk for streamming
            # or t should't be too lang to predict none blank
            t = t + 1
            per_frame_noblk = 0
    # print(len(result))
    print(context_decoder_labels_padded)
    print(result)
    # print(hyps)
    dist=edit_distance(context_decoder_labels_padded,result)
    # print(edit_distance(context_decoder_labels_padded,result))
    return [hyps], dist,result

def basic_greedy_search2(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    context_list: torch.Tensor = torch.IntTensor([0]),
    context_lengths: torch.Tensor = torch.IntTensor([0]),
    n_steps: int = 64,
    context_filter_state: str= 'off',
    context_decoder_labels_padded: torch.Tensor = torch.IntTensor([0]),
    pvresult=[],
) -> List[List[int]]:
    # fake padding
    padding = torch.zeros(1, 1).to(encoder_out.device)
    # sos
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1).to(encoder_out.device)
    cache = model.predictor.init_state(1,
                                       method="zero",
                                       device=encoder_out.device)
    new_cache: List[torch.Tensor] = []
    t = 0
    hyps = []
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = n_steps
    per_frame_noblk = 0
    prev_out_hw=1
    result=[]
    # print(context_list)
    # quit()
    bias_hidden = model.context_bias.forward_bias_hidden(context_list, context_lengths)
    context_list_empty=torch.zeros((1, 1), dtype=torch.int)
 
    # context_list_empty=context_list_empty.unsqueeze(0)
    context_lengths_empty=context_lengths[0]
    context_lengths_empty=context_lengths_empty.unsqueeze(0)
    bias_hidden_empty = model.context_bias.forward_bias_hidden(context_list_empty, context_lengths_empty)
    encoder_out_unbiased=encoder_out.clone()
    encoder_out , encoder_out_bias= model.context_bias.forward_encoder_bias(bias_hidden, encoder_out)
    encoder_out_empty , encoder_out_bias_empty= model.context_bias.forward_encoder_bias(bias_hidden_empty,encoder_out_unbiased)
    print(context_lengths_empty)
    print(context_lengths)
    context_decoder_labels_padded=context_decoder_labels_padded.squeeze(0)

    # print(encoder_out- encoder_out_empty)
    # print(torch.eq(bias_hidden, bias_hidden_empty))

        # print("____________________________________________")
        # quit()
    # print("")
    flag=0
    token_id=1
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        encoder_out_step_empty = encoder_out_empty[:, t:t + 1, :]# [1, 1, E]
        if prev_out_nblk:
            step_outs = model.predictor.forward_step(pred_input_step, padding,
                                                     cache)  # [1, 1, P]
            pred_out_step, new_cache = step_outs[0], step_outs[1]
            pred_out_step2, _ = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)

            if context_filter_state=='on':
                hw_output = model.context_bias.forward_hw_pred(bias_hidden,pred_out_step2)
                hw_output=hw_output.squeeze()
                values, indices = hw_output.topk(1)
                # print(pred_out_step)
                # print(pred_out_step.shape)
                print(hw_output)
                # print(int (indices.item()),end="")
                # if 1:
                gate=int (indices.item())
                if flag==1:
                    gate=1
                    flag =0
                if  random() < .5:
                # if token_id<len(pvresult):
                #     gate=pvresult[token_id]
                # token_id+=1
                # if gate==0:
                    pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden_empty, pred_out_step)
                    # encoder_out_step=encoder_out_step_empty
                    prev_out_hw=0
                else:
                    pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)
                    prev_out_hw=1

            else:
                pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)

        if prev_out_hw:
            joint_out_step = model.joint(encoder_out_step,
                                pred_out_step)  # [1,1,v]
        else:
            joint_out_step = model.joint(encoder_out_step_empty,
                                            pred_out_step)  # [1,1,v]
        joint_out_probs = joint_out_step.log_softmax(dim=-1)

        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
        if joint_out_max != model.blank:
            hyps.append(joint_out_max.item())
            result.append(prev_out_hw)
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            # state_m, state_c =  clstate_out_m, state_out_c
            cache = new_cache
        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == model.blank:
                prev_out_nblk = False
            # TODO(Mddct): make t in chunk for streamming
            # or t should't be too lang to predict none blank
            t = t + 1
            per_frame_noblk = 0
    # print(len(result))
    print(context_decoder_labels_padded)
    print(result)
    dist=edit_distance(context_decoder_labels_padded,result)
    print(edit_distance(context_decoder_labels_padded,result))
    return [hyps],dist

def basic_greedy_search_both(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    context_list: torch.Tensor = torch.IntTensor([0]),
    context_lengths: torch.Tensor = torch.IntTensor([0]),
    n_steps: int = 64,
    context_filter_state: str= 'off',
    context_decoder_labels_padded: torch.Tensor = torch.IntTensor([0]),

) -> List[List[int]]:
    # fake padding
    padding = torch.zeros(1, 1).to(encoder_out.device)
    # sos
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1).to(encoder_out.device)
    cache = model.predictor.init_state(1,
                                       method="zero",
                                       device=encoder_out.device)
    new_cache: List[torch.Tensor] = []
    t = 0
    hyps = []
    result=[]
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = n_steps
    per_frame_noblk = 0
    prev_out_hw=1

    # print(context_list)
    # quit()
    bias_hidden = model.context_bias.forward_bias_hidden(context_list, context_lengths)
    context_list_empty=torch.zeros((1, 1), dtype=torch.int)
 
    # context_list_empty=context_list_empty.unsqueeze(0)
    context_lengths_empty=context_lengths[0]
    context_lengths_empty=context_lengths_empty.unsqueeze(0)
    bias_hidden_empty = model.context_bias.forward_bias_hidden(context_list_empty, context_lengths_empty)
    encoder_out_unbiased=encoder_out.clone()
    encoder_out , encoder_out_bias= model.context_bias.forward_encoder_bias(bias_hidden, encoder_out)
    encoder_out_empty , encoder_out_bias_empty= model.context_bias.forward_encoder_bias(bias_hidden_empty,encoder_out_unbiased)
    go_back_flag=0
    go_back_end=-1
    result=[]
    cache_list=[]
    step_list=[]
    input_list=[]
    last_t=0
    # print("")
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        encoder_out_bias_step=encoder_out_bias[:, t:t + 1, :]  # [1, 1, E]
        encoder_out_step_empty = encoder_out_empty[:, t:t + 1, :]# [1, 1, E]
        if prev_out_nblk:
            step_outs = model.predictor.forward_step(pred_input_step, padding,
                                                     cache)  # [1, 1, P]
            pred_out_step, new_cache = step_outs[0], step_outs[1]
            step_list.append(pred_out_step)
            cache_list.append(cache)
            input_list.append(pred_input_step)
            pred_out_step_unbiased=pred_out_step.clone()
            pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)
            hw_output = model.context_bias.forward_hw_pred_both(encoder_out_bias_step,predictor_out_bias_step)
            hw_output=hw_output.squeeze()
            values, indices = hw_output.topk(1)
            # print(pred_out_step)
            # print(pred_out_step.shape)
            print(hw_output)
            gate=int (indices.item())
            if context_filter_state=='on':
                if go_back_flag==0:
                    if gate==0:
                        result.append(0)
                        last_t=t
                        # prev_out_hw=0
                    else:
                        if len(result)!=0 and result[-1]==0:
                            go_back_end=t
                            t=last_t
                            go_back_flag=1
                            result.pop()
                            hyps.pop()
                            input_list.pop()
                            # prev_out_nblk = True
                            per_frame_noblk = per_frame_noblk - 1
                            step_list.pop()
                            cache_list.pop()
                            pred_out_step=step_list[-1]
                            cache=cache_list[-1]
                            pred_input_step = input_list[-1]
                            continue
                        result.append(1)
                        # prev_out_hw=1
                else:
                    result.append(1)
                    if t>=go_back_end:
                        go_back_flag=0

            else:
                result.append(1)

            if result[-1]==0:
                pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden_empty, pred_out_step_unbiased)
            
            # else:
            #     pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)
        if result[-1]==1:
            joint_out_step = model.joint(encoder_out_step,
                                pred_out_step)  # [1,1,v]
        else:
            joint_out_step = model.joint(encoder_out_step_empty,
                                            pred_out_step)  # [1,1,v]
        joint_out_probs = joint_out_step.log_softmax(dim=-1)

        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
        if joint_out_max != model.blank:
            hyps.append(joint_out_max.item())
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            # state_m, state_c =  clstate_out_m, state_out_c
            cache = new_cache

        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == model.blank:
                prev_out_nblk = False
            # TODO(Mddct): make t in chunk for streamming
            # or t should't be too lang to predict none blank
            t = t + 1
            per_frame_noblk = 0
    context_decoder_labels_padded=context_decoder_labels_padded.squeeze(0)
    dist=edit_distance(context_decoder_labels_padded,result)
    print(context_decoder_labels_padded)
    print(result)
    return [hyps], dist
