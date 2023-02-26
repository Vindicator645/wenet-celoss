from typing import List

import torch


def basic_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    context_list: torch.Tensor = torch.IntTensor([0]),
    context_lengths: torch.Tensor = torch.IntTensor([0]),
    n_steps: int = 64,
    context_filter_state: str= 'off',

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
    # print(context_list)
    # quit()
    bias_hidden = model.context_bias.forward_bias_hidden(context_list, context_lengths)
    context_list_empty=context_list[0]
    context_list_empty=context_list_empty.unsqueeze(0)
    context_lengths_empty=context_lengths[0]
    context_lengths_empty=context_lengths_empty.unsqueeze(0)
    bias_hidden_empty = model.context_bias.forward_bias_hidden(context_list_empty, context_lengths_empty)
    encoder_out_unbiased=encoder_out.clone()
    encoder_out , encoder_out_bias= model.context_bias.forward_encoder_bias(bias_hidden, encoder_out)

    # print("")
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        if prev_out_nblk:
            step_outs = model.predictor.forward_step(pred_input_step, padding,
                                                     cache)  # [1, 1, P]
            pred_out_step, new_cache = step_outs[0], step_outs[1]
            if context_filter_state=='on':
                hw_output = model.context_bias.forward_hw_pred(bias_hidden,pred_out_step)
                hw_output=hw_output.squeeze()
                values, indices = hw_output.topk(1)

                # print(hw_output)
                # print(int (indices.item()),end="")
                if (int (indices.item()))==0:
                    pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)
                else:
                    pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden_empty, pred_out_step)
            else:
                pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)
    
        joint_out_step = model.joint(encoder_out_step,
                                     pred_out_step)  # [1,1,v]
        # print(joint_out_step.shape)
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

    return [hyps]

def basic_greedy_search_both(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    context_list: torch.Tensor = torch.IntTensor([0]),
    context_lengths: torch.Tensor = torch.IntTensor([0]),
    n_steps: int = 64,
    context_filter_state: str= 'off',

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
    # print(context_list)
    # quit()
    bias_hidden = model.context_bias.forward_bias_hidden(context_list, context_lengths)
    context_list_empty=context_list[0]
    context_list_empty=context_list_empty.unsqueeze(0)
    context_lengths_empty=context_lengths[0]
    context_lengths_empty=context_lengths_empty.unsqueeze(0)

    bias_hidden_empty = model.context_bias.forward_bias_hidden(context_list_empty, context_lengths_empty)

    encoder_out_unbiased=encoder_out.clone()
    encoder_out , encoder_out_bias= model.context_bias.forward_encoder_bias(bias_hidden, encoder_out)
    encoder_out_empty , encoder_out_bias_empty= model.context_bias.forward_encoder_bias(bias_hidden_empty, encoder_out)
    # print("")
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        encoder_out_bias_step=encoder_out_bias[:, t:t + 1, :]  # [1, 1, E]
        encoder_out_step_empty = encoder_out_empty[:, t:t + 1, :]# [1, 1, E]
        if prev_out_nblk:
            step_outs = model.predictor.forward_step(pred_input_step, padding,
                                                     cache)  # [1, 1, P]
            pred_out_step, new_cache = step_outs[0], step_outs[1]
            pred_out_step_unbiased=pred_out_step.clone()
            pred_out_step, predictor_out_bias_step = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step)
            hw_output = model.context_bias.forward_hw_pred_both(encoder_out_bias_step,predictor_out_bias_step)
            hw_output=hw_output.squeeze()
            values, indices = hw_output.topk(1)
            if (int (indices.item()))==0:
                pred_out_step, predictor_out_bias_step_empty = model.context_bias.forward_predictor_bias(bias_hidden, pred_out_step_unbiased)
                encoder_out_step=encoder_out_step_empty
   
        joint_out_step = model.joint(encoder_out_step,
                                     pred_out_step)  # [1,1,v]
        # print(joint_out_step.shape)
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

    return [hyps]
