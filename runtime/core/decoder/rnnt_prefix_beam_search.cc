// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "decoder/rnnt_prefix_beam_search.h"
#include "torch/script.h"
#include "torch/torch.h"
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "utils/log.h"
#include "utils/utils.h"
int blank=0;
namespace wenet {
RnntPrefixBeamSearch::RnntPrefixBeamSearch(const RnntPrefixBeamSearchOptions& opts,std::shared_ptr<AsrModel> new_model_)
: opts_(opts) {
  // model_=new_model_->Copy();
  model_=new_model_;

    // const RnntPrefixBeamSearchOptions& opts,
    // const std::shared_ptr<ContextGraph>& context_graph)
    // : opts_(opts), context_graph_(context_graph) {
  Reset();
}


void RnntPrefixBeamSearch::Reset() {
  outputs_.clear();
  abs_time_step_ = 0;
  wenet::Sequence seq;
  // prefix_score.s = 0.0;
  // prefix_score.ns = -kFloatMax;
  // prefix_score.v_s = 0.0;
  // prefix_score.v_ns = 0.0; 
  std::vector<int> empty;
  beam_init.clear();
  beam_init.push_back(model_->getEmptySequence(blank));
  //tensor convert to vector
  outputs_.emplace_back(empty);
  hypotheses_.emplace_back(empty);
  // likelihood_.emplace_back(prefix_score.total_score());
  times_.emplace_back(empty);
}

static bool PrefixScoreCompare(
    const wenet::Sequence& s1, const wenet::Sequence& s2) {
  return s1.score > s2.score;
}
static bool PrefixCompare(const wenet::Sequence& s1, const wenet::Sequence& s2) {
	return s1.hyp < s2.hyp;
}
// void UpdateOutputs(const std::pair<std::vector<int>, PrefixScore>& prefix){}
void RnntPrefixBeamSearch::UpdateOutputs(
    const wenet::Sequence& prefix) {
  // const std::vector<int>& input = prefix.hyp;
  // const std::vector<int>& start_boundaries = prefix.second.start_boundaries;
  // const std::vector<int>& end_boundaries = prefix.second.end_boundaries;

  // std::vector<int> output;
  // int s = 0;
  // int e = 0;
  // for (int i = 0; i < input.size(); ++i) {
  //   if (s < start_boundaries.size() && i == start_boundaries[s]) {
  //     output.emplace_back(context_graph_->start_tag_id());
  //     ++s;
  //   }
  //   output.emplace_back(input[i]);
  //   if (e < end_boundaries.size() && i == end_boundaries[e]) {
  //     output.emplace_back(context_graph_->end_tag_id());
  //     ++e;
  //   }
  // }
  // outputs_.emplace_back(output);
}

// void RnntPrefixBeamSearch::Reset() {
//   hypotheses_.clear();
//   likelihood_.clear();
//   cur_hyps_.clear();
//   viterbi_likelihood_.clear();
//   times_.clear();_back(output);
// }

void RnntPrefixBeamSearch::UpdateHypotheses(
    const std::vector<wenet::Sequence>& hpys) {
  // cur_hyps_.clear();
  // outputs_.clear();
  beam_init.clear();
  hypotheses_.clear();
  likelihood_.clear();
  viterbi_likelihood_.clear();
  times_.clear();
  for (auto& item : hpys) {
    // cur_hyps_[item.first] = item.second;
    UpdateOutputs(item);
    hypotheses_.emplace_back(std::move(item.hyp));
    // likelihood_.emplace_back(item.second.total_score());
    // viterbi_likelihood_.emplace_back(item.second.viterbi_score());
    // times_.emplace_back(item.second.times());
  }
}

// Please refer https://robin1001.github.io/2020/12/11/Rnnt-search
// for how Rnnt prefix beam search works, and there is a simple graph demo in
// it.

// void cache_to_batch(){
//   state_ms=torch.zeros(beam_init.size(),beam_init[0].h.size());
//   state_cs=torch.zeros(0);
//   for(int i=0;i<beam_init();i++){
//     state_ms[i]=beam_init[i].h;
//     state_cs[i]=beam_init[i].c;
// }
void RnntPrefixBeamSearch::Search(const std::vector<int>& hyp) {
}
void RnntPrefixBeamSearch::Search(const std::vector<std::vector<float>>& logp) {
}
void RnntPrefixBeamSearch::Search(const std::vector<std::vector<float>>& ctc_log_probs,const std::vector<torch::Tensor>&encoder_outs) {
  if (ctc_log_probs.size() == 0) return;
  //2. init beam using wenet::Sequence to save beam unit

  int maxlen=encoder_outs.size();
  //  3. start decoding (notice: we use breathwise first searching)
  //  !!!! In this decoding method: one frame do not output multi units. !!!!
  //  !!!!    Experiments show that this strategy has little impact      !!!!
  for (int t = 0; t < maxlen; ++t, ++abs_time_step_) {
    // auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    // auto tensor = torch::from_blob(value.data(), {int64_t(value.size())}, opts).clone();
    // # 3.1 building input
    // # decoder taking the last token to predict the next token
    torch::Tensor input_hyp_tensor=torch::zeros(beam_init.size());
    for(int i=0;i<beam_init.size();i++){
      input_hyp_tensor[i]=beam_init[i].hyp.back();
    }
    // build score tensor to do torch.add() function
    torch::Tensor scores=torch::zeros(beam_init.size());
    for(int i=0;i<beam_init.size();i++){
      scores[i]=beam_init[i].score;
    }  
    //cache_to_batch();
    
    // 3.2 forward decoder
    torch::Tensor logp;
    std::vector<torch::Tensor> cache_batch;
    torch::Tensor new_cache_h,new_cache_c;
    model_->forward_decoder_one_step(encoder_outs[t],input_hyp_tensor,cache_batch,logp,new_cache_h,new_cache_c);
    // 3.3 shallow fusion for transducer score
    //  and ctc score where we can also add the LM score

    torch::Tensor ctc_log_probs_tensor=torch::zeros({ctc_log_probs.size(), ctc_log_probs[0].size()}, torch::kFloat);
    for(int ii=0;ii<ctc_log_probs.size();ii++){
      for(int jj=0;jj<ctc_log_probs[0].size();jj++){
        ctc_log_probs_tensor[ii][jj]=ctc_log_probs[ii][jj];
      }
    }
    logp=torch::log(opts_.transducer_weight*torch::exp(ctc_log_probs_tensor)+opts_.ctc_weight*torch::exp(logp));
    // 3.4 first beam prune
    // std::vector<float> topk_score;
    // std::vector<int32_t> topk_index;    
    torch::Tensor topk_score;
    torch::Tensor topk_index;
    // TopK(logp_t, first_beam_size, &topk_score, &topk_index);
    auto tp=torch::topk(logp,opts_.first_beam_size);
    topk_index=std::get<0>(tp);
    topk_score=std::get<1>(tp);
    // 3.5 generate new beam (N*N)
    std::vector<wenet::Sequence>beam_A;
    for(int j=0;j<beam_init.size();j++){
      wenet::Sequence base_seq=beam_init[j];
      for(int k=0;k<opts_.first_beam_size;k++){
        //blank: only update the score
        if(topk_index[j][t].item<int>()==blank){
          // wenet::Sequence new_seq;
          // new_seq.hyp=base_seq.hyp;
          // new_seq.score=scores[j][k];
          // new_seq.h=base_seq.h;
          // new_seq.c=base_seq.c;
          wenet::Sequence new_seq={base_seq.hyp,scores[j][k].item<float>(),base_seq.h,base_seq.c};
          beam_A.push_back(new_seq);
        }else{//other unit: update hyp score statement and last
          std::vector<int> hyp_new=base_seq.hyp;
          hyp_new.push_back(topk_index[topk_index[j,k].item<int>()].item<int>());
          wenet::Sequence new_seq={hyp_new,scores[j,t].item<float>(),new_cache_h[j],new_cache_c[j]};
          beam_A.push_back(new_seq);
        }
      }
    }
    //3.6 prefix fusion
    std::vector<wenet::Sequence> fusion_A;
    std::sort(beam_A.begin(),beam_A.end(),PrefixCompare);
    for(int j=0;j<beam_A.size();j++){
      while(j+1<beam_A.size()&&beam_A[j].hyp==beam_A[j+1].hyp){
        beam_A[j+1].score=LogAdd(beam_A[j+1].score,beam_A[j].score);
        j++;
      } 
      fusion_A.push_back(beam_A[j]);
    }
    //4. second pruned
    std::sort(fusion_A.begin(),fusion_A.end(),PrefixScoreCompare);
    fusion_A.resize(opts_.second_beam_size);
    // while(fusion_A.size()>opts_.second_beam_size){fusion_A.pop_back()};
    UpdateHypotheses(fusion_A);
  }
}


void RnntPrefixBeamSearch::FinalizeSearch() { UpdateFinalContext(); }

void RnntPrefixBeamSearch::UpdateFinalContext() {
  if (context_graph_ == nullptr) return;
  // CHECK_EQ(hypotheses_.size(), cur_hyps_.size());
  CHECK_EQ(hypotheses_.size(), likelihood_.size());
  // We should backoff the context score/state when the context is
  // not fully matched at the last time.
  // for (const auto& prefix : hypotheses_) {
  //   // PrefixScore& prefix_score = cur_hyps_[prefix];
  //   if (prefix_score.context_state != 0) {
  //     prefix_score.UpdateContext(context_graph_, prefix_score, 0,
  //                                prefix.size());
  //   }
  // }
  // std::vector<std::pair<std::vector<int>, PrefixScore>> arr(cur_hyps_.begin(),
  //                                                           cur_hyps_.end());
  // std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

  // // Update cur_hyps_ and get new result
  // UpdateHypotheses(arr);
}

} // DECODER_RNNT_GREEDY_SEARCH_H_
