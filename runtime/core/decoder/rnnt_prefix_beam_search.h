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


#ifndef DECODER_RNNT_PREFIX_BEAM_SEARCH_H_
#define DECODER_RNNT_PREFIX_BEAM_SEARCH_H_
#include "torch/script.h"
#include "torch/torch.h"
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
// #include "decoder/ctc_prefix_beam_search.h"
#include "decoder/context_graph.h"
#include "decoder/search_interface.h"
#include "utils/utils.h"
#include "decoder/torch_asr_model.h"

namespace wenet {

struct RnntPrefixBeamSearchOptions {
  int blank = 0;  // blank id
  int first_beam_size = 10;
  int second_beam_size = 10;
  float transducer_weight=0.7;
  float ctc_weight=0.3;
};

class RnntPrefixBeamSearch : public SearchInterface {
 public:
  // explicit RnntPrefixBeamSearch(
  //     const RnntPrefixBeamSearchOptions& opts,
  //     const std::shared_ptr<ContextGraph>& context_graph = nullptr);
  explicit RnntPrefixBeamSearch(const RnntPrefixBeamSearchOptions& opts,std::shared_ptr<AsrModel> new_model_);
  void Search(const std::vector<int>& hpy) override;
  void Search(const std::vector<std::vector<float>>& logp) override;
  void Search(const std::vector<std::vector<float>>& logp,const std::vector<torch::Tensor> &encoder_outs) override;
 
  void Reset() override;
  void FinalizeSearch() override;
  SearchType Type() const override { return SearchType::rnntPrefixBeamSearch; }
  // void UpdateOutputs(const std::pair<std::vector<int>, PrefixScore>& prefix);
  void UpdateOutputs(const wenet::Sequence& prefix);
  // void UpdateHypotheses(
  //     const std::vector<std::pair<std::vector<int>, PrefixScore>>& hpys);
  void UpdateHypotheses(
   const std::vector<wenet::Sequence>& hpys);

  void UpdateFinalContext();

  const std::vector<float>& viterbi_likelihood() const {
    return viterbi_likelihood_;
  }
  const std::vector<std::vector<int>>& Inputs() const override {
    return hypotheses_;
  }
  const std::vector<std::vector<int>>& Outputs() const override {
    return outputs_;
  }
  const std::vector<float>& Likelihood() const override { return likelihood_; }
  const std::vector<std::vector<int>>& Times() const override { return times_; }
  private:
  int abs_time_step_ = 0;
  std::vector<Sequence> beam_init;
  std::shared_ptr<AsrModel> model_;

  // N-best list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<float> viterbi_likelihood_;
  std::vector<std::vector<int>> times_;
  torch::Tensor state_ms;
  torch::Tensor state_cs;
  // std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
  std::shared_ptr<ContextGraph> context_graph_ = nullptr;
  // Outputs contain the hypotheses_ and tags like: <context> and </context>
  std::vector<std::vector<int>> outputs_;
  const RnntPrefixBeamSearchOptions& opts_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(RnntPrefixBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_BEAM_SEARCH_H_
