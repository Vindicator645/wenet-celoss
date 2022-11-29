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


#include "decoder/rnnt_greedy_search.h"
#include "torch/script.h"
#include "torch/torch.h"
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "utils/log.h"
#include "utils/utils.h"

namespace wenet {

RnntGreedySearch::RnntGreedySearch() {
  Reset();
}

void RnntGreedySearch::Reset() {
  hypotheses_.clear();
  likelihood_.clear();
  times_.clear();
  outputs_.clear();
  abs_time_step_ = 0;
  std::vector<int> empty;
  outputs_.emplace_back(empty);
  hypotheses_.emplace_back(empty);
  likelihood_.emplace_back(0);
  times_.emplace_back(empty);
}

void RnntGreedySearch::UpdateOutputs(const std::vector<int>& output) {
  outputs_.emplace_back(output);
}

void RnntGreedySearch::UpdateHypotheses(const std::vector<int>& hyp) {
  std::vector<int> n_hyp = hypotheses_[0];
  std::vector<int> merg_hyp;
  merg_hyp.insert(merg_hyp.end(), n_hyp.begin(), n_hyp.end());
  merg_hyp.insert(merg_hyp.end(), hyp.begin(), hyp.end());

  outputs_.clear();
  hypotheses_.clear();
  likelihood_.clear();
  times_.clear();
  
  UpdateOutputs(merg_hyp);
  hypotheses_.emplace_back(merg_hyp);
  likelihood_.emplace_back(0.0);
  times_.emplace_back(std::vector<int>{0});
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in
// it.
void RnntGreedySearch::Search(const std::vector<int>& hyp) {
  UpdateHypotheses(hyp);
}

void RnntGreedySearch::FinalizeSearch() { UpdateFinalContext(); }

void RnntGreedySearch::UpdateFinalContext() {
}

void RnntGreedySearch::Search(const std::vector<std::vector<float>>& logp) {
}
void RnntGreedySearch::Search(const std::vector<std::vector<float>>& logp,const std::vector<torch::Tensor> &encoder_outs) {
}

} // namespace wenet
