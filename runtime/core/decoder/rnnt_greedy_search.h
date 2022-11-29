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


#ifndef DECODER_RNNT_GREEDY_SEARCH_H_
#define DECODER_RNNT_RGEEDY_SEARCH_H_
#include "torch/script.h"
#include "torch/torch.h"
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "decoder/context_graph.h"
#include "decoder/search_interface.h"
#include "utils/utils.h"

namespace wenet {

class RnntGreedySearch : public SearchInterface {
 public:
  explicit RnntGreedySearch();

  void Search(const std::vector<int>& hpy) override;
  void Search(const std::vector<std::vector<float>>& logp) override;
  void Search(const std::vector<std::vector<float>>& logp,const std::vector<torch::Tensor> &encoder_outs)override;
  void Reset() override;
  void FinalizeSearch() override;
  SearchType Type() const override { return SearchType::rnntGreedySearch; }
  void UpdateOutputs(const std::vector<int>& output);
  void UpdateHypotheses(const std::vector<int>& hyp);
  void UpdateFinalContext();

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

  // N-best list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<std::vector<int>> times_;

  std::shared_ptr<ContextGraph> context_graph_ = nullptr;
  // Outputs contain the hypotheses_ and tags like: <context> and </context>
  std::vector<std::vector<int>> outputs_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(RnntGreedySearch);
};

}  // namespace wenet

#endif  // DECODER_RNNT_GREEDY_SEARCH_H_
