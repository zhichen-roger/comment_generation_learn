#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import os
import sys
import pathlib

import torch
import jieba

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

import config
from model import PGN
from dataset import PairDataset
from utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs


class Predict():
    @timer(module='initalize predicter')
    def __init__(self):
        self.DEVICE = config.DEVICE

        dataset = PairDataset(config.data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)

        self.vocab = dataset.build_vocab(embed_file=config.embed_file)

        self.model = PGN(self.vocab)
        self.stop_word = list(
            set([
                self.vocab[x.strip()] for x in
                open(config.stop_word_file,encoding='utf-8'
                     ).readlines()
            ]))
        self.model.load_model()
        self.model.to(self.DEVICE)

    def greedy_search(self,
                      x,
                      max_sum_len,
                      len_oovs,
                      x_padding_masks):
        """Function which returns a summary by always picking
           the highest probability option conditioned on the previous word.

        Args:
            x (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            len_oovs (Tensor): Numbers of out-of-vocabulary tokens.
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).

        Returns:
            summary (list): The token list of the result summary.
        """

        # Get encoder output and states.Call encoder forward propagation
        encoder_output, encoder_states = self.model.encoder(
            replace_oovs(x, self.vocab), self.model.decoder.embedding)

        # Initialize decoder's hidden states with encoder's hidden states.
        decoder_states = self.model.reduce_state(encoder_states)

        # Initialize decoder's input at time step 0 with the SOS token.
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(self.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS]
        coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)
        # Generate hypothesis with maximum decode step.
        while int(x_t.item()) != (self.vocab.EOS) \
                and len(summary) < max_sum_len:

            context_vector, attention_weights, coverage_vector = \
                self.model.attention(decoder_states,
                                     encoder_output,
                                     x_padding_masks,
                                     coverage_vector)
            p_vocab, decoder_states, p_gen = \
                self.model.decoder(x_t.unsqueeze(1),
                                   decoder_states,
                                   context_vector)
            final_dist = self.model.get_final_distribution(x,
                                                           p_gen,
                                                           p_vocab,
                                                           attention_weights,
                                                           torch.max(len_oovs))
            # Get next token with maximum probability.
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            decoder_word_idx = x_t.item()
            summary.append(decoder_word_idx)
            x_t = replace_oovs(x_t, self.vocab)

        return summary

#     @timer('best k')
    def best_k(self, beam, k, encoder_output, x_padding_masks, x, len_oovs):
        """Get best k tokens to extend the current sequence at the current time step.

        Args:
            beam (untils.Beam): The candidate beam to be extended.
            k (int): Beam size.
            encoder_output (Tensor): The lstm output from the encoder.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.
            x (Tensor): Source token ids.
            len_oovs (Tensor): Number of oov tokens in a batch.

        Returns:
            best_k (list(Beam)): The list of best k candidates.

        """
        # use decoder to generate vocab distribution for the next token
        x_t = torch.tensor(beam.tokens[-1]).reshape(1, 1)
        x_t = x_t.to(self.DEVICE)

        # Get context vector from attention network.
        context_vector, attention_weights, coverage_vector = \
            self.model.attention(beam.decoder_states,
                                 encoder_output,
                                 x_padding_masks,
                                 beam.coverage_vector)

        # Replace the indexes of OOV words with the index of OOV token
        # to prevent index-out-of-bound error in the decoder.

        p_vocab, decoder_states, p_gen = \
            self.model.decoder(replace_oovs(x_t, self.vocab),
                               beam.decoder_states,
                               context_vector)

        final_dist = self.model.get_final_distribution(x,
                                                       p_gen,
                                                       p_vocab,
                                                       attention_weights,
                                                       torch.max(len_oovs))
        # Calculate log probabilities.
        log_probs = torch.log(final_dist.squeeze())
        # Filter forbidden tokens.
        if len(beam.tokens) == 1:
            forbidden_ids = [
                self.vocab[u"这"],
                self.vocab[u"此"],
                self.vocab[u"采用"],
                self.vocab[u"，"],
                self.vocab[u"。"],
            ]
            log_probs[forbidden_ids] = -float('inf')
        # EOS token penalty. Follow the definition in
        # https://opennmt.net/OpenNMT/translation/beam_search/.
        log_probs[self.vocab.EOS] *= \
            config.gamma * x.size()[1] / len(beam.tokens)

        log_probs[self.vocab.UNK] = -float('inf')
        # Get top k tokens and the corresponding logprob.
        topk_probs, topk_idx = torch.topk(log_probs, k)

        # Extend the current hypo with top k tokens, resulting k new hypos.
        best_k = [beam.extend(x,
                  log_probs[x],
                  decoder_states,
                  coverage_vector) for x in topk_idx.tolist()]

        return best_k

    def beam_search(self,
                    x,
                    max_sum_len,
                    beam_width,
                    len_oovs,
                    x_padding_masks):
        """Using beam search to generate summary.

        Args:
            x (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            beam_width (int): Beam size.
            max_oovs (int): Number of out-of-vocabulary tokens.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.

        Returns:
            result (list(Beam)): The list of best k candidates.
        """
        # run body_sequence input through encoder. Call encoder forward propagation
        encoder_output, encoder_states = self.model.encoder(
            replace_oovs(x, self.vocab), self.model.decoder.embedding)
        coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)
        # initialize decoder states with encoder forward states
        decoder_states = self.model.reduce_state(encoder_states)

        # initialize the hypothesis with a class Beam instance.

        init_beam = Beam([self.vocab.SOS],
                         [0],
                         decoder_states,
                         coverage_vector)

        # get the beam size and create a list for stroing current candidates
        # and a list for completed hypothesis
        k = beam_width
        curr, completed = [init_beam], []

        # use beam search for max_sum_len (maximum length) steps
        for _ in range(max_sum_len):
            # get k best hypothesis when adding a new token

            topk = []
            for beam in curr:
                # When an EOS token is generated, add the hypo to the completed
                # list and decrease beam size.
                if beam.tokens[-1] == self.vocab.EOS:
                    completed.append(beam)
                    k -= 1
                    continue
                for can in self.best_k(beam,
                                       k,
                                       encoder_output,
                                       x_padding_masks,
                                       x,
                                       torch.max(len_oovs)
                                      ):
                    # Using topk as a heap to keep track of top k candidates.
                    # Using the sequence scores of the hypos to campare
                    # and object ids to break ties.
                    add2heap(topk, (can.seq_score(), id(can), can), k)

            curr = [items[2] for items in topk]
            # stop when there are enough completed hypothesis
            if len(completed) == beam_width:
                break
        # When there are not engouh completed hypotheses,
        # take whatever when have in current best k as the final candidates.
        completed += curr
        # sort the hypothesis by normalized probability and choose the best one
        result = sorted(completed,
                        key=lambda x: x.seq_score(),
                        reverse=True)[0].tokens
        return result

    @timer(module='doing prediction')
    def predict(self, text, tokenize=True, beam_search=True):
        """Generate summary.

        Args:
            text (str or list): Source.
            tokenize (bool, optional):
                Whether to do tokenize or not. Defaults to True.
            beam_search (bool, optional):
                Whether to use beam search or not.
                Defaults to True (means using greedy search).

        Returns:
            str: The final summary.
        """
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)
        len_oovs = torch.tensor([len(oov)]).to(self.DEVICE)
        x_padding_masks = torch.ne(x, 0).byte().float()
        if beam_search:
            summary = self.beam_search(x.unsqueeze(0),
                                       max_sum_len=config.max_dec_steps,
                                       beam_width=config.beam_size,
                                       len_oovs=len_oovs,
                                       x_padding_masks=x_padding_masks)
        else:
            summary = self.greedy_search(x.unsqueeze(0),
                                         max_sum_len=config.max_dec_steps,
                                         len_oovs=len_oovs,
                                         x_padding_masks=x_padding_masks)
        summary = outputids2words(summary,
                                  oov,
                                  self.vocab)
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


if __name__ == "__main__":
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))
    # Randomly pick a sample in test set to predict.
    with open(config.test_data_path, 'r', encoding='utf-8') as test:
        # picked = random.choice(list(test))
        picked = '户外 服装 情侣 拼接 男女 皮肤 风衣 快速 防护 晒衣服 女款 天蓝 玫红 模特 展示 ， 细节 展示 ， 产品 展示 ， 拥抱 夏季 ， 轻 时尚 ， 拼 出色 ， 防 泼水 面料 ， 我们 ' \
                 '的 优点 ， 夏季 与 你 更 亲密 ， 抗撕裂 ， 还有 更 多 ， 舒适 透气 ， 保护 你 我们 更 用心 ! ， 我们 都 有 ! ， 轻盈 舒适 ， 拼色 皮肤 衣带 你 ， 附赠 收纳 ' \
                 '包 ， 科技 快干 ， 户外 防护 ， 拼色 时尚 轻薄 皮肤 风衣 ， 女款 ， 防 泼水 ， 透气 测试 展示 ， 男款 ， 第三步 ， I 女款 ， 第四步 ， 第二步 ， 让 你 与 夏天 ' \
                 '更 亲近 ， 接缝 细节 出众 ， 时尚 拼色 ， 防 泼水 / 户外 防护 1 ， 应有 的 ， 游泳池 ， 户外 ， 一件 好 的 皮肤 衣 ， 即可 快速 晒干 ， 畅享 户外 乐趣 ， ， ' \
                 '30 ℃ 以上 通风 环境 ， 阳光 照射 ， 大 特点 ， 对 阳光 有 较 好 的 折射 、 反射 、 散射 ， 立体 剪裁 ， 基本 信 ， 性能 ， 从而 达到 防护 作用 ， 把 收纳袋 ' \
                 '口 拉紧 ， ， 产品 特点 ， 男款 / 女款 ， 准备 好 皮肤 衣 和 ， 夏季 自 在 穿行 ， 海滩 ， 皮肤 风衣 收纳 步骤 ， 颜色 : 男款 ， 透气 ， 袋中 ， 立即 抢 ， ' \
                 '收纳袋 ， 透气性 : ， 把 整理 好 的 皮肤 ， 拿出 收纳袋 ， 把 皮肤 ， 收纳 步骤 完成 ， 主 面料 : 锦 訾 面料 ， 运动 更 随心 ， 名称 : ， 衣 竖向 放入 收纳 ' \
                 '， 户外 穿行 ， 经过 防护 处理 的 多 微孔 纤维 组成 ， ， 防晒尺码 S 适用人群 女士 材质 锦涤 功能 防风 款式 标准型 <sep>舒爽 透气 ， 快速 散热 ， 让 身体 畅快 ' \
                 '吸收 ， 轻盈 面料 ， 丝柔 触感 ， 亲肌 柔顺 ， 轻松 出行 ， 抗撕裂 面料 ， 让 你 运动 更 舒心 。 '
        source, ref = picked.strip().split('<sep>')
    print('source: ', source.replace(" ", ""), '\n')
    greedy_prediction = pred.predict(source.split(),  beam_search=False)
    print('greedy: ', greedy_prediction.replace(" ", ""), '\n')
    beam_prediction = pred.predict(source.split(),  beam_search=True)
    print('beam: ', beam_prediction.replace(" ", ""), '\n')
    print('ref: ', ref.replace(" ", ""), '\n')
