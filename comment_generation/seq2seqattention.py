#!usr/bin/env python
# encoding:utf-8
from __future__ import division

'''
__Author__:沂水寒城
功能：基于开源模块sumy的简单文本摘要
文本摘要方法参考学习可以借鉴阮一峰下面的文章：
http://www.ruanyifeng.com/blog/2013/03/automatic_summarization.html
'''

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer

# 设置输出句子总数
SENTENCES_COUNT = 1


def urlContentSummary(url, language):
    '''
    基于URL内容的文本的摘要方法
    '''
    parser = HtmlParser.from_url(url, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)


def plainTextSummary(data, language):
    '''
    基于明文数据内容的摘要方法
    '''
    parser = PlaintextParser.from_string(data, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)


def fileDataSummary(files, language):
    '''
    基于文件数据内容的摘要方法
    '''
    parser = PlaintextParser.from_file(files, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)


if __name__ == "__main__":
    data = """
            后秀黑标系列梭织皮肤衣运动风衣男长袖开衫外套拉链跑步衣空灵灰再运用时装精致版型，打造适合365天的空，隐形拉链，这是美国的一个小众运动品牌，它是发汗服的，隐藏式帽子，腋下连体裁片，作为可以日常通勤的服装，定位城市机能的黑标，所有颜色，产品指数，透气，鼻祖，1999年成立之初主要是为了帮助运动，产品信息/ProductInformation，进口反光胶膜，360度反光标识，品名:，弹力，多样功能随意转换，梭织皮肤衣，连体裁片，低调，质:锦訾;氨訾，产品展示/Show，厚薄，矿物黑Black，适中，优良，为夜跑者提供更多安全性，双隐形拉链，多样化装载，腋下采用动态，产品系列:黑标系列，宽松，高弹，良好，普通，紧身，加厚，修身，无弹，微弹，胸围，袖长，平铺测量参考(单位cm)，肩宽，测量数据，袖口宽，且功能强大，腰围，轻盈，矿物黑，衣长，抗UV，自由呼吸，防水，防风，四面弹上市时间2018年夏季尺码M适用人群男士版型修身型。
            """

    plainTextSummary(data, 'chinese')
    print('=' * 10)

    #fileDataSummary('test.txt', 'chinese')