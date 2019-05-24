# coding=utf-8
import gensim
import smart_open


# 读取预料文件，并对读取的句子做预处理
def readCorpusforDoc(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            # 训练集处理
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            # 测试集处理，不同之处是加上标签
            else:
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

# 训练doc2vec模型
def doc2vec(corpus_file, save_model_name):
    train_corpus = list(readCorpusforDoc(corpus_file))
    # 训练doc2vec模型，vector_size：词向量维度， min_count:采集的词出现的最小频率
    model = gensim.models.doc2vec.Doc2Vec(vector_size=256, min_count=1, epochs=100)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    # 保存模型
    model.save(save_model_name)

# 读取预料文件
def readCorpusforWord(filename):
    for line in open(filename):
        yield line.split()

# 训练word2vec模型
def word2vec(corpus_file, save_model_name):
    train_corpus = list(readCorpusforWord(corpus_file))
    # 训练模型
    model = gensim.models.Word2Vec(train_corpus, size=256, window=2, sample=1e-4, \
                                   negative=5, hs=0, sg=1, iter=1000, min_count=1)
    # 保存模型
    model.save(save_model)


if __name__ == '__main__':
    # 模型训练的语料文件
    train_file = '/home/zwj/Desktop/NLP/doc2vec/train_data/genres.txt'
    # 保存的模型名称
    save_model = '/home/zwj/Desktop/NLP/doc2vec/train_data/genres_256.model'
    # 训练模型
    doc2vec(train_file, save_model)
    # word2vec(train_file, save_model)
