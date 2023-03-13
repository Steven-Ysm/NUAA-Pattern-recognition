
from data import build_corpus
from evaluate import hmm_train_eval
import time

def main():
    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估ｈｍｍ模型
    print("正在训练评估HMM模型...")
    starttime = time.time()
    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )
    endtime = time.time()
    print("训练时间为：", endtime - starttime)

if __name__ == "__main__":
    main()
