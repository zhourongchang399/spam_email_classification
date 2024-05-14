## Running experiments

选择数据集（cn_email,en_email）
parser.add_argument("--dataset", type=str, default='en_email', help='dataset')
读取数据集大小
parser.add_argument("--amount", type=int, default=2000, help='the number of example')
种子
parser.add_argument("--seed", type=int, default=23, help='Random seed')
训练集大小
parser.add_argument("--size", type=float, default=0.2, help='test_size')
是否启用tfidf
parser.add_argument("--tfidf", action='store_true', help='tfidf')
拉普拉斯平滑指数
parser.add_argument("--alpha", type=float, default=1.0, help='Laplace Smoothing')
类别不平衡率
parser.add_argument("--ir", type=float, default=1, help='Imbalance ratio')

### spam-email classification tfidf:
- python3 main.py --dataset=cn_email --tfidf --amount=10000 --alpha=0.1