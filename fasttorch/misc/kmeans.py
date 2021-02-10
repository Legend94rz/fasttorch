from sklearn.cluster import KMeans as _KMeans
import torch as T


class KMeans:
    def __init__(self, n_clusters=20, max_iter=None, verbose=True, device='cpu'):
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = T.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = T.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, T.argmin(self.dists, (0)))
            if T.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        labels = T.empty((x.shape[0],)).long().to(self.device)
        dists = T.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = T.sum(T.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = T.argmin(dist)
            dists = T.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = T.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = T.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = T.cat([centers, T.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        self.representative_samples = T.argmin(self.dists, (0))