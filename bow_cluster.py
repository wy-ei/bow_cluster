import numpy as np
import random
import math
import logging
import pickle
from collections.abc import Iterable
from collections import Counter, defaultdict
from sklearn.utils import check_random_state

__all__ = ['BowCluster']


class Item:
    def __init__(self, tokens):
        self.tokens = set(tokens)
        self.proba = None
        self.label = None

    def __repr__(self):
        return '{}:{}'.format(self.label, self.tokens)

class Center:
    def __init__(self, init_item, max_features=50):
        self.tokens = set(init_item.tokens)
        self.items = []
        self.max_features  = max_features 
        self.label = None
        self.counter = Counter()
        self.i = 0
        
    def reset(self):
        self.counter.clear()
        self.items = []

    def adjust(self):
        new_tokens = {w for w, c in self.counter.most_common(self.max_features )}
        diff = len(new_tokens ^ self.tokens)
        self.tokens = new_tokens
        return diff

    def add(self, item):
        self.counter.update(item.tokens)
        self.items.append(item)

    def post_process(self):
        self.adjust()
        for item in self.items:
            item.label = self.label

    def _similarity(self, item):
        """
        compute the similarity between two sets of words
        """
        words_1, words_2 = self.tokens, item.tokens
        n_words_1, n_words_2 = len(words_1), len(words_2)

        if n_words_1 == 0 or n_words_2 == 0:
            return 0

        common_words = words_1 & words_2
        n_common_words = len(common_words)
        if n_common_words == 0:
            return 0

        return n_common_words / math.sqrt(n_words_1 * n_words_2)
    
    def similarity(self, item_or_items):
        if isinstance(item_or_items, Iterable):
            items = item_or_items
            dist = np.empty(len(items))
            for i, item in enumerate(items):
                dist[i] = self._similarity(item)
            return dist
        else:
            item = item_or_items
            return self._similarity(item)
    
    def distance(self, item_or_items):
        return 1 - self.similarity(item_or_items)


class BowCluster:
    def __init__(self, n_clusters=200,
                 init='auto', tol=200,
                 max_iter=5, min_cluster_size=1,
                 center_max_features=50, random_state=42):

        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size
        self.center_max_features = center_max_features
        
        self.random_state = check_random_state(random_state)
        self.cluster_centers = None
    
    @property
    def cluster_centers_(self):
        words_list = []
        for center in self.cluster_centers:
            words_list.append(center.tokens)
        return words_list
        
    def _init_cluster_centers(self, items):
        init_items = []

        if self.init == 'auto':
            init_items = self._k_centers(items, n_clusters=self.n_clusters)
        elif self.init == 'random':
            init_items = self.random_state.choice(items, self.n_clusters)
        elif isinstance(self.init, list):
            init_items = [Item(tokens) for tokens in self.init[:self.n_clusters]]
            k = self.n_clusters - len(self.init)
            if k > 0:
                init_items += self.random_state.choice(items, k)
        else:
            raise ValueError('parameter error: init has wrong value')
            
        cluster_centers = []
        for item in init_items:
            center = Center(item, max_features=self.center_max_features)
            cluster_centers.append(center)
            
        return cluster_centers
    
    def _k_centers(self, items, n_clusters):
        """
        实现 kmeans++ 的思想，得到聚类的初始化中心
        """

        # 每次从 n_candidate 个样本中寻找聚类中心
        n_candidate = 2 + int(np.log(n_clusters))
        n_selected_items = min(2000, n_clusters * 5)
        items = self.random_state.choice(items, n_selected_items)
        
        centers = []

        # 先随机选一个
        centers.append(Center(self.random_state.choice(items)))

        min_distance = centers[0].distance(items)
        min_total_distance = min_distance.sum()    

        for _ in range(1, n_clusters):
            # 按概率挑 n_candidate 个样本作为候选的聚类中心
            rand_vals = self.random_state.random_sample(n_candidate) * min_total_distance
            candidate_ids = np.searchsorted(np.cumsum(min_distance), rand_vals)
            candidate_center = [Center(items[i]) for i in candidate_ids]

            best_candidate = None
            best_distance = None
            best_total_distance = None

            for i in range(n_candidate):
                distance = candidate_center[i].distance(items)
                new_distance = np.minimum(min_distance, distance)
                new_total_distance = new_distance.sum()

                if (not best_candidate) or new_total_distance < min_total_distance:
                    best_candidate = candidate_center[i]
                    best_distance = new_distance
                    best_total_distance = new_total_distance

            centers.append(best_candidate)
            min_distance = best_distance
            min_total_distance = best_total_distance

        return centers

    def fit(self, corpus):
        
        items = [Item(tokens) for tokens in corpus]

        cluster_centers = self._init_cluster_centers(items)
        
        for i_iter in range(self.max_iter):
            for center in cluster_centers:
                center.reset()
                
            self._put_items_in_centers(cluster_centers, items)

            diff = 0
            for center in cluster_centers:
                diff += center.adjust()
            
            print("iteration {}, diff: {}".format(i_iter + 1, diff))
            if diff < self.tol:
                break

        cluster_centers = self._remove_less_items_cluster_center(cluster_centers)

        self.cluster_centers = cluster_centers

        self._post_process()

        self.labels_ = [item.label for item in items]

    def predict(self, corpus):
        labels = []
        items = [Item(tokens) for tokens in corpus]
        for item in items:
            center = max(self.cluster_centers, key=lambda center: center.similarity(item))
            labels.append(center.label)
        return labels
    
    def predict_proba(self, corpus):
        return self.transform(corpus)
    
    def transform(self, corpus):
        X = np.empty((len(corpus), len(self.cluster_centers)))
        items = [Item(words) for words in corpus]
        
        for i, center in enumerate(self.cluster_centers):
            X[:, i] = center.similarity(items)

        return X
        
    def _put_items_in_centers(self, cluster_centers, items):
        for item in items:
            center = self._find_best_fit_center(cluster_centers, item)
            sim = center.similarity(item)
            item.proba = sim
            center.add(item)

    def _post_process(self):
        for label, center in enumerate(self.cluster_centers):
            center.label = label
            center.post_process()
        

    def _remove_less_items_cluster_center(self, cluster_centers):
        less_items_centers = [center for center in cluster_centers if len(center.items) < self.min_cluster_size]
        cluster_centers = [center for center in cluster_centers if len(center.items) >= self.min_cluster_size]        
        
        for center in less_items_centers:
            for item in center.items:
                best_fit_center = self._find_best_fit_center(cluster_centers, item)
                best_fit_center.add(item)

        return cluster_centers

    def save(self, filepath):
        result = []
        for center in self.cluster_centers:
            result.append({
                'tokens': center.tokens,
                'max_features': center.max_features,
                'label': center.label
            })
        with open(filepath, mode='wb') as fout:
            pickle.dump(result, fout)

    def restore(self, filepath):
        with open(filepath, mode='rb') as fin:
            result = pickle.load(fin)

        cluster_centers = []
        for center_data in result:
            item = Item(center_data['tokens'])
            center = Center(item, max_features=center_data['max_features'])
            center.label = center_data['label']
            cluster_centers.append(center)
        self.cluster_centers = cluster_centers 

    def _find_best_fit_center(self, centers, item):
        return max(centers, key=lambda center: center.similarity(item))