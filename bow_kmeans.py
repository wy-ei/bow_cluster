import random
import math
import logging
import pickle
from collections import Counter, defaultdict

# __all__ = ['BowKMeans']

class Item:
    def __init__(self, keywords):
        self.keywords = set(keywords)
        self.label = None

    def __repr__(self):
        return '{}:{}'.format(self.label, self.keywords)


class Center:
    def __init__(self, init_item, max_features=50):
        self.keywords = set(init_item.keywords)
        self.items = []
        self.max_features  = max_features 
        self.label = None
        self.counter = Counter()
        
    def reset(self):
        self.counter.clear()
        self.items = []

    def adjust(self):
        new_keywords = {w for w, c in self.counter.most_common(self.max_features )}
        diff = len(new_keywords ^ self.keywords)
        self.keywords = new_keywords
        return diff

    def add(self, item):
        self.counter.update(item.keywords)
        self.items.append(item)

    def post_process(self):
        self.adjust()
        for item in self.items:
            item.label = self.label
        # self.items.sort(key=lambda item: len(item.keywords & self.keywords), reverse=True)

    def compute_similarity(self, other):
        n_words_1 = len(self.keywords)
        n_words_2 = len(other.keywords)

        if n_words_1 == 0 or n_words_2 == 0:
            return 0

        common_words = self.keywords & other.keywords
        n_common_words = len(common_words)
        if n_common_words == 0:
            return 0

        return n_common_words / math.sqrt(n_words_1 * n_words_2)


class BowKMeans:
    def __init__(self, n_clusters=200,
                 init='random', tol=200,
                 max_iter=4, min_cluster_size=1,
                 center_max_features=50):

        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size
        self.center_max_features = center_max_features


    def _init_cluster_centers(self, items):
        init_items = []

        if self.init == 'random':
            init_items = random.choices(items, k=self.n_clusters)
    
        elif isinstance(self.init, list):
            init_items = [Item(keywords) for keywords in self.init]
            k = self.n_clusters - len(self.init)
            if k > 0:
                init_items += random.choices(items, k=k)
            init_items = init_items[:self.n_clusters]

        cluster_centers = []
        for item in init_items:
            center = Center(item, max_features=self.center_max_features)
            cluster_centers.append(center)
            
        return cluster_centers

    def fit(self, keywords_list):
        
        items = [Item(keywords) for keywords in keywords_list]

        cluster_centers = self._init_cluster_centers(items)

        for i_iter in range(self.max_iter):
            for center in cluster_centers:
                center.reset()

            for i_item, item in enumerate(items, 1):
                if i_item % 2001 == 0:
                    center.adjust()
                    
                nearest_center = None
                max_sim = -1
                for center in cluster_centers:
                    sim = center.compute_similarity(item)
                    if sim > max_sim:
                        max_sim = sim
                        nearest_center = center

                nearest_center.add(item)

            diff = 0
            for center in cluster_centers:
                diff += center.adjust()

            print("iteration {}, diff: {}".format(i_iter + 1, diff))
            if diff < self.tol:
                break

        cluster_centers = self._remove_less_items_cluster_center(cluster_centers)

        self.cluster_centers_ = cluster_centers

        self._post_process()

        self.labels_ = [item.label for item in items]

    def _post_process(self):
        for label, center in enumerate(self.cluster_centers_):
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
        for center in self.cluster_centers_:
            result.append({
                'keywords': center.keywords,
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
            item = Item(center_data['keywords'])
            center = Center(item, max_features=center_data['max_features'])
            center.label = center_data['label']
            cluster_centers.append(center)
        self.cluster_centers_ = cluster_centers 

    def predict(self, keywords_list):
        labels = []
        items = [Item(keywords) for keywords in keywords_list]
        for item in items:
            center = max(self.cluster_centers_, key=lambda center: center.compute_similarity(item))
            labels.append(center.label)
        return labels

    def _find_best_fit_center(self, centers, item):
        return max(centers, key=lambda center: center.compute_similarity(item))
