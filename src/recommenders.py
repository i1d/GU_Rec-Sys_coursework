import pandas as pd
import numpy as np
import time

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from src.metrics import precision_at_k


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, gs_params, weighting=True):
        self.data = data
        self.gs_params = gs_params

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T


        
        
        # test_size_weeks = 3

        self.data_train = data[data['week_no'] < data['week_no'].max() - 3]
        self.data_test = data[data['week_no'] >= data['week_no'].max() - 3]
        
        self.result_train = self.data_train.groupby('user_id')['item_id'].unique().reset_index()
        self.result_train.columns=['user_id', 'actual']
        
        self.result_test = self.data_test.groupby('user_id')['item_id'].unique().reset_index()
        self.result_test.columns=['user_id', 'actual']
 

        # self.model = self.fit(self.user_item_matrix)
        self.model = self.gs_fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)


    @staticmethod
    def _prepare_matrix(data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model
    
    
    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[999999]],
                                        recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]   # удалим юзера из запроса

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    
    def gs_fit(self, user_item_matrix, num_threads=4):
        res_test = []
        res_train = []
        s = []
        fit_time = []
        get_rec_time = []
        gs_models = []
        model_params = []
        d_conclusion = dict()
        r_test = self.result_test.copy()
        r_train = self.result_train.copy()
        step = 1
        steps = 1
        for p in self.gs_params.values():
            steps *= len(p)
        for factor in self.gs_params['factor']:
            for l_reg in self.gs_params['l_reg']:
                for itr in self.gs_params['itr']:
                    print(f'Step {step} of {steps}. Fitting model for factor={factor}, l_reg={l_reg}, iterations={itr}:', end='')
                    s.append(f'als_f-{factor}_lr-{l_reg}_i-{itr}')
                    fit_start = time.time() 
                    gs_model = AlternatingLeastSquares(factors=factor,
                                                    regularization=l_reg,
                                                    iterations=itr, 
                                                    calculate_training_loss=True, 
                                                    use_gpu=False,
                                                    use_native=True,
                                                    use_cg=True,
                                                    random_state=42)
                    gs_model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True) # На вход item-user matrix
                    fit_end = time.time()
                    fit_time.append(round(fit_end - fit_start, 2))
                    print(f'Fit time: {fit_time[-1]} sec.')

                    get_rec_start = time.time()
                    r_test[s[-1]] = r_test['user_id'].apply(lambda x: self._get_recommendations(x, gs_model))
                    r_train[s[-1]] = r_train['user_id'].apply(lambda x: self._get_recommendations(x, gs_model))
                    get_rec_end = time.time()
                    get_rec_time.append(round(get_rec_end - get_rec_start, 2))
                    print(f'Get recommendation time: {get_rec_time[-1]} sec.\n')

                    res_test.append(r_test.apply(lambda row: precision_at_k(row[s[-1]], row['actual']), axis=1).mean())
                    res_train.append(r_train.apply(lambda row: precision_at_k(row[s[-1]], row['actual']), axis=1).mean())

                    gs_models.append(gs_model)
                    model_params.append({'factor': factor, 'l_reg': l_reg, 'itr': itr})
                    step += 1
        d_conclusion['params'] = s
        d_conclusion['result_train'] = res_train
        d_conclusion['result_test'] = res_test
        d_conclusion['fit_time'] = fit_time
        d_conclusion['get_rec_time'] = get_rec_time
        d_conclusion['cv_model'] = gs_models
        d_conclusion['model_params'] = model_params
        d_c = pd.DataFrame(d_conclusion)
        d_c = d_c.sort_values('result_train', ascending=False)
        print(d_c)
        best_model = d_c['cv_model'][0]

        return best_model
