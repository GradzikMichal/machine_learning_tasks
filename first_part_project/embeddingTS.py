import pandas as pd
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def embeddingTS():
    import tensorflow as tf
    from tensorflow.keras import layers

    articles = pd.read_csv('../resources/articles.csv')
    # customers = pd.read_csv('../resources/customers.csv')
    transactions = pd.read_csv('../resources/transactions_train.csv')
    merged_df = pd.merge(
        articles[['article_id', 'product_type_no', "colour_group_code", 'index_group_no', 'garment_group_no']],
        transactions[['customer_id', 'article_id', 't_dat']], on='article_id')
    merged_df = merged_df.sort_values(by=['t_dat'], ascending=[True])
    merged_df = merged_df[['customer_id', 'product_type_no', "colour_group_code", 'index_group_no', 'garment_group_no']]
    merged_df = merged_df.drop_duplicates()
    # print(merged_df.columns)
    # Index(['product_code', 'article_id', 'product_type_no', 'colour_group_code',
    #       'index_group_no', 'garment_group_no', 'customer_id'],
    #      dtype='object')
    prod_type_uniq = merged_df["product_type_no"].unique()
    colour_code_uniq = merged_df["colour_group_code"].unique()
    index_group_no_uniq = merged_df["index_group_no"].unique()
    garment_group_no_uniq = merged_df["garment_group_no"].unique()
    prod_type_list = tf.feature_column.categorical_column_with_vocabulary_list("product_type_no", prod_type_uniq)
    colour_group_list = tf.feature_column.categorical_column_with_vocabulary_list("colour_group_code", colour_code_uniq)
    index_group_list = tf.feature_column.categorical_column_with_vocabulary_list("index_group_no", index_group_no_uniq)
    garment_group_list = tf.feature_column.categorical_column_with_vocabulary_list("garment_group_no",
                                                                                   garment_group_no_uniq)

    prod_type_embedding_column = tf.feature_column.embedding_column(prod_type_list, dimension=3)
    colour_group_embedding_column = tf.feature_column.embedding_column(colour_group_list, dimension=3)
    index_group_embedding_column = tf.feature_column.embedding_column(index_group_list, dimension=3)
    garment_group_embedding_column = tf.feature_column.embedding_column(garment_group_list, dimension=2)

    value_dict = {"product_type_no": merged_df["product_type_no"].values}
    feature_layer = layers.DenseFeatures(prod_type_embedding_column)
    tensor_obj = feature_layer(value_dict)
    feature_matrix = tensor_obj.numpy()
    df_columns = pd.DataFrame(feature_matrix,
                              columns=[ "Prod_type1"
                                  "Prod_type1", "Prod_type2","Prod_type3"
                                       ])
    merged_df = merged_df.drop(columns='product_type_no')
    merged_df[['Prod_emb1', 'Prod_emb2', 'Prod_emb3']] = df_columns
    #merged_df[['Prod_emb1']] = df_columns

    df_columns = None
    feature_matrix = None
    tensor_obj = None
    feature_layer = None

    #value_dict = {"colour_group_code": merged_df["colour_group_code"].values}
    #feature_layer = layers.DenseFeatures(colour_group_embedding_column)
    #tensor_obj = feature_layer(value_dict)
    #feature_matrix = tensor_obj.numpy()
    #df_columns = pd.DataFrame(feature_matrix,
    #                          columns=[ "Colour_type1"
    #                              "Colour_type1", "Colour_type2", "Colour_type3"
    #                                   ])
    merged_df = merged_df.drop(columns='colour_group_code')
    #merged_df[['Color_emb1', 'Color_emb2', 'Color_emb3']] = df_columns
    #merged_df[['Color_emb1']] = df_columns

    df_columns = None
    feature_matrix = None
    tensor_obj = None
    feature_layer = None

    value_dict = {"index_group_no": merged_df["index_group_no"].values}
    feature_layer = layers.DenseFeatures(index_group_embedding_column)
    tensor_obj = feature_layer(value_dict)
    feature_matrix = tensor_obj.numpy()
    df_columns = pd.DataFrame(feature_matrix,
                              columns=[ "Index_group1"
                                  "Index_group1", "Index_group2", "Index_group3"
                                       ])
    merged_df = merged_df.drop(columns='index_group_no')
    merged_df[['Index_emb1', 'Index_emb2', 'Index_emb3']] = df_columns
    #merged_df[['Index_emb1']] = df_columns

    df_columns = None
    feature_matrix = None
    tensor_obj = None
    feature_layer = None

    value_dict = {"garment_group_no": merged_df["garment_group_no"].values}
    feature_layer = layers.DenseFeatures(garment_group_embedding_column)
    tensor_obj = feature_layer(value_dict)
    feature_matrix = tensor_obj.numpy()
    df_columns = pd.DataFrame(feature_matrix,
                              columns=["Garment_group1"
                                  "Garment_group1", "Garment_group2"
                                       ])
    merged_df = merged_df.drop(columns='garment_group_no')
    merged_df[["Garment_group1", "Garment_group2"]] = df_columns
    #merged_df[["Garment_group1"]] = df_columns
    del df_columns
    del feature_matrix
    del tensor_obj
    del feature_layer
    del prod_type_list
    del index_group_list
    del garment_group_list
    del colour_group_list

    #print(merged_df.columns)
    '''
    Index(['customer_id', 'product_type_no', 'colour_group_code', 'index_group_no',
           'garment_group_no', 'Prod_emb1', 'Prod_emb2', 'Prod_emb3', 'Color_emb1',
           'Color_emb2', 'Color_emb3', 'Index_emb1', 'Index_emb2', 'Index_emb3',
           'Garment_group1', 'Garment_group2', 'Garment_group3'],
          dtype='object')
    '''
    #print(merged_df)
    merged_df = merged_df.dropna()
    merged_df = merged_df.groupby('customer_id').filter(lambda x: x['customer_id'].count() >= 20)
    #print(merged_df)
    merged_df.to_csv('../resources/merged.csv', index=False)


if __name__ == '__main__':
    #embeddingTS()
    merged_df = pd.read_csv('../resources/merged.csv')
    articles_vector = merged_df[['Prod_emb1', 'Prod_emb2', 'Prod_emb3', 'Index_emb1', 'Index_emb2', 'Index_emb3',
           'Garment_group1', 'Garment_group2']]
    #articles_vector = merged_df[['Prod_emb1', 'Color_emb1',
    #                                    'Index_emb1',
    #                                    'Garment_group1']]
    articles_vector = articles_vector.drop_duplicates()
    articles_vector = articles_vector.to_numpy()
    last_transactions = merged_df.groupby('customer_id').tail(1)
    previous_items = merged_df.drop(index = last_transactions.index)
    previous_items = previous_items.groupby('customer_id').mean()
    customer_idx = previous_items.index
    #print(last_transactions[['customer_id']].nunique()) #275364
    #print(previous_items.head(1).to_numpy())

    #knn = NearestNeighbors(n_neighbors=500, algorithm='auto')
    #knn.fit(articles_vector)
    #def get_neighbors(id):
    numOfUser = previous_items.size
    res = []
    aaa = 0
    for uid in range(numOfUser):
        vector = previous_items.loc[customer_idx[uid]]
        vector = vector.to_numpy()
        #neigh = knn.kneighbors([vector], 10, return_distance=True)
        result = np.dot(articles_vector, vector)
        sort = np.sort(result)[::-1]
        top10 = sort[:20]
        idx = []
        for i in top10:
            idx.append(np.argwhere(result == i))
            op = 0
            for id in idx:
                if (articles_vector[id][0][0] == last_transactions.loc[last_transactions['customer_id'] == customer_idx[uid]].to_numpy()[0][1:]).all():
                    op = 1
                    res.append(1)
                    break
            if op == 1:
                break
            print(sum(res), sum(res)/numOfUser, aaa)
        aaa = aaa + 1
'''
-0.81806964 -0.5250745 -0.5612142 0.03700546 -0.21183196 0.080402866
  0.563251 0.0643402]]
  
    [-0.38957423  0.25237995  0.09822659 - 1.1488198 - 0.68397534
     - 0.6391043   1.0721966 - 0.3751195 - 0.39959148  0.2895533
     0.9924948]
'''
