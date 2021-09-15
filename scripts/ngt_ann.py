import ngtpy
import random
import numpy as np
import requests

timestamp = 1631581478
query_vectors = np.load(f'/usr/local/src/iur/output/query-vec-{timestamp}.npy')
query_authors = np.load(f'/usr/local/src/iur/output/query-authors-vec-{timestamp}.npy')
target_vectors = np.load(f'/usr/local/src/iur/output/target-vec-{timestamp}.npy')
target_authors = np.load(f'/usr/local/src/iur/output/target-authors-vec-{timestamp}.npy')

def run_ann_local():
    print('Running NGT locally')
    # Find this many top hits
    search_size = 64
    
    print(query_vectors.shape)
    
    ngtpy.create(b"indices/temp_index", query_vectors.shape[1], distance_type='Cosine')
    index = ngtpy.Index(b"indices/temp_index")
    index.batch_insert(query_vectors)
    index.save()
    
    # Compute rank
    num_queries = query_authors.shape[0]
    ranks = np.zeros((num_queries), dtype=np.int32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
        
    for query_index in range(num_queries):
        query = target_vectors[query_index]
        author = query_authors[query_index]
        search_results = index.search(query, search_size)
        # print(search_results)
        # break
    
        distances = [element[1] for element in search_results]
        indices_in_sorted_order = [element[0] for element in search_results]
    
        labels_in_sorted_order = target_authors[indices_in_sorted_order]
        rank = np.where(labels_in_sorted_order == author)
        assert len(rank) == 1
        if len(rank[0]) == 0:
            rank = num_queries-1
        else:
            rank = rank[0] + 1.
        ranks[query_index] = rank
        reciprocal_rank = 1.0 / float(rank)
        reciprocal_ranks[query_index] = (reciprocal_rank)
    
    print(index.get_num_of_distance_computations())
    
    #print(ranks)
    metrics = {
        'MRR': np.mean(reciprocal_ranks),
        'MR': np.mean(ranks),
        'min_rank': np.min(ranks),
        'max_rank': np.max(ranks),
        'median_rank': np.median(ranks),
        'recall@1': np.sum(np.less_equal(ranks,1)) / np.float32(num_queries),
        'recall@2': np.sum(np.less_equal(ranks,2)) / np.float32(num_queries),
        'recall@4': np.sum(np.less_equal(ranks,4)) / np.float32(num_queries),
        'recall@8': np.sum(np.less_equal(ranks,8)) / np.float32(num_queries),
        'recall@16': np.sum(np.less_equal(ranks,16)) / np.float32(num_queries),
        'recall@32': np.sum(np.less_equal(ranks,32)) / np.float32(num_queries),
        'recall@64': np.sum(np.less_equal(ranks,64)) / np.float32(num_queries),
        'num_queries': num_queries,
        'num_targets': target_authors.shape[0]
    }
    
    # Display results
    results = "\n".join([f"{k} {v}" for k, v in metrics.items()])
    print(results)



def run_ann_service():
    print('Running NGT via service')
    # Find this many top hits    
    search_size = 64
    
    print(query_vectors.shape)

    # for vector in query_vectors:
    #     response = requests.post('http://localhost:5000/api/v1/indices/test_index_1/objects/', data=vector.tobytes(), headers={'Content-type': 'application/octet-stream'})
    #     response.raise_for_status()
    #     print(response.json())
    #     break

    # Pick the index to search on
    base_api_url = 'http://localhost:5000/api/v1'
    index_name = 'temp_index'
    url = f"{base_api_url}/indices/{index_name}/search?size={search_size}"

    # Compute rank
    num_queries = query_authors.shape[0]
    ranks = np.zeros((num_queries), dtype=np.int32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
        
    for query_index in range(num_queries):
        query = target_vectors[query_index]
        author = query_authors[query_index]
        # search_results = index.search(query, search_size)

        # temp file
        temp_query_file = '/tmp/temp_query.npy'

        # Write out the search query
        with open(temp_query_file, 'wb') as save_file:
            np.save(save_file, query)

        # Read and post data
        with open(temp_query_file, 'rb') as save_file:
            search_results = requests.post(
                url,
                files={
                    'query_file': save_file
                },
                data={
                    'size': search_size
                }
            )

        search_results.raise_for_status()
        # print(search_results.json())
        # exit()


        distances = [element['distance'] for element in search_results.json()]
        indices_in_sorted_order = [element['object_id'] for element in search_results.json()]
    
        labels_in_sorted_order = target_authors[indices_in_sorted_order]
        rank = np.where(labels_in_sorted_order == author)
        assert len(rank) == 1
        if len(rank[0]) == 0:
            rank = num_queries-1
        else:
            rank = rank[0] + 1.
        ranks[query_index] = rank
        reciprocal_rank = 1.0 / float(rank)
        reciprocal_ranks[query_index] = (reciprocal_rank)
    
    # print(index.get_num_of_distance_computations())
    
    #print(ranks)
    metrics = {
        'MRR': np.mean(reciprocal_ranks),
        'MR': np.mean(ranks),
        'min_rank': np.min(ranks),
        'max_rank': np.max(ranks),
        'median_rank': np.median(ranks),
        'recall@1': np.sum(np.less_equal(ranks,1)) / np.float32(num_queries),
        'recall@2': np.sum(np.less_equal(ranks,2)) / np.float32(num_queries),
        'recall@4': np.sum(np.less_equal(ranks,4)) / np.float32(num_queries),
        'recall@8': np.sum(np.less_equal(ranks,8)) / np.float32(num_queries),
        'recall@16': np.sum(np.less_equal(ranks,16)) / np.float32(num_queries),
        'recall@32': np.sum(np.less_equal(ranks,32)) / np.float32(num_queries),
        'recall@64': np.sum(np.less_equal(ranks,64)) / np.float32(num_queries),
        'num_queries': num_queries,
        'num_targets': target_authors.shape[0]
    }
    
    # Display results
    results = "\n".join([f"{k} {v}" for k, v in metrics.items()])
    print(results)



if __name__ == '__main__':
    run_ann_service()
    # run_ann_local()
