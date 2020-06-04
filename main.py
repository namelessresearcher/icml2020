from functions import *


#-------------------------------import dataset------------------------
G1 = snap.LoadEdgeList(snap.PUNGraph, "ego-gplus.txt", 0, 1)
G = snap.ConvertGraph(snap.PUNGraph, G1)
print G.GetNodes(), ' nodes'
print G.GetEdges(), ' edges' 
print snap.GetTriads(G), ' triangles'


#-----------------------------triangle estimator demo-----------------------
actual = snap.GetTriads(G)
noise_model = 'main'

k = 100
it = 100
p = 0.5


results = []

for i in tqdm(range(it)):

    Gs = gen_sample(G,p, noise_model)  
    val =  estimator_triangles(G,Gs,k,noise_model,p, use_one_sample=True,filtering=0.5, random_queries=False, learn_weights=False)
    results.append(val)

error = calc_error(results, actual) 
print error 


#-----------------------------edge estimator demo-----------------------
actual = G.GetEdges()
noise_model = 'main'
k = 100
it = 100
p = 0.5


results = []

for i in tqdm(range(it)):
    Gs = gen_sample(G,p, noise_model)  
    val =  estimator_edges(G,Gs,k,p, use_one_sample=True,filtering=0.5, random_queries=False)
    results.append(val)

error = calc_error(results, actual) 
print error 

    

