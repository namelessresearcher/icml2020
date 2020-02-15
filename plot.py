import numpy as np
import matplotlib.pyplot as plt

#precomputed data used in the paper
y = np.array([7.452930027717221, 6.428576119005332, 5.330663735125866, 4.2314253 ,2.749511267242876,1.2725809983205592 ,0.7653683909468516])
x = np.array([391.0,253.0,172.0,113.0,52.0,10.0,3.0])/50000
error = np.array([0.514232,0.454315,0.42634,0.36354,0.25425,0.1114323,0.074382])
after_queries = np.array([1.7596553, 1.52432, 1.415523, 1.245235, 0.84324,0.61432444,0.5855629104])
error_after_queries = np.array([0.168523, 0.125243, 0.115243, 0.0745328, 0.0315243, 0.02415320,0.0143243])

#to avoid type 3 fonts
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True


fig, ax = plt.subplots()
ax.plot(x,y, marker='o')
ax.fill_between(x, y-error, y+error, alpha=0.15, antialiased=True,label='_nolegend_')

ax.plot(x,after_queries, marker='o')
ax.fill_between(x, after_queries-error_after_queries, after_queries+error_after_queries, alpha=0.1, antialiased=True,label='_nolegend_')

ax.legend(['0 queries','100 queries'],shadow=True)
ax.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
plt.xlabel('Graph irregularity', fontsize=13)
plt.ylabel('Relative error (\%)', fontsize=13)
sizex = 5
fig.set_size_inches(sizex,sizex/1.2777)
fig.tight_layout()
plt.savefig('regularity.pdf', format='pdf')

