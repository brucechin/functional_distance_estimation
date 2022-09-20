import numpy as np



start_range = 5
end_range = 10 
for i in range(start_range, end_range):
    d = 2 * i 
    n = 2**(2*i)
    k = i 
    dataset = []
    for j in range(n):
        row = np.random.rand(d) * 2 - 1.0
        row = row/np.linalg.norm(row, ord = 2)
        dataset.append(row)
    
    sketch = np.random.normal(loc = 0, scale = 1)
    sketch = sketch * (1/np.sqrt(k))
    