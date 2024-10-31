import numpy as np

def process_signature(signature, center=True, add_noise=False):
    signature = signature.copy()
    signature.columns = ["i","v"]
    # optionally noise can be added as a fraction of the expression values
    if add_noise:
        signature.iloc[:,1] = signature.iloc[:,1] + np.random.normal(signature.shape[0])/(np.mean(np.abs(signature.iloc[:,1]))*100000)
    signature = signature.sort_values("v", ascending=False).set_index("i")
    signature = signature[~signature.index.duplicated(keep='first')]
    if center:
        signature.loc[:,"v"] -= np.mean(signature.loc[:,"v"])            
    abs_signature = np.array(np.abs(signature.loc[:,"v"]))
    return signature, abs_signature