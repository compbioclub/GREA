import numpy as np

####NEW CODE####

def process_signature(sig_val, center=True, add_noise=False, **kwargs):
    # optionally noise can be added as a fraction of the expression values
    if add_noise:
        sig_val += np.random.normal(sig_val.shape)/(np.mean(np.abs(sig_val, axis=0))*100000)
    if center:
        sig_val -= sig_val.mean(axis=0)
    return sig_val


def process_ss_signature(signature, center=True, add_noise=False):
    signature = signature.copy()

    # optionally noise can be added as a fraction of the expression values
    if add_noise:
        for i in range(signature.shape):
            signature.iloc[:,i] = signature.iloc[:,i] + \
            np.random.normal(signature.shape[0])/(np.mean(np.abs(signature.iloc[:,i]))*100000)
    if center:
        signature -= signature.values.mean(axis=0)   
    abs_signature = signature.abs()
    return signature, abs_signature


def process_nss_signature(signature, center=True, add_noise=False):
    signature = signature.reset_index()
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