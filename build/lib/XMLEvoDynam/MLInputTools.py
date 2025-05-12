import numpy as np
import matplotlib.pyplot as plt 
import MDAnalysis as mda
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd

class MLInputTools:
    
    def construct_feature_names(GROUP1, GROUP2):
        columns = np.empty(int(np.shape(GROUP1)[0]*np.shape(GROUP2)[0]), dtype=object)
        c_num = 0
        for g1 in range(len(GROUP1)):
            for g2 in range(len(GROUP2)):
                columns[c_num] = f"{GROUP1[g1]}_{GROUP2[g2]}"
                c_num = c_num + 1
        return columns
    
    def euclidean_distance(value):
        return (value[0]**2 + value[1]**2 + value[2]**2)**(1/2)
    
    def import_trr(reference_structure, trajectory):
        u = mda.Universe(reference_structure, trajectory)
        return u
    
    def compute_com_distance(u, pace, batch=False, batchStart = -1, batchEnd = -1):
    
        n_residues = len(u.residues)
        if batch is True and batchStart != -1 and batchEnd != -1:
            n_frames = len(u.trajectory[batchStart:batchEnd:pace])
            com_array = np.zeros((n_frames, n_residues, 3))
            for i, ts in tqdm(enumerate(u.trajectory[batchStart:batchEnd:pace]), total=n_frames, desc="Residue COM"):
                for j, res in enumerate(u.residues):
                    com_array[i, j, :] = res.atoms.center_of_mass()
            return com_array
    
        elif batch is False:
            n_frames = len(u.trajectory[::pace])
            com_array = np.zeros((n_frames, n_residues, 3))
            batchStart, batchEnd = 0, n_frames
    
            for i, ts in tqdm(enumerate(u.trajectory[batchStart:batchEnd:pace]), total=n_frames, desc="Residue COM"):
                for j, res in enumerate(u.residues):
                    com_array[i, j, :] = res.atoms.center_of_mass()
            return com_array
    
    
    def compute_feature_matrix(com_array, group1, group2, pace):
        feature_list = []
        for frame in tqdm(range(com_array.shape[0]), total=com_array.shape[0], desc="Feature Matrix"):
            coms = com_array[frame]
            D = cdist(coms[group1], coms[group2], metric='euclidean')
            feature_list.append(D)
    
        feature_list = np.array(feature_list)
        feature_array = feature_list.reshape((np.shape(feature_list)[0], np.shape(feature_list)[1]*np.shape(feature_list)[2]))
        return feature_array
    
    
    def weights_to_bias(wfile):
        bKT=2.5
        wk = np.exp(np.loadtxt(wfile, comments="#")[:,-1]/bKT)
        return wk/np.sum(wk)
    
    def get_dt_label(nFrames):
        return np.arange(0,nFrames,1).reshape((nFrames,1))
    
    def get_identity_label(nFrames, label):
        return np.full(nFrames, label, dtype=int).reshape((nFrames,1))
    
    def get_final_XML_input(features, bias, replica, dt):
        return np.concatenate((features, bias, replica, dt), axis=1)
    
    def get_batch_endpoints(batch_size, nFrames):
        batch_start = np.ceil(np.arange(0, 1, batch_size)*nFrames).astype(int)
        batch_end = np.ceil(np.arange(batch_size, 1+batch_size, batch_size)*nFrames).astype(int)
        return batch_start, batch_end
    
    def get_batch_points(nBatches, nFrames):
        frames = np.arange(0,nFrames,1)
        batchFrames = []
        batchStart = np.round(np.linspace(0, nFrames, nBatches+1)).astype(int)
        for batch in range(len(batchStart)-1):
            batchFrames.append(frames[batchStart[batch]:batchStart[batch+1]])
        return batchFrames
    
    def construct_XML_input(reference, trajectory, weights, pace, g1, g2, featureLabel, fOut, nBatches = -1):

        
        #print(f"Generating Column Names")
        #column_names = MLInputTools.construct_feature_names(g1, g2)
    
        print(f"Import Trajectory")
        t = MLInputTools.import_trr(reference, trajectory)
        nFramesAnalysis, nFramesTotal = len(t.trajectory[::pace]), len(t.trajectory)
        
        print(f"Genearte labels and timesteps")
        replica_labels = MLInputTools.get_identity_label(nFramesTotal, featureLabel)
        timestep_labels = MLInputTools.get_dt_label(nFramesTotal)
    
        print(f"Compute bias")
        if weights is not False:
            bias = MLInputTools.weights_to_bias(weights).reshape((nFramesTotal, 1))
        else:
            bias = np.ones(nFramesTotal).reshape((nFramesTotal, 1))
        if nBatches != -1:
    
            batch_pts = MLInputTools.get_batch_points(nBatches, nFramesTotal)
    
    
            for batch in tqdm(range(len(batch_pts)), desc="Batch Number"):
                cBatchStart, cBatchEnd = batch_pts[batch][0], batch_pts[batch][-1]
                com_array = MLInputTools.compute_com_distance(t, pace, batch=True, batchStart = cBatchStart, batchEnd = cBatchEnd)
                distance_matrix = MLInputTools.compute_feature_matrix(com_array, g1, g2, pace)
                np.save(f"{fOut}_batch{batch}.npy", MLInputTools.get_final_XML_input(distance_matrix, bias[cBatchStart:cBatchEnd:pace], replica_labels[cBatchStart:cBatchEnd:pace], timestep_labels[cBatchStart:cBatchEnd:pace]))
    
        elif nBatches == -1:
            print(f"Compute center of masses")
            com_array = MLInputTools.compute_com_distance(t, pace, batch=False)
    
            print(f"Compute feature values")
            distance_matrix = MLInputTools.compute_feature_matrix(com_array, g1, g2, pace)
            
            print(f"Use memmap to get the final dataset")
            output = MLInputTools.get_final_XML_input(distance_matrix, bias[::pace], replica_labels[::pace], timestep_labels[::pace])
            outputfile = np.memmap(f"{fOut}.npy", dtype='float32', mode='w+', shape=np.shape(output))
            for line in range(len(output)):
                outputfile[line] = output[line]
            outputfile.flush()
            print(f"ML Input file successfully generated at: {fOut}")
