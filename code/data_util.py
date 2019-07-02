import torch
import torch.nn as nn
import numpy as np
import sys
import os
from io import BytesIO
import gzip
from glob import glob
from tqdm import tqdm
from random import shuffle

# BASE_DATA_DIR = '../Sem 2 block 3/MLQS/Assignment/ML4QS/PythonCode/data/assgn3_data/interpolation'
BASE_DATA_DIR = 'C:/Users/asus/Desktop/MS in AI/Sem 2/Sem 2 block 3/MLQS/Assignment/ML4QS/PythonCode/data/assgn3_data/interpolation'

FEATURES_TO_REMOVE = ['discrete:wifi_status:is_not_reachable', 'discrete:wifi_status:is_reachable_via_wwan', 'discrete:wifi_status:missing', 'lf_measurements:light', 
                      'lf_measurements:pressure', 'lf_measurements:proximity_cm', 'lf_measurements:proximity', 'lf_measurements:relative_humidity', 
                      'lf_measurements:battery_level', 'lf_measurements:screen_brightness', 'lf_measurements:temperature_ambient', 'location:min_speed', 'location:max_speed', 
                      'audio_properties:normalization_multiplier', 'watch_heading:mean_cos', 'watch_heading:std_cos', 'watch_heading:mom3_cos', 'watch_heading:mom4_cos', 
                      'watch_heading:mean_sin', 'watch_heading:std_sin', 'watch_heading:mom3_sin', 'watch_heading:mom4_sin', 'watch_heading:entropy_8bins', 
                      'discrete:app_state:is_active', 'discrete:app_state:is_inactive', 'discrete:app_state:is_background', 'discrete:app_state:missing', 
                      'discrete:battery_plugged:is_ac', 'discrete:battery_plugged:is_usb', 'discrete:battery_plugged:is_wireless', 'discrete:battery_plugged:missing', 
                      'discrete:battery_state:is_unknown', 'discrete:battery_state:is_unplugged', 'discrete:battery_state:is_not_charging', 
                      'discrete:battery_state:is_discharging', 'discrete:battery_state:is_charging', 'discrete:battery_state:is_full', 'discrete:battery_state:missing', 
                      'discrete:on_the_phone:is_True', 'discrete:on_the_phone:missing', 'discrete:ringer_mode:is_normal', 'discrete:ringer_mode:is_silent_no_vibrate', 
                      'discrete:ringer_mode:is_silent_with_vibrate', 'discrete:ringer_mode:missing']
        
    
    
    
def parse_header_of_csv(csv_str):
	# Isolate the headline columns:
	headline = csv_str[:csv_str.index(b'\n')]
	columns = headline.split(b',')

	# print(columns)
	# The first column should be timestamp:
	# assert columns[0] == b'timestamp'
	# The last column should be label_source:
	assert columns[-1].startswith(b'label_source')
	
	# Search for the column of the first label:
	for (ci,col) in enumerate(columns):
		if col.startswith(b'label:'):
			first_label_ind = ci
			break
		pass

	# Feature columns come after timestamp and before the labels:
	feature_names = columns[1:first_label_ind]
	# Then come the labels, till the one-before-last column:
	label_names = columns[first_label_ind:-1]
	for (li,label) in enumerate(label_names):
		# In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
		assert label.startswith(b'label:')
		label_names[li] = label.replace(b'label:',b'')
		pass
	
	return (feature_names,label_names)

def parse_body_of_csv(csv_str,n_features):
	# Read the entire CSV body into a single numeric matrix:
	# full_table = np.loadtxt(io.BytesIO(csv_str),delimiter=',',skiprows=1)
	full_table = np.genfromtxt(BytesIO(csv_str),delimiter=',')
	
	# Timestamp is the primary key for the records (examples):
	timestamps = full_table[:,0].astype(int)
	
	# Read the sensor features:
	X = full_table[:,1:(n_features+1)]
	
	# Read the binary label values, and the 'missing label' indicators:
	trinary_labels_mat = full_table[:,(n_features+1):-1] # This should have values of either 0., 1. or NaN
	M = np.isnan(trinary_labels_mat) # M is the missing label matrix
	Y = np.where(M,0,trinary_labels_mat) > 0. # Y is the label matrix
	
	return (X,Y,M,timestamps)

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
def read_user_data(uuid):
	# user_data_file = '../data/%s.features_labels.csv.gz' % uuid
	user_data_file = BASE_DATA_DIR + '/%s.features_labels.csv.gz' % uuid

	# Read the entire csv file of the user:
	with gzip.open(user_data_file,'rb') as fid:
		csv_str = fid.read()
		pass

	(feature_names,label_names) = parse_header_of_csv(csv_str)
	n_features = len(feature_names)
	(X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features)

	return (X,Y,M,timestamps,feature_names,label_names)



def get_full_name(all_user_data, prefix):
    user = []
    for u in all_user_data:
        for pre in prefix:
            if u.startswith(pre):
                user.append(u)
    return user


def load_data_from_uuid(uuid):
    
    x_list, y_list, m_list, timestamps_list = [], [], [], []
    
    for id in tqdm(uuid):
        (X,Y,M,timestamps,feature_names,label_names) = read_user_data(id)
        x_list.append(X)
        y_list.append(Y)
        m_list.append(M)
        timestamps_list.append(timestamps)
    
    return (x_list, y_list, m_list, timestamps_list, feature_names, label_names)



def feat_and_user_prep():
    print("="*70 + "\n\t\t\t DATA and FEATURE prep\n" + "="*70)
    
    
    if os.path.exists(BASE_DATA_DIR):
        # all_user_data = os.listdir(BASE_DATA_DIR)
        all_user_data = [u.split("/")[-1].split("\\")[-1].rsplit(".",3)[0] for u in sorted(glob(BASE_DATA_DIR + "/*.features_labels.csv.gz"))]
        users_to_remove = get_full_name(all_user_data, ["7D9BB", "96A3"])
        print("\nUsers removed = ", users_to_remove)
        [all_user_data.remove(u) for u in users_to_remove]
    else:
        print("[!] User data path does not exist..!!")
    
    
    # DATA-SPLITS
    val_users = ["0E618", "2C32", "4FC321", "5EF64"]
    test_users = ["9DC38", "11B5E", "74B86", "83CF6", "9759", "A76A", "CDA3", "7CE37510"]
    train_users = []
    
    data_split = {"val" : get_full_name(all_user_data, val_users),
                  "test" : get_full_name(all_user_data, test_users)}
    
    data_split["train"] = [u for u in all_user_data if all([u not in val for key,val in data_split.items()])]
    
    print("\nTrain users = ", len(data_split["train"]))
    print("Test users = ", len(data_split["test"]))
    print("Val users = ", len(data_split["val"]))
    
    
    ###### Loading Data as per above specifications ########
    print("\n" + "-"*40 + "\nLoading Data from user-IDs\n" + "-"*40)
    
    for key,_ in data_split.items():
        print("\nLoading ...  {}".format(key))
        data_split[key] = load_data_from_uuid(data_split[key])

    
    return data_split

def create_data(data_split, seq_len=100, normalize_features=False, instance_weight_exp=0.5):
    (X,Y,M,timestamps,feature_names,label_names) = data_split
    return DATASET(X, Y, M, timestamps, feature_names, label_names, seq_len, normalize_features=normalize_features, instance_weight_exp=instance_weight_exp)


class DATASET():
    def __init__(self, X, Y, M, timestamps, feature_names, label_names, seq_len, normalize_features=False, instance_weight_exp=0.5):
        self.X = X
        self.Y = Y
        self.M = M 
        self.timestamps = timestamps
        self.feature_names = feature_names
        self.label_names = label_names
        self.seq_len = seq_len
        self.prepare_data(normalize_features, instance_weight_exp=instance_weight_exp)
        
    def prepare_data(self, normalize_features, instance_weight_exp=0.5):
        self.user_sizes = [x.shape[0] for x in self.X]
        self.user_sizes.insert(0,0)
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)
        self.M = np.concatenate(self.M, axis=0)
        self.timestamps = np.concatenate(self.timestamps, axis=0)
        
        # self.valid_indices = []
        # for u_id in range(len(self.user_sizes)-1):
        #     self.valid_indices += list(range(self.user_sizes[u_id], self.user_sizes[u_id] + self.user_sizes[u_id+1]-self.seq_len+1))
        
        #########  Making sure we have no missing timesteps in a sequence (of length seq_len)  ################# 
        print("\nCalculating missing timesteps..")
        self.valid_indices = list(range(sum(self.user_sizes) - self.seq_len + 1))
        self.continuous_timestamps = ((self.timestamps[1:] - self.timestamps[:-1]) <= 120)
              
        indices_copy = self.valid_indices[:]
        for i in indices_copy:
            if np.any(self.continuous_timestamps[max(i-1,0):max(i+self.seq_len-2, 0)] == 0):
                self.valid_indices.remove(i)
                
        ######## Removing unwanted features and their data from the dataset  ###########
        valid_feats = [i for i in range(len(self.feature_names)) if self.feature_names[i].decode("utf-8") not in FEATURES_TO_REMOVE]
        self.X = self.X[:, valid_feats]
        self.feature_names = [self.feature_names[i] for i in valid_feats]
        
        # Replacing NaNs with 0 for processing
        self.X = np.where(np.isnan(self.X), 0, self.X)
        
        self.feature_count = self.X.shape[1]
        self.label_count = self.Y.shape[1]
        
        ######## Apply instance weighting for unbalanced classes #########
        print("Calculating Instance weighting...")
        self.pos_label_per_class = np.sum(self.Y * (1 - self.M), axis=0)
        self.neg_label_per_class = np.sum((1-self.Y) * (1 - self.M), axis=0)
        self.pos_weights = 1 / (2 * (self.pos_label_per_class /(1 - self.M).sum(axis=0)))  # Inverse ratio of (valid)positive class examples to the total valid class examples
        self.neg_weights = 1 / (2 * (self.neg_label_per_class /(1 - self.M).sum(axis=0)))  # Inverse ratio of (valid)negative class examples to the total valid class examples
        self.pos_weights = np.power(self.pos_weights, instance_weight_exp)
        self.neg_weights = np.power(self.neg_weights, instance_weight_exp)
        
        shuffle(self.valid_indices)
        self.current_pos = 0
        
        
        
    def get_batch(self, batch_size):
        batch_x = np.zeros((batch_size, self.seq_len, self.feature_count))
        batch_y = np.zeros((batch_size, self.seq_len, self.label_count))
        batch_inst_wts = np.zeros((batch_size, self.seq_len, self.label_count))
         
        for i in range(batch_size):
            index_pos = self.valid_indices[self.current_pos]
            self.current_pos+=1
            if self.current_pos>= len(self.valid_indices):
                shuffle(self.valid_indices)
                self.current_pos = 0
             
            batch_x[i,:,:] = self.X[index_pos : index_pos + self.seq_len,:]
            batch_y[i,:,:] = self.Y[index_pos : index_pos + self.seq_len,:]
            batch_inst_wts = (1 - self.M[index_pos : index_pos + self.seq_len,:]) * (batch_y[i,:,:]) * (self.pos_weights) + (1 - batch_y[i,:,:]) * self.neg_weights
             
        
        return batch_x, batch_y, batch_inst_wts
        


    def print_statistics(self):
        print("Number of users: %d" % (len(self.user_sizes)-1))
        print("Number of timepoints: %d" % (self.X.shape[0]))
        print("Number of training examples: %d" % (len(self.valid_indices)))
        print("Number of features: %d" % (self.feature_count))
        print("Number of labels: %d" % (self.label_count))
        print("Sequence length: %d" % (self.seq_len))


# uuid = '7CE37510-56D0-4120-A1CF-0E23351428D2'
# (X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid)
# print("\nUser {}'s data shape: {}".format(uuid, X.shape))
# print(X[10].shape)
    
