"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Script for partioning into simples shapes
"""
import os.path
import sys
import numpy as np
import argparse
import laspy
from timeit import default_timer as timer
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from copy import deepcopy, copy
sys.path.append("/partition/cut-pursuit/build/src")
sys.path.append("/partition/ply_c")
#sys.path.append("./partition")
import libcp
import libply_c
from graphs import *
#from provider import *

parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--ROOT_PATH', default='/data/dales')
parser.add_argument('--dataset', default='dales', help='s3dis/sema3d/your_dataset')
parser.add_argument('--k_nn_geof', default=45, type=int, help='number of neighbors for the geometric features')
parser.add_argument('--k_nn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
parser.add_argument('--lambda_edge_weight', default=1., type=float, help='parameter determine the edge weight for minimal part.')
parser.add_argument('--reg_strength', default=0.5, type=float, help='regularization strength for the minimal partition')
parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')
parser.add_argument('--voxel_width', default=0.03, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--ver_batch', default=0, type=int, help='Batch size for reading large files, 0 do disable batch loading')
parser.add_argument('--cutoff', default=0, type=int, help='minimum cluster size')
parser.add_argument('--speed', default=4, type=int, help='speed mode for cut pursuit')
parser.add_argument('--overwrite', default=0, type=int, help='Wether to read existing files or overwrite them')
args = parser.parse_args()

#path to data
root = args.ROOT_PATH+'/'
#list of subfolders to be processed
if args.dataset == 's3dis':
    folders = ["Area_1/", "Area_2/", "Area_3/", "Area_4/", "Area_5/", "Area_6/"]
    n_labels = 13
elif args.dataset == 'sema3d':
    folders = ["test_reduced/", "test_full/", "train/"]
    n_labels = 8
elif args.dataset == 'custom_dataset':
    folders = ["train/", "test/"]
    n_labels = 10 #number of classes
elif args.dataset == 'dales':
    folders = ["train_feat/"]
    n_labels = 8 #number of classes
else:
    raise ValueError('%s is an unknown data set' % args.dataset)

times = [0,0,0] #time for computing: features / partition / spg

if not os.path.isdir(root + "clouds"):
    os.mkdir(root + "clouds")
if not os.path.isdir(root + "features"):
    os.mkdir(root + "features")
if not os.path.isdir(root + "superpoint_graphs"):
    os.mkdir(root + "superpoint_graphs")

#features = ["Planarity","Scattering","Verticality"]#,"NormalX","NormalY"]#,"NormalZ"]
#features = ["Verticality"]#,"NormalZ"]

def write_partitions_to_las(ori_file_path, new_file_path, partition_ids,xyz=None, geof=None):
    with laspy.open(ori_file_path) as fh:
        in_file = fh.read()

        # Create a new las file with the same header and point data as the input file
        #out_file = laspy.create(point_format=in_file.header.point_format, file_version=in_file.header.version)
        

        #out_file = laspy.create("with_labels.las", header=in_file.header)
        header =copy(in_file.header)
        if xyz is not None:
            header.scale = np.array([1e-6,1e-6,1e-6])
            header.offset = np.array([0.,0.,0.])
            header.point_count = 0 #len(xyz)
        # Add a new dimension called clusterid to the output file
        for i in range(len(partition_ids)):
            header.add_extra_dim(laspy.ExtraBytesParams(name=f"myclusterid_{i}", type=np.uint32)) #define_new_dimension(name="clusterid", data_type=5, description="Cluster ID")
            header.add_extra_dim(laspy.ExtraBytesParams(name=f"myclusterid_vis_{i}", type=np.uint8))
        if geof is not None:
            num_feats = geof.shape[1]
            for i in range(num_feats):
                header.add_extra_dim(laspy.ExtraBytesParams(name=f"geof_{i}", type=np.float32))
        

        out_file = laspy.LasData(header)
        # Assign the values from the labels variable to the clusterid dimension
        #labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]  # Replace with your own labels data
        #out_file.points = in_file.points
        if xyz is None:
            out_file.x = in_file.x
            out_file.y = in_file.y
            out_file.z = in_file.z
            out_file.classification = in_file.classification
        else:
            out_file.x = xyz[:,0]#*10000.
            out_file.y = xyz[:,1]#*10000.
            out_file.z = xyz[:,2]#*10000.
        #out_file.classification = in_file.classification
        for i in range(len(partition_ids)):
            out_file[f"myclusterid_{i}"] = partition_ids[i]
            out_file[f"myclusterid_vis_{i}"] = np.mod(partition_ids[i],10).astype(np.uint8)
        #print(xyz.shape, partition_ids.shape, )
        if geof is not None:
            num_feats = geof.shape[1]
            for i in range(num_feats):
                out_file[f"geof_{i}"] = geof[:,i].astype(np.float32)
        
        out_file.write(new_file_path)

def partition_file(file_path, feature_list, args, min_partition_id=0, max_size=None):
    partitions = make_partitions(file_path, feature_list, args, max_size)
    partitions += min_partition_id

    write_partitions_to_las(file_path, file_path+"-cluster_5feat_larger.laz", [partitions.copy()], geof=None, xyz=None)#
    return len(np.unique(partitions))


def make_partitions(file_path, feature_list, args, feature_list2=None, args2=None, max_size=None):
    with laspy.open(file_path) as fh:
        las = fh.read()
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        print(xyz.min(0))
        xyz =  (xyz - xyz.min(0)).astype('float32').copy()
        partitions = partition_step(feature_list, args, las, xyz)
        print(f"found {len(np.unique(partitions))} partitions")

        if feature_list2 is not None:
            components = np.unique(partitions)
            partitions2 = np.zeros_like(partitions)
            num_components2 = 0

            for component in components:
                idx = (partitions == component )
                num_points = idx.sum()
                if num_points>args.k_nn_geof and num_points>max_size:
                    comp_partitions = partition_step(feature_list2, args2, las, xyz[idx], idx)
                    
                    partitions2[idx] = comp_partitions + num_components2
                    num_components2 +=len(np.unique(comp_partitions))
                else:
                    partitions2[idx] = num_components2
                    num_components2 += 1
        # elif max_size is not None:
        #     components = np.unique(partitions)
        #     partitions2 = np.zeros_like(partitions)
        #     num_components2 = 0

        #     for component in components:
        #         idx = (partitions == component )
        #         num_points = idx.sum()
        #         if num_points>args.k_nn_geof and num_points>max_size:
        #             comp_partitions = fps_nn(xyz[idx], idx)
                    
        #             partitions2[idx] = comp_partitions + num_components2
        #             num_components2 +=len(np.unique(comp_partitions))
        #         else:
        #             partitions2[idx] = num_components2
        #             num_components2 += 1
    if feature_list2 is not None:
        return partitions, partitions2
    else:
        return partitions

def partition_step(feature_list, args, las, xyz, idx=None):
    geof = np.vstack([las[f] for f in feature_list]).transpose().astype('float32')
    geof[:,2] *= 2.
    if idx is not None:
        geof = geof[idx]
    _, partitions = compute_partitions(args, xyz=None, xyz_pruned=xyz, features_pruned=geof.copy())
    return partitions

def partition_file_2step(file_path, feature_list, feature_list2, args,args2):
    partitions, partitions2 = make_partitions(file_path, feature_list, args, feature_list2=feature_list2, args2=args2)
    

    write_partitions_to_las(file_path, file_path+"-cluster_2step.laz", [partitions.copy(),partitions2.copy()], geof=None, xyz=None)#



def partition_file_hierarchical(file_path, feature_list, args):
    with laspy.open(file_path) as fh:
        las = fh.read()
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        print(xyz.min(0))
        xyz =  (xyz - xyz.min(0)).astype('float32').copy()
        voxel_widths = [0.5]

        voxel_width = voxel_widths[0]
        geof = np.vstack([las[f] for f in feature_list]).transpose().astype('float32').copy()

        #if voxel_width>0.:
        xyz_pruned = libply_c.prune(xyz, voxel_width, np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), np.array(1,dtype='u1'), 0,0)[0].copy()
        #else:
        #    xyz_pruned = xyz.copy()
        _, partitions = compute_partitions(args, xyz, xyz_pruned, None)
        components = np.unique(partitions)
        partitions2 = np.zeros_like(partitions)
        num_components2 = 0

        for component in components:
            idx = (partitions == component )
            if idx.sum()>args.k_nn_geof:
                comps, comp_partitions = compute_partitions(args, xyz = None, xyz_pruned = xyz[idx].copy(), features_pruned=geof[idx].copy())
                
                partitions2[idx] = comp_partitions + num_components2
                num_components2 += len(comps)
            else:
                partitions2[idx] = num_components2
                num_components2 += 1

    write_partitions_to_las(file_path, file_path+"-cluster-hierar.laz", [partitions.copy(), partitions2.copy()], geof=None, xyz=None)#

def compute_partitions(args, xyz, xyz_pruned,  kd_tree=None, features_pruned=None):
    
    
     
    graph_nn, target_fea = compute_graph_nn_2(xyz_pruned, args.k_nn_adj, args.k_nn_geof)
    if features_pruned is None:
        print("-- compute geofs")
        features_pruned = libply_c.compute_geof(xyz_pruned, target_fea.copy(), args.k_nn_geof).astype('float32').copy()
    graph_nn["edge_weight"] = np.array(1. / ( args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
    print("-- cut-pursuit")
    components, in_component = libcp.cutpursuit(features_pruned, graph_nn["source"], graph_nn["target"]
                                        , graph_nn["edge_weight"], args.reg_strength, cutoff=args.cutoff, speed=args.speed)
    
    #file_path = "/data/dales/train_feat/5080_54435_mini.las"
    #write_partitions_to_las(file_path, file_path+f"-cluster_{len(xyz)}.las", [in_component.copy()], geof=features_pruned, xyz=xyz_pruned)
    if xyz is not None:
        kd_tree = build_kd_tree(xyz, kd_tree)
        partition_ids = interpolate_partition_ids(xyz, xyz_pruned, in_component)
        return components, partition_ids
    else: 
        return components, in_component

def interpolate_partition_ids(xyz, xyz_pruned, partition_ids_pruned,  kd_tree_pruned=None):
    """compute the knn graph"""
    kd_tree_pruned = build_kd_tree(xyz_pruned, kd_tree_pruned)

    neighbors = kd_tree_pruned.query(xyz, k=1, return_distance=False)
    partition_ids = partition_ids_pruned[neighbors.flatten()] 
    return partition_ids

def build_kd_tree(xyz, kd_tree):
    if kd_tree is None:
        kd_tree = KDTree(xyz, leaf_size=30, metric='euclidean')
    return kd_tree

def normalize(xyz):
    xyz = xyz - xyz.min(0) 
    xyz =  xyz / xyz.max(0)
    return xyz

#filename = "/data/dales/train_feat/5080_54435_mini.las"
features = ["Planarity","Scattering","Verticality"]#,"NormalX","NormalY"]#,"NormalZ"]
features2 = features+["NormalX","NormalY"]
args2 = deepcopy(args)
args2.reg_strength = 0.9*args.reg_strength
num_total_partitions = 0
for i in range(1,9):
    filename = f"/data/dales/train_feat/5135_54495_{i}.laz" #"5080_54435.laz"
    #partition_file_2step(filename, feature_list=features,feature_list2=features2, args=args,args2=args2)
    num_partitions = partition_file(filename, feature_list=features2, args=args, min_partition_id = num_total_partitions)
    num_total_partitions += num_partitions

