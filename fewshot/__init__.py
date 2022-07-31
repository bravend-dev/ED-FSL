from .proto import PrototypicalNetwork
from .matching import MatchingNetwork
# from .mlman import MLMAN
from .relation import RelationNetwork
from .induction import InductionNetwork
from .proto_att import AttPrototypicalNetwork
from .similar_trainer import SimilarFSLTrainer
# from .metaopnethead import *
from .logistic_regression import *
from .melr_network import MELRNetwork, MELRNetworkNoAttention, MELRNetworkNoLoss
from .dproto import DProto
from .melrplus import MELRPlus

from .trainer import FSLTrainer
from .trainer_wsd import WSDFSLTrainer
from .trainer_melr import MELRFSLTrainer
from .trainer_tree_consistency import TreeConsistencyFSLTrainer
from .trainer_melr_tc import MELRTCFSLTrainer
from .reparameterize_trainer import ReparameterizeTrainer
