from .q_learner import QLearner
from .qatten_learner import QattenLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qtran_transformation_learner import QLearner as QTranTRANSFORMATIONLearner
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .qplex_curiosity_vdn_learner_cds import QPLEX_curiosity_vdn_Learner_cds

from .fast_q_learner import fast_QLearner

from .qplex_curiosity_vdn_learner import QPLEX_curiosity_vdn_Learner
from .qplex_curiosity_vdn_learner_ind import QPLEX_curiosity_vdn_Learner_ind
from .qplex_curiosity_vdn_learner_ca import QPLEX_curiosity_vdn_Learner_ca




REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qtran_transformation_learner"] = QTranTRANSFORMATIONLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["qplex_curiosity_vdn_learner_cds"] = QPLEX_curiosity_vdn_Learner_cds

REGISTRY["qplex_curiosity_vdn_learner_ind"] = QPLEX_curiosity_vdn_Learner_ind
REGISTRY["qplex_curiosity_vdn_learner_ca"] = QPLEX_curiosity_vdn_Learner_ca

REGISTRY["fast_QLearner"] = fast_QLearner
REGISTRY['qplex_curiosity_vdn_learner']=QPLEX_curiosity_vdn_Learner





