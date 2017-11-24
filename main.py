import numpy as np
from utils.options import Options
from utils.factory import ModelDict, AgentDict

opt=Options()
np.random.seed(opt.seed)

model_prototype = ModelDict[opt.model_type]

agent = AgentDict[opt.agent_type](opt.agent_params,
        model_prototype = model_prototype)

if opt.mode == 1:
    agent.fit_model()
elif opt.mode == 2:
    agent.test_model()
elif opt.mode == 3:
    agent.generate_model()
