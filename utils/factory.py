#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Do 23 Nov 2017 17:26:51 CET
Info:
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from core.models.deeplab_resnet import Res_Deeplab
ModelDict={"Deeplab": Res_Deeplab}

from core.agents.deeplab import DeeplabAgent
AgentDict={"Deeplab": DeeplabAgent}
