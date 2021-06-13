# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:46:28 2021

@author: CYLin
"""


import HydroCNHS

Path = r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest\model_loadtest_NoABM.yaml"
#Path = r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest\model_loadtest.yaml"
Model = HydroCNHS.loadModel(Path)


#%%
def formSimSeq(Node, BackTrackingDict, GaugedOutlets):
    """A recursive function, which keep tracking back upstream nodes until reach the most upstream one.
    """
    SimSeq = []
    def trackBack(node, SimSeq, BackTrackingDict, TempDict = {}, AddNode = True):
        if AddNode:
            SimSeq = [node] + SimSeq
        if BackTrackingDict.get(node) is not None:
            gaugedOutlets = [o for o in BackTrackingDict[node] if o in GaugedOutlets] 
            if len(gaugedOutlets) >= 1:
                # g > 1 or len(g) < len(all) => update TempDict
                if len(gaugedOutlets) > 1 or len(gaugedOutlets) < len(BackTrackingDict[node]): 
                    # Get rank of each g
                    rank = []
                    for g in gaugedOutlets:
                        upList = BackTrackingDict.get(g)
                        if upList is None:
                            rank.append(0)
                        else:
                            rank.append(len(upList))
                    gmax = gaugedOutlets[rank.index(max(rank))]
                    # Update TempDict: delete node and update others.
                    gaugedOutlets.remove(gmax)
                    TempDict.pop(gmax, None)    # if 'key' in my_dict: del my_dict['key']
                    for g in gaugedOutlets:
                        TempDict[g] = node
                    # Call trackBack with gmax and TempDict (recursive)
                    SimSeq, TempDict, BackTrackingDict = trackBack(gmax, SimSeq, BackTrackingDict, TempDict)
                    
                elif len(gaugedOutlets) == 1 and len(BackTrackingDict[node]) == 1:
                    SimSeq, TempDict, BackTrackingDict = trackBack(gaugedOutlets[0], SimSeq, BackTrackingDict, TempDict)
                    TempDict.pop(gaugedOutlets[0], None)
                    # Search TempDict and jump backward to add other tributary.
                    # reverse TempDict
                    rTempDict = {}
                    for g in TempDict:
                        if rTempDict.get(TempDict[g]) is None:
                            rTempDict[TempDict[g]] = [g]
                        else:
                            rTempDict[TempDict[g]].append(g)
                    if rTempDict != {}:
                        # Replace BackTrackingDict
                        for g in rTempDict:
                            BackTrackingDict[g] = rTempDict[g]
                        ToNode = SimSeq[min([SimSeq.index(i) for i in rTempDict])]
                        SimSeq, TempDict, BackTrackingDict = trackBack(ToNode, SimSeq, BackTrackingDict, {}, False)
            else:
                for up in BackTrackingDict[node]:
                    SimSeq, TempDict, BackTrackingDict = trackBack(up, SimSeq, BackTrackingDict, TempDict)    
        return SimSeq, TempDict, BackTrackingDict 
    SimSeq, TempDict, BackTrackingDict = trackBack(Node, SimSeq, BackTrackingDict)
    return SimSeq
#%%

BackTrackingDict = {"G":["g7","g8","R1"],
                    "g7":["R1"],
                    "R1":["V1"],
                    "V1":["g1","g2","g3","g4","g5","g6"],
                    "g6":["g1","g2","g3","g4","g5"],
                    "g3":["g1","g2"],
                    "g2":["g1"],
                    "g5":["g4"],
                    "g8":["g7","R1"]}
GaugedOutlets = ["g1","g2","g3","g4","g5","g6","g7","g8","G"]
Node = "G"


SimSeq = formSimSeq(Node, BackTrackingDict, GaugedOutlets)




