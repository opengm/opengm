import numpy

def visualizeGm(gm,plotUnaries=True,plotFunctions=False,plotNonShared=False,layout='neato',iterations=1000,show=True,relNodeSize=1.0):
    """
    visualize a graphical model with matplotlib , networkx and graphviz

    Keyword arguments:
      - plotUnaries -- plot unaries (default: ``True``)
      - plotFunctions -- plot functions (default: ``False``)
      - plotNonShared -- plot non shared functions (default: ``False`` )
      - layout -- used layout to generate node positions: (default: ``\'neato\'`` )
         - ``\'spring\'`` :
            "spring model'' layout which should be used only for very small graphical models (  ``|`` V ``|`` + ``|`` F ``|`` <  50 )
         - ``\'neato\'`` (needs also python graphviz) :
            "spring model'' layouts.  This is the default tool to use if the graph is not too large (  ``|`` V ``|`` + ``|`` F ``|`` <  100 )
            and you don't know anything else about it. 
            Neato attempts to minimize a global energy function, 
            which is equivalent to statistical multi-dimensional scaling.  
         - ``\'fdp\'`` (needs also python graphviz) :
            "spring model'' layouts similar to those of neato, 
            but does this by reducing forces rather than working with energy.
         - ``\'sfdp\'`` (needs also python graphviz) :
            multiscale version of fdp for the layout of large graphs.
         - ``\'twopi\'`` (needs also python graphviz) :
            radial layouts, after Graham Wills 97. 
            Nodes are placed on concentric circles depending their distance from a given root node.
         - ``\'circo\'`` (needs also python graphviz) :
            circular layout, after Six and Tollis 99, 
            Kauffman and Wiese 02. This is suitable for certain diagrams of multiple cyclic structures, 
            such as certain telecommunications networks.  
        - show : show the graph or supress showing  (default=True)
        - relNodeSize : relative size of the notes must be between 0 and 1.0 .
    """
    try:
        import networkx as nx
    except:
        raise TypeError("\"import networkx as nx\" failed")    
    try:
        import matplotlib.pyplot as plt
    except:
        raise TypeError("\" import matplotlib.pyplot as plt\" failed")   
    try:
        from networkx import graphviz_layout
    except:
        raise TypeError("\"from networkx import graphviz_layout\" failed")    
    # set up networkx graph
    G=nx.Graph()
    # node lists
    varNodeList=[x for x in xrange(gm.numberOfVariables)]
    factorNodeList=[]#[x for x in xrange(gm.numberOfVariables,gm.numberOfVariables+gm.numberOfFactors)]
    functionNodeList=[]
    # edge list
    factorVarEdgeList=[]
    factorFunctionEdgeList=[]
    # labels
    varLabels=dict( )
    factorLabels=dict( )
    functionLabels=dict( )
    # factor index to node index
    fi_ni=dict()
    for vi in xrange(gm.numberOfVariables):
        G.add_node(vi)
        varLabels[vi]=str(vi)
    # starting node index 
    nodeIndex=gm.numberOfVariables
    # for factor function adj.
    function_dict=dict()
    for fi in xrange(gm.numberOfFactors):
        factor=gm[fi]
        # unarie factor
        if(factor.numberOfVariables==1 and plotUnaries==True):
            # add node
            G.add_node(fi)
            factorNodeList.append(nodeIndex)
            factorLabels[nodeIndex]=str(fi)
            fi_ni[fi]=nodeIndex
            # add edge
            G.add_edge(nodeIndex,factor.variableIndices[0],weight=2)
            factorVarEdgeList.append( (nodeIndex,factor.variableIndices[0]  ) )
            # increment node index
            nodeIndex+=1    
        # high order factor
        elif(factor.numberOfVariables>1):
            # add node
            G.add_node(fi)
            factorNodeList.append(nodeIndex)
            factorLabels[nodeIndex]=str(fi)
            fi_ni[fi]=nodeIndex
            # add edges
            for vi in factor.variableIndices:
                G.add_edge(nodeIndex,vi,weight=0.5)
                factorVarEdgeList.append( (nodeIndex,vi) )
            # increment node index
            nodeIndex+=1
        # set up function factor adj. 
        if(plotFunctions):
            if(factor.numberOfVariables>1 or plotUnaries==True):
                fid=(factor.functionIndex,factor.functionType)
                if bool(fid in function_dict ) == False:
                    factorSet=set()
                else :
                    factorSet = function_dict[fid]
                factorSet.add(fi)
                function_dict[fid]=factorSet

    for fid in function_dict.keys():
        
        factorSet = function_dict[fid]
        if(len(factorSet)>1 or plotNonShared==True):
            # add node
            G.add_node(fid)
            functionNodeList.append(nodeIndex)
            functionLabels[nodeIndex]=str(fid[0])+'-'+str(fid[1])
            # node weight (for better results)
            weight=0.001
            if(len(factorSet)==1):
                weight=1.5
            # add edges
            for fi in factorSet:

                factorNodeIndex=fi_ni[fi]
                G.add_edge(factorNodeIndex,nodeIndex,weight=weight)
                factorFunctionEdgeList.append((factorNodeIndex,nodeIndex))
            # increment node index
            nodeIndex+=1

    print "get node position..."
    if  layout=='spring':
        pos=nx.spring_layout(G,dim=2,weight='weight',iterations=iterations) 
    elif layout=='dot':
        pos=graphviz_layout(G,prog='dot')
    elif layout=='neato':
        pos=graphviz_layout(G,prog='neato')
    elif layout=='fdp':
        pos=graphviz_layout(G,prog='fdp')
    elif layout=='sfdp':
        pos=graphviz_layout(G,prog='sfdp')
    elif layout=='twopi':
        pos=graphviz_layout(G,prog='twopi')
    elif layout=='circo':
        pos=graphviz_layout(G,prog='circo')
    else:
        try:
            pos=graphviz_layout(G,prog=layout)
        except :
            raise NameError("unknown layout : "+layout)
    from networkx import graphviz_layout
    print "....done "



    nodeSize=200.0*relNodeSize
    fontSize=12.0*relNodeSize

    if plotFunctions:
        nx.draw_networkx(G,pos,node_size=nodeSize*2.0,nodelist=functionNodeList,fontSize=fontSize,withLabels=False,labels=functionLabels,node_color='gray',font_color='k',edgelist=[],node_shape='d',font_size=fontSize*0.75)
        nx.draw_networkx_edges(G,pos,alpha=0.9,width=1*relNodeSize,edgelist=factorFunctionEdgeList,style='dotted')

    nx.draw_networkx(G,pos,node_size=nodeSize,nodelist=factorNodeList,fontSize=fontSize,withLabels=False,labels=factorLabels,node_color='k',font_color='w',edgelist=[],node_shape='s',font_size=fontSize)
    nx.draw_networkx(G,pos,node_size=nodeSize,nodelist=varNodeList,fontSize=fontSize,withLabels=False,labels=varLabels,node_color='w',font_color='k',edgelist=[],node_shape='o',font_size=fontSize)

    nx.draw_networkx_edges(G,pos,alpha=0.6,width=1*relNodeSize,edgelist=factorVarEdgeList,style='solid')
    plt.subplots_adjust()
    plt.axis('off')
    #plt.savefig("gm.png") # save as png
    if show:
        plt.show() # display