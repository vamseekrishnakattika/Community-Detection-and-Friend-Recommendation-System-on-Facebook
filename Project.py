# Databricks notebook source
# Part 1 - Fetching input fb data and creating a graph
from igraph import *
from pyspark.sql import Row
inputFile = sc.textFile("/FileStore/tables/facebook_combined.txt")

#retrieve vertex and edges from the data set
def getVertex(entry, i):
  row = entry.split(' ')
  return int(row[i])

def getEdge(entry):
  row = entry.split(' ')
  return int(row[0]),int(row[1])

#create vertex and edges RDD
vertext1RDD = inputFile.map(lambda x: getVertex(x,0)).cache().distinct()
vertext2RDD = inputFile.map(lambda x: getVertex(x,1)).cache().distinct()
vertexRDD = vertext1RDD.union(vertext2RDD).distinct()
vertexCount = vertexRDD.count()
edgesRDD = inputFile.map(getEdge).cache()
edgeCount = edgesRDD.count()
print 'Vertex count: %d' %vertexCount
print 'First 5 Vertices: %s' % vertexRDD.takeOrdered(5)
print ('Edge count: %d' %edgeCount)
print 'First 5 Edges: %s' % edgesRDD.take(5)

#Build igraph with vertices and edges from the dataset
vertices = vertexRDD.collect()
edges = edgesRDD.collect()
graph = Graph(vertex_attrs={"label":vertices}, edges=edges, directed=False)


#Part 2 Dataset analysis and cleaning on the built graph 
def getMean(param):
  return reduce(lambda x, y: x + y, param) / len(param)

print 'Graph connected: ',graph.is_connected(mode=STRONG)
networkDiameter = graph.diameter(directed=False, unconn=True, weights=None)
networkBetweeness = graph.betweenness(vertices=None, directed=False, cutoff=None, weights=None, nobigint=True)
meanNetworkBetweeness= getMean(networkBetweeness)
networkDegrees = graph.degree()
meanNetworkDegree= getMean(networkDegrees)
print 'Network Diameter: %d' %networkDiameter
print 'Mean Network Betweenness : ', meanNetworkBetweeness
print 'Mean Network Degree : %d' %meanNetworkDegree

from operator import add
networkDegreesRDD = sc.parallelize(networkDegrees)
counts = networkDegreesRDD.map(lambda x: (x, 1)).reduceByKey(add)
output = counts.take(10)
print 'Degree, Count (First 10 sample)'
for (degree, count) in output:
  print("%s , %i" % (degree, count))

#identify insignificant nodes and eliminating them
islands = []
islandDegrees = []
for v in vertices:
  friendsList = graph.neighbors(vertex=v, mode=ALL)
  friendCount = len(friendsList);
  if (friendCount < 2):
    islands.append(v)
    islandDegrees.append(graph.degree(v))
print 'Islands : ' , set(islands)
print 'IslandDegrees : ', set(islandDegrees)
print 'Deleting the islands from the graph'
graph.delete_vertices(islands)

# Finding new Edges, vertices and identify most significant nodes, >300 friend count 
newVertices = []
newEdges = []
coreNodes = []
coreDegrees = []
for v in graph.vs:
    newVertices.append(v["label"])
    vertexDegree = graph.degree(v)
    if(vertexDegree > 300):
      coreNodes.append(v.index)
      coreDegrees.append(vertexDegree)
for e in graph.es:
    newEdges.append(e.tuple)
meanCoreDegree = getMean(coreDegrees)
print 'New vertex count after deleting islands: ' ,len(set(newVertices))    
print 'New Edge count after deleting islands: ' ,len(set(newEdges))    
print 'Core Nodes: ', set(coreNodes)
print 'Mean Core Degree: ', meanCoreDegree

#sub graph focussing on a core node which was identified as significant node
coreNode = 0
coreNodeFriends = graph.neighbors(vertex=coreNode, mode=ALL)
CoreNodeFriendsOfFriends = graph.neighborhood(vertices=coreNode, order=2, mode=ALL)
print 'Core Node ', coreNode, ' friends count: ', len(coreNodeFriends)
print 'Core Node ', coreNode, ' friends of friends count: ', len(CoreNodeFriendsOfFriends)

coreNodeFriends.append(coreNode)
CoreNodeGraph = graph.subgraph(coreNodeFriends, implementation = "auto")

#identify cliques on the subgraph
coreNodeCliques = CoreNodeGraph.maximal_cliques(min =2 , max =10)
# print coreNodeCliques

# Part 3 -  Finding tight communities using fast fast greedy cluster 
fastGreedy = CoreNodeGraph.community_fastgreedy()
fastGreedycluster = fastGreedy.as_clustering()
print fastGreedycluster.modularity
print fastGreedycluster

#community detection with centrality based approach using edge betweeness
communities = CoreNodeGraph.community_edge_betweenness(directed=False)
edgeBtwcluster = communities.as_clustering()
print edgeBtwcluster.modularity
print edgeBtwcluster

#community detection using walk trap algorithm
walkTrap = CoreNodeGraph.community_walktrap() 
walkTrapcluster = walkTrap.as_clustering()
print walkTrapcluster.modularity
print walkTrapcluster

#community detection using info map algorithm
infoMap = CoreNodeGraph.community_infomap()
print infoMap.modularity
print infoMap.as_cover()

# Part 4 preparing mutual friends count
# Map Phase - finding mutual friends count from Graph
def returnTuple(entry):
  row = entry.split(' ')
  return int(row[0]),int(row[1]),-1

egoRDD = inputFile.map(returnTuple)
mutualFriends=[]
def generate(x):
  toNodes=[]
  for row in egoRDD.collect():
    if row[0]==x:
      toNodes.append(row[1])
  for i in range(0,len(toNodes)-1):
    mutualFriends.append([toNodes[i],toNodes[i+1],1])
# Reduce phase
prev = -1
for row in egoRDD.collect():
  if row[0]!=prev:
    generate(row[0])
  prev=row[0]
  
def predict(entry):
  return (entry[0],entry[1]),entry[2]
  
mutualFriends_stg1 = sc.parallelize(mutualFriends)
mutualFriendsRDD = mutualFriends_stg1.map(predict)
sortedMutualFriendRDD = mutualFriendsRDD.reduceByKey(lambda a,b:a+b).sortBy(lambda a: -a[1])

print 'Sorted Mutual friends list top 10', sortedMutualFriendRDD.take(10)


# Part 5 - Friend Recommendation
#Select one user for whom friend suggestion has to be made
userId=115
#Filter mutual friend list for the selected user
suggestions_1 = sortedMutualFriendRDD.filter(lambda x:x[0][0]==userId).map(lambda x:(x[0][1],x[1]))
suggestions_2 = sortedMutualFriendRDD.filter(lambda x:x[0][1]==userId).map(lambda x:(x[0][0],x[1]))
suggestions = suggestions_1.union(suggestions_2)
suggestions_sorted = suggestions.sortBy(lambda x:-x[1])
suggestions_RDD = suggestions_sorted.map(lambda x:x[0])
# print suggestions_RDD.collect()

#Get all friends of user given and removing them from suggestions
friends_1= egoRDD.filter(lambda x:x[0]==userId).map(lambda x:x[1])
friends_2= egoRDD.filter(lambda x:x[1]==userId).map(lambda x:x[0])
friends = friends_1.union(friends_2)
# print friends.collect()
already_friends = suggestions_RDD.intersection(friends)
finalSuggestions = suggestions_RDD.subtract(already_friends)
print 'Friend suggestions using mutual friend count for ', userId ,' is ' ,finalSuggestions.collect()

#Narrowing down the suggestions based on tight communities formed using fast greedy method
suggestion_list = finalSuggestions.collect()
community_based_suggestion=[]
for cluster_index in range(8):
  for member in suggestion_list:
    if member in fastGreedycluster[cluster_index] and userId in fastGreedycluster[cluster_index]:
      community_based_suggestion.append(member)

print 'Friend Suggestions using tight community cluster for ', userId ,' is ' , community_based_suggestion