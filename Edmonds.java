import java.util.ArrayList;

public class Edmonds {

   private ArrayList<Node> cycle;

   public AdjacencyList getMinBranching(Node root, AdjacencyList list){
       AdjacencyList reverse = list.getReversedList();
       // remove all edges entering the root
       if(reverse.getAdjacent(root) != null){
           reverse.getAdjacent(root).clear();
       }
       AdjacencyList outEdges = new AdjacencyList();
       // for each node, select the edge entering it with smallest weight
       for(Node n : reverse.getSourceNodeSet()){
           ArrayList<Edge> inEdges = reverse.getAdjacent(n);
           if(inEdges.isEmpty()) continue;
           Edge min = inEdges.get(0);
           for(Edge e : inEdges){
               if(e.weight < min.weight){
                   min = e;
               }
           }
           outEdges.addEdge(min.to, min.from, min.weight);
       }

       // detect cycles
       ArrayList<ArrayList<Node>> cycles = new ArrayList<ArrayList<Node>>();
       cycle = new ArrayList<Node>();
       getCycle(root, outEdges);
       cycles.add(cycle);
       for(Node n : outEdges.getSourceNodeSet()){
           if(!n.visited){
               cycle = new ArrayList<Node>();
               getCycle(n, outEdges);
               cycles.add(cycle);
           }
       }

       // for each cycle formed, modify the path to merge it into another part of the graph
       AdjacencyList outEdgesReverse = outEdges.getReversedList();

       for(ArrayList<Node> x : cycles){
           if(x.contains(root)) continue;
           mergeCycles(x, list, reverse, outEdges, outEdgesReverse);
       }
       return outEdges;
   }

   private void mergeCycles(ArrayList<Node> cycle, AdjacencyList list, AdjacencyList reverse, AdjacencyList outEdges, AdjacencyList outEdgesReverse){
       ArrayList<Edge> cycleAllInEdges = new ArrayList<Edge>();
       Edge minInternalEdge = null;
       // find the minimum internal edge weight
       for(Node n : cycle){
           for(Edge e : reverse.getAdjacent(n)){
               if(cycle.contains(e.to)){
                   if(minInternalEdge == null || minInternalEdge.weight > e.weight){
                       minInternalEdge = e;
                       continue;
                   }
               }else{
                   cycleAllInEdges.add(e);
               }
           }
       }
       // find the incoming edge with minimum modified cost
       Edge minExternalEdge = null;
       int minModifiedWeight = 0;
       for(Edge e : cycleAllInEdges){
           int w = e.weight - (outEdgesReverse.getAdjacent(e.from).get(0).weight - minInternalEdge.weight);
           if(minExternalEdge == null || minModifiedWeight > w){
               minExternalEdge = e;
               minModifiedWeight = w;
           }
       }
       // add the incoming edge and remove the inner-circuit incoming edge
       Edge removing = outEdgesReverse.getAdjacent(minExternalEdge.from).get(0);
       outEdgesReverse.getAdjacent(minExternalEdge.from).clear();
       outEdgesReverse.addEdge(minExternalEdge.to, minExternalEdge.from, minExternalEdge.weight);
       ArrayList<Edge> adj = outEdges.getAdjacent(removing.to);
       for(Iterator<Edge> i = adj.iterator(); i.hasNext(); ){
           if(i.next().to == removing.from){
               i.remove();
               break;
           }
       }
       outEdges.addEdge(minExternalEdge.to, minExternalEdge.from, minExternalEdge.weight);
   }

   private void getCycle(Node n, AdjacencyList outEdges){
       n.visited = true;
       cycle.add(n);
       if(outEdges.getAdjacent(n) == null) return;
       for(Edge e : outEdges.getAdjacent(n)){
           if(!e.to.visited){
               getCycle(e.to, outEdges);
           }
       }
   }
}

import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Collection;

public class AdjacencyList {

   private Map<Node, ArrayList<Edge>> adjacencies = new HashMap<Node, ArrayList<Edge>>();

   public void addEdge(Node source, Node target, int weight){
       ArrayList<Edge> list;
       if(!adjacencies.containsKey(source)){
           list = new ArrayList<Edge>();
           adjacencies.put(source, list);
       }else{
           list = adjacencies.get(source);
       }
       list.add(new Edge(source, target, weight));
   }

   public ArrayList<Edge> getAdjacent(Node source){
       return adjacencies.get(source);
   }

   public void 	(Edge e){
       adjacencies.get(e.from).remove(e);
       addEdge(e.to, e.from, e.weight);
   }

   public void reverseGraph(){
       adjacencies = getReversedList().adjacencies;
   }

   public AdjacencyList getReversedList(){
       AdjacencyList newlist = new AdjacencyList();
       for(ArrayList<Edge> edges : adjacencies.values()){
           for(Edge e : edges){
               newlist.addEdge(e.to, e.from, e.weight);
           }
       }
       return newlist;
   }

   public Set<Node> getSourceNodeSet(){
       return adjacencies.keySet();
   }

   public Collection<Edge> getAllEdges(){
       ArrayList<Edge> edges = new ArrayList<Edge>();
       for(List<Edge> e : adjacencies.values()){
           edges.addAll(e);
       }
       return edges;
   }
}

public class Edge implements Comparable<Edge> {
   
   final Node from, to;
   final int weight;
   
   public Edge(final Node argFrom, final Node argTo, final int argWeight){
       from = argFrom;
       to = argTo;
       weight = argWeight;
   }
   
   public int compareTo(final Edge argEdge){
       return weight - argEdge.weight;
   }
}

public class Node implements Comparable<Node> {
   
   final int name;
   boolean visited = false;   // used for Kosaraju's algorithm and Edmonds's algorithm
   int lowlink = -1;          // used for Tarjan's algorithm
   int index = -1;            // used for Tarjan's algorithm
   
   public Node(final int argName) {
       name = argName;
   }
   
   public int compareTo(final Node argNode) {
       return argNode == this ? 0 : -1;
   }
}
