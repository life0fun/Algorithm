/*
 * package com.haijin.facebull
 * This is traveling salesman problem.
 * find a path that connect all the nodes, hamilton cycle. 
 * hence there exists a path between any two nodes.
 *
 * you can do it with Edmond's algorithm, finding the MST in a directed graph.
 * A MST(n-1 edges, n node) rooted at X is just one MIN in-bound edge for node other than root.
 * http://www.ce.rit.edu/~sjyeec/dmst.html
 * To find min # of edges in scss, using cycle contracting algorithm. Approximating MEG.
 * DFS, find cycle of nodes(visited,cisiting,notvisited) and contracted cycles. Eventually get a ham cycle.
 * 
 * if there is a cycle, break the cycle by replacing one edge from inside the cycle with one 
 * external inbound cycle with min extra delta. 
 *     Min(cost(e) - (cost(i)-min(cost(all internal edges))
 *     
 *     http://www.facebook.com/topic.php?uid=15325934266&topic=4990#topic_top
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeMap;

public class facebull {
	private static final boolean DEBUG = true;
	public static void log(String... args) {
		if(DEBUG){
			for(String s : args){
				System.out.println(s);
			}
		}
	}
	
	public static class Node {
		protected final String mName;
		protected int mColor;  // white clean, gray, visiting, black, visited
		
		public Node(final String name){
			mName = name;
			mColor = 0; 
		}
		public String toString(){ return mName;}
		
		@Override  // override equal and hashcode together
        public boolean equals(Object o) {
            if ( this == o ) return true;
            if (! (o instanceof Node)) return false;
            return mName.equals(((Node)o).mName);
		}
		@Override
        public int hashCode() {
            return mName.hashCode();
        }
	}
	
	public static class Edge implements Comparable<Edge> {
		protected final Node mSrc, mDst;
		protected final String mMachine;
		protected int mCost;
		protected int mOriginCost;
		
		public Edge(Node src, Node dst, final int cost, final int origcost, final String machine){
			mSrc = src;
			mDst = dst;
			mMachine = machine;
			mCost = cost;
			mOriginCost = origcost;
		}
		
		public Edge(Edge e){
			mSrc = e.mSrc; mDst = e.mDst; mMachine = e.mMachine;mCost = e.mCost; mOriginCost = e.mOriginCost;
		}
		
		public String toString(){
			//return mSrc+"(" + mSrc.mColor + ") ->" + mDst+"(" + mDst.mColor + ")"  + "(" + mMachine + "=" + mCost + ")";
			return mSrc+ "->" + mDst  + "(" + mMachine + "=" + mCost + ":" + mOriginCost + ")";
		}
		
		/**
		 * return 0 if the values are the same.
		 */
		public int compareTo(final Edge e){
			if(this.mCost == e.mCost && this.mSrc.toString().equals(e.mSrc.toString()) && this.mDst.toString().equals(e.mDst.toString())){
				return 0;
			}
			return 1;   
		}
		
		@Override  // override equal and hashcode together
        public boolean equals(Object o) {
            if ( this == o ) return true;
            if (! (o instanceof Edge)) return false;
            return 0 == compareTo((Edge)o);
		}
		@Override
        public int hashCode() {
            return mSrc.hashCode() * mDst.hashCode();
        }
	}
	
	/* the value comparator for hashmap */
	class ValueComparator implements Comparator {
		  Map<Edge, Integer> base;
		  public ValueComparator(Map<Edge, Integer> base) {this.base = base; }

		  public int compare(Object a, Object b) {
		    if((Integer)base.get(a) < (Integer)base.get(b)) { return -1;} 
		    else if((Integer)base.get(a) == (Integer)base.get(b)) {return 0;} 
		    else { return 1;}
		  }			
	}
	
	/**
	 * A direct MST consists of maps of edges with src node as key, as well as dst node as key.
	 * Each MST contains 4 maps, src, dst edges maps, and the MST of src/dst.
	 * 
	 * for each recursion of each node as root, one instance of this is created and the result MST in stored. 
	 */
	public static final class DirectedMST {
				
		// A MST has four Maps for a graph
		protected Map<Node, ArrayList<Edge>> mSrcNodeEdgeMap = new HashMap<Node, ArrayList<Edge>>();  // src node as key, out-bound edges, sorted by cost.
		protected Map<Node, ArrayList<Edge>> mDstNodeEdgeMap = new HashMap<Node, ArrayList<Edge>>();  // dst node as key, in-bound edges
		protected Map<Node, ArrayList<Edge>> mSrcNodeMSTEdgeMap = new HashMap<Node, ArrayList<Edge>>();  // src node as key, out-bound edges
		protected Map<Node, ArrayList<Edge>> mDstNodeMSTEdgeMap = new HashMap<Node, ArrayList<Edge>>();  // dst node as key, in-bound edges
		
		
		public DirectedMST() { }
		
		// static factory method
		public static DirectedMST deepCopy(final DirectedMST mst){
			DirectedMST newmst = new DirectedMST();
			
			for(ArrayList<Edge> edges : mst.mSrcNodeEdgeMap.values()){
				for(Edge e : edges){
					//newmst.insertEdge(e.mSrc, e, newmst.mSrcNodeEdgeMap);
					newmst.insertEdge(e.mSrc, e.mSrc, e.mDst, e.mCost, e.mOriginCost, e.mMachine, newmst.mSrcNodeEdgeMap);
				}
			}
			for(ArrayList<Edge> edges : mst.mDstNodeEdgeMap.values()){
				for(Edge e : edges){
					//newmst.insertEdge( e.mDst, e, newmst.mDstNodeEdgeMap);
					newmst.insertEdge(e.mDst, e.mSrc, e.mDst, e.mCost, e.mOriginCost, e.mMachine, newmst.mDstNodeEdgeMap);
				}
			}
			newmst.mSrcNodeMSTEdgeMap.clear();
			newmst.mDstNodeMSTEdgeMap.clear();
			return newmst;
		}
		
		/**
		 * insert one edge into the map, with raw src/dst node and cost
		 */
		public void insertEdge(Node key, Node src, Node dst, final int cost, final int origcost, final String machine, Map<Node, ArrayList<Edge>> map){
			ArrayList<Edge> l = null;
			Edge newedge = new Edge(src, dst, cost, origcost, machine);
			l = map.get(key);  // will return null upon KeyError
			if ( null == l){
				l = new ArrayList<Edge>();
				map.put(key, l);
			}
			l.add(newedge);
		}
		
		// insert the min edge into the map
		public void insertEdge(Node key, Edge minedge, Map<Node, ArrayList<Edge>> map){
			ArrayList<Edge> l = null;
			if(!map.containsKey(key)){
				l = new ArrayList<Edge>();
				map.put(key,l);  // at the time here, l is not populated.
			}else{
				l = map.get(key);
			}
			l.add(minedge);
		}
		
		/* after a must buy edge is decided(all MSTs confirmed), update the edge cost to 0*/
		public void resetEdgeCost(Edge e){
			for(Edge edge : mSrcNodeEdgeMap.get(e.mSrc)){
				if(e.compareTo(edge) == 0){ edge.mCost = 0; log("reset edge:"+edge.toString()); break;}
			}
			for(Edge edge : mDstNodeEdgeMap.get(e.mDst)){
				if(e.compareTo(edge) == 0){ edge.mCost = 0; log("reset edge:"+edge.toString()); break;}
			}
		}
		
		/**
		 * find the min edge for a node
		 */
		public Edge findNodeMinEdge(final Node e, final Map<Node, ArrayList<Edge>> map){
			int mincost = Integer.MAX_VALUE;
			ArrayList<Edge> edges = null;
			Edge minedge = null;
			edges = map.get(e);
			
			for(Edge arch : edges){
				if(arch.mCost < mincost){
					mincost = arch.mCost;
					minedge = arch;
					log(" node:" + e.toString() + " minedge: "+minedge.toString());
				}
			}
			return minedge;
		}
		
		/**
		 * A cycle with nodes in the set, find the min edge within a set that forms a cycle
		 */
		public Edge findSetMinEdge(final Set<Node> set, HashSet<Edge> allExternalInEdges){
			Edge minEdge = null;
			int minCost = Integer.MAX_VALUE;
			
			log("findSetMinEdge: " + set.toString());
			dumpMap(mDstNodeMSTEdgeMap);
			for(Node n : set){
				// only one in-bound edge to a node in MST, and the set that contains the root already being filtered out.
				Edge e = mDstNodeMSTEdgeMap.get(n).get(0);  // in-bound MST edge of a node,
				log("findSetMinEdge: node: " + n.toString() + " edges:" + mDstNodeMSTEdgeMap.get(n) + " edge:" + e.toString());
				if(minEdge == null || e.mCost < minCost){
					minEdge = e;
					minCost = e.mCost;
					log("minCost:" + minCost + " minEdge:" + minEdge.toString());
				}
				for(Edge edge : mDstNodeEdgeMap.get(n)){  // all the in bound edge to this node add to external in edge for this MST cycle.
					if(!set.contains(edge.mSrc) && edge.compareTo(e) != 0){  // edges that are not min-edge, and are from external
						allExternalInEdges.add(edge);
					}
				}
			}
			return minEdge;
		}
		
		/**
		 * DFS traverse MST(edges). return a set of node that is reachable from the root.
		 * The same as Min-cut BFS, one call will recursively find one cycle from the root.
		 */
		public Set<Node> DFSMST(Node root, final Map<Node, ArrayList<Edge>> srckeymap, Set<Node> cycle){
			cycle.add(root);
			root.mColor = 1;
			// node might not have out-bound edge, be careful.
			if(srckeymap.get(root) != null){
				for(Edge e : srckeymap.get(root)){ // all this node's out edgs
					//log("traversing edge: " + e.toString());
					cycle.add(e.mDst);   // reachable, the two node belongs in one cycle.
					if(e.mDst.mColor == 0){  // DFS if the node at the other end not visited, the map is InEdgeMap!!!
						DFSMST(e.mDst, srckeymap, cycle);
					}
				}
			}
			return cycle;
		}
		
		@Deprecated  // DFS find cycle in a directed Graph
		public Node DFS(Node curnode, Map<Node, ArrayList<Edge>> srckeymap, ArrayList<Node> path){
			curnode.mColor = 1;
			path.add(curnode);  // curnode has next, add current node to path, recursion.
			if(srckeymap.get(curnode) != null){
				for(Edge e : srckeymap.get(curnode)){  // for each next
					if(path.contains(e.mDst)){
						path.add(e.mDst);
						log("cycle detected and started:" + path.toString());
						return e.mDst;  
					}else{
						Node cyclenode = DFS(e.mDst, srckeymap, path);  //recursion on next
						if(cyclenode != null){
							return cyclenode;
						}
					}
				}
			}
			path.remove(curnode);  // remove curnode from the path after exploring all next.
			return null;  // curnode does not have any next, nnull cycle node.
		}
		
		/**
		 * detect any cycle in MST, the map is MST src key map and has n nodes, and n-1 edges.
		 * result stored in the maps of this class(DirectedMST).
		 */
		public ArrayList<Set<Node>> detectAndConsolidateCycles(Node root, Map<Node, ArrayList<Edge>> srckeymap){
			boolean cyclemerged = false;
			ArrayList<Set<Node>> cycles = new ArrayList<Set<Node>>(); // top level all cycles.
			ArrayList<Node> path = new ArrayList<Node>();

			// this loop will find all the cycles in MST
			for( Node n : srckeymap.keySet()){  // all the node in MST that has out-bound edges.
				//if(n.mColor == 1) continue;
				//path.clear();
				//Node cyclenode = DFS(n, srckeymap, path);
				//if(cyclenode != null){ log("cycle:" + path.toString()); }
				if(n.mColor == 0){
					cyclemerged = false;
					Set<Node> cycle = new HashSet<Node>();    // create a new set object
					DFSMST(n, srckeymap, cycle); // each call will DFS and populate to cycle all the reachable node rooted at this node.
					for(Set<Node> s : cycles){   // is this found cycle part of any existing cycle ?
						log("merge cycles: current cycle:" + cycle.toString() + " Existing: cycle:" + s.toString());
						if(!Collections.disjoint(s, cycle)){
							log(" merge cur cycle : "+cycle.toString() + " => to " + s.toString());
							s.addAll(cycle);
							cyclemerged = true;
							break;  // find this cycle connect to an existing cycle, no need to loop more.
						}
					}
					if(!cyclemerged){    // the cycle rooted at this node not belong to any existing cycle.
						cycles.add(cycle);
						log("stand alone new cycle rooted at :" + n.mName + " :: " + cycle.toString());
					}
				}
			}
			for(Set<Node> s : cycles){ log("All cycles: one is: " + s.toString()); }  // take a log
			
			// loop thru all the cycles, merge each of them into one big connected cycle.
			// for each cycle, merge it to cycle that contains root.
			for(int i=0;i<cycles.size();i++){   // do not use fail-fast iterator due to concurrentmodificationException
				Set<Node> s = cycles.get(i);
				log("merge cycle :" + s.toString());
				if(s.contains(root)){
					continue;  // root does not have any incoming edges, skip it!!!
				}
				mergeDisjointSet(s, cycles);    // repeat for each standalone cycle.
			}
			return cycles;
		}
			
		/**
		 * merge one disjoint set to the set where root sit, and forms the MST. the set has n nodes and n-1 edge.
		 * break the cycle by replacing its min internal edge with min external edge.
		 */
		public void mergeDisjointSet(Set<Node> set, ArrayList<Set<Node>> cycles){
			int adjustCost = 0;
			Edge minEdgeFromOut = null;
			int minCost = Integer.MAX_VALUE;
			HashSet<Edge> allExternalInEdges = new HashSet<Edge>();
			Edge minEdgeInCycle = findSetMinEdge(set, allExternalInEdges);
			log("Set minedge :"+minEdgeInCycle + " Set:" + set.toString() + " AllExtInEdge:" + allExternalInEdges.toString());
            if(allExternalInEdges.size() == 0){   // no external edge into the set, done ?
                return;
            }
			// find out the min cost replacement edge that come from outside to this cycle.
			for(Edge e : allExternalInEdges){
				Node dstNode = e.mDst;
				// if take this ext edge and take out the Inter Edge entering into the node pointed by this ext edge, what is the min extra delta. 
				adjustCost = e.mCost - (mDstNodeMSTEdgeMap.get(dstNode).get(0).mCost - minEdgeInCycle.mCost);
				log("adjcost="+ adjustCost + " internal Entering into dstnode="+dstNode + " To dstnode extEdge: "+e.toString());
				if(minEdgeFromOut == null || adjustCost < minCost){
					minEdgeFromOut = e;
					minCost = adjustCost;
				}
			}
			log("Min in-bound External Edge:" + minEdgeFromOut.toString() + " mincost :" + minCost);
			
			// now found the min edge from outside, eliminate the cycle by removing the edge to the node inside the cycle
			Edge edgeToRemove = mDstNodeMSTEdgeMap.get(minEdgeFromOut.mDst).get(0);
			mDstNodeMSTEdgeMap.get(edgeToRemove.mDst).clear();
			log("contract cycle: replace edge:" + edgeToRemove.toString() + "by ext edge :" + minEdgeFromOut.toString());
			
			// a node in MST can only have one incoming edge, but might have multiple out-going edge
			// while modify a list while looping, must use iterator
			Iterator<Edge> it = mSrcNodeMSTEdgeMap.get(edgeToRemove.mSrc).iterator();  
			while(it.hasNext()){  // remove the out-going mst edge from the removed node
				if(it.next().compareTo(edgeToRemove) == 0){   // find the edge that choose to remove
					it.remove();   // remove the last ele returned by this iterator
					break;
				}
			}
			// now add minExtEdge back
			insertEdge(minEdgeFromOut.mSrc, minEdgeFromOut, mSrcNodeMSTEdgeMap);
			insertEdge(minEdgeFromOut.mDst, minEdgeFromOut, mDstNodeMSTEdgeMap);
			
			// finally, need to merge two cycles after adding edge from outside
			for(int i=0;i<cycles.size();i++){   // do not use fail-fast iterator due to concurrentmodificationException
				Set<Node> s = cycles.get(i);
				if(s.contains(minEdgeFromOut.mSrc)){
					set.addAll(s);  // merge ext cycle into current cycle.  cur cycle still in global cycles.
					cycles.remove(s);
					break;   // break out once done, no further looping.
				}
			}
		}
		
		/**
		 * populate node incoming edge map
		 */
		public void createMapWithDestNodeAsKey(Map<Node, ArrayList<Edge>> srckeymap, Map<Node, ArrayList<Edge>> dstkeymap){
			dstkeymap.clear();
			for(ArrayList<Edge> edges : srckeymap.values()){
				for(Edge e : edges){
					insertEdge(e.mDst, e, dstkeymap);   // just add this edge object into new collection.
				}
			}
		}
		
		public void dumpMSTMap(Node root){
			log(" - - - - - graph edges [] in-bound edge to root---------");
			for(Node key : mSrcNodeEdgeMap.keySet()){
				log("Node:"+key.toString()+" >>> " + mSrcNodeEdgeMap.get(key).toString());
			}
			for(Node key : mDstNodeEdgeMap.keySet()){
				log("Node:"+key.toString()+" <<< " + mDstNodeEdgeMap.get(key).toString());
			}
			log(" - - - - - - solution: root:  " + root + " ------------");
			for(Node key : mSrcNodeMSTEdgeMap.keySet()){
				log("Node:"+key.toString()+" >>> " + mSrcNodeMSTEdgeMap.get(key).toString());
			}
		}
		public void dumpMap(Map<Node, ArrayList<Edge>> map){
			log(" ----  dump map start -------");
			for(Node n : map.keySet()){
				for(Edge e : map.get(n))
					log("key:" + n.toString() + " edge:" + e.toString());
			}
			log(" ---- dump map end -------");
		}
	}
	
	/**
	 * constructor
	 */
	public facebull(){
		// mGraph is the gloabl map stores the init edges. deepcopy to each node's result maps when solving that node. 
		mGraph = new DirectedMST();
	}
	
	/**
	 * find MST for a single node start from root.
	 */
	public DirectedMST findDirectedMST(Node root, DirectedMST graph){
		log("--- start solution for node :" + root.toString() + " -----");
		// make a copy of in edge map, will populate the result MST.
		DirectedMST resultmst = DirectedMST.deepCopy(graph); 
		Edge minedge = null;
		
		//graph.dumpMap(resultmst.mDstNodeEdgeMap);
		
		// 1. discard arc entering into the root, if any. MST has n nodes, n-1 edges, so discard root's first.
		if(resultmst.mDstNodeEdgeMap.get(root) != null){
			resultmst.mDstNodeEdgeMap.get(root).clear();   // ArrayList<Edge> exists but is empty now.
		}
		// 2. select the min entering arc for each node, every node must have at least one entering arc as this is a connected graph.
		for(Node e : resultmst.mDstNodeEdgeMap.keySet()){
			minedge = resultmst.findNodeMinEdge(e, resultmst.mDstNodeEdgeMap);
			if(minedge != null){
				log(" --- Min edge into node : " + e.mName + " Edge: " + minedge.toString());
				resultmst.insertEdge(minedge.mSrc, minedge, resultmst.mSrcNodeMSTEdgeMap);
			}
		}
		// 3. we got the MST with n node, n-1 entering arc, populate the map with dst node as key(in-bound)
		resultmst.createMapWithDestNodeAsKey(resultmst.mSrcNodeMSTEdgeMap, resultmst.mDstNodeMSTEdgeMap);
		
		// 4. detect circle. result MST has n node and n-1 edges, traverse it, n disjointsets represent n cycles.
		cleanAllNodeColor();
		resultmst.detectAndConsolidateCycles(root, resultmst.mSrcNodeMSTEdgeMap);  // only inEdgeMap key is reversed. all other maps have src as key.
		
		resultmst.dumpMSTMap(root);
		log("--- End solution for node" + root.toString() + " -----");
		return resultmst;
	}
	
	public void addSolution(Map<Node, ArrayList<Edge>> map){
		for(ArrayList<Edge> edges : map.values()){
			for(Edge e : edges){
				if(mInBoundEdgeCountMap.containsKey(e)){
					mInBoundEdgeCountMap.put(e, (mInBoundEdgeCountMap.get(e).intValue()+1));
				}else{
					mInBoundEdgeCountMap.put(e, (1));
				}
				if(!mSolutionEdges.contains(e)){
					mSolutionEdges.add(e);
					mCurCost += e.mOriginCost;
					log("add solution edge:" + e.toString());
				}
			}
		}
	}
	
	/**
	 * connect src/dst of solution edges and converted to list of nodes start with cycle start, end with cycle end. 
	 * 2-3, 3-1 => [2,3,3,1]
	 */
	public void solutionEdgsCycles(){
		while(mSolutionEdges.size() > 0){
			Edge e = mSolutionEdges.remove(0);
			ArrayList<Edge> cycle = new ArrayList<Edge>();
			cycle.add(e);
			solutionEdgsDFS(cycle);
			for(Edge rm : cycle){
				mSolutionEdges.remove(rm);
			}
			mSolCycles.add(cycle);
		}
	}
	
	public void solutionEdgsDFS(ArrayList<Edge> cycle){
		Edge headedge = cycle.get(0);
		Edge tailedge = cycle.get(cycle.size()-1);
		for(Edge e : mSolutionEdges){
			if(e.mDst == headedge.mSrc){
				cycle.add(0, e);
				solutionEdgsDFS(cycle);
			}
			if(e.mSrc == tailedge.mDst){
				cycle.add(e);
				solutionEdgsDFS(cycle);
			}
		}
		return;
	}
	
	// assume buying this edge, what is the min set of solutions.
	public void onePassOneEdge(Edge e, DirectedMST graph){
		DirectedMST g = DirectedMST.deepCopy(graph);
		g.resetEdgeCost(e);
		findAllDirectedMST(g);
		log(" ==== start one recursion: edge:" + e.toString() + " ======");
		for(Edge se : mSolutionEdges){
			log(" Solution Edge: " + se.toString());
		}
		log(" ==== end one recursion: mCurCost=" + mCurCost + " optimal cost=" + mOptimalCost + " ======");
		
		if( mCurCost < mOptimalCost){
			mOptimalCost = mCurCost;
			mCurEdge = e;
			mOptimalSolution.clear();
			log("+++ one optimal solution:" + mOptimalCost + " +++++++"); 
			for(Edge se: mSolutionEdges) { 
				e.mCost = e.mOriginCost;   // mCost revert from setting of 0.
				mOptimalSolution.add(new Edge(se)); 
				log(se.toString());
			}
			log("+++ end one optimal solution +++++++"); 
		}
	}
	
	/**
	 * for the final solution, each node in the path can only have one arc in and one arc out.
	 * the union of MST is already a set of edges and wont have two same arcs. Just select distinctive nodes from both src and dst side.
	 * Optimization: after finding a MST of one node, continue on node that does not have an out-bound edge.
	 */
	public void consolidateSolution(DirectedMST graph){
		boolean morework = false;
		// first, get one solution edges
		mOptimalCost = findAllDirectedMST(graph);
		log(" ==== start one recursion of all solution Edges ======");
		for(Edge e : mSolutionEdges){
			log(" Solution Edge: " + e.toString());
			mOptimalSolution.add(e);
		}
		log(" ==== end one recursion: mCurCost=" + mCurCost + " optimal cost=" + mOptimalCost + " ======");

		// now test each edge in the solution and find optimal.
		List<Edge> optimaledge = new ArrayList<Edge>(mOptimalSolution);
		while(true){
			morework = false;
			for(Edge e : optimaledge){  
				if(e.mCost != 0){    // only work on edge that hasnt been reduced.
					onePassOneEdge(e, graph);
					//morework = true;
				}
			}
			if(morework)
				graph.resetEdgeCost(mCurEdge); // got to buy the cur optimal edge.
			else
				break;
		}
		return;
	}
	
	public void showSolution(List<Edge> sol){
		log(" ==== global optimal solution =========");
		int sum = 0;
		ArrayList<Integer> machines = new ArrayList<Integer>();
		for(Edge e : sol){
			log(e.toString());
			sum += e.mOriginCost;
			machines.add(new Integer(e.mMachine.substring(1)));
		}
		Collections.sort(machines);
		System.out.println(sum);
		for(int i=0;i<machines.size()-1;i++){
			System.out.print(machines.get(i));
			System.out.print(" ");
		}
		System.out.println(machines.get(machines.size()-1));
	}
	
	/**
	 * for each node in graph, find its mst
	 */
	public int findAllDirectedMST(DirectedMST graph) {
		mSolutionEdges.clear();  // updated inside add solution.
		mCurCost = 0;
		DirectedMST resultmst = null;
		for(Node n : graph.mSrcNodeEdgeMap.keySet()){
			//if(n.toString().equals("N6"))
			   resultmst = findDirectedMST(n, graph);
			   addSolution(resultmst.mSrcNodeMSTEdgeMap);  // the result MST stored in this instance's map.
		}
		return mCurCost;
	}
	
	// insert one edge read in from input file, each line is an edge
	public void insertEdge(String line){
		String[] arr = line.split("\\s+");  // tab or one more space
		Node src = mAllNodes.get(arr[1].trim());
		if(src ==  null){
			src = new Node(arr[1].trim());
			mAllNodes.put(arr[1].trim(), src);
		}
		Node dst = mAllNodes.get(arr[2]);
		if(dst ==  null){
			dst = new Node(arr[2].trim());
			mAllNodes.put(arr[2].trim(), dst);
		}
		mGraph.insertEdge(src, src, dst, Integer.parseInt(arr[3]), Integer.parseInt(arr[3]), arr[0], mGraph.mSrcNodeEdgeMap);
	}
	
	public void readInGraph(String file) throws IOException{
		in = new BufferedReader(new FileReader(file));
		String line;
		while ((line = in.readLine()) != null){
			log(line);
            if(line.length() > 0)
			    insertEdge(line);
		}
		// populate the in-bound edges(reverse map of out-bound edgs) after inserting into src key map.
		mGraph.createMapWithDestNodeAsKey(mGraph.mSrcNodeEdgeMap, mGraph.mDstNodeEdgeMap);
		in.close();
	}
	
	public void cleanAllNodeColor(){
		for(Node n : mAllNodes.values()){
			n.mColor = 0;
		}
	}
	
	public void unitTest(){
		for(String s : testdata){
			insertEdge(s);
		}
		mGraph.createMapWithDestNodeAsKey(mGraph.mSrcNodeEdgeMap, mGraph.mDstNodeEdgeMap);
	}
	private void eat(String str) {
        st = new StringTokenizer(str);
    }
   
    String next() throws IOException {
        while (!st.hasMoreTokens()) {
            String line = in.readLine();
            if (line == null) {
                return null;
            }
            eat(line);
        }
        return st.nextToken();
    }

    int nextInt() throws IOException {
        return Integer.parseInt(next());
    }

    long nextLong() throws IOException {
        return Long.parseLong(next());
    }

    double nextDouble() throws IOException {
        return Double.parseDouble(next());
    }

	public static void main(String[] args) throws IOException {
		if(args.length < 1){
			System.out.println("Usage: java facebull filename");
			//System.exit(1);
		}
		facebull fb = new facebull();
		//fb.unitTest();
		fb.readInGraph(args[0]);
		
		// now do find 
		fb.mGraph.dumpMSTMap(null);
		
		fb.consolidateSolution(DirectedMST.deepCopy(fb.mGraph));
		
		fb.showSolution(fb.mOptimalSolution);
	}
	
	// member variables
	private BufferedReader in;
    private PrintWriter out;
    private StringTokenizer st;

	protected DirectedMST mGraph = null;
	protected HashMap<String, Node> mAllNodes = new HashMap<String, Node>();
	// in-bound edge for each node as key, and the sort by value map. 
	Map<Edge, Integer> mInBoundEdgeCountMap = new HashMap<Edge, Integer>();
	protected Map<Edge, Integer> mInBoundEdgeCountSortedMap = null;
	protected List<Edge> mSolutionEdges = new ArrayList<Edge>();
	List<Edge> mOptimalSolution = new ArrayList<Edge>();
	Edge mCurEdge = null;
	int mCurCost = 0;
	int mOptimalCost = Integer.MAX_VALUE;
	// the final solution cycles, with each cycle start, end connected with edges
	protected ArrayList<ArrayList<Edge>>  mSolCycles = new ArrayList<ArrayList<Edge>>();
	
	protected String[] testdata = {    // correct answser: (1->2, 2->6, 6->5, 5->4, 4->3, 3->1)
		"M1	N1	N2	1",
		"M2	N1	N6	5",
		"M3	N2	N3	8",
		"M4	N4	N2	7",
		"M5	N2	N6	2",
		"M6	N3	N1	10",
		"M7	N3	N5	6",
		"M8	N5	N4	3",
		"M9	N6	N5	11",
		"M10 N6	N4	9",
		"M11 N4	N3	4",
	};
	
	/*
	 * test cases 4.

// test case 4: answ: 912 [ 1 5 6 7 10 ]
M1 C1 C2 619
M2 C1 C3 837
M3 C1 C4 765
M4 C2 C1 850
M5 C2 C3 69
M6 C2 C4 106
M7 C3 C1 1
M8 C3 C2 841
M9 C3 C4 877
M10 C4 C1 117
M11 C4 C2 580
M12 C4 C3 873
	
// test case 5.  answ: 1686. [ 1 15 17 27 36 40 45 53 63 70 ]
M1 C1 C2 578
M2 C1 C3 961
M3 C1 C4 521
M4 C1 C5 492
M5 C1 C6 498
M6 C1 C7 613
M7 C1 C8 989
M8 C1 C9 905
M9 C2 C1 611
M10 C2 C3 576
M11 C2 C4 697
M12 C2 C5 710
M13 C2 C6 747
M14 C2 C7 932
M15 C2 C8 206
M16 C2 C9 443
M17 C3 C1 109
M18 C3 C2 723
M19 C3 C4 929
M20 C3 C5 597
M21 C3 C6 125
M22 C3 C7 382
M23 C3 C8 53
M24 C3 C9 955
M25 C4 C1 844
M26 C4 C2 326
M27 C4 C3 246
M28 C4 C5 200
M29 C4 C6 474
M30 C4 C7 265
M31 C4 C8 691
M32 C4 C9 498
M33 C5 C1 749
M34 C5 C2 558
M35 C5 C3 531
M36 C5 C4 13
M37 C5 C6 102
M38 C5 C7 235
M39 C5 C8 63
M40 C5 C9 16
M41 C6 C1 573
M42 C6 C2 715
M43 C6 C3 465
M44 C6 C4 190
M45 C6 C5 132
M46 C6 C7 393
M47 C6 C8 671
M48 C6 C9 505
M49 C7 C1 641
M50 C7 C2 198
M51 C7 C3 170
M52 C7 C4 927
M53 C7 C5 33
M54 C7 C6 176
M55 C7 C8 472
M56 C7 C9 919
M57 C8 C1 800
M58 C8 C2 572
M59 C8 C3 358
M60 C8 C4 956
M61 C8 C5 520
M62 C8 C6 765
M63 C8 C7 342
M64 C8 C9 843
M65 C9 C1 663
M66 C9 C2 390
M67 C9 C3 386
M68 C9 C4 803
M69 C9 C5 200
M70 C9 C6 11
M71 C9 C7 447
M72 C9 C8 852
 
	 *
	 */
}
