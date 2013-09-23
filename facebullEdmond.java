/*
 * package com.haijin.facebull
 * This is travelling salesman problem.
 * find a path that connect all the nodes, hamilton cycle. 
 * hence there exists a path between any two nodes.
 *
 * you can do it wiht Edmond's algorithm, finding the MST in a directed graph.
 * http://www.ce.rit.edu/~sjyeec/dmst.html
 * for each node except from root, select the min in-bound edge. n nodes, n-1 edges.
 * if there is a cycle, break the cycle by replacing one edge from inside the cycle with one 
 * external inbound cycle with min extra delta. 
 *     Min(cost(e) - (cost(i)-min(cost(all internal edges))
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

public class facebull {
	private static final boolean DEBUG = false;
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
		protected Node mSrc, mDst;
		protected final String mMachine;
		protected final int mCost;
		
		public Edge(Node src, Node dst, final int cost, final String machine){
			mSrc = src;
			mDst = dst;
			mMachine = machine;
			mCost = cost;
		}
		
		public String toString(){
			return mSrc+"(" + mSrc.mColor + ") ->" + mDst+"(" + mDst.mColor + ")"  + "(" + mMachine + "=" + mCost + ")";
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
	}
	
	/**
	 * A direct MST consists of maps of edges with src node as key, as well as dst node as key.
	 */
	public static final class DirectedMST {
		// A MST has four Maps for a graph
		protected Map<Node, ArrayList<Edge>> mSrcNodeEdgeMap = new HashMap<Node, ArrayList<Edge>>();
		protected Map<Node, ArrayList<Edge>> mDstNodeEdgeMap = new HashMap<Node, ArrayList<Edge>>();  // reverse mapping
		protected Map<Node, ArrayList<Edge>> mSrcNodeMSTEdgeMap = new HashMap<Node, ArrayList<Edge>>();
		protected Map<Node, ArrayList<Edge>> mDstNodeMSTEdgeMap = new HashMap<Node, ArrayList<Edge>>();
		
		public DirectedMST() { }
		
		// static factory
		public static DirectedMST deepCopy(final DirectedMST mst){
			DirectedMST newmst = new DirectedMST();
			
			for(ArrayList<Edge> edges : mst.mSrcNodeEdgeMap.values()){
				for(Edge e : edges){
					newmst.insertEdge(e.mSrc, e, newmst.mSrcNodeEdgeMap);
				}
			}
			for(ArrayList<Edge> edges : mst.mDstNodeEdgeMap.values()){
				for(Edge e : edges){
					newmst.insertEdge( e.mDst, e, newmst.mDstNodeEdgeMap);
				}
			}
			newmst.mSrcNodeMSTEdgeMap.clear();
			newmst.mDstNodeMSTEdgeMap.clear();
			return newmst;
		}
		
		/**
		 * insert one edge into the map
		 */
		public void insertEdge(Node key, Node src, Node dst, final int cost, final String machine, Map<Node, ArrayList<Edge>> map){
			ArrayList<Edge> l = null;
			Edge newedge = new Edge(src, dst, cost, machine);
			l = map.get(key);  // will return null upon KeyError
			if ( null == l){
				l = new ArrayList<Edge>();
				map.put(key, l);
			}
			l.add(newedge);
		}
		
		public void insertEdge(Node key, Edge newedge, Map<Node, ArrayList<Edge>> map){
			ArrayList<Edge> l = null;
			if(!map.containsKey(key)){
				l = new ArrayList<Edge>();
				map.put(key,l);  // at the time here, l is not populated.
			}else{
				l = map.get(key);
			}
			l.add(newedge);
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
				}
			}
			return minedge;
		}
		
		/**
		 * find the min edge within a set that forms a cycle
		 */
		public Edge findSetMinEdge(final Set<Node> set, HashSet<Edge> allExternalInEdges){
			Edge minEdge = null;
			int minCost = Integer.MAX_VALUE;
			
			log("findSetMinEdge: " + set.toString());
			dumpMap(mDstNodeMSTEdgeMap);
			for(Node n : set){
				// only one edge to a node in the mst, and set with root already filtered.
				Edge e = mDstNodeMSTEdgeMap.get(n).get(0);
				log("findSetMinEdge: node: " + n.toString() + " edges:" + mDstNodeMSTEdgeMap.get(n) + " edge:" + e.toString());
				if(minEdge == null || e.mCost < minCost){
					minEdge = e;
					minCost = e.mCost;
					log("minCost:" + minCost + " minEdge:" + minEdge.toString());
				}
				for(Edge edge : mDstNodeEdgeMap.get(n)){  // all the in bound edge to node n in the cycle in MST.
					if(!set.contains(edge.mSrc) && edge.compareTo(e) != 0){  
						allExternalInEdges.add(edge);
					}
				}
			}
			return minEdge;
		}
		
		/**
		 * traverse mst(edges) and return a set of node that is strongly connected components.
		 */
		public Set<Node> DFSMST(Node root, final Map<Node, ArrayList<Edge>> srckeymap, Set<Node> cycle){
			cycle.add(root);
			root.mColor = 1;
			// node might not have out-bound edge, be careful.
			if(srckeymap.get(root) != null){
				for(Edge e : srckeymap.get(root)){ // all this node's out edgs
					//log("traversing edge: " + e.toString());
					cycle.add(e.mDst);   // the two node belongs in one cycle.
					if(e.mDst.mColor == 0){  // the map is InEdgeMap!!!
						DFSMST(e.mDst, srckeymap, cycle);
					}
				}
			}
			return cycle;
		}
		
		// DFS find cycle in a directed Graph
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
		 * detect any cycle in MST, the map has n nodes, and n-1 edges.
		 */
		public ArrayList<Set<Node>> detectAndConsolidateCycles(Node root, Map<Node, ArrayList<Edge>> srckeymap){
			boolean cyclemerge = false;
			ArrayList<Set<Node>> cycles = new ArrayList<Set<Node>>();
			ArrayList<Node> path = new ArrayList<Node>();

			// this loop will find all the cycles in MST
			for( Node n : srckeymap.keySet()){  // all the node in mst that has out-bound edges.
				//if(n.mColor == 1) continue;
				//path.clear();
				//Node cyclenode = DFS(n, srckeymap, path);
				//if(cyclenode != null){ log("cycle:" + path.toString()); }
				if(n.mColor == 0){
					cyclemerge = false;
					Set<Node> cycle = new HashSet<Node>();    // create a new set object
					DFSMST(n, srckeymap, cycle); // populate the new set object
					for(Set<Node> s : cycles){  // is this found cycle part of any existing cycle ?
						log("existing cycle:" + s.toString() + " and current cycle " + cycle.toString());
						if(!Collections.disjoint(s, cycle)){
							log("adding merged cycle:"+cycle.toString() + " => to " + s.toString());
							s.addAll(cycle);
							cyclemerge = true;
							break;  // find this cycle connect to an existing cycle, no need to loop more.
						}
					}
					if(!cyclemerge){    // this cycle not belong to any existing cycle.
						cycles.add(cycle);
						log("adding new cycle:" + cycle.toString());
						for(Set<Node> s : cycles){ log("now cycles:" + s.toString()); }
					}
				}
			}
			
			// contract cycles in disjoint sets.
			for(int i=0;i<cycles.size();i++){   // do not use fail-fast iterator due to concurrentmodificationException
				Set<Node> s = cycles.get(i);
				log("contract cycle :" + s.toString());
				if(s.contains(root)){
					continue;  // root does not have any incoming edges, skip it!!!
				}
				mergeDisjointSet(s, cycles);
			}
			return cycles;
		}
			
		/**
		 * merge one disjointset to set where root sit, and forms the MST. the set has n nodes and n edge. the se
		 */
		public void mergeDisjointSet(Set<Node> set, ArrayList<Set<Node>> cycles){
			int adjustCost = 0;
			Edge minEdgeFromOut = null;
			int minCost = Integer.MAX_VALUE;
			HashSet<Edge> allExternalInEdges = new HashSet<Edge>();
			Edge minEdgeInCycle = findSetMinEdge(set, allExternalInEdges);
			log("minedge:"+minEdgeInCycle + "AllExtInEdge:" + allExternalInEdges.toString());
			// find out the min cost replacement edge that come from outside to this cycle.
			for(Edge e : allExternalInEdges){
				Node dstNode = e.mDst;
				// if take this ext edge and take out the Inter Edge to the node pointed by this ext edge, what is the min extra delta. 
				adjustCost = e.mCost - (mDstNodeMSTEdgeMap.get(dstNode).get(0).mCost - minEdgeInCycle.mCost);
				log("adjcost="+ adjustCost + " dstnode="+dstNode + " extEdge: "+e.toString());
				if(minEdgeFromOut == null || adjustCost < minCost){
					minEdgeFromOut = e;
					minCost = adjustCost;
				}
			}
			log("Min Outer Edge:" + minEdgeFromOut.toString() + " mincost :" + minCost);
			
			// now found the min edge from outside, eliminate the cycle by removing the edge to the node inside the cycle
			Edge edgeToRemove = mDstNodeMSTEdgeMap.get(minEdgeFromOut.mDst).get(0);
			mDstNodeMSTEdgeMap.get(edgeToRemove.mDst).clear();
			log("remove edge:" + edgeToRemove.toString() + "by ext edge :" + minEdgeFromOut.toString());
			
			// a node in MST can only have one incoming edge, but might have multiple out-going edge
			// while modify a list while looping, must use iterator
			Iterator<Edge> it = mSrcNodeMSTEdgeMap.get(edgeToRemove.mSrc).iterator();
			while(it.hasNext()){
				if(it.next().compareTo(edgeToRemove) == 0){
					it.remove();
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
					set.addAll(s);  // merge two cycles
					cycles.remove(s);
					break;
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
		
		public void dumpMap(){
			for(Node key : mSrcNodeEdgeMap.keySet()){
				log("Node:"+key.toString()+" >>> " + mSrcNodeEdgeMap.get(key).toString());
			}
			log(" - - - - - -  - - -  -  ---- -  -  -");
			for(Node key : mDstNodeEdgeMap.keySet()){
				log("Node:"+key.toString()+" <<< " + mDstNodeEdgeMap.get(key).toString());
			}
			log(" - - - - - - solution  -  ---- -  -  -");
			for(Node key : mSrcNodeMSTEdgeMap.keySet()){
				log("Node:"+key.toString()+" >>> " + mSrcNodeMSTEdgeMap.get(key).toString());
			}
		}
		public void dumpMap(Map<Node, ArrayList<Edge>> map){
			log(" ----  dump start -------");
			for(Node n : map.keySet()){
				for(Edge e : map.get(n))
					log("key:" + n.toString() + " edge:" + e.toString());
			}
			log(" ---- dump end -------");
		}
	}
	
	/**
	 * constructor
	 */
	public facebull(){
		mGraph = new DirectedMST();
	}
	
	/**
	 * find MST for a single node start from root.
	 */
	public void findDirectedMST(Node root){
		log("--- Start solution for node" + root.toString());
		// make a copy of in edge map
		DirectedMST resultmst = DirectedMST.deepCopy(mGraph);
		Edge minedge = null;
		
		// 1. discard arc entering into the root, if any. MST has n nodes, n-1 edges, so discard root's first.
		if(resultmst.mDstNodeEdgeMap.get(root) != null){
			resultmst.mDstNodeEdgeMap.get(root).clear();   // ArrayList<Edge> exists but is empty now.
		}
		// 2. select the min entering arc for each node, every node must have at least one entering arc as this is connected graph.
		for(Node e : resultmst.mDstNodeEdgeMap.keySet()){
			minedge = resultmst.findNodeMinEdge(e, resultmst.mDstNodeEdgeMap);
			if(minedge != null){
				log("Min edge in cycle : " + minedge.toString());
				resultmst.insertEdge(minedge.mSrc, minedge, resultmst.mSrcNodeMSTEdgeMap);
			}
		}
		// 3. we got the mst with n node, n-1 entering arc, populate the map with dst node as key(in-bound)
		resultmst.createMapWithDestNodeAsKey(resultmst.mSrcNodeMSTEdgeMap, resultmst.mDstNodeMSTEdgeMap);
		
		// 4. detect circle. result MST has n node and n-1 edges, traverse it, n disjointsets represent n cycles.
		cleanAllNodeColor();
		resultmst.detectAndConsolidateCycles(root, resultmst.mSrcNodeMSTEdgeMap);  // only inEdgeMap key is reversed. all other maps have src as key.
		
		resultmst.dumpMap();
		addSolution(resultmst.mSrcNodeMSTEdgeMap);
		log("--- End solution for node" + root.toString());
	}
	
	public void addSolution(Map<Node, ArrayList<Edge>> map){
		for(ArrayList<Edge> edges : map.values()){
			for(Edge e : edges){
				mSolutionEdges.add(e);
			}
		}
	}
	
	// for the final solution, each node in the path can only have one arc in and one arc out.
	// the union of mst is already set and wont have two same arcs. Just select distinct nodes from both src and dst side.
	public HashSet<Edge> consolidateSolution(){
		HashSet<Edge> sol = new HashSet<Edge>();
		ArrayList<Node> srcNodes = new ArrayList<Node>();
		ArrayList<Node> dstNodes = new ArrayList<Node>();
		ArrayList<Edge> edges = new ArrayList<Edge>();
		for(Edge e : mSolutionEdges){   // result is already a set, no two same edges.
			srcNodes.add(e.mSrc);
			dstNodes.add(e.mDst);
			edges.add(e);
		}
		for(int i=0;i<srcNodes.size();i++){
			Node n = srcNodes.get(i);
			if(srcNodes.lastIndexOf(n) == srcNodes.indexOf(n)){
				sol.add(edges.get(i));
			}
		}
		for(int i=0;i<dstNodes.size();i++){
			Node n = dstNodes.get(i);
			if(dstNodes.lastIndexOf(n) == dstNodes.indexOf(n)){
				sol.add(edges.get(i));
			}
		}
		return sol;
	}
	
	public void showSolution(HashSet<Edge> sol){
		log("---- global solution ---");
		int sum = 0;
		ArrayList<Integer> machines = new ArrayList<Integer>();
		for(Edge e : sol){
			log(e.toString());
			sum += e.mCost;
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
	public void findAllDirectedMST() {
		for(Node n : mGraph.mSrcNodeEdgeMap.keySet()){
			//if(n.toString().equals("N6"))
			   findDirectedMST(n);
		}
	}
	
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
		mGraph.insertEdge(src, src, dst, Integer.parseInt(arr[3]), arr[0], mGraph.mSrcNodeEdgeMap);
	}
	
	public void readInGraph(String file) throws IOException{
		log("reading config file:" + file);
		in = new BufferedReader(new FileReader(file));
		String line;
		while ((line = in.readLine()) != null){
			log(line);
			insertEdge(line);
		}
		in.close();

		// populate reverse mapping immediately.
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
			System.exit(1);
		}
		facebull fb = new facebull();
		fb.unitTest();
		//fb.readInGraph(args[0]);
		fb.mGraph.dumpMap();
		fb.findAllDirectedMST();
		fb.showSolution(fb.consolidateSolution());
	}
	
	// member variables
	private BufferedReader in;
    private PrintWriter out;
    private StringTokenizer st;

	protected DirectedMST mGraph = null;
	protected HashMap<String, Node> mAllNodes = new HashMap<String, Node>();
	protected Set<Edge> mSolutionEdges = new HashSet<Edge>();
	protected String[] testdata = {    // correct answser: (1->2, 2->3, 2->6, 3->5, 5->4)
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
}
