//
// Prism algorithm, the same as dijkstra. 
// use a priqueue to store nodes, keyed by dist from current node.
// so init all node dist[i] to max. extractMin(), relax all its nb,
// re-insert nb to heap with new value.
// in dijkstra, dist[k] = min(dist[cur] + cost[cur,k])
// here, dist[k] = min(dist[k], cost(cur,k))
// also, color a node after processed, move from one unvisited set to visited set.

class Prism {
  public static class Node implements Comparable {
    private int value;
    private Node parent;
    private bool visited;

    public Node(int val){
      value = val;
      parent = null;
      visited = false;
    }

    @Override
    public int compareTo(Object o){
      Node n = (Node) o;
      return this.value - o.value;
    }
  }

  public static class Edge {
    Node from, to;
  }

  PriorityQueue<Node> Q = new PriorityQueue<Node>();

  /**
   * generate mst for the G, as a map, with adjacent edge list
   */
  int MST(Node root, Map<Node, List<Edeg> G) {
    Q.add(root);
    root.visited = true;

    while(Q.size()){
      cur = Q.poll();
      for(nb : G.get(cur)){
        // dijkstra dist[nb] > dist[cur] + cost(cur,nb)
        if not nb.visited and nb.value > cost(cur,nb):
            nb.value = cost(cur,nb)
            nb.parent = cur
            Q.add(nb)
      }
      cur.visited = true;
    }
  }
}



// Kruskal's algorithm finds a minimum spanning tree for a connected, weighted, undirected graph.
// http://algowiki.net/wiki/index.php?title=Kruskal's_algorithm
//
// graph is repr by adj hashmap. first sort all edges, and take each edge out,
// disjoin set merge edge.from and edge.to.
public class Kruskal {

  public ArrayList<Edge> getMST(Node[] nodes, ArrayList<Edge> edges){
      // sort all edges, and greedy to take min edge one by one.
      java.util.Collections.sort(edges);
      ArrayList<Edge> MST = new ArrayList<Edge>();
       
      DisjointSet<Node> nodeset = new DisjointSet<Node>();
      nodeset.createSubsets(nodes);

      for(Edge e : edges){
          if(nodeset.find(e.from) != nodeset.find(e.to)){
              nodeset.merge(nodeset.find(e.from), nodeset.find(e.to));
              MST.add(e);
          }
      }
      return MST;
  }
}


/**
 *  A disjoint sets ADT.  Performs union-by-rank and path compression.
 *  Implemented using arrays.  
 *  Elements are represented by ints, numbered from zero.
 **/

public class DisjointSets {

  // shall use hashmap, key is node, and value is its parent
  private int[] array;
  map<Node, Node> root = new HashMap<Node, Node>();

  /**
   *  Construct a disjoint sets object.
   *
   *  @param numElements the initial number of elements--also the initial
   *  number of disjoint sets, since every element is initially in its own set.
   **/
  public DisjointSets(int numElements) {
    array = new int [numElements];
    for (int i = 0; i < array.length; i++) {
      array[i] = -1;   // parent init to -1.
      Node = new Node(i);
      root.put(Node, null);
    }
  }

  /**
   *  union() unites two disjoint sets into a single set.  A union-by-rank
   *  heuristic is used to choose the new root.  This method will corrupt
   *  the data structure if root1 and root2 are not roots of their respective
   *  sets, or if they're identical.
   *
   *  @param root1 the root of the first set.
   *  @param root2 the root of the other set.
   **/
  public void union(int root1, int root2) {
    if (array[root2] < array[root1]) {
      array[root1] = root2;             // root2 is taller; make root2 new root
    } else {
      if (array[root1] == array[root2]) {
        array[root1]--;            // Both trees same height; new one is taller
      }
      array[root2] = root1;       // root1 equal or taller; make root1 new root
    }
  }

  Node union(Node n1, Node n2){
    Node p1 = root.get(n1);
    Node p2 = root.get(n2);
    if(p2) root.put(n1, p2);
    if(p1) root.put(n2, p1);
  }

  /**
   *  find() finds the (int) name of the set containing a given element.
   *  Performs path compression along the way.
   *
   *  @param x the element sought.
   *  @return the set containing x.
   **/
  public int find(int x) {
    if (array[x] < 0) {
      return x;                         // x is the root of the tree; return it
    } else {
      // Find out who the root is; compress path by making the root x's parent.
      array[x] = find(array[x]);
      return array[x];                                       // Return the root
    }
  }

  /**
   *  main() is test code.  All the find()s on the same output line should be
   *  identical.
   **/
  public static void main(String[] args) {
    int NumElements = 128;
    int NumInSameSet = 16;

    DisjointSets s = new DisjointSets(NumElements);
    int set1, set2;

    for (int k = 1; k < NumInSameSet; k *= 2) {
      for (int j = 0; j + k < NumElements; j += 2 * k) {
        set1 = s.find(j);
        set2 = s.find(j + k);
        s.union(set1, set2);
      }
    }

    for (int i = 0; i < NumElements; i++) {
      System.out.print(s.find(i) + "*");
      if (i % NumInSameSet == NumInSameSet - 1) {
        System.out.println();
      }
    }
    System.out.println();
  }
}
