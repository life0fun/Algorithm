// Graph UF ? n^2 ? slide win+Heap ? Mergesort ? bisect ? two ptrs ? 2^n backtrack ? 

// Subset selection: Decision Tree, O(2^n), DP[]: O(n^2), SlidingWin+MaxHeap: O(nlgn);  F[i]=F[i-k]+a/i; cur=pre+a/i; prepre=pre;
// 1. Abstract Invariants to tracked state, data structure(PQ, TreeMap). At I, what's given, what's answer ? 
// 2. Bisect: l: parked to first abs big(>target), insert point; [0..l-1]<target, discard; bucket sort. arr[i] vs arr[arr[i]];
// 4. Sliding win mono stk: later smaller arr_r cancels earlier larger in stk; subary(l,r) or dist(l,r). Global L watermark trap. 
// 4. sliding win: PQ/Treeset/maxminstk; fix/var width; Invert circular or two ends to max of middle slide win;
// 3. Two pointers ary: mov min(arr,l,r); {while(arr/l <= minH){l++;r--} minH=min(l,r)};
// 4. DFS(root, prefix, collector); prune; prefix to pass down aggregates at leaf; No Memoization with prefix recur down.
// 4. List dfs(arr, off): no prefix, leaf echo back parent aggregates all paths and ret; Powerset node sum(dfs(inc), dfs(skip));
// 6. K lists processing. merge, priority q. reasoning by add one upon simple A=[min/max], B=[min, min+1, min+k...] 
// 7. Decision tree Search: dfs(i) => dfs(i+1, prefix, collector); while(node || !stk.empty){ recur push; done pop; parent agg children and ret agg state up};
// 8. DP tracks max _end_/_start_ incl/excl i; max _end_ at i, dp[i]=Max(dp[i-k] for k=0..i); max_start_ at i, for(n-1..0) dp[i]+=dp[next]; Init Boundary properly.
// 10. BFS/DFS slow. use PriorityQueue to sort min and prune. DP[i_aggregate] > DFS/BFS > Recur(off, prefix);
// 11. MergeSort: process each pair, turn N^2 to NlgN. all pairs(i, j) comp skipping some when merge sorted halves. Count of Range Sum within.
// 5. Coin change order matters, outer loop v, inner loop each coins, each face value. dp[v]+=dp[v-a/i]; if order not matter, outer loop each coin; 
// 12. Knapsack: Recur incl/skip i into subset; Each constrain a dim; matrix[i,w]: aggre each dim; cell[i,wt, val], or dp[w,p] += dp[w-wi][p-ip];
// 13. Recur(arr,off,prefix) {for(i=off;i++){recur(i+1, prefix+arr[i]}}; outer loop over all coins, recur i+1 prefix into sub-recur; i was incl/excl in outer i iter.
// 14. Recur(i)=1+Recur(i+1) vs Recur(i, global_res): only update global when reaching leaf. 
// 14. TreeMap/MinMaxStk with mono min/max stk to track invar in a dynamic win; move L to keep win invars when consume R.; 
// 12. i-j: i is consumed? j is new; the # of eles from i+1 to j, excl i; len-1-i: # of ele exclude consumed i; len-i: # of suffix incl not consumed i; incl i;

// Loop move LR ? Recur subset pos+1 ? DP[i+1] ? Exit Check First; Boundary(l==r, arr[l]==arr[r]) and End state(off == len, arr[pos] < 0), move smaller edges !!!
// Recursion: (start, end, context) as args, ret Agg of sub states. recur(start, end). Pratt(cur_idx, _parent_min_bp_as_stop_condition_); 
// DP !! boundaries(i=1,j=head) !! as the partial val is carried to tree DFS recur next candidates. 
// 1. discovered state update, boundary check, build candidates, recur, visited aggregation.
// 1. Backtrack dfs search carrying partial tree at k, construct candidates k+1, loop recur each candidate.
// 2. The first thing in Recur(A, i, B, j) is exit end state boundary checks. return A.size()==i.
// 3. Boundary check and state track aggregations. Dp[i,j,k] = min/max(DP[i-1,j-1,k-1]+cost, ...);
// 4. Recur with local copy to each candidates, aggregate the result. dp[i]=max(dp[i-1][branch1], dp[i-1][branch2], ...)
// 5. the optimal partial tree is transitive updated from subtree thus updates the global and carry on.

// 5. Loop Init/Range/Exit: Init boundary, range, state update, Loop final state(sold, bot, empty ary) clean up.
// [l..r]: len/width=(r-l+1); r-l+1==Win for [l..r] boundary. [l..r) half open, len=r-l. len-i, discard[0..i-1], the suffix len incl i, [i..len)
// Inverse(int sorted[], int[] inverse), arr[sorted[0]] is the max. inv[i] is not arr[i]'s inverse. it's arr[sorted[i]]'s;   
// Backtrack(i,j): DFS recur branch out each next state(incl/excl); Recur(i+1, prefix state, Collector). Restore prefix after recursion!! 
// Dfs Done: child return End state, parent Aggregate max.
//  1. BFS with PQ sort by price, dist. BellFord, shortest path.
//  2. DFS spread out with local copy and recur with backtrack and prune.
//  3. Permutation, at each offset, swap suffix to offset as prefix for next candidate. recur with new offset. Restore Prefix after recursion.
//  4. DP for each sub len/Layer, merge left / right.
//  5. Recur Fn must Memoize intermediate result.

// BTree ADT Enum type and variant values {EMPTY, TreeNode}, TreeNode{ATOM(char), Entry(op, TreeNode(lhs), TreeNode(rhs))}
// Loop child Recursion, Exhaust child recursions as tree dfs parent->child.
// steps: dfs_enter(parent), [process_edge, recur(child) if !disc], dfs_exit_update(parent). 
// Pratt: while(op.lbs > loop_stop_cond) { consume_op; recur(new_op); concat(rhs); peek(next_op)}

// DP carries down Accumulated state and selections. State(buy/sell/rest) is the max of all options. 
//  bot @ day_i is the max(bot/i-1/k, sod/k-1, rest/i-1/k-1). accumulated profit passed each round.
//  rest/i/k is available from prev r days' sod/i-r/k;  
//  bot/i/k = max(bot/i-1/k, sod/i-1/k-1 - a/i, rst/i-1/k-1 - a/i)
//  sod/i/k = max(sod/i-1/k, bot/i-1/k   + a/i)
//  rst/i/k = sod/i-1/k\

// DP[i,j] for aggregation pass down. incl/excl, Agg(left, prev_left, prev).
// dp[i,j] = Max(dp[i-1,j-1] + dp[i,j-1] + dp[i, j-1]); Max(prev_left, pre, upper);
// Reuse single row from i to i+1, stage dp[i] as the prev_left for next dp[i+1]. 
// prepre/k-1, prepre, pre/k-1, pre, cur/k-1, cur, next

// 1. Total number of ways, dp[i] += sum(dp[i-1][v])
// 2. Expand. dp[i][j] = 1 + dp[i+1][j-1]
// after adding more items, rite seg is fixed, need to recur on left seg.
//   dp[i] = max( dp[j] for j in 1..i-1,)
//   for i in 1..slot:
//     for v in 1..V  or for v in [1..V], all possible WTs
//       dp[i] += dp[i-v]  enum all possible v at slot i
//       // or max min
//       dp[i] = min(dp[j+1] + 1, dp[i])
//       dp(i) = ( dp(i-1) * ways(i) ) + ( dp(i-2) *ways(i-1, i) )
//       tot = max(prepre + cur, pre)

// 3. when both left,rite need to recur. i->k, k->j, enum all 2 segs. 
// each segs recur divide to 2 segs. 
//   for gap len range from i..n, for i range from 1..n. 
//     dp[i,j] = min( dp[i, j-1] + dp[i+1, j] )
//     dp[i,j] = min( dp[i,j], dp[i,k] + dp[k+1,j] + arr[i]*arr[k]*arr[j])
//     dp[i,j,k] = for m in i..j: max(dp[i+1,m-1,0] + dp[m,j,k+1]) when a[i]=a[m]
//     dp[i,j,k] = max(dp[i+m, j, k-1]+(arr[i+m]-arr[i]))

// 4. Recur(i, prefix, Collector) VS. DP[i,j]: next step candidates preparation and dfs.
//   Loop each next=i+1 -> N, swap(i, next), incl/excl current i, carrying [prefix + i], recur(i+1, prefix+i, Collector).
//   or just process head_i, Recur(i+1, prefix + incl_excl_i, collector)
// Loop child recursion on linear. Loop until { new_child = recur}
// DP[i,j], intermediate gap=1..n, dp[i,gap] + dp[gap,j]; 3 loops, gap, i, j.
// Outer loop coins, inner loop each value. loop multiple copies before done with each coin. unique comb;
// outer loop each value, inner loop coins, no need multi copy loops. not unique. [2,3], [3,2] count as diff.

// Formalize the problem into DP recursion. dp[i] _start_/_end_ at i; Loop backwards if dp[i] related to dp[i+k];
// BFS/DFS recur search, incl/excl cur element; permutation swap to get N^2. K-group, K-similar, etc.
// Merge sort to Reduce to N2 comparsions or lgN, break into two segs, recur in each seg.
// Stack PQ to track window invariant. Later bigger cancel prev smaller. mono inc/desc.    
// 1. Bucket sort / bucket merge. A[i] stores value = index value+1
// 5. Ary divide into two parts, up[i]/down[i], wiggly sort, or longest ^ mountain.

// knapsack matrix[i,w]: loop each item, each W, incl/excl i, dp[i][w]=max(dp[i-1][w-w_i]+v_i, dp[i-1][w])
// cnt ways: (profit>P, w<W): update matrix upon each ip/iw, makes cells within [0..P][0..W-iw] contribute to dp[p+ip][w+iw];  
// for p in 0..P, for w in 0..W-iw, dp[p+ip,w+iw] += dp[p][w]; 
// binpacking: NP hard. multiple bin. First fit, first unfit, best fit. etc.

// """ Intervals scheduling, TreeMap, Segment(st,ed) is interval.
// Sort by start, use stk to filter overlap. Greedy can work.
// sort by finish(start<finish). no stk. track global end. end=max(iv.end).
// Interval Array, add/remove range all needs consolidate new range.
// TreeMap to record each st/ed, slide in start, val+=1, slide out end, val-=1. sum. find max overlap.
// Interval Tree, find all overlapping intervals. floorKey/ceilingKey/submap/remove on add and delete.

// Trie and Suffix tree. https://leetcode.com/articles/short-encoding-of-words/
// Node stores outgoing edges { Map<HeadChar,Edge> edges; suffix_idx} Edge {startidx,endidx,endNode}; Edge(st,ed) is label.
// Split edge by inserting new internal nodes contains the branch edges. root->leaf path: arr[suffix_idx:end]

// """ Graph: Adj edge list: Map<Node, LinkedList<Set<Node>>, recur thru all edges visiting nodes.
// DFS(root, prefix, collector): Recur only __undisc__. DFS done, all child visited, memorize path[cur]. 
// DFS Edge: Tree: !discovered, Back: !processed anecestors in stk. Forward: already found by. Cross: Directed to already traversed component.
// Edge exact once: dfs(parent,me), dfs(me, !discovered); Back:(in stk !processed, excl parent). processed_me is done when node is processed.  
// SCC: DFS Back-edge merge earliest, or two-DFS: label vertex by completion. DFS reversed G from highest unvisited. each dfs run finds a component.
// List All paths: DFS(v, prefix, collector). ret paths or collect at leaves. BFS(v,prefix): no enqueue traverse when nb in prefix. 
// - - - - -
// Graph Types: Directed/Acyclic/Wegithed(negative). Edge Types: Tree, Back, Forward, Cross(back_to_dfs-ed_component) 
// a) DFS/BFS un-weighted, b) Relaxation Greedy for weighted, c) DP[i,k,j] all pairs paths.
// Dijkstra/MST_prim: pick a root, PQ<[sort_by_dist, node, ...]>; shortest path to all. Kruskal. UnionFind shortest edges.
// Dijkstra no negative edges. dist[parent] no longer min if (parent-child) < 0 reduce weights. 
// Single src Shortest path to all, Bellman: loop |v| times { for each edge(u,v,e), d[v] = min(d[u] + e)}
// DAG: topsort. linear dijkstra. DFS+memorize all subpaths, parent is frozen by order. parents min before expanding subgraph.   
// MST(UnionFind): SCC. Shortest path changes when each weight+delta as the path is multihops.
// All paths: BFS(prefix), (exactly) k hops. nb not in prefix, or DFS(cur, prefix), or DP[i,j,k] = d[i,u,k-1]+e(u,j);
// All Pairs: all i,j pair via each intermeidate k, d[i,j]=d[i,k]+d[k,j]. Directed, Path[i,k,j] != Path[j,k,i]. List Paths
// Shortest cycle(girth): all pair shortest path(i,j),(j,i);
// - - - - -
// DP vs DFS. for k=1..N, foreach edge(u,v), DP[v,k]=DP[u,k-1]+(u,v). 
// From root, DFS(v, prefix, collector) {dfs(nb, prefix+v, collector)} recur nb collecting. 
// BFS(v) {while [v, path] = Q.pop() {Q.enq(nb, path+v)}}
// 

(max) Matching Edge: max set of edges no share vertdx. pairing vertex. polynomial. some indepset NP-hard problems can be reduced to it.
Bipartite Matching: max edges/pairs no sharing vertex. sizeof (max matching) = min(vertex cover),
IndepSet. vertices No conflicting, vertex Color. Backtracking(incl, excl, recur)/Greedy. Tree Heuristic: leaf nodes, repeat.
Vertex Color: no conflicting. 4-color thereom. Backtracking. start lowest-degree node, delete adjs. Repeat. 
Vertex Cover: min vertex cover all edges complement Max(IndepSet). start highest-degree, delete adjs, repeat. lgN worse than optimal.
Chromatic number: min(vertex color), Min(# of IndepSet), NP-Complete. construct graph from 3-SAT set, reduce to 3-SAT(each clause=true in a set), NP-Hard.
Workers-Jobs Scheduling: max pairs set not vertex sharing edges. NP-Hard. Heuristic: start with leaves, remove, repeat.
Scheduling/Clique/Vertex color: conflicting nodes have edges, Sorted by End time.
Edge color: no two same color edges share a vertex. A person can not talk to two others at the same time.
All NP-hard solved by backtracking, dfs each candidates carry down or ret partial results. 

Max Flow from s-t defines the largest matching in G.
Max Flow: When a reverse edge is used, remove its forward edge found in prev BFSs from the matching edge sets.
// Weighted Edges Graph: Max Matching(Disjoint Edges), BFS, Increase Reduce Residual/Flow of Forward Reverse path edges.
Max Matching Flow: each bfs path Reduces edges Forward flow and Increases reverse edges Residual flow. 
  rGraph: add reverse edges(j,i) of each (i,j). each edge has residual/flow. init (i,j) residual/flow=cap/0, (j,i)=0/0
  BFSs(st, ed, parent): a path with edges residual>0; min path vol; A reverse edge can be used as its residue can > 0.
  Augment_path(st, ed, vol) { parent(ed)->ed: residual-vol/flow+vol; ed->parent(ed): residual+vol/0), recur(st, parent[ed], vol)


// Pratt Parsing: https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
// pub enum Token { ATOM(char), OP(char), ... }; token variants.
// Lexer { tokens: vec<Token>, fn next() -> Optional<Token>; }
// Parser: TokenTree enum { Atom, Cons(Op, Vec<TokenTree>), BTree(op, TokenTree(lhs), TokenTree(rhs)}) }
//
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
// Orthogonal Optimization of subqueries and Aggregation.
// Correlated subquery rule-based unnesting: CorrelatedJoin(left=input, external query, rite=subquery correlated to external query column);
// repeatedly pushes correlated join down based on these rules until it is converted into common join.
Query parser parses SQL => Relation(table) Algebra => Query Tree => Physical Plan Operator(Join order, Pred push down) => Execution Plan(a chain of Operators)
Query Tree: QRoot = Select selList FROM fromList Where cond(attr cond subquery)
1. Rewrite correlated subquery by Join/Aggregation: outer query to a 2-argment selection(R x R-cond). Replace inner subquery with Join/Aggregation.
2. 2-argument selection operator re-write: Sigma(R x R-cond) : select all from R that satifiy R-cond(contains Relations).
   Example: Where attr in (select ...) or Where attr > (select ...)
3. translate the outer query using the 2-argument selection hence the inner subquery rewrite as Join.
4. re-write 2-arg selection rule: replace select+cartesion product by a Join(R with R-cond on keys from R-cond subquery).
5. Correlated subquery rule: inner query to a query plan, outer query to a 2-arg selection, re-write(replace by a Join) 
6. Correlated subquery rule-based unnesting: conversion to Join and Aggregation with Correlation conditions
Eventually, physical operator is a single chain of operators with current operator pull input = next operator output.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// dept max salary, left outer join itself null bigger than your e.id. No group by.
SELECT D.Name AS Department , E.Name AS Employee, E.Salary 
from Department D, Employee E LEFT OUTER JOIN Employee e2 on (e2.departmentid = E.departmentid and E.salary <= e2.salary) 
where e2.id IS NULL and E.DepartmentId = D.id;

// greatest-n-per-group query using left outer join. as only 1 records has null bigger, no group needed.
SELECT t1.*
FROM transactions t1 LEFT OUTER JOIN transactions t2 
ON (t1.acct_id = t2.acct_id AND t1.trans_date < t2.trans_date)
WHERE t2.acct_id IS NULL;

// top 2 per dept, left outer join.
SELECT D.Name AS Department, E.Name AS Employee, E.Salary 
from Department D, Employee E LEFT OUTER JOIN Employee e2 
on (e2.departmentid = E.departmentid and E.salary <= e2.salary) 
where E.DepartmentId = D.id 
group by e.name 
having count(*) <=2 

// Top 3 salary, dup not allowed.
SELECT D.Name as Department, E.Name as Employee, E.Salary 
  FROM Department D, Employee E, Employee E2  
  WHERE D.ID = E.DepartmentId and E.DepartmentId = E2.DepartmentId and 
  E.Salary <= E2.Salary     // there only will have at most 3 row from joined E2 that has salary higher than me.
  group by D.ID, E.Name 
  having count(distinct E2.Salary) <= 3   // count(*)
  order by D.Name, E.Salary desc

// sort the list by x dim. recur left, right.  
def KdTree(pointList, height):
  // Node {point, left, rite}
  cur_dim = height % DIMS
  sort(pointList, (a, b)->{a[cur_dim] < b[cur_dim]})
  root = Node(median(pointList))
  root.left = kdTree(pointList[0..pivot), k+1)
  root.rite = kdTree(pointList[pivot+1,), k+1)
  return root

def find_nearest(root, query_point, best=None, best_distance=float('inf'), depth=0):
  if root is None: return best, best_distance

  dim = depth % 2
  current_point = Point(*root.point)
  current_distance = euclidean_distance(query_point, current_point)

  if best is None or current_distance < best_distance:
      best = current_point
      best_distance = current_distance

  if query_point[dim] < root.point[dim]: // recur down to left plan
      child_branch = root.left
      opposite_branch = root.right
  else:
      next_branch = root.right
      opposite_branch = root.left

  best, best_distance = find_nearest(child_branch, query_point, best, best_distance, depth + 1)

  // If query point is closer to plane other plan. A dim plane is split by root in the median.
  if abs(query_point[dim] - root.point[dim]) < best_distance:
      best, best_distance = find_nearest(opposite_branch, query_point, best, best_distance, depth + 1)

  return best, best_distance
  
// 1. pick a pivot, put to end
// 2. start j one idx less than pivot, which is end-1.
// 3. park i,j, i to first > pivot(i++ if l.i <= pivot), j to first <pivot(j-- if l.j > pivot)
// 4. swap i,j and continue until i<j breaks
// 4. swap pivot to i, the boundary of divide 2 sub string.
// 5. 2 substring, without pivot in, recursion.
def qsort(arr, lo, hi):
  def swap(arr, i, j):  arr[i],arr[j] = arr[j],arr[i]
  def partition(arr, l, r):
    pivot = arr[r]
    wi = l
    for i in xrange(l,r):    // not incl r
      if arr[i] <= pivot:
        swap(arr, wi, i)
        wi += 1
    swap(arr, wi, r)
    return wi
  if lo < hi:  // recur only when lo and hi are diff ele.
    p = partition(arr, lo, hi)
    qsort(arr, lo, p-1)  // split into 3 segs, [0..p-1, p, p+1..n]
    qsort(arr, p+1, hi)
  return arr
//  qsort([5,8,5,4,5,1,5], 0, 6)

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// linear + Binsect
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
static int kadan(int[] arr) {
  int spanStart = 0, spanEnd = 0, curmax, gmax;
  for (int i = 0; i < arr.length; i++ ) {
    curmax = Math.max(curmax + arr[i], arr[i]);
    curmax = Math.max(curmax, 0) + arr[i]; // arr[i] incl prev only it > 0;
    gmax = Math.max(gmax, curmax);
    curmin = Math.min(curmin, 0) + arr[i]; // take prev to arr[i] if it is negative;
    gmin = Math.min(gmin, curmin);
  wraped_max = Math.max(gmax, sum-gmin);
  return gmax;
}

// LC 1749. Maximum Absolute Sum of Any Subarray, Track highest positive and lowest negative;
int maxAbsoluteSum(int[] arr) {
  int gmax=0,curmin=0,curmax=0,tmp;
  for(int i=0;i<arr.length;i++) {
    curmax=Math.max(arr[i], curmax+arr[i]);
    curmin=Math.min(arr[i], curmin+arr[i]);
    gmax=Math.max(gmax, Math.max(curmax, Math.abs(curmin)));
  }
  return gmax;
}
// track max[i]/min[i] up to a/i, swap max/min when a/i < 0; this handles 0 gracefully.
int maxProduct(int[] arr) {
  int curmax=arr[0], curmin=arr[0], gmax=curmax;
  for(int i=1;i<arr.length;++i){
    // swap max/min when arr[i]<0 before prod.
    if(arr[i] < 0) { int tmp=curmax;curmax=curmin;curmin=tmp; }
    curmax = Math.max(curmax*arr[i], arr[i]); // always include arr/i as it can be a max ending at i;
    curmin = Math.min(curmin*arr[i], arr[i]);
    gmax=Math.max(gmax, curmax);
  }
  return gmax;
}

// Number of Subarrays with Bounded Maximum, when a[i] > R, i restart a new subary. else, subary include a[i]
int numSubarrayBoundedMax(int arr[], int L, int R) {
  int result=0, left=-1, right=-1;
  for (int i=0; i<A.size(); i++) {
    if (A[i] > R)  left = i;     // i restart a new subary
    if (A[i] >= L) right = i;    // 
    result += right-left;
  }
  return result;
}

// longest subarray with an equal number of 0 and 1. Prefix sum stay the same from i..j.
// http://www.geeksforgeeks.org/largest-subarray-with-equal-number-of-0s-and-1s/
int findMaxLength(int[] arr) {
  Map<Integer, Integer> valoff = new HashMap<>();
  int prefix = 0, maxlen = 0;
  valoff.put(0, -1);
  for(int i=0;i<arr.length;++i) {
    prefix += arr[i] == 0 ? -1 : 1;
    if(valoff.containsKey(prefix)) {
      maxlen = Math.max(maxlen, i-valoff.get(prefix));
    } else {valoff.putIfAbsent(prefix, i);}
  }
  return maxlen;
}
maxSubaryEqualZeroOne(new int[]{1, 0, 1, 1, 1, 0, 0});

// two passes, from L->R: f[i]=f[i-1]; then L<-R,  f[i]=f[i+1]
int candy(int[] arr) {
  int[] f = new int[arr.length]; f[0] = 1;
  for(int i=1;i<arr.length;i++) {
    if(f[i]==0) f[i]=1;
    if(arr[i-1] < arr[i]) {f[i] = f[i-1]+1;}
  }
  for(int i=arr.length-2;i>=0;i--) {
    if(arr[i] > arr[i+1]) {f[i] = Math.max(f[i], f[i+1]+1);}
  }
}
int maxTurbulenceSize(int[] A) {
  int ans = 1; int l = 0;
  for (int r = 1; r < N; ++r) {
    int c = Integer.compare(A[r-1], A[r]);
    if (c == 0) { l = r; } // reset the left start point l to i;
    else if (r == N-1 || c * Integer.compare(A[r], A[r+1]) != -1) { // same sign.
      ans = Math.max(ans, r - l + 1);
      l = r; // reset l to r.
    }
  }
  return ans;
}
// LC 80;
int removeDuplicates(int[] arr) { // at most two dups;
  int pos=1;
  for(int i=2;i<arr.length;i++) {
    if(arr[i] != arr[pos] || arr[i] != arr[pos-1]) {
      arr[pos+1] = arr[i];
      pos++;
    }}
  return pos+1;
}

def minjp(arr):  // use single var to track, no need to have ary.
  mjp = 1;  curmaxRite = arr[0];  nextMaxRite = arr[0];   // seed with value of first.
  for i in xrange(1,n):
    if i <= curMaxRite: // i is reachable form current jump,
      nextMaxRite = Math.max(nextMaxRite, i+arr[i])
    else:   // when approaching to current jp
      mjp += 1
      curMaxRite = nextMaxRite
    if curMaxRite >= n:
      return mjp
assert minjp([1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9]) == 3

// Track the step k when reaching A[i], so next steps is k-1,k,k+1. stones[i] value is idx that has a stone. 
// Map<Integer, Set<Integer>> and branch next jumps.
boolean canCross(int[] stones) {
  if (stones.length == 0) { return true;}
  // each stone's idx, associate with a set of next jumps.
  HashMap<Integer, HashSet<Integer>> posJumps = new HashMap<Integer, HashSet<Integer>>(stones.length);
  posJumps.put(0, new HashSet<Integer>()); posJumps.get(0).add(1);  // first jmp to first pos
  for (int i = 1; i < stones.length; i++) { posJumps.put(stones[i], new HashSet<Integer>() ); }  // hashset store how many prev jmps when reaching this stone.
  for (int i = 0; i < stones.length - 1; i++) {
    for (int step : posJump.get(stones[i])) { int nextReach = iStop + step;
      if (nextReach == stones[stones.length - 1]) {  return true; }
      // put pre jump into nextReach's next step set.
      HashSet<Integer> set = posJumps.get(nextReach);  // 在 nextReach 上有一个 set.
      if (set != null) {
        set.add(step);   // add step into next reach's set, so we can jmp from there.
        if (step - 1 > 0) set.add(step - 1);
        set.add(step + 1);
      }
    }
  }
  return false;
}

int canCompleteCircuit(int[] gas, int[] cost) {
  int[] net = new int[gas.length];
  int sum = 0, start=0;
  for(int i=0;i<gas.length;++i) { net[i] = gas[i] - cost[i]; sum += net[i]; }
  if(sum < 0) return -1; 
  sum = 0;
  for(int i=0;i<gas.length;++i) {
    sum += net[i];
    if(sum < 0) {start = -1; sum = 0;} 
    else if(start < 0) { start = i; }
  }
  return start;
}

// l ptr to the First Larger(<); Lift l when <=, OR if no lift when ==, [0..l-1] discardable.
// drag down r when arr/r > target, when l,m,r, make r<l, r ptr to smaller.
static int Bisect(int[] arr, int target, boolean insert_at_end) {
  int l=0, r=arr.length-1;
  while (l <= r) {
    int m = l+(r-l)/2;
    if (insert_at_end) {
      if (arr[m] <= target) { l=m+1;} // lift l when eqs for insert_at_end;
      else { r = m-1;}
    } else { // insert at beg, drag down r when arr/m eqs target.
      if (arr[m] < target) { l=m+1; } // only lift l when arr/m absolutely < target.
      else { r = m-1;}
    }
  }
  return l; // check l not oob arr.length;
}

static int BisectClosest(int[] arr, int l, int r, int target) {
  while(l<=r) {
    if(arr[m] < target) { l = m+1;} else {r = m-1;}
  }
  if(l>=arr.length) return arr.length-1;
  return l == 0 ? l : Math.min(arr[l], l-1);
}

static int bsearchRotate(int[] arr, int target) {
  while(l <= r) {
    while(l<r && arr[l] == arr[l+1]) l+=1; // move over dups
    while(l<r && arr[r] == arr[r-1]) r-=1;
    int m = l + (r-l)/2;
    if(arr[m] == target) return m;
    if(arr[l] <= arr[m]) {  // <=, ortherwise, [4,2] will fail as [4,4] is sorted but [4,2] is not sorted.
      if(arr[l] <= target && target < arr[m]) { r = m-1;}
      else {l = m+1;}
    } else {
      if(arr[m] < target && target <= arr[r]) { l = m+1;}
      else {r = m-1;}
    }
  }
  return -1;  
}
assert bsearchRotate(new int[]{3, 4, 5, 1, 2}, 2) == 5;
assert bsearchRotate(new int[]{4,5,6,7,8,1,2,3}, 8) == 4
assert bsearchRotate(new int[]{8,4,5,6,7}, 4) == 1

// for any rotated min/max, check [l..r] is sorted first !!
int rotatedMin(int[] arr) {
  int l=0,r=arr.length-1;
  while(l<r) {
    int m=l+(r-l)/2;
    if(arr[l] <= arr[m] && arr[m] <= arr[r]) return arr[l];
    if(arr[m] > arr[m+1]) return arr[m+1];
    if(arr[l] <= arr[m]) { l=m+1; }
    else { r=m;}
  }
  return arr[l];
}
// find min in rotated. / / with dup, advance mid, and check on each mov """
static int rotatedMin(int[] arr) {
  int l = 0, r = arr.length-1;
  while (l < r) {
    while (l < r && arr[l] == arr[l+1]) l += 1;
    while (l < r && arr[r] == arr[r-1]) r -= 1;
    int m = (l+r)/2;
    if(arr[l] <= arr[m] && arr[m] <= arr[r]) return arr[l];
    if (arr[m] > arr[m+1]) { return arr[m+1]; }  // check m, m+1
    if (arr[l] <= arr[m]) {  l = m+1; }  // rotate point must in the right. 
    } else { r = m; }  //  not m-1, [3,1,2]
  }
  return l;
}
System.out.println(rotateMin(new int[]{4,5,0,1,2,3}));
rotatedMin(new int[]{2,2,1,2,2,2,2,2})
rotatedMin(new int[]{2,2,2,2,2,2,1,2})

// find the max in a ^ ary,  m < m+1 """
static int rotatedMax(int[] arr) {
  int l = 0, r = arr.length-1;
  while (l < r) {  // l and r wont be the same, m+1 always valid.
    int m = (l+r)/2;
    if(arr[l] <= arr[m] && arr[m] <= arr[r]) return arr[r];
    if (arr[m] < arr[m+1]) { l = m+1; }
    else if (arr[m] > arr[m+1]) { r = m; }
  }
  return l;
}
print rotatedMax([8, 10, 20, 80, 100, 200, 400, 500, 3, 2, 1])
print rotatedMax([10, 20, 30, 40, 50])
print rotatedMax([120, 100, 80, 20, 0])

// LC 962; no need leftmin, just rmax;
int maxDist(int[] arr) {
  int[] rmax = new int[arr.length]; rmax[arr.length-1]=arr[arr.length-1];
  for (int i=arr.length-2; i>=0; i--) {
    rmax[arr.length-i] = Math.max(rmax[i+1], arr[i]);
  }
  int l = 0, r = 0, maxdist = 0;   // two counters to break while loop.
  // stretch r as r from left to rite is mono decrease.
  while (l < sz && r < sz) {
    if (rmax[r] >= arr[l]) {
      maxdist = Math.max(maxdist, r-l+1);  // calculate first, mov left second.
      r += 1;
    } else { l += 1; }
  }
  return maxdist;
}

// container most water: track each bin **first** left higher. not left max higher. 
// how left/rite ptrs avoid bisect of stk ? upon each bin, need find the first bigger.
// left ptr incrs until bigger. because r from the end, it captures the length.
int maxArea(int[] arr) {
  int l = 0, r=arr.length-1, maxw = 0;
  while(l<r) {
    gmax = Math.max(gmax, Math.min(arr[l], arr[r])*(r-l));
    // stretch L or R to bigger area.
    if(arr[l] <= arr[r]) l++; else r--;
  }
  return maxw;
}

int largestRectangleArea(int[] arr) {
  Deque<Integer> stk = new ArrayDeque<>();
  int[] leftedge = new int[arr.length], riteedge = new int[arr.length];
  for(int i=0;i<arr.length;++i) {
    while(stk.size() > 0 && arr[i] < arr[stk.peekLast()]) { // left's rite edge is found. pop it. No need to track it. 
      riteedge[stk.peekLast()] = i;
      stk.removeLast();  // out of stk, rite edge.
    }
    leftedge[i] = stk.size() > 0 ? stk.peekLast() : -1; // after popping, cur's left edge fixed.
    stk.addLast(i);
  }
  // fix the rite edges of all bins in stk to the end.
  while(stk.size() > 0) {riteedge[stk.removeFirst()] = arr.length;}
  int gmax = 0;
  for(int i=0;i<arr.length;++i) {
    gmax = Math.max(gmax, arr[i]*(riteedge[i]-leftedge[i]-1));
  }
  return gmax;
}
// stk to store left edge. rite edge is when arr[i] < stk.peekLast();
int largestRectangleArea(int[] arr) {
  int gmax = 0;
  Deque<Integer> stk = new ArrayDeque<>(); // store the idx, arr[stk[i]]
  for(int i=0;i<=arr.length;++i) {
    if(stk.size()==0) {stk.add(i); continue;}
    int height = i==arr.length ? 0 : arr[i];
    while(stk.size() > 0 && height < arr[stk.peekLast()]) {
      int popi = stk.removeLast();
      int left = stk.size()==0 ? 0 : stk.peekLast()+1; // +1
      gmax = Math.max(gmax, arr[popi]*(i-left));
    }
    stk.add(i);
  }
  return gmax;
}

int trapWater(int[] arr) {
  { // for each bin, calculate its left/rite edges.
    int[] leftmaxi = new int[arr.length], ritemaxi = new int[arr.length];
    int leftmax=0, ritemax = 0;
    for(int i=0;i<arr.length;++i) { 
      leftmax = Math.max(leftmax, arr[i]); leftmaxi[i] = leftmax; 
      rmitemax = Math.max(ritemax, arr[arr.length-1-i]); ritemaxi[i] = ritemax;
    }
    int tot = 0;
    for(int i=0;i<arr.length;++i) { tot += Math.min(leftmaxi[i], ritemaxi[i]) - arr[i]; }
    return tot;
  } 
  { // two ptrs from both sides, squeeze minH(left, rite edges) to peak. increase minH. 
    int l=0,r=arr.length-1,minH=0,water=0;
    while(l<r) {  // move smaller edge; 
      minH=Math.min(arr[l],arr[r]) // calculate k that is next to minH; [l..r]
      { if(arr[l] == minH) { k=l; while(arr[k]<=minH){tot+=diff;k++} l=k;}
        else {k=r; while(arr[k]<=minH){tot+=diff;k--} r=k;}
      }
      {
        while(l<r && arr[l] <= minH) { water += minH-arr[l++];}
        while(l<r && arr[r] <= minH) { water += minH-arr[r--];}
      }
    }
  }
  { // find peak first. then squeeze left and rite separately.
    while(k<maxhidx) {
      while(k<maxhidx && arr[k] <= arr[l]) { tot+=arr[l]-arr[k]; k++;}
      l=k;
    }
  }
}

// put 0 at head, 2 at tail, 1 in the middle, wr+1/wb-1
int[] sortColors(int[] arr) {
  int lidx=0,ridx=arr.length-1,i=0; // 0's idx from head and 2's index from tail;
  while(i<=ridx) {
    if(arr[i] == 0) { swap(arr, i, lidx); lidx++; i++; continue;}
    if(arr[i] == 2) {swap(arr, i, ridx); ridx--;} // no move off i. recheck arr/i again.
    else {i++;}  // arr[i]==1, no move around.
  }
  return arr;
}
sortColor([2,0,1])
sortColor([1,2,1,0,1,2,0,1,2,0,1])

// loop invariant: a[0:low] < 1, a[low:mid]=1, a[mid:high]=2
// init: low=0, mid=0, high=n-1
int[] threeWayParition(int[] arr) {
  low=0,high=arr.length-1,mid=0;
  for(mid=low; mid<high; mid++) {  // stop entering when mid==high
    int cur = a[mid];
    if(a[mid] < 1) {
      a[mid] = a[low]; a[low] = cur;  low++;
    } else if (a[mid] > 1) {
      a[mid] = a[high]; a[high] = cur; high--; 
    }
}}

void sortColors(int[] arr) {
  int lw = 0, rw = arr.length-1;
  for(int i=0;i<=rw;) {
      if(arr[i] == 0) {
          swap(arr, lw, i);
          lw++; ++i;
      } else if(arr[i] == 2) {
          swap(arr, rw, i);
          rw--;
      } else { ++i;}
      System.out.println("i " + i + " lw " + lw + " rw " + rw + " " + Arrays.toString(arr));
  }
}

void rainbowSort(int[] colors, int left, int rite, int startColor, int endColor) {
  if (colorFrom == colorTo) { return; }
  if (left >= right) { return; }
  int colorMid = (colorFrom + colorTo) / 2;
  int l = left, r = right;
  while (l < r) {
    while (l < r && colors[l] <= colorMid) { l++;  }
    while (l < r && colors[r] > colorMid) { r--;  }
    if (l < r) {
      int temp = colors[l]; colors[l] = colors[r]; colors[r] = temp;
      l++; r--;
    }
  }
  rainbowSort(colors, left, r,     colorFrom,    colorMid);
  rainbowSort(colors, l,    right, colorMid + 1, colorTo);
}

// build Lmin[], and use R stack from rite to left, compare stk peek to lmin[r], not arr[r]
boolean oneThreetwoPattern(int[] arr) {
  int[] lmin = new int[arr.length];
  lmin[0] = arr[0];
  Stack<Integer> stk = new Stack<>();
  for (int i=1;i<arr.length;i++) { lmin[i] = Math.min(lmin[i-1, arr[i]); }
  for (int r=arr.length-1;r>=0;r--) {
    if (arr[r] > lmin[r]) {  // only when 1 < 3, then find 2.
      while (!stk.isEmpty() && stk.peek() <= lmin[r]) { stk.pop(); }  // stk top < lmin.r, no use, pop.
      if (!stk.isEmpty() && stk.peek() < arr[r]) { return true; } // after pop, if stk not empty, and its top < arr[r]
      stk.push(arr[r]);
    }
  }
}
// Maximum Sum of 3 Non-Overlapping Subarrays; enum middle subary from [i,i+k]; sum left and rite.
// dp[i] is max sum that mid ary is [i..i+k], max left subary 0..i, max rite subary i+k..n.
static int maxSumOfThreeSubarrays(int[] arr, int subarylen) {
  int sz = arr.length, leftmaxsofar = 0, ritemaxsofar = 0, allmax = 0;
  int[] dp = new int[sz]; int[] prefixsum = new int[sz]; int[] leftMax = new int[sz]; int[] riteMax = new int[sz];
  prefixsum[0] = arr[0];
  for (int i=1;i<sz;i++) { prefixsum[i] += prefixsum[i-1] + arr[i]; }

  leftmaxsofar = prefixsum[subarylen-1]; // rolling track leftmax
  leftMax[subarylen-1] = leftmaxsofar;
  for (int i=subarylen; i<sz; i++) { // each subary [i-k, k]
    leftmaxsofar = Math.max(leftmaxsofar, prefixsum[i] - prefixsum[i-subarylen]);
    leftMax[i] = leftmaxsofar;
  }
  ritemaxsofar = prefixsum[sz-1] - prefixsum[sz-1-subarylen];
  riteMax[sz-subarylen] = ritemaxsofar;
  for (int i=sz-1-subarylen; i>=subarylen; i--) {
    ritemaxsofar = Math.max(ritemaxsofar, prefixsum[i+subarylen-1] - prefixsum[i-1]);
    riteMax[i] = ritemaxsofar;
  }
  for (int i=subarylen; i<sz-subarylen; i++) {
    dp[i] = Math.max(dp[i], (prefixsum[i+subarylen-1] - prefixsum[i-1]) + leftMax[i-1] + riteMax[i+subarylen]);
    allmax = Math.max(allmax, dp[i]);
  }
  return allmax;
}
maxSum3Subary(new int[]{1,2,1,2,6,7,5,1}, 2);   // 23, [1, 2], [2, 6], [7, 5], no greedy [6,7]

// LC 769 max # of chunk; [0..k] prefix Max > [k..n] suffix Min; not sum.
// arr[i] either form a new grp, or merge to pre grp. To form a new grp, it must > pre max; 

// while sliding, count up first, then count down, then add up+down. Reset counters at turning point.
int longestMountainSubary(int[] arr) {
  int maxlen =0, up = 0, down = 0;
  { for (int i=1;i<arr.length;i++) {
    // down > 0, means we are in down phase, when switching to up phase, reset.
    if (down > 0 && arr[i-1] < arr[i] || arr[i-1] == arr[i]) {  // turning point. reset counters.
      up = down = 0;    // reset when encouter turning points
    }
    if (arr[i-1] < arr[i]) up++;
    if (arr[i-1] > arr[i]) down++;
    if (up > 0 && down > 0 && up + down + 1 > maxlen) { maxlen = up + down + 1;}
    return maxlen;
  }
  {
    int lstart=-1, rend=-1, maxw=0; 
    for(int i=1;i<arr.length;i++) {
      if(arr[i] > arr[i-1]) {
        if(lstart == -1) lstart=i-1;
        else if(rend != -1) { lstart = i-1; rend=-1;} // reset when \ chaged to /
      } else if(arr[i] < arr[i-1]) {
        if(lstart != -1) { rend = i; maxw = Math.max(maxw, rend-lstart+1); }
      } else {lstart = -1; rend=-1; }
    }
    return maxw;
  }
}

// Longest mountain = LUp[i] + RDown[i]; find Long Increasing Seq [0..i] and [i..N] 
int insertPos(List<Integer> arr, int target) {
  int l=0, r=arr.size()-1;
  while(l<=r) {
    int m=l+(r-l)/2;
    if(arr.get(m) == target) return m;
    if(arr.get(m) < target) { l=m+1;}
    else {r=m-1;}
  }
  if(l==arr.size()) arr.add(target);
  else arr.set(l, target);
  return l; // less than target, [..l), exclude l;
}
int minimumMountainRemovals(int[] arr) {
  int[] leftLIS = new int[arr.length], riteLIS = new int[arr.length];
  List<Integer> list = new ArrayList<>(); 
  list.add(arr[0]); leftLIS[0]=0;
  for(int i=1;i<arr.length;i++) {leftLIS[i] = insertPos(list, arr[i]); }
  list.clear(); list.add(arr[arr.length-1]);
  for(int i=arr.length-2;i>=0;i--) {riteLIS[i] = insertPos(list, arr[i]); }
  int maxlen = 0;
  for(int i=1;i<arr.length-1;i++) {
    if(leftLIS[i] > 0 && riteLIS[i] > 0) {
      maxlen = Math.max(maxlen, (leftLIS[i]+riteLIS[i]+1));
    }
  }
  return arr.length-maxlen;
}

// # wiggle subsequence
// # up[] track longest up end at i, down[i], longest down end at i.
// # if arr[i] > arr[k], up[i] = down[k] + 1
static int longestWiggleSubSequence(int[] arr) {
  int sz = arr.length;
  int[] up = new int[sz];
  int[] down = new int[sz];
  for (int i=1;i<sz;i++) {
    if      (arr[i-1] < arr[i]) { up[i] = up[i-1] + 1;  down[i] = down[i-1]; }
    else if (arr[i-1] > arr[i]) { up[i] = down[i-1];    down[i] = down[i-1] + 1; }
    else {                        down[i] = down[i-1];  up[i] = down[i-1]; }
  }
  return Math.max(up[sz-1], down[sz-1]);
}

static int longestConsecutive(int[] nums) {
  Set<Integer> num_set = new HashSet<Integer>();
  for (int num : nums) { num_set.add(num); }

  int longestStreak = 0;
  for (int num : num_set) {
    if (!num_set.contains(num-1)) {  // only move forward from start node
      int currentNum = num;
      int currentStreak = 1;
      while (num_set.contains(currentNum+1)) {
        currentNum += 1;
        currentStreak += 1;
      }
      longestStreak = Math.max(longestStreak, currentStreak);
    }
  }
  return longestStreak;
}
// LC 2498 swappable when diff < limit. Sort. distribute arr/i to last grp, or new grp.
List<Integer> smallestBySwapping(int[] arr, int limit) {
  Arrays.sort(arr);
  List<ArrayDeq<Integer>> grps = new ArrayList<>();
  for(int i=0;i<arr.length;i++) {
    if(grps.size() <= 0 || arr[i]-grps.get(-1).peekLast() > limit) {
      grps.add(new ArrayList<>());
    }
    grps.get(-1).addLast(arr[i]);
    grp_idx[i] = grps.size()-1;
  }
  for(int i=0;i<arr.length;i++) {
    res.add(grp_idx[i].removeFirst());
  }
}

// dp[i,j] : LAP of arr[i:j] with i is the anchor, and k is the left and j is the rite. 
// Try each i anchor.  k <- i -> j,  dp[i,j] = dp[k,i]+1, iff arr[k]+arr[j]=2*arr[i]
def longestArithmathProgress(arr):
  mx = 0
  dp = [[0]*len(arr) for i in xrange(len(arr))]
  // boundary, dp[i][n] = 2, [arr[i], arr[n]] is an AP with len 2
  for i in xrange(len(arr)-2, -1, -1):   dp[i][len(arr)-1] = 2
  // i is the middle of the 3, 
  for i in xrange(len(arr)-2, -1, -1): // start from n-2
    k = i-1, j = i+1
    while k >= 0 and j < len(arr):
      if 2*arr[i] == arr[k] + arr[j]:
        dp[k][i] = dp[i][j] + 1
        print arr[k],arr[i],arr[j], " :: ", k, i, j, dp[k]
        mx = max(mx, dp[k][i])
        k -= 1, j += 1
      elif 2*arr[i] > arr[k] + arr[j]:  j += 1
      else:                             k -= 1
    if k >= 0:  // if k is out, then [i:j] as header 2.
      dp[k][i] = 2
      k -= 1
  return mx
print longestArithmathProgress([1, 7, 10, 15, 27, 29])
print longestArithmathProgress([5, 10, 15, 20, 25, 30])

def LAP(arr):
  mx = 0
  dp = [[0]*len(arr) for i in xrange(len(arr))]
  for i in xrange(1,len(arr)):      dp[0][i] = 2
  for i in xrange(1,len(arr)-1):
    k,j = i-1,i+1
    while k >= 0 and j <= len(arr)-1:
      if 2*arr[i] == arr[k]+arr[j]:
        dp[i][j] = dp[k][i] + 1
        mx = max(mx, dp[i][j])
        k -= 1
        j += 1
      elif 2*arr[i] > arr[k]+arr[j]:
        j += 1
      else:
        if dp[k][i] == 0:
          tab[k][i] = 2
        k -= 1
  return mx
print LAP([1, 7, 10, 15, 27, 29])
print LAP([5, 10, 15, 20, 25, 30])

// List<List<Integer>> LinkedList<>();
List<List<Integer>> threeSum(int[] num) {
  Arrays.sort(num);
  List<List<Integer>> res = new LinkedList<>(); 
  for (int i = 0; i < num.length-2; i++) {
    if (i == 0 || (i > 0 && num[i] != num[i-1])) {
      int lo = i+1, hi = num.length-1, sum = 0 - num[i];
      while (lo < hi) {
        if (num[lo] + num[hi] == sum) {
          res.add(Arrays.asList(num[i], num[lo], num[hi]));
          while (lo < hi && num[lo] == num[lo+1]) lo++;
          while (lo < hi && num[hi] == num[hi-1]) hi--;
          lo++; hi--;
        } 
        else if (num[lo] + num[hi] < sum) lo++;
        else hi--;
      }
    }
  }
  return res;
}

// LC 1658. Reduce X to Zero from both ends; convert to find max mid ary size;

// iterate each cur of arr, bisect each ele into dp[] sorted array.
// cur bigger than all in dp, append to dp end, increase lislen.
// enhancement is use parent[] to track the parent for each node in the ary.
int lengthOfLongestIncreaseSequence(int[] arr) {
  int[] dp = new int[arr.length];  // LoopInvar: track lis[i]; Pre: dp[i] = 0; post: dp[n] = max
  int lislen = 0;   // dp[] is empty initially.
  for (int cur : arr) {
    int lo = 0, hi = lislen;     // lislen is 0 for now.
    while (lo <= hi) {  
      int m = (lo + hi) / 2;
      if (dp[m] < cur)  lo = m + 1;
      else              hi = m - 1;
    }
    dp[lo] = cur;  // updte insertion point lo cur;
    if (lo == lislen) { ++lislen; }  // bisect indicate append to DP end. increase size.
  }
  return lislen;
}

// return -1 if a shall precedes b, ab > ba; otherwise 1, ba > ab;
int compare(int a, int b) {
  if(a == -1) { return 1; } else if (b == -1) {return -1;}
  String astr = Integer.toString(a), bstr = Integer.toString(b);
  // return (astr+bstr).compareTo(bstr+astr);
  for(int i=0,j=0;i<astr.length() || j<bstr.length();) {
      if(i == astr.length()) {i = 0;}
      else if(j == bstr.length()) { j= 0;}
      if(i<astr.length() && j<bstr.length()) {
          if(astr.charAt(i) < bstr.charAt(j)) {return 1;}
          else if(astr.charAt(i) > bstr.charAt(j)) { return -1;}
          else { i++;j++; continue;}
      }
  }
  return -1;  // both equals. 
}

int parent(int idx) { return (idx-1)/2;}
int[] children(int idx) { return new int[]{2*idx+1, 2*idx+2};}
void swap(int[] arr, int i, int j) { int tmp = arr[i]; arr[i]=arr[j];arr[j]=tmp;}
int maxof(int[] arr, int i, int j) { 
  if(compare(arr[i], arr[j]) > 0) { return j; } else {return i;}
}
void siftup(int[] arr, int idx) {
  while(idx > 0) {
    int parent = parent(idx);
    if(maxof(arr, parent, idx) == idx) {
        swap(arr, parent, idx);
        idx = parent;
    } else { break;}
  }
}
void siftdown(int[] arr, int idx, int end) {
  while(idx < end) {
    int[] child = children(idx);
    int maxidx = idx; 
    if(child[0] <= end) { maxidx = maxof(arr, maxidx, child[0]); }
    if(child[1] <= end) { maxidx = maxof(arr, maxidx, child[1]); }
    if(maxidx == idx) return;
    swap(arr, idx, maxidx);
    idx = maxidx;
  }
}
void heapify(int[] arr) {
  int m = arr.length/2;
  while(m >=0) {
    siftdown(arr, m, arr.length-1);
    m--;
  }
}
String largestNumber(int[] arr) { // arr is heapsort by heapify to a max heap.
  heapify(arr); int k = 0; StringBuilder sb = new StringBuilder();
  while(arr[0] != -1) {
    sb.append(arr[0]);
    arr[0] = -1;
    k++;
    swap(arr, 0, arr.length-k);
    siftdown(arr, 0, arr.length-1-k);
  }
  String ret = sb.toString();
  if(ret.charAt(0) == '0') return "0";
  return ret;
}

// http://stackoverflow.com/questions/5212037/find-the-recur-largest-sum-in-two-arrays
//    The idea is, take a pair(i,j), we need to consider both [(i+1, j),(i,j+1)] as next val
//    in turn, when considering (i-1,j), we need to consider (i-2,j)(i-1,j-1), recursively.
//    By this way, we have isolate subproblem (i,j) that can be recursively checked.
def recurMaxSum(p, q, kth):
  color = [[0 for i in xrange(len(q)) ] for i in xrange(len(p)) ]
  outl = []
  heap = MaxHeap()
  heap.insert(p[0]+q[0], 0, 0)
  color[0][0] = 1

  for i in xrange(k):
    [maxv, i, j] = heap.head()
    outl.append((maxv, i, j))
    heap.extract()

    // insert [i+1,j] and [i, j+1]
    if i+1 < len(p) and not color[i+1][j]:
      color[i+1][j] = 1
      heap.insert(p[i+1]+q[j], i+1, j)
    if j+1 < len(q) and not color[i][j+1]:
      color[i][j+1] = 1
      heap.insert(p[i]+q[j+1], i, j+1)
  print 'result ', outl

def testrecurMaxSum():
  p = [100, 50, 30, 20, 1]
  q = [110, 20, 5, 3, 2]
  KthMaxSum(p, q, 7)

// max Envelop [][] lis. 1. Arrays.sort by width, 2. Arrays.binarySearch, 3. update DP[idx], 4. return lislen.
int maxEnvelopes(int[][] envelopes) {
  if (envelopes == null || envelopes.length == 0 || enveloes[0] == null ||  envelopes[0].length != 2) return 0;
 
  Arrays.sort(envelopes, new comparator<int[]>((a,b) -> {
    if (a[0] == b[0]) return b[1] - a[1];
    else              return a[0] - b[0];
  }));
    
  int lislen = 0;  // init to 0. when sect ret lo = lislen, that's the end. will lislen++;
  int[] lis = new int[envelopes.length];
  for (int[] e : envelopes) {
    int idx = Arrays.binarySearch(lis, 0, lislen, e[1]);
    if (idx < 0) {   idx = -(idx+1); } // binary search return insertion pos
    lis[idx] = e[1];    // update with smaller value.
    if (idx == lislen) { lislen++;  }
  }
  return lislen;
}

// the max num can be formed from k digits from a, preserve the order in ary.
// key point is drop digits along the way.
// Stack to save the branch(drop/not-drop) presere the order. 用来后悔 track each branch point.
static int maxkbits(int[] arr, int k ) {
  int alen = arr.length, total = 0;
  Deque<Integer> deq = new ArrayDeque<>();
  deq.addLast(arr[0]);
  for(int i=1;i<alen;++i) {
    if (arr[i] < deq.peek()) { deq.addLast(arr[i])}
    else { // pop deq until deq.len + alen-i == k or deq.peek > arr[i]
      while(!deq.isEmpty() && deq.size() + alen -i > k && deq.peek() < arr[i]) dep.pollLast();
      deq.addLast(arr[i]);
    }
  }
  while(!deq.isEmpty()) { total = total*10 + deq.pollFirst(); }
  return total;
}
print reduce(lambda x,y: x*10+y, maxKbits([3,9,5,6,8,2], 2))

String removeKdigits(String num, int k) {
  Deque<Character> stk = new ArrayDeque<>(); // stk shall never empty !
  char[] arr = num.toCharArray();
  int i=0;
  while(i<arr.length) {
    if(i==0 || k==0 || stk.size()==0 || stk.size()>0 && arr[i]>=stk.peekLast()) {
      if(stk.size()!=0 || arr[i]!='0') stk.addLast(arr[i]); 
      i++; continue;
    }
    while(stk.size() > 0 && stk.peekLast() > arr[i]) {
      stk.removeLast(); k--;
      if(k==0) break;
    }
  }
  StringBuilder sb = new StringBuilder();
  while(stk.size() > k) {
    sb.append(stk.removeFirst());
  }  
  return sb.length() > 0 ? sb.toString() : "0";
}

// max k digit num can be formed from a, b, preserve the order in each ary 
// idea is i digits from a, k-i digits from b, then merge.
def maxNum(a,b,k):
  def merge(a,b):  # merge two sorted ary
    return [max(a,b).pop(0) for _ in a+b]
  mxnum = 0
  for i in xrange(k+1):
    lout = merge(maxKbits(a,i), maxKbits(b,k-i))
    v = reduce(lambda x,y:x*10+y, lout)
    mxnum = max(mxnum,v)
  return mxnum
print maxNum([3, 4, 6, 5], [9, 1, 2, 5, 8, 3], 5)

// # for each i, 2 options, swap / notswap, need to use two state to track them.
// swap[i] means swap a.i/b.i, keep means no swap a.i/b.i
// if a.i > a.i-1 and b.i > b.i-1, then keep.i = keep.i-1. swap.i = swap.i-1 + 1. if swap, swap both.
// if b.i > a.i-1 and a.i > b.i-1, 
//    swap.i = min(swap.i, keep.i-1 + 1)   // swap i, or swap i-1
//    keep.i = min(keep.i, swap.i-1)
int minSwap(int[] A, int[] B) {
  // track j=k notswap: notswap, s: swapped
  int keep = 0, swap1 = 1;
  for (int i = 1; i < A.length; ++i) {
    int notswap2 = Integer.MAX_VALUE, swap2 = Integer.MAX_VALUE;
    if (A[i-1] < A[i] && B[i-1] < B[i]) {         // not need to swap
      notswap2 = Math.min(notswap2, keep);  // the same as pre value of not swap
      swap2 = Math.min(swap2, swap1 + 1);       // or if swap, pre also need swap.
    }
    if (A[i-1] < B[i] && B[i-1] < A[i]) {
      notswap2 = Math.min(notswap2, swap1);     // either pre swap
      swap2 = Math.min(swap2, keep + 1);    // or pre not swap, current swap.
    }
    keep = notswap2; 
    swap1 = swap2;
  }
  return Math.min(n1, s1);
}    

// next greater rite circular: idea is to go from rite, pop stk top when top is less than cur.
// to overcome circular, do 2 passes, as when second pass, stk has the greater rite stk from prev
int[] nextGreater(int[] arr) {
  Deque<Integer> stk = new ArrayDequeu<>();
  int[] nextGreater = new int[sz];
  int sz = arr.length, i = 0;
  for(int i=sz-1; i>=0; i--) {   // from rite -> left.
    while (!stk.isEmpty() && arr[i] >= stk.peekLast()) {  // clean up stk when stk top is no use.
      stk.pollLast();
    }
    nextGreater[i] = stk.isEmpty() ? -1 : arr[stk.peekLast()];
    stk.offerLast(i);
  }
  for(int i=sz-1;i>=0;i--) {
    while (!stk.isEmpty() && arr[i] >= stk.peekLast()) { stk.pollLast(); }
    nextGreater[i] = stk.isEmpty() ? -1 : arr[stk.peekLast()];
  }
}

// cut a line cross least brick.
// map key=rite_edge_width, value=num_layers has that edge. edge with max layers is where cut line cross least.
int leastBricks(List<List<Integer>> wall) {
  Map</*x=*/Integer, /*bricks=*/Integer> xcuts = new HashMap();
  int count = 0;
  for (List<Integer> row : wall) {
    int rowWidth = 0;   // when start a new row, init counter
    for (int i = 0; i < row.size() - 1; i++) {   
      rowWidth += row.get(i);     // grow rowWidth for each brick in the row.
      //map.compute(rowWidth, (k, v) -> v += 1);
      xcuts.put(rowWidth, xcuts.getOrDefault(rowWidth, 0) + 1);
      count = Math.max(count, xcuts.get(rowWidth));
    }
  }
  return wall.size() - count;
}

// Bisect to find left <= arr/i, arr/j <= rite;
int[] edges(List<Integer> arr, int left, int rite) {
  int l=0,r=arr.size()-1, lidx=-1, ridx=-1;
  while(l<=r) {
    int m=l+(r-l)/2;
    // if(arr.get(m) == left) { l=m; break; }  //  no need
    if(arr.get(m) < left) {l=m+1;} else {r=m-1;}
  }
  lidx=l;

  l=0;r=arr.size()-1;
  while(l<=r) {
    int m=l+(r-l)/2;
    if(arr.get(m) <= rite) {l=m+1;} else {r=m-1;}
  }
  ridx = l-1;
  return new int[]{lidx, ridx};
}
int segK(char[] arr, int l, int r, int k, Map<Character, List<Integer>> map, Deque<int[]> segs) {
  for(int i=l;i<=r;i++) {
    List<Integer> ilist = map.get(arr[i]);
    int[] ilr = edges(ilist, l, r);
    if(ilr[1]-ilr[0]+1 < k) {
      int start = l;
      for(int j=ilr[0];j<=ilr[1];j++) {  // further split
        segs.addLast(new int[]{start, ilist.get(j)-1});
        start = ilist.get(j)+1;
      }
      segs.addLast(new int[]{start, r});
      return 0;
    }
  }
  return r-l+1;
}

// Partial sort by partition to find Kth element. partition sort is O(n). Find Median. Be careful of pivot.
// http://www.ardendertat.com/2011/10/27/programming-interview-questions-10-kth-largest-element-in-array/
int[] partition(int[] arr, int l, int r) {
  int widx = l, segmax = Integer.MIN_VALUE, segmin = Integer.MAX_VALUE;
  // need a real value from the arry to make part arr[0..k] <= arr[k],
  // int pivot = Math.floorDiv(arr[l] + arr[r], 2); 
  int pivotidx = l+(r-l)/2;
  swap(arr, r, pivotidx);
  for(int i=l;i<r;++i) {
    if(arr[i] <= arr[r]) { // arr[r] == pivot
      swap(arr, i, widx); widx++;
      segmax = Math.max(segmax, arr[i]); segmin = Math.min(segmin, arr[i]);
    }
  } // [l..widx-1] the same if min=max;
  swap(arr, widx, r); // widx ptr to the last ele >= arr[r]; 
  // [l..widx] <= arr[r]; [l..widx-1] the same 
  if(widx==r && segmin==segmax) return new int[]{widx-1, segmin};
  return new int[]{widx, -1};
}
int findKth(int[] arr, int l, int r, int k) {
  {
    int left=l, minmax, ascendk = arr.length-k; // 1st desc = length = last asc = idx in ascend;
    while(l <= r) { // to update left even wehn l==r
      int[] left_minmax = partition(arr, l, r);
      left=left_minmax[0];minmax=left_minmax[1];
      if(left == ascendk) break;
      if(left < ascendk) { l = left+1;}
      else {
        // [l..left] the same, return any if fine;
        if(minmax != -1) { if(l <= ascendk) break; }
        else { r = left-1;}
      }
    }
    return arr[left];
  }
  {  // recursion
    int[] left_minmax = partition(arr, l, r);
    int left=left_minmax[0], minmax=left_minmax[1], ascendk=arr.length-k;
    if(left == ascendk) { return arr[left]; } 
    else if (left < ascendk) { return findRecur(arr, left+1, r, k); }
    else {
      if(minmax > 0) return minmax; 
      else return findRecur(arr, l, left-1, k); 
    }
  }
}

static int kth(int[] arr, int[] brr, int k) {
  int asz = arr.length, bsz = brr.length;
  if (asz > bsz) { return kth(brr, arr, k); }
  if (asz == 0) return brr[k-1];
  if (k == 1) { return Math.min(arr[0], brr[0]); }
  
  int aidx = Math.min(asz, k/2);   // take half
  int bidx = Math.min(bsz, k/2);    // a+b = k-1
  // must -1, as we take half of 
  if (arr[aidx-1] > brr[bidx-1]) {    // discard half every time.
      int[] subb = Arrays.copyOfRange(brr, bidx, brr.length);
      return kth(arr, subb, k-bidx);  
  } else {
      int[] suba = Arrays.copyOfRange(arr, aidx, arr.length);
      return kth(suba, brr, k-aidx);
  }
}
System.out.println(kth(new int[]{2, 3, 6, 7, 9}, new int[]{1, 4, 8, 10}, 4));

static int kth(int[] arr, int[] brr, int k) {
  if (arr.length > brr.length) return kth(brr, arr, k);
  if (arr.length == 0) { return brr[k-1]; }
  if (k == 1) { return Math.min(arr[0], brr[0])}
  int pa = Math.min(k/2, arr.length);
  int pb = k - pa;
  if (arr[pa-1] < brr[pb-1]) {
    recur(Arrays.copyOfRange(arr, pa, arr.length), brr, k-pa);
  } else {
    recur(arr, Arrays.copyOfRange(brr, pb, brr.length), k-pb);
  }
}

int onePassTotPairsWithinDist(int[] arr, int dist) {
  int l=0,r=l,totpairs=0;
  while(l<=r && r<arr.length) {
    if(arr[r]-arr[l] <= dist) { totpairs += (r-l); r++;}
    else { l++;}
  }
  return totpairs;
}

// kth Pair; binary search dist that # of pair=k; each search scan ary count pairs whose diff < dist; 
// Kth needs minHeap sorting, or bucket sorting; better just do binary search.
int smallestDistancePair(int[] arr, int k) {
  int onePassTotPairs(int[] arr, int dist) {
    int l=0,r=l,tot=0;
    while(l<=r && r<arr.length) {
      if(arr[r]-arr[l] <= dist) { tot += (r-l); r++;} 
      else {l++;}
    }
    return tot;
  }
  Arrays.sort(arr); 
  int lo_bound=0,hi_bound=arr[arr.length-1]-arr[0], kthdist=(int)1e9;
  while(lo_bound <= hi_bound) {
    int mid=lo_bound+(hi_bound-lo_bound)/2;
    int tot = onePassTotPairsWithinDist(arr, mid); // 
    if(tot < k) { lo_bound=mid+1;}
    else { kthdist = Math.min(kthdist, mid); hi=mid-1;  }
  }
  if(kthdist==(int)1e9) return 0; else return kthdist;
}

// LC 1508: binsearch k smallest subary sum using sliding window;
// expand r++ when winsum < target, else l++; count # of subary.
// if # of subary > k, bump target; tot=subarysum(rite)-left;
double findMedianSortedArrays(vector<int> A, vector<int> B) {
  int m = A.size(), n = B.size();
  if (A.size() > B.size()) return findMedianSortedArrays(B, A);
  int imin = 0, imax = m, half = (m + n + 1) / 2, i, j, num1, num2;
  while (imin <= imax) {
    i = (imin + imax) / 2;
    j = half - i;
    if (j > 0 && i < m && B[j - 1] > A[i])
      imin = i + 1;
    else if (i > 0 && j < n && A[i - 1] > B[j])
      imax = i - 1;
    else {
      if (!i) num1 = B[j - 1];
      else if (!j) num1 = A[i - 1];
      else num1 = max(A[i - 1], B[j - 1]);
      break;
    }
  }
  if ((m + n) % 2) return num1;
  if (i == m) num2 = B[j];
  else if (j == n) num2 = A[i];
  else num2 = min(A[i], B[j]);
  return (num1 + num2) / 2.0;
}
int[] bisectMedian(int[] arra, int al, int[] arrb, int bl, int discard) {
  int am=al,bm=bl;
  while(al < arra.length && bl < arrb.length && al+bl < discard) { // exclude al, bl, how many discarded
    int remain = discard-(al+bl); // al is not processed, hence al is excluded, [0..l] len is l+1;
    if(arra.length - al <= arrb.length-bl) {  // len - al, len from [al..len); al is [0..al), len-al: len [al..len), suffix from al;
      am = Math.min((al+arra.length-1)/2, al+remain-1);
      bm = Math.max(bl, bl+(remain-1-(am-al+1))); // -1 to count bl, [al..am] inclusive can be discarded.
    } else {
      bm = Math.min((bl+arrb.length-1)/2, bl+remain-1);
      am = Math.max(al, al+(remain-1-(bm-bl+1)));
    }
    //System.out.println(String.format("remain %d al %d am %d bl %d bm %d", remain, al, am, bl, bm));
    if(arra[am] <= arrb[bm]) { al = am+1; } // al can mov over ar;
    else { bl = bm+1; }
  }
  if(al >= arra.length) {
    return new int[]{-1, discard-arra.length};
  } else if(bl >= arrb.length) {
    return new int[]{discard-arrb.length, -1};
  } else {
    return new int[]{al, bl};
  }
}

double findMedianSortedArrays(int[] arra, int[] arrb) {
  int totlen = arra.length+arrb.length;
  if(arra.length==0 || arrb.length==0) { return median(arra) + median(arrb); }
  int[] idx = bisectMedian(arra, 0, arrb, 0, (totlen-1)/2);  
  int first=0, second=0, discarded=0;
  if(idx[0] == -1) {
    discarded = arra.length+idx[1];
    first = arrb[idx[1]];
    second = (idx[1]+1==arrb.length) ? first : arrb[idx[1]+1];
  } else if(idx[1] == -1) {
    discarded = arrb.length+idx[0];
    first = arra[idx[0]];
    second = (idx[0]+1 == arra.length) ? first : arra[idx[0]+1];
  } else {
    discarded = idx[0]+idx[1];
    first = Math.min(arra[idx[0]], arrb[idx[1]]);
    if(first == arra[idx[0]]) {
      second = Math.min(arrb[idx[1]], idx[0]+1 < arra.length ? arra[idx[0]+1] : arrb[idx[1]]);
    } else {
      second = Math.min(arra[idx[0]], idx[1]+1 < arrb.length ? arrb[idx[1]+1] : arra[idx[0]]);
    }
  }
  if(totlen % 2 == 0) {
    return (first+second)/2.0;
  } else {
    if(discarded == totlen/2) return first; else return second;
  }
}

// - - - - - - - - - - - - - - - - - - - - -  - - -
// Bucket sort: min gap between points; bucket width, # of bucket. 
// sort eles into buckets so buckets are partially sorted. 
// 1+ ele per bucket. inside buckets tracks aggregation of eles(min/max).
// - - - - - - - - - - - - - - - - - - - - -  - - -
static void swap(int[] arr, int i, int j) { int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; }
static int[] buckesort(int[] arr) {
  int l = 0; int sz = arr.length; int r = sz -1;
  while (l < sz) {
    while (arr[l] != l+1 && arr[l] >= 0) {
      int lv = arr[l];
      if (arr[lv-1] == -999) {arr[lv-1] = -1; arr[l] = -999; }
      else if (arr[lv-1] < 0 ) { arr[lv-1] -= 1; arr[l] = -999; } 
      else { swap(arr, l, lv-1); arr[lv-1] = -1; }
    }
    if (arr[l] == l+1) {arr[l] = -1;}
    l += 1;
  }
  System.out.println("arr -> " + Arrays.toString(arr));
  return arr;
}

int firstMissingPositive(int[] arr) {  // arr[i] == i+1, arr[arr[i]-1]=arr[i]-1
  {
    for(int i=0;i<arr.length;++i) { if(arr[i] < 0) arr[i]=0; } // zero out all negatives;

    for(int i=0;i<arr.length;++i) { // mark the exist of ividx, arr[ividx-1] as negative, incl 0; 
      int ividx = Math.abs(arr[i]);
      if(ividx == 0 || ividx > arr.length) { continue; } // skip, not a valid idx
      if(arr[ividx-1]==0) arr[ividx-1]=-ividx; // strictly mark to neg to indicate ividx exists;
      if(arr[ividx] > 0) arr[ividx]*=-1; // mark to neg indicating ividx exists. No-ops if prev ividx already marked.
    }
    for(int i=0;i<arr.length;++i) { if(arr[i] >= 0) return i+1;}
  }
  { // bucket[i]=i+1, swap arr[i] as ividx until arr[ividx-1] is ividx invariant enforced. No mark arr[i] to neg !!
    for(int i=0;i<arr.length;++i) {
      if(arr[i] <= 0 || arr[i] > arr.length) { continue; } // skip i, out of range.
      while(arr[i] > 0 && arr[i] <= arr.length && arr[arr[i]-1] != arr[i]) {
        swap(arr, i, arr[i]-1);  // swap idx=(arr[i]-1) sets arr[idx-1] set to idx;
      }
    }
    for(int i=0;i<arr.length;++i) { if(arr[i] != i+1) return i+1;}
  }
  return arr.length+1;
}
// arr[i] is the offset to arr[arr[i]]. arr[i] must be [0..n]; arr[arr[i]] can be anything.
int findDuplicate1(int[] nums) {
  int rep = 0;
  for(int i=0;i<nums.length;++i) {
    int idx = Math.abs(nums[i]);
    if (nums[idx] < 0) { rep = idx; break;}
    nums[idx] = -nums[idx];            
  }
  for(int i=0;i<nums.length;++i) {
    nums[i] = Math.abs(nums[i]);
  }
  return ret;
}
int findDuplicate(int[] arr){
  for(int i=0;i<arr.length;i++){
    if(arr[Math.abs(arr[i])-1] < 0) return Math.abs(arr[i])
    arr[Math.abs(arr[i])-1] *= -1;
  }
  return -1;
}

// longest dup substring: binary search max len from sz/2;
int sublenWin(String s, int winlen, int start) {
  Set<String> set = new HashSet<>();
  for(int i=start;i<=s.length()-winlen;i++) {
    String ss = s.substring(i, i+winlen);
    if(set.contains(ss)) { return i; } 
    else { set.add(ss); }
  }
  return -1;
}
String longestDupSubstring(String s) {
  int l=0,r=s.length()-1, maxlen=0, maxi=0;
  while(l<=r) {
    int m=l+(r-l)/2;
    int pos = sublenWin(s, m+1, 0);
    if(pos >= 0) { maxlen=m+1; maxi=pos; l=m+1; } 
    else { r=m-1; }
  }
  if(maxlen==0) return "";
  return s.substring(maxi, maxi+maxlen);
}

// max repetition num, bucket sort. change val to -1, dec on every reptition.
//     bucket sort, when arr[i] is neg, means it has correct pos. when hit again, inc neg.
//     when dup processed, set it to max num to avoid process it again.
def maxrepeat(arr):
  def swap(arr, i, j):     arr[i], arr[j] = arr[j], arr[i]
  for i in xrange(len(arr)):
    while arr[i] >= 0 and arr[i] != i and arr[i] != sys.maxint:
      v = arr[i]
      if arr[v] > 0:  // first time swapped to.
        swap(arr, i, v)
        arr[v] = -1
      else:           // target already swapped, it is repeat. incr repeat cnt.
        arr[v] -= 1
        arr[i] = sys.maxint  // i is repeat node. mark it.
    if arr[i] == i:
      arr[i] = -1
  return arr
// maxrepeat([1, 2, 2, 2, 0, 2, 0, 2, 3, 8, 0, 9, 2, 3])
def repmiss(arr):
  def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
  for i in xrange(len(arr)):
    while arr[i] > 0 and arr[i] != i+1:
      v = arr[i]-1
      if arr[v] > 0:
        swap(arr, i, v)
        arr[v] = -arr[v]  // flip to mark as present.
      else:               // arr[v-1] < 0:
        print "dup at ", i, arr
        break
  return arr
buckesort(new int[]{4,2,5,2,1});

""" bucket tab[arr[i]] for i,j diff by t, and idx i,j with k """
def containsNearbyAlmostDuplicate(self, nums, k, t):
  if t < 0: return False
  n = len(nums); d = {};  w = t + 1
  for i in xrange(n):
    m = nums[i] / w
    if m in d:   return True
    if m - 1 in d and abs(nums[i] - d[m - 1]) < w:  return True
    if m + 1 in d and abs(nums[i] - d[m + 1]) < w:  return True
    d[m] = nums[i]
    if i >= k: del d[nums[i - k] / w]
  return False

class Solution { // bucket sort. Each bucket tracks min/max.
  class Bucket { int min; int max; int left; int rite; int idx; }
  int maximumGap(int[] arr) {
    // width = (max-min)/n-1; width must take floor. buckets = n-1; // (max-min)/width
    // say 1,5,9, w=8/2=4; width=4, [1..5][6..10][11..15], buckets=9/5=2;
    // 1,3,6,9, 8/3=2, 1-3, 4-6, 7-9
    // width: 5/3=1, 1236, [1,2][3,4][5,6] num bucket=(max-min)/(width+1)
    // [l..l+width]; len=width+1; [l+i*(len), l+(i+1)*len-1]]; 
    // x's bucket idx: (x-min)/(width+1); when x=min+w, idx = 0; so /(w+1);
    int gmin=Integer.MAX_VALUE, gmax=Integer.MIN_VALUE, sz=arr.length;
    for(int i=0;i<arr.length;i++) {
      gmin = Math.min(gmin, arr[i]);
      gmax = Math.max(gmax, arr[i]);
    }
    if(sz==1) return 0;
    int width = (gmax-gmin)/(sz-1);
    int nbuckets = sz-1; // (int)Math.ceil((gmax-gmin+1)/(width+1));
    Bucket[] buckets = new Bucket[nbuckets];
    for(int i=0;i<nbuckets;i++) {
      buckets[i] = new Bucket();
      buckets[i].min = Integer.MAX_VALUE;
      buckets[i].max = Integer.MIN_VALUE;
      buckets[i].left = gmin+i*width + 1;
      buckets[i].rite = buckets[i].left + width;
    }
    for(int i=0;i<sz;i++) {
      int bidx = (arr[i]-gmin)/(width+1);
      buckets[bidx].min = Math.min(buckets[bidx].min, arr[i]);
      buckets[bidx].max = Math.max(buckets[bidx].max, arr[i]);
    }
    int curmin = buckets[0].min, curmax=buckets[0].max, gap=curmax-curmin;
    for(int i=1;i<nbuckets;i++) {
      if(buckets[i].min != Integer.MAX_VALUE) {
        gap = Math.max(gap, buckets[i].min-curmax);
      }
      if(buckets[i].max != Integer.MIN_VALUE) {
        curmax = buckets[i].max;
      }
    }
    return gap;
  }
}

// - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - -- - - 
// Merge sort to convert N^2 to Nlgn. all pair(i,j) comp, skipping some during merge.
// inverse count(right smaller) with merge sort. sort A, instead of sorted another indirect lookup, 
// Rank[i] stored idx of Arr, an indirection layer. not the actual A[i]. max =/= A[n] but A[rank[n]]
// - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - -- - - 
class MergeInverse {
  static void MergeInverse2(int[] arr, int left, int rite, int[] sorted, int[] inverse) {
    if (left == rite) { return; }
    int mid = left + (rite-left)/2;
    int leftlen = mid-left+1, ritelen=rite-mid;
    int[] segment_sorted = new int[leftlen+ritelen];
    MergeInverse2(arr, left, mid, sorted, inverse);
    MergeInverse2(arr, mid+1, rite, sorted, inverse);
    // each left rite subary is sorted, iterate via sorted[0..leftlen], use l,r from 0;
    // inverse[] must on sorted order, hence inverse[sorted[l]]. left sub[9,5] is itered as [5,9]
    int lidx = left, ridx = mid+1, k = 0;
    while(lidx <= mid && ridx <= rite) { }
    while(lidx <= mid) {}
    while(ridx <= rite) {}
    for(int l=0,r=0; l<leftlen || r<ritelen;) {
      if (l==leftlen) { segment_sorted[l+r] = mid+1+r; r++; } 
      else if (r==ritelen) {
        segment_sorted[l+r] = left+l;
        inverse[sorted[left+l]] += r;
        l++;
      } else if (arr[sorted[left+l]] < arr[sorted[mid+1+r]]) {
        segment_sorted[l+r] = left+l;
        inverse[sorted[left+l]] += r;
        l++;
      } else {
        segment_sorted[l+r] = mid+1+r;
        r++;
      }
    }
    for(int i=0;i<leftlen+ritelen;i++) {
      sorted[left+i] = segment_sorted[i];
    }
  }
}

// count the number of range sums of an ary that lie in [lower, upper] inclusive.
// 1. map prefixsum[i] to its counts. from n..0, find all prefix[i]-[lower..upper].
// 2. any prefix pair(i,j) is a range. count all pairs within low, upper. N^2.
// 3. merge sort to compare any pair. some pairs are skipped during merge sorted halves.
void merge(long[] prefix, int la, int lb, int rb, long[] staging) {
  int ra=lb-1, si=0;
  while(la <= ra && lb <= rb) {
    if(prefix[la] <= prefix[lb]) {staging[si++]=prefix[la++];}
    else {staging[si++]=prefix[lb++];}
  }
  while(la<=ra) {staging[si++]=prefix[la++];}
  while(lb<=rb) {staging[si++]=prefix[lb++];}
}
int mergeSort(long[] prefix, int l, int r, int lo, int up) {
  int tot=0;
  if(l==r) { if(lo <= prefix[l] && prefix[l] <= up) return 1; else return 0;}
  long[] staging = new long[r-l+1];
  int m=l+(r-l)/2;
  int ltot = mergeSort(prefix, l, m, lo, up);
  int rtot = mergeSort(prefix, m+1, r, lo, up);
  tot += (ltot+rtot);
  
  int rl=m+1,rr=m+1;
  for(int i=l;i<=m;i++) { // compare pair(l, rl..rr); count matched pair(l,r)
    while(rl<=r && prefix[rl] - prefix[i] < lo) rl++;  // rl parked to the first bigger than lo, inclusive.
    while(rr<=r && prefix[rr] - prefix[i] <= up) rr++; // rr parked to the first bigger than up, exclusive.
    tot += rr-rl; // after loop break, rr parked over up, pointed to the end, not inclusive.
  }
  merge(prefix, l, m+1, r, staging); // shall populate staging during the merge loop.
  for(int i=l;i<=r;i++) { prefix[i] = staging[i-l]; }
  return tot;
}
int countRangeSum(int[] arr, int lo, int up) {
  long[] prefix = new long[arr.length];
  for(int i=0;i<arr.length;++i) {
    prefix[i] = (i==0 ? 0 : prefix[i-1]) + arr[i];
  }
  return mergeSort(prefix, 0, arr.length-1, lo, up);
}  

// ------------------------------------------------------------------------------------
// Graph traverse: visited map[time], Tree no need visited. BFS longest path.
// Q variations: stk, Heap, dist by mx_weight, path_min; A* f(n) = g(n) + h(n);
// Grid to DP: Row[i,j] = Max(prevRow[i,j], max(prev[i, 0..j] foreach col j);
// LC 2127: dep network. find a cycle with the longest path. Topsort prereq: DAG.
// ------------------------------------------------------------------------------------
/**
 * parent[], entry_time[], discovered[], processed[], reachable_ancestor[];
 * 1. travesal; bfs update global var. dfs recursive aggregate context and ret;
 * 2. visit order matters, (parent > child), topology sort
 * 3. process_node_enter/exit, track invariant on enter/exit. discved/visited
 * 4. process edge: tree_edge, back_edge, cross_edge;
 *    a) not visited, process edge, set parent. dfs children
 *    b) in stack, process edge. not parent, no dfs.
 *    c) visited, must be directed graph, otherwise, cycle detected, not a DAG.
 * 5. update parent state when dfs down; earliest_reachable_anecestor;
 */
dfs(root, parent, context={disc, visited, parents, entry_time}, collector):
  if not root: return
  process_vertex_early(v) // { discved = true; entry_time=now}
  rank.root = gRanks++   // set rank global track.
  for c in root.children:
    if not discved[c]:
      discved[c]=true; parents[c] = root; 
      process_edge(root, c, tree_edge)  // tree edge before dfs.
      dfs(c, parent=root)    // then recur dfs
    else if !visited[c] or directed:
      process_edge(root, c, back_edge)  // back edge, cycle.
    else if visited[c]:      // must be DAG, Forward edge.
      process_edge(root, c, cross_edge) // 
  visited[root] = true
  process_vertex_late(v);
def edge_classification(x, y): // edge x->y
  if(parent[y]==x) return Tree;
  if(discovered[y] && !processed[y]) return BACK; // an ancestor that prev seen;
  if(processed[y] && entry_time[y] > entry_time[x]) return FORWARD; // x->z->y, x->y
  if(processed[y] && entry_time[y] < entry_time[x]) return CROSS; // sort of back edge.
// dfs(v) process v's edges; update v's reachable_ancestor;
def process_edge(x, y): // 
  class = edge_classification(x,y) // 
  if (class == TREE) tree_out_degree[x] = tree_out_degree[x] + 1
  if ((class == BACK) && (parent[x] != y)) // x->y is a back edge; exclude parent->x;
    if (entry_time[y] < entry_time[reachable_ancestor[x]] // update x's reachable_ancestor. 
      reachable_ancestor[x] = y; // Y is the earliest ancestor reachable from X.
// all children explored, recur ret. update parent's and cut node reachable_ancestor;
 def process_vertex_late(v):
  int time_v;    // earliest reachable time for v;
  int time_parent;  // earliest reachable time for parent[v]
  if (parent[v] < 1)   // no need post/late processing for root.
    if (tree_out_degree[v] > 1)
      return;
  root = (parent[parent[v]] < 1); /* is parent[v] the root? */
  if ((reachable_ancestor[v] == parent[v]) && (!root))
    printf("parent articulation vertex: %d \n",parent[v]);
  if (reachable_ancestor[v] == v) // parent[v] is a cut node as no back edge from v;
    printf("bridge articulation vertex: %d \n",parent[v]);
    if (tree_out_degree[v] > 0) // if v is not a leaf, v is also a cut node;
      printf("bridge articulation vertex: %d \n",v);
  // after dfs(v) done, update v's parent's reachable_ancestor with v's reachable_ancestor;
  time_v = entry_time[reachable_ancestor[v]];
  time_parent = entry_time[reachable_ancestor[parent[v]]];
  if (time_v < time_parent)  // update v's parent's reachable_anecestor based on v's.
    reachable_ancestor[parent[v]] = reachable_ancestor[v];
  
def dfs_two_color(G, start):
  def process_edge(from, to):
    if color[from] == color[to]:
      bipartite = false
      return
    color[to] = complement(color[from])

def topology_sort(G):
  sorted = []
  for v in g where visited[v] == false:
    dfs(v)
  def process_vertex_late(v):
    sorted.append(v) // pre-requisit at the head.

void dfsClone(Node node, Node parent, Map<Node, Node> clone) {
  if(node == null) return;
  Node node_clone = new Node(node.val);
  clone.put(node, node_clone);
  if(parent != null) { clone.get(parent).neighbors.add(node_clone); }

  for(Node child : node.neighbors) {
    if(!clone.containsKey(child)) {
      dfsClone(child, node, clone);
    } else {
      clone.get(node).neighbors.add(clone.get(child));
    }
  }
}
static Node flatten(Node root) {
  Stack<Node> stk = new Stack<>(); // stk rite before descend to left;
  // LoopInvar: stk left, rite <= left, mov to rite; if no rite, pop stk.
  while (root != null || stk.size() > 0 ) {
    if (root != null) {
      if (root.rite != null) { stk.add(root.rite); }   // set aside rite to stk, will come back later.
      root.rite = root.left;
      root = root.rite;           // descend
    } 
    else { root = stk.pop(); }
  }
  return root;
}
// Nested loop version. Inner loop to pop stk until cur.rite not null.
static Node flatten(Node root) {
  while(root != null) {
    if (root.left) { stk.add(root); root = root.left; continue; }
    else {
      loop(poping stk until find the cur whose rite not null) {
        visit(root);
        if (root.rite) { root = root.rite; break;}
        else {
          if (!stk.empty()) { cur = stk.pop(); continue; }
          else { root = null;  break; }
        }
      }
    }
  }
}

// find the root of a Min Heigth Tree. Remove leaves layer by layer.
// MHT can only have at most 2 roots. 3 roots folds to 1 root.
List<Integer> MinHeightBfsRemoveLeaves(Map<Integer, Node> nodes) {
  Deque<Node> q = new ArrayDeque<>();
  int remains = nodes.size();
  while(remains > 2) { // MinHeightTree can only have at most 2 root. 3 roots will fold to 1 root.
    for(Node n : nodes.values()) {
      if(!n.visited && n.removedChildren.size()+1==n.children.size()) {
        q.addLast(n);
      }
    }
    while(q.size() > 0) {
      Node hd = q.removeFirst();  hd.visited = true; remains--;
      for(Node c : hd.children) { 
        if(!c.visited) { c.removedChildren.add(hd); }  }
    }
  }
  List<Integer> roots = new ArrayList<>();
  for(Node n : nodes.values()) { if(!n.visited) roots.add(n.id);}
  return roots;
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// TrieNode, stores edges in a map keyed prefix string, vs Node[26]
// { Map<PrefixString,Edge> edges; suffix_idx} Edge {startidx,endidx,toNode}; Edge(st,ed) is label.
// For abcd, abce, no need to create node b as ab is the common prefix. split abc edge and create node b when adding abx.
// Split edge of parent node, the new edge point to the new TriNode. root->leaf path: arr[suffix_idx:end]
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
class TrieNode { // a map of edges keyed by prefix string; split edge when prefix branch.
  class Edge { int start_idx, end_idx, TriNode to_node; }
  Map<PrefixString, Edge> out_edges; // just a map of edges in each TrieNode.
  TrieNode[] children;   // children[i] => new TrieNode
  String leftword;           // leaf word
  TrieNode() {  children = new TrieNode[26]; }
}
TriNode InsertString(TriNode root, String s, int st) {
  int ed=s.length(); // split s by the longest common prefix edge vs s. s suffix a new edge; 
  for(;ed>st;ed--) { if(root.out_edges.containsKey(s.substring(st, ed)) break; )}
  if(ed==st) { 
    TriNode leaf = new TriNode(s, st, ed); Edge e = new Edge(st, ed, leaf); 
    root.out_edges.put(s.substring(st, ed), e); return leaf; } 
  else { Edge e = root.out_edges.get(s.substring(st,ed+1)); 
    return InsertString(e.to_node, s, ed);
  }
}
TriNode build(String[] dict) {
  TriNode root = new TriNode();
  for(int i=0;i<dict.length;i++) {
    TriNode curnode = root; String dword = dict[i];
    for(int j=0;j<dword.length();j++) {
      InsertString(root, s, 0);
    }
  }
}
String replaceWords(List<Strinf> roots, String sentence) {
  TrieNode trieRoot = new TrieNode();
  for (String root: roots) {
    TrieNode cur = trieRoot;
    for (char letter: root.toCharArray()) {
      if (cur.children[letter - 'a'] == null) {  # create child branch
        cur.children[letter - 'a'] = new TrieNode(); 
      }  
      cur = cur.children[letter - 'a'];
    }
    cur.leftword = root;   // word set at leaf level;
  }

  StringBuilder ans = new StringBuilder();
  for (String word: sentence.split("\\s+")) {
    if (ans.length() > 0) ans.append(" ");

    TrieNode cur = trieRoot;
    for (char letter: word.toCharArray()) {
      if (cur.children[letter - 'a'] == null || cur.word != null) break; // this leaf
      cur = cur.children[letter - 'a'];
    }
    ans.append(cur.word != null ? cur.word : word);
  }
  return ans.toString();
}
// build a trie using words in the dict.
String replaceWords(List<String> roots, String sentence) {
  TrieNode trieRoot = new TrieNode();
  // insert all words into trie, iterative, 2 for loops. 
  for (String word: roots) {
    TrieNode curnode = trieRoot;
    for (char letter: word.toCharArray()) {
      if (curnode.children[letter - 'a'] == null) cur.children[letter - 'a'] = new TrieNode();
      curnode = curnode.children[letter - 'a'];
    }
    cur.word = word;
  }
  StringBuilder ans = new StringBuilder();
  // for each word, for each letter, find trieNode.
  for (String word : sentence.split("\\s+")) {
    if (ans.length() > 0)  ans.append(" ");
    TrieNode cur = trieRoot;
    for (char letter: word.toCharArray()) {
        if (cur.children[letter - 'a'] == null || cur.word != null)  // no children, or leaf
            break;
        cur = cur.children[letter - 'a'];
    }
    ans.append(cur.word != null ? cur.word : word);
  }
  return ans.toString();
}

// map sum, a map backed by Trie, dfs to collect aggregations of all children
class MapSum {
  TrieNode insert(TriNode root, String word) {
    TrieNode cur = root;
    for (char c : word.toCharArray()) {
      if (cur.children[c] == null) { cur.children[c] = new TrieNode();}
      cur = cur.children[c];
    }
    cur.isWord = true;
    cur.word = word;
  }
  int sum(String prefix) {
    TrieNode curr = root;
	  for (char c : prefix.toCharArray()) {
	    cur = curr.children.get(c);
	    if (cur == null) { return 0; }
    }
    return dfs(curr);
  }
  private int dfs(TrieNode root) {  // return sum.
    int sum = 0;
    for (char c : root.children.keySet()) {    // dfs all children
      sum += dfs(root.children.get(c));
    }
    return sum + root.value;
  }
}
int minExtraChar(String s, String[] dictionary) {
  Set<String> dset = new HashSet<>();
  for(int i=0;i<dictionary.length;i++) dset.add(dictionary[i]);
  int[] f=new int[s.length()]; // # of skips
  for(int i=0;i<s.length();i++){
    f[i] = 1 + (i>0 ? f[i-1] : 0);
    for(int j=i;j>=0;j--) {
      String jistr = s.substring(j, i+1);
      //if(suffix.contains(jistr)) { break;}
      if(dset.contains(s.substring(j,i+1))) {
        f[i] = Math.min(f[i], (j > 0 ? f[j-1] : 0));
      }
    }
  }
  return f[s.length()-1];
}

// segment sum(i..j) = PathSum[0..j]-[0..i];
int PathSum(TreeNode root, int target, long prefix, Map<Long, Integer> prefixSumMap) {
  if (root == null) return 0;
  int tot = 0; 
  long pathPrefix = prefix + root.val;
  tot += prefixSumMap.getOrDefault(pathPrefix - target, 0);
  prefixSumMap.compute(pathPrefix, (k,v) -> v == null ? 1 : v+1);
  tot += PathSum(root.left, target, pathPrefix, prefixSumMap);
  tot += PathSum(root.right, target, pathPrefix, prefixSumMap);
  prefixSumMap.compute(pathPrefix, (k,v) -> v = v-1);
  prefixSumMap.remove(pathPrefix, 0);
  return tot;
}
// Leaf Nodes Pairs within dist;
Pairs dfs(TreeNode root, int dist) {
  Pairs retpairs = new Pairs(), lpairs=null, rpairs=null;
  if(root.left == null && root.right==null) { retpairs.leaves.add(0); return retpairs;}
  if(root.left != null) {lpairs = dfs(root.left, dist); retpairs.pairs += lpairs.pairs;}
  if(root.right != null) {rpairs = dfs(root.right, dist); retpairs.pairs += rpairs.pairs;}
  if(lpairs != null && lpairs.leaves != null && rpairs!=null && rpairs.leaves !=null) {
    for(int l : lpairs.leaves) { // iter left and rite and sum all pairs within dist;
      if(l+1 < dist) retpairs.leaves.add(l+1);
      for(int r : rpairs.leaves) {
        if(r+1 < dist) retpairs.leaves.add(r+1);
        if(l+r <= dist) retpairs.pairs++;
      }
    }
  }
  return retpairs;
}

// two errors: 1. did not max subtree incl/excl. 2. shall max incl/excl of the subtree, max(l.[incl|excl])+max(r.[incl/excl]) vs. max([l|r].incl, [l|r].excl)
class NodeMax {
  int overallmax;  // max of incl_root and excl root;
  int incl_root; // defintely incl root, child must excl root; +child.excl_root
  int excl_root; // definitely excl root; max of child's overallmax;
}
NodeMax dfs(TreeNode root, TreeNode parent) {
  NodeMax s = new NodeMax();
  if(root==null) return s;
  NodeMax lchild = dfs(root.left, root);
  NodeMax rchild = dfs(root.right, root);
  s.incl_root = root.val + lchild.excl_root + rchild.excl_root;
  s.excl_root = lchild.overallmax + rchild.overallmax;
  s.overallmax = Math.max(lchild.overallmax+rchild.overallmax, root.val + lchild.excl_root + rchild.excl_root);
  return s;
}
int rob(TreeNode root) {
  NodeMax max = dfs(root, null);
  // return Math.max(max.incl_root, max.excl_root);
  return max.overallmax;
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - -------
// Tree dfs: int dfs(root, parent, prefix, collector); or Tree iter while(cur != null || !stk.empty())
// ret tot upon each dfs child done so tot is aggregated at root;
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - -------
boolean isValidBST(TreeNode cur, long lo, long hi) { // lo/hi replace the need to detect cur is left/rite.
  if(cur==null) return true;
  if(cur.val >= hi || cur.val <= lo) return false; // cur must between lo < cur < hi
  // reduce hi when recur left, expand lo when recur right.
  return isValidBST(cur.left, cur, lo, cur.val) && isValidBST(cur.right, cur, cur.val, hi);
}
boolean isValidBST(TreeNode root) {
  return recur(root.left, root, Long.MIN_VALUE, root.val) && recur(root.right, root, root.val, Long.MAX_VALUE);
}
def common_parent(root, n1, n2):
  if not root: return None
  if root is n1 or root is n2: return root
  left = common_parent(root.left, n1, n2)
  rite = common_parent(root.rite, n1, n2)
  return root if left and rite else return left if left else rite.

// max of left subtree, or first parent whose right subtree has me
def findPrecessor(node):
  if node.left:                return findLeftMax(node.left)
  while root.val > node.val:   root = root.left  // smaller preced must be in left tree
  while findRightMin(root) != node:  root = root.right
  return root

String serialize(TreeNode root) {
  StringBuilder sb = new StringBuilder();
  if(root == null) {sb.append("$,"); return sb.toString();}
  sb.append(root.val + ",");
  sb.append(serialize(root.left));
  sb.append(serialize(root.right));
  return sb.toString();
}

// tab[i] = num of bst for ary 1..i, tab[i] += tab[k]*tab[i-k] '''
def numBST(n):
  tab = [0]*(n+1)
  tab[0] = tab[1] = 1
  for i in xrange(2,n):
    for k in xrange(1,i+1):
      tab[i] += tab[k-1]*tab[i-k]   # each lchild combine with each rchild. lxr
  return tab[n]

NodeIdx recur(String[] arr, int idx) {
  {
    if(idx >= arr.length) return new NodeIdx(null, idx);
    String nstr = arr[idx];
    if(nstr.equals("$")) { return new NodeIdx(null, idx);}
    TreeNode n = new TreeNode(Integer.parseInt(nstr));
    NodeIdx left = recur(arr, idx+1);
    n.left = left.node;
    NodeIdx rite = recur(arr, left.idx+1);
    n.right = rite.node;
    return new NodeIdx(n, rite.idx);
  }
  {
    Deque<WrapNode> stk = new ArrayDeque<>();
    WrapNode root = null;
    for(int i=0;i<arr.length;++i) {
      if(arr[i].length() == 0) { continue; }
      if(arr[i].equals("$")) {
        if(stk.size() == 0) return null;
        if(!stk.peekLast().leftdone) {
          stk.peekLast().leftdone = true;
        } else if(!stk.peekLast().ritedone) {
          stk.peekLast().ritedone = true;
        }
        // recursively poping stk.
        while(stk.size() > 0 && stk.peekLast().leftdone && stk.peekLast().ritedone) {
          stk.removeLast();
        }
      } else {
        WrapNode n = new WrapNode(new TreeNode(Integer.parseInt(arr[i])));
        if(stk.size() == 0 ) {root = n; stk.addLast(n); continue; }
        if(!stk.peekLast().leftdone) {
          stk.peekLast().node.left = n.node;
          stk.peekLast().leftdone = true;
        } else if(!stk.peekLast().ritedone) {
          stk.peekLast().node.right = n.node;
          stk.peekLast().ritedone = true;
        } else { System.out.println("not possbile " + nstr + stk.peekLast().node.val); return null; } 
        stk.addLast(n);
      }
    }
    return root.node;
  }
}

class Node {int value, rank;  Node left, rite; }
static Node morrisInOrder(Node root) {
  Node pre = null, cur = root;
  Stack<Node> output = new Stack<>();
  while (cur != null) {  // while cur not null if not recursion.
    while(cur.left != null) { // descend to left one by one after fixing left's ritemost back to root.
      tmp = cur.left;
      // desc to rite most of the left to thread back to root.
      while(tmp.rite != null && tmp.rite != cur) { tmp = tmp.rite; }
      if(tmp.rite == null)  tmp.rite=cur; else tmp.rite = null;
      // after fixing left's ritemost. descend to left
      pre = cur; cur=cur.left;
    }
    output.add(cur);  // visit cur after left is leaf. follow its thread back to move to parent.
    pre = cur;
    cur = cur.rite;
  }
  return output;
}

// find pair node in bst tree sum to a given value, lgn.
// first, lseek to lmin, rseek to rmax, stk along the way, then back one by one.
// down to both ends(lstop/lcur), with stk to rem parent, then mov rite or left """
def BstNodePair(root, target):
  lstk,rstd = deque(),deque()  # use lgn stk to store tree path parent
  lcur,rcur = root,root; lstop,rstop = False,False
  while True:
    while not lstop:
      if lcur: lstk.append(lcur); lcur = lcur.left;
      else:
        if not len(lstk): lstop = 1;
        else: lcur = lstk.pop(); lval=lcur; lcur=lcur.rite; lstop=1;
    while not rstop:
      if rcur: rstk.append(rcur); rcur = rcur.rite;
      else:
          if not len(rstk): rstop = 1;
          else: rcur = lstk.pop(); rval = rcur; rcur = rcur.left; rstop = 1
    if lval + rval == target:
      return lcur,rcur
    if lval + rval > target: rstop = False
    else:  lstop = False
  return lcur,rcur

// recur done collects # of nodes in subtrees. 
int[] kthSmallestRecur(TreeNode root, int k) {
  if(root == null) return new int[]{0, -1};
  int[] left = recur(root.left, k);
  if(left[1] >= 0) return left; // found
  if(left[0]+1 == k) return new int[]{left[0]+1, root.val};
  int[] rite = recur(root.right, k-left[0]-1);
  if(rite[1] >= 0 ) return rite;
  else return new int[]{left[0]+rite[0]+1, -1}; // return pair [num_children, found]
}
int kthSmallestLoop(TreeNode root, int k) {
  Deque<TreeNode> stk = new ArrayDeque<>();
  TreeNode cur = root;
  while(cur != null || stk.size() > 0) {
    while(cur != null) { stk.addLast(cur); cur = cur.left; continue;} // all the way to stk left child;
    cur = stk.removeLast(); // no more left, pop stk and visit cur;
    --k;
    if(k==0) break;
    cur = cur.right; // after visit, move to right.
  }
  return cur.val;
}

// given a root, and an ArrayDeque (initial empty). iterative. root is not pushed to stack at the begining.
List<Integer> preorderTraversal(TreeNode root) {
  List<Integer> result = new ArrayList<>();
  Deque<TreeNode> stk = new ArrayDeque<>();
  TreeNode cur = root;
  while (!stk.isEmpty() || cur != null) {
    while(cur != null) {
      stk.push(cur);
      result.add(cur.val);  // visit(cur); // Pre-order
      cur = cur.left;
    }
    
    TreeNode node = stk.pop();
    visit(cur);  // In-order
    cur = node.right;   // after popping from the stk, alway go rite.
  }
}

// just reverse preorder of root, rite, left.
// a visited map. pop, if not visited, visit, push back in, and push children; visited, post order process; 
List<Integer> postorderTraversalTwoStk(TreeNode root) {
  Deque<Integer> result = new ArrayDeque<>();
  Deque<TreeNode> stack = new ArrayDeque<>();
  TreeNode cur = root;
  while(!stack.isEmpty() || cur != null) {
    while(cur != null) {
        stack.offerLast(cur);
        result.offerFirst(cur.val);  
        cur = cur.right;               // go right first. Reverse the process of preorder
    }
    // cur is null via prev parent.rite; pop parent from stk top. recur parent.left 
    TreeNode node = stack.pop();
    cur = node.left; 
  }
  return result;
}

// ------------------------------------------------------------------------
// sequential consume next char to parse text into Expr. new Stk per Recur;
// ------------------------------------------------------------------------
boolean recurMatch(String s, int start, Map<String, Integer> dict) {
  if(start == s.length()) return true;
  int j = start;
  while(j < s.length() && s.charAt(j) == s.charAt(start)) { ++j;} // skip dups
  if(j-start > 4 && skipDups(s.charAt(start), j-start, dict)) {
    System.out.println("skipping " + s.charAt(start) + " to " + j);
    start = j;
  }
  if(start == s.length()) return true;
  for(int k=start;k<=s.length();++k) {
    if(dict.containsKey(s.substring(start, k))) {
      if(recurMatch(s, k, dict)) return true;
    }
  }
  // Map is not mean for iteration.
  // for(Map.Entry<String, Integer> e : dict.entrySet()) {
  //   if(start+e.getKey().length() <= s.length() && 
  //       e.getKey().equals(s.substring(start, start+e.getKey().length()))) {
  //     if(match(s, start+e.getKey().length(), dict)) return true;
  //   }
  // }
  return false;
}

// recur with substring, or recur with soffset, poffset.
static boolean regMatch(String s, String p) {
    int psz = p.length(), ssz = s.length();
    // check pattern string first.
    if (psz == 0) {                       return ssz == 0; }
    if (psz == 1 && p.charAt(0) == '*') { return true; }
    if (ssz == 0)                         return false;

    char shd = s.charAt(0), phd = p.charAt(0);
    if (phd != '*') {   // not *, easiest first
        if (phd != '?' && shd != phd) { return false; }
        return regMatch(s.substring(1), p.substring(1));    # dfs recur
    }
    // now phd is *, recur to match all 0..n substring, for loop or while loop ?
    for (int i=0; i < s.length(); i++) {
        if (regMatch(s.substring(i), p.substring(1))) { return true; }
    }
    return false;
}
String s = "adceb", p = "*a*b";
System.out.println("matching "  + regMatch(s, p));

// Recur Ary, Empty ary(start=end, idx=a.size) vs single element(start=end)
// Loop Range: idx[0..sz], idx=size, empty substr is valid input.
// Loop body applies to range; ensure idx valid when loop ends.
bool RegMatch(std::string_view txt, int ti, std::string_view p, int pi) {
  // Check Exit boundary first.
  if (pi == p.size()) { return ti == txt.size(); }
  if (pi + 1 == p.size()) {  // last item to match.
      return (ti + 1 == txt.size() && (p.at(pi) == '.' || txt.at(ti) == p.at(pi)));
  }
  if (p.at(pi+1)) != '*') {
    return (txt.at(ti) == p.at(pi) || p.at(pi) == '.') && RegMatch(txt, ti+1, p, pi+1)
  } else {  // pattern is X*, * match head [0..N] times.
      // if head not equal, consume p, match 0 times.
      if (p.at(pi) != '.' && p.at(pi) != txt.at(ti)) { return RegMatch(txt, ti, p, pi+2); }
      // Loop inv: inside loop body: ti+k points to repeated char. outside, k points to next or end;  
      int k = 0;  // Loop Range [ti+k, ti+k < size] 
      // while (ti + k < txt.size() && txt.at(ti+k) == txt.at(ti)) {
      //     if (RegMatch(txt, ti+k, p, pi+2)) return true;
      //     k++;
      // }
      for(int k=0; ti+k < txt.size() && txt.at(ti)==txt.at(ti+k); k++) {
          if (RegMatch(txt, ti+k, p, pi+2)) return true;
      }
      for(int repeat_idx=ti; repeat_idx < txt.size() && txt.at(repeat_idx) == txt.at(ti); repeat_idx++) {
        if (RegMatch(txt, repeat_idx, p, pi+1)) return true;
      }
      // Loop Exit: head not repeat, or the end of txt. repeat_idx= next_head || txt.size(); 
      return (RegMatch(txt, repeat_idx, p, pi+2));
  }
};

static boolean RegMatch(char[] arr, int i, char[] pat, int j) {
  if ( i == arr.length) {
    if (j == pat.length || (j+2 == pat.length && pat[j+1] == '*')) { return true;}
    return false;
  } else if ( j == pat.length) { return false; }
  // both i and j are valid
  if (j+1 < pat.length && pat[j+1] == '*') {
    if (pat[j] == '.' || arr[i] == pat[j]) {
      int k = i;
      for(; k < arr.length && arr[i] == arr[k]; k++) {
        if (RegMatch(arr, k, pat, j+2)) return true;
      }
      // when out, k == arr.length or arr[k] != arr[i]
      return RegMatch(arr, k, pat, j+2);
    }
    return RegMatch(arr, i, pat, j+2);
  } else {
    if (pat[j] == '.' || arr[i] == pat[j]) { return RegMatch(arr, i+1, pat, j+1);}
    return false;
  }
}

// 1, If p.charAt(j) == s.charAt(i) :  dp[i][j] = dp[i-1][j-1];
// 2, If p.charAt(j) == '.' : dp[i][j] = dp[i-1][j-1];
// 3, If p.charAt(j) == '*': here are two sub conditions:
//     1  if p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2]  //in this case, a* only counts as empty
//     2  if p.charAt(i-1) == s.charAt(i) or p.charAt(i-1) == '.':
//                 dp[i][j] = dp[i-1][j]    //in this case, a* counts as multiple a 
//               or dp[i][j] = dp[i][j-1]   // in this case, a* counts as single a
//               or dp[i][j] = dp[i][j-2]   // in this case, a* counts as empty
boolean regMatch(String s, String p) {
  int ssz = s.length(), psz = p.length();
  int count = 0;
  // quick check whether there is star.
  for (int i = 0; i < psz; i++) { if (p.charAt(i) == '*') count++; }  # many wildcard as 1
  if (count==0 && ssz != psz) return false;
  else if (psz - count > ssz) return false;
  
  boolean[] match = new boolean[ssz+1];
  match[0] = true;
  for (int i = 0; i < ssz; i++) {    match[i+1] = false;   }
  // iterate all pattern chars
  for (int i = 0; i < psz; i++) {
    if (p.charAt(i) == '*') {
      for (int j = 0; j < ssz; j++) {
          match[j+1] = match[j] || match[j+1]; 
      }
    } else {
      // from ssz end back, because prev dfs already calculated.
      for (int j = ssz-1; j >= 0; j--) {
          match[j+1] = (p.charAt(i) == '?' || p.charAt(i) == s.charAt(j)) && match[j];
      }
      match[0] = false;
    }
  }
  return match[ssz];
}
// dp version, https://mp.weixin.qq.com/s/ZoytuPt5dfP5pMODbuKnCQ
// dp[0][0] = 1; dp[i][j] = dp[i - 1][j - 1]; OR dp[i][j] |= dp[i][j - 2]; OR dp[i][j] |= dp[i - 1][j];
// check pattern exhausted boundary first. check non wildcard first. for wildcard, while p=s, s++, Skip *, p+=2;
static boolean regmatch(char[] s, char[] p, int soff, int poff) {
  int ssz = s.length, psz = p.length;
  if (poff == psz) {       return soff == ssz;}
  if (p[poff+1] != '*') {  return (s[soff] == p[poff] || p[poff] == '.') && regmatch(s, p, soff+1, poff+1); }
  // pattern is wildcard, try each recursive
  while (s[soff] == p[poff] || p[poff] == '.') {
    if (regmatch(s, p, soff, poff+2)) { return true; }
    soff += 1;  // out-of-bound soff
  }
}

//https://github.com/labuladong/fucking-algorithm/blob/master/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%B3%BB%E5%88%97/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E4%B9%8B%E6%AD%A3%E5%88%99%E8%A1%A8%E8%BE%BE.md
bool dp(string& s, int i, string& p, int j) {
  if (s[i] == p[j] || p[j] == '.') {  // 匹配，检查 j+1 ?= *
    if (j < p.size() - 1 && p[j + 1] == '*') {
        // 1.1 通配符匹配 0 次或多次
        return dp(s, i, p, j + 2) || dp(s, i + 1, p, j);
    } else {
        // 1.2 常规匹配 1 次
        return dp(s, i + 1, p, j + 1);
    }
  } else {  // 不匹配, 检查 j+1 ?= *, 匹配 0 次 
    if (j < p.size() - 1 && p[j + 1] == '*') {
        // 2.1 通配符匹配 0 次
        return dp(s, i, p, j + 2);
    } else {
        // 2.2 无法继续匹配
        return false;
    }
  }
}

// (2*(3-4))*5 != 2*((3-4)*5) all possible groupings; bind with pre recur not all possible; 
// bind left/rite not work for all combinations; dfs(pos, pre_v, pre_op, s); 
List<Integer> diffWaysToCompute(int l, int r, String s, List<Integer>[][] cache) {
  List<Integer>[][] dp = new ArrayList[n][n];
  for (int length = 3; length <= n; length++) {
    for (int start = 0; start + length - 1 < n; start++) {
      processSubexpression(expression, dp, start, start + length - 1);
    }
  }
  List<Integer> res = new ArrayList<>();
  if(cache[l][r] != null) {return cache[l][r]; }
  for(int k=l;k<=r;k++) {
    if(isOp(s, k)) {
      for(Integer lh : diffWaysToCompute(l, k-1, s)) {
        for(Integer rh : diffWaysToCompute(k+1, r, s)) {
          res.add(eval(s.charAt(k), lh, rh));
        }
      }
    }
  }
  if(res.size()==0) { res.add(Integer.parseInt(s.substring(l, r+1))); }
  return res;
}
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Pratt parser, while(hd) { eval(hd, recur(hd) }
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// // 2 + 3 * 4 - 5 /（6 + 7）/ 8 - 9; recur stop arg is __parent_min_bp__. recur ret when cur op < parent_op, bind to parent.
// fn Pratt(lexer, parent_op_bindpower) -> S { // set to min of all op, will consum all lexer tokens. 
//   // steps: { Prep parent, peek op, loop until op.lbp < min}
//   cur_lhs = match(lexer.next());
//   cur_op = lexer.peek();  // look ahead peek the next op
//   while (cur_op.lbp > parent_op_bindpower) {  // cur_lhs bind to cur_op, continue;
//     lexer.pop();  // if recur, consume head
//     rhs = recur(parent_op_bindpower = cur_op.lbp);  // child recur ret, parent dfs done.
//     lhs = cons(cur_op, lhs, rhs); // child dfs done as a new TreeNode.
//     cur_op = lexer.peek();  // look ahead prep loop condition.
//   }
//   return lhs;
// }
// ({parent: null}, 2, +,  ? ) 
//          ({parent: +}, 3, *, ? )
//                 ({parent: *}, 4, '+, ? )
//          ({parent: +}, (3*4), '-, ? )
//     
SExpr ParseInternal(src, idx, parent_op) {
  lhs = src.charAt(idx);  // lexer.pop();
  cur_op = src[idx];
  while (cur_op.lbp > parent_op.rbp)) { /* loop stops when cur op < parent_op, then all prefix can bind to lhs */
    cur_op = l.pop();
    rhs, nextidx = parseInternal(src, idx, cur_op); // recur to consume strong ops down.
    lhs = (op, lhs, rhs); // update lhs to contain consumed parts.
    cur_op = src[nextidx]; // new cur_op after recur returns;
  }
  return lhs;
}

class Solution {
  class Expr {
    int val; int idx;  // idx to the op just after number token.
    Expr(int val, int idx) { this.val=val;this.idx=idx;}
  }
  int skipEmpty(char[] arr, int idx) {
    while(idx < arr.length && arr[idx] == ' ') idx++;
    return idx;
  }
  int getInt(char c) {
    if(c >= '0' && c <= '9') { return c-'0';}
    return -1;
  }
  int[] getNumAndNextOpIdx(char[] arr, int idx) {
    int num = 0;
    while(idx < arr.length && getInt(arr[idx]) != -1) { 
      num = num*10+getInt(arr[idx]); idx++; 
    }
    return new int[]{num, skipEmpty(arr, idx)};
  }
  int eval(int lhs, char op, int rhs) {
    switch (op) { case '+': return lhs+rhs; case '-': return lhs-rhs; case '*': return lhs*rhs; case '/': return lhs/rhs;}
    return -1;
  }
  // how power the op binds to a number to the right next recur vs. to the left parent.
  int bp(char op) { 
    switch (op) { case '(', ')': return 1; case '+', '-': return 2; case '*', '/': return 3; }
    return -1;
  }
  // pos points to the operant; get this recur's operand and operator; 
  // recur rets when curOp not strong than parent, lhs bind to parentOp; otherwise, lhs binds to curOp, recur;
  Expr pratt(char[] arr, int pos, int parentOpBp) {
    int[] numIdx = getNumAndNextOpIdx(arr, pos);
    int lhs = numIdx[0], opidx=numIdx[1];
    while(opidx < arr.length && opbp(arr[opidx]) > parentOpBp) { // if curOp is strong than parent op, lhs shall bind to curOp;
      Expr rhs = pratt(arr, opidx+1, opbp(arr[opidx]));
      lhs = eval(lhs, arr[opidx], rhs.val); // no stk for nested. evaled inplace;
      opidx = rhs.idx;
    }
    return new Expr(lhs, opidx);
  }
  int calculate(String s) {
      char[] arr = s.toCharArray();
      return pratt(arr, 0, -1).val;
  }
  // using stk is better, stk<Expr> top always be updated by expr[i];
  calculate("(3-(5-(8)-(2+(9-(0-(8-(2))))-(4))))");
}

// LC 1106, Parsing a Boolean Expr. LC 726. Number of Atoms
// stk<List<Expr>>, consume expr[i], update stk top if not new expr; otherwise stk.add(new Expr(op, idx));
Expr parse(int i, String expr) {
  Expr curexp = new Expr(i);
  Deque<Expr> stk = new ArrayDequeue<>();
  while(char nxc = next(expr)) {
    if(nxc == either('|(, &(, !(')) {
      char paren = next(expr);
      stk.add(new Expr(OR/AND/NOT));
      continue;
    } else if(nxt == ')') {
      Expr top = stk.remove();
      for(boolean b : top.list()) {
        bval = OR_AND_NOT(top.op, val, b);
      }
      if(stk.size() > 0) { stk.peek().list.add(bval); } 
      else { return bval;}
    }
    else if(nxt == ',') {  } // nothing to do, union when recur done;
    else if(nxt == 't f') {
      stk.peek().addAtom(nxt);
    }
  }
}
Map<String, Integer> expand(int recur_start, int l, String s) {
  Deq<Map<Expr>> stk;
  stk.push(new HashMap<>());
  while(nxt = s.next())) {
    if(nxt == '(') { stk.add(new HashMap<>()); continue; }
    else if(nxt==')') {
      Map curmap = stk.removeLast();
      if(isDigit(s.peekNext())){ count=s.next(); }
      for(String atom : curmap.keySet()) {
        curmap.put(atom, curmap.get(atom)*cnt);
      }
      mergeMap(stk.peek(), curmap);
    } else if(nxt==UPPER_CHAR){
      while(isDigit(s.peek())){ cnt += s.next();}
      stk.top().putIfAbsent(nxt);  // aggregate into stk top;
      stk.top().compute(nxt, (k,v)->v+cnt);
    }
}}
class Solution {
  class ExpRes {
    String str_; int edidx;
    ExpRes(String s, int i) { this.str_ = s; this.edidx=i;}
  }
  int getNum(char i) { if(i >= '0' && i<='9') { return i-'0';} return -1; }
  boolean isLetter(char c) { return c >= 'a' && c <= 'z'; }
  ExpRes expand(char[] arr, int numidx) { // need to keep global next index;
    ExpRes res = null;
    int i=numidx, repn=0;
    while(i<arr.length && getNum(arr[i]) >= 0) { repn = repn*10 + getNum(arr[i]);i++; }
    i++; // skip the '[', recur starts;
    String reps = "";
    StringBuilder sb = new StringBuilder();
    while(arr[i] != ']') { // recusion starts with `N[`, continue child recursion till the end `]`;
      while(i<arr.length && isLetter(arr[i])) { sb.append(arr[i]); i++; }
      if(getNum(arr[i]) >= 0) { // see a num, expand sub N[] again;
        res = expand(arr, i);  // recur child expand to bind num[num] 
        sb.append(res.str_);
        i = res.edidx + 1;
      }
    }
    StringBuilder retsb = new StringBuilder();
    for(int k=0;k<repn;k++) { retsb.append(sb); }
    return new ExpRes(retsb.toString(), i); // eval of num[..] done; i parked at post ]
  }
  String decodeString(String s) {
    char[] arr = s.toCharArray();
    Deq<ExpRes> stk = new ArrayDequeue<>(); 
    ExpRes res = null;
    int i = 0;
    StringBuilder sb = new StringBuilder();
    // Main loop consumes stream. add stk for new n[] recursion;
    while(i < arr.length) {
      while(i<arr.length && isLetter(arr[i])) { sb.append(arr[i]); i++; }
      if(i<arr.length && getNum(arr[i]) >= 0) {
        while(i<arr.length && getNum(arr[i]) >= 0) {
          repn = repn*10 + getNum(arr[i]);
          i++;
        }
        stk.add(new ExpRes(i, repn)); continue;
      } else if(arr[i] != ']') { 
        while(i<arr.length && isLetter(arr[i])) { 
          stk.top.sb.append(arr[i]); i++; 
        }
      } else if (arr[i]==']') {
        Expr cur = stk.top.pop();
        for(int k=0;k<stk.top.repn;k++) { 
          stk.top.sb.append(cur.sb);
        }
      }
    }
    return stk.top.sb.toString();
  }
  String decodeString("2[abc]3[cd]ef");
}
class Solution {
  class Expansion { // each {, ,} recur a new Expansion, union items in expansion sep by ,;
    int start, end;
    List<Expansion> childexps;
    boolean combination;
    List<String> expanded;
    Expansion parent;

    Expansion(int start) { 
      this.childexps=new ArrayList<>(); this.expanded=new ArrayList<>(); 
      this.start=start; this.combination=true; 
    }
    void addChildExpr(Expansion subexpr) { this.childexps.add(subexpr);}
    List<String> expand() { // expand all children expansions;
      if(expanded.size() > 0) return expanded;
      if(!combination) {
        Set<String> set = new TreeSet<>();
        for(Expansion e : childexps) {
          for(String s : e.expand()) { set.add(s); }
        }
        expanded.clear();
        for(String s : set) { expanded.add(s); }
      } else {
        expand(0, "", expanded);
      }
      return expanded;
    }
    void expand(int off, String prefix, List<String> out) {
      if(off >= this.childexps.size()) return;
      List<String> curexp = this.childexps.get(off).expand();
      if(off==this.childexps.size()-1) {
        for(String s : curexp) { out.add(prefix+s);  }
        return;
      }
      for(String s : curexp) { expand(off+1, prefix+s, out); }
    }
    String toString() { return String.format("Expansion from %d %d subexps %d comb ? %b expanded %s", start, end, childexps.size(), combination, String.join(",", expanded));}
  }
  
  // Each token belongs to the current expansion; A child exp starts at { or letter, ends at , or }; 
  // recur {}; stk child exps, {}a{}e; merge child exp upon , or };
  // int parse(char[] arr, int off, Expansion curexp) { // curexp starts at offset. root starts at -1 to include {}{} 
  Expansion parse(char[] arr, int off) { // curexp starts at offset. root starts at -1 to include {}{} 
    Expansion curexp = new Expansion(off);
    Deque<Expansion> nested = new ArrayDeque<>();  // a stk to store all child expansion of curexp;
    int idx = off+1;  // consume the next token of curexp.
    while(idx < arr.length) { // this parse loop until closing }; 
      if(arr[idx] == '{') {  // recur a child expansion upon {; Or push to global stk;
        Expansion childexp = parse(arr, idx, childexp); // recur child exp of childexp, start at {  till the closing }.
        nested.addLast(childexp); // child recursion done, stash to the curexp's stk until closing } to trigger consolidate.
        idx = childexp.end+1;
        continue;
      } 
      // upon closing symbols, merge children expansions as a single expansion to parent. 
      if(arr[idx] == '}' || arr[idx] == ',') { // recur done and merge nested. { trigger recursion.
        if(nested.size() > 1) { // {a{},{}e}
          // A new parent expansion to merge in stk exps. The new parent is a child of the current expansion that the , } belongs to.
          Expansion merged = new Expansion(nested.peekFirst().start); 
          while(nested.size()>0) { merge.addChildExpr(nested.pollFirst()); } // merge all in stk expansions;
          merge.end = idx;
          merge.expand();
          curexp.addChildExpr(merge);
        } else {
          curexp.addChildExpr(nested.pollFirst()); // add single child expansion to cur exp as , } are tokens of cur exp;
        }
        curexp.end = idx;
        if(arr[idx] == '}') { // closing child {}, recursion done; return;
          return curexp;
        } // comma ends a child expansion. continue the parent.
        curexp.combination = false;
        idx++; continue;
      }
      // atom child expansion with letters; no recursion;
      Expansion letterexp = new Expansion(idx);
      String s = "";
      while(idx < arr.length && arr[idx] >='a' && arr[idx]<='z') { s += arr[idx++];}
      letterexp.expanded.add(s);
      nested.addLast(letterexp);
    }
    while(nested.size()>0) { curexp.add(nested.pollFirst());}
    return idx+1;
  }
  List<String> braceExpansionII(String expression) {
    char[] arr = expression.toCharArray();
    // Expansion exp = new Expansion(-1);
    // the expansion rules(comb, union) is the same for outer and inner, hence consolidate into one parse fn.
    Expansion exp = parse(arr, -1);
    return exp.expand();
  }
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Flattern nested lists. stk iters of each nested list has a iter
// the current nested list's iter recursively call into child's hasNext
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
class NestedIterator implements Iterator<Integer> {
  int pos = 0; // cursor to the current item in the list 
  int expandedpos = -1; // not yet expanded.  
  NestedIterator current_nested_child_iter; // the current active child iter;
  List<NestedInteger> list;
  // Deque<NestedIterator> stk;  // no need stk for recursion.

  NestedIterator(List<NestedInteger> nestedList) {
    pos = 0; expandedpos=-1; // not expanded on construction. Lazy expansion;
    list = nestedList;
  }
  // idempotent; will only expand once when pos is a list;
  void expandList(int pos) {  // expand only when pos not yet expanded. 
    if(expandedpos < pos && !list.get(pos).isInteger()) {
      current_nested_child_iter = new NestedIterator(list.get(pos).getList()); // empty list ok;
      expandedpos = pos;  // list[pos] is expanded if it is a nested list;
    } 
  }
  @Override
  Integer next() { // hasNext already checked pos != end; as no next when pos == end ! 
    if(list.get(pos).isInteger()) { pos++; return list.get(pos-1).getInteger(); }
    return current_nested_child_iter.next(); // hasNext already expanded the list when pos is a list;
  }
  @Override
  boolean hasNext() { // loop until next available iter has next.
    while(true) {  // loop to skip empty list !
      if(pos>=list.size()) return false;
      if(list.get(pos).isInteger()) return true;  
      expandList(pos);  // when cur pos is a nested list, expand it;
      if(current_nested_child_iter.hasNext()) return true;
      pos++; // pos already expanded but done. advance pos and retry pos;
    }
  }
}

// ------------------------------------------------------------
// sliding win: subary/conti: slide l/r to track bad actors; convert prefix+suffix to sliding win stats.
// slide R in, update win minmax. later cancels prev bigger items in stk, or aggregate tot; 
// keep invariants by advance l. for pair, L..m..R; Kth pair by bisect distance, each bisect slide win entire ary.
// To be able to determinstically move l, fix uniq chars in each win. Multi pass to expand uniq char window. 
// When not subary nor Subseq, sort to leverage some greedy; otherwise, Knapsack DP matrix filling. DP is already O(nk); 
// -----------------------------------------------------------
// MaxHeap subset selection: when no constrains on both dimensions, sort one dim, maxheap the other.
// the maxsum/outlier/invariance(max/min/sum). When pq is used, only poll the max ! 
// -----------------------------------------------------------
// slide r, mov l when a.r is not needed. 
static String anagram(String arr, String p) {
  Map<Character, Integer> winMap = new HashMap<>();
  Map<Character, Integer> patternMap = new HashMap<>();
  for (int i=0;i<p.length();i++) { patternMap.compute(p.charAt(i), (k,v)->v==null ? 1: v+1); }
  for(int l = 0, r=0; r<arr.length(); r++){
    if (!patternMap.containsKey(arr.charAt(r))) { l = r+1; winMap.clear(); continue; }
    winMap.put(ic, winMap.getOrDefault(ic, 0)+1);
    if (winMap.get(ic) > patternMap.get(ic)) {
      while (arr.charAt(l) != ic) { winMap.compute(arr.charAt(l), (k,v)->v--);l++;}
      winMap.put(ic, winMap.get(ic)-1);
    }
  }
  return null;
}
System.out.println(anagram("hello", "lol"));

// LC 2962, count subarys max at least k 
long countSubarrays(int[] nums, int k) {
  int maxv=0,maxcnt=0;
  for(int i=0;i<nums.length;i++) { maxv = Math.max(maxv, nums[i]);}
  int l=0,r=l, cnt=0; long tot=0;
  while(r<nums.length) {
    if(nums[r]==maxv) {cnt++;}
    while(cnt==k) { // when loop exits, cnt < k;
      if(nums[l]==maxv) cnt--;
      l++;  // shrink l until maxv cnt < k; 
    }
    tot += l; // after l>0, cnt=k, count subarys on the rite even arr[r] != max
    r++;
  }
  return tot;
}
// LC 907. sum min of all subarys. RangeMin vs. RangeSum. RangeSum with fenwick
int sumSubarrayMins(int[] arr) {
  int tot=0, cur=0;
  Deque<Integer> stk = new ArrayDeque<>(); // left edge: first smaller on stk, rite: first smaller.
  for(int i=0;i<=arr.length;i++) {
    if(stk.size()==0) {stk.addLast(i); continue;}
    if(i==arr.length) cur=0; else cur=arr[i]; // append a min last to clear up stk;
    while(stk.size() > 0 && cur < arr[stk.peekLast()]) { // pop bigger top. count it contribution.
      int topidx = stk.removeLast(); // pop stk, calculate its left edge. rite edge is i;
      int left_len = topidx-(stk.size() > 0 ? stk.peekLast() : -1); // len: (left..topidx]
      int rite_len = i-topidx; // rite len: [topidx, i);
      tot += (rite*left)*arr[topidx];
    }
    stk.addLast(i);
  }
  return tot;
}
// k win contains dups of a_i; track a_i last occur idx and tot freq to min win size; 
int subarraysWithDistinctK(int[] arr, int k) {
  Map<Integer,Integer> cnt = new HashMap<>();
  // no need this map. just move kl when win size=k; and set l=kl when size back to k;
  Map<Integer,Integer> lastidx = new HashMap<>(); // track last appear idx of arr_I;
  int l=0,minkwin_start=l,tot=0; // [l...minkwin_start...r]; aaab, l=0,kl=2;
  for(int r=0;r<arr.length;r++) {
    cnt.compute(arr[r], (key,v)->v==null?1:v+1); // first, consume r and update aggregation.
    lastidx.put(arr[r], r); // now arr/r slides;
    while(cnt.size() > k) { // aaabbc, mov l to b;
      cnt.compute(arr[l], (key,v)->v-1);
      if(cnt.get(arr[l])==0) cnt.remove(arr[l]);
      l++;
      if(l>minkwin_start) minkwin_start=l; // minkwin_start 
    }
    if(cnt.size() == k) { // aaab, lseek l=0, kl=2
      while(cnt.get(arr[minkwin_start]) > 1 && minkwin_start < lastidx.get(arr[minkwin_start])) minkwin_start++; // push left furthest
      tot += minkwin_start-l+1; // aaab, aaaba 
    }
  }
  // just mov kl when win size > k; set l=kl when advance win left;
  for(int r=0;r<arr.length;r++) {
    cnt.compute(arr[r], (key,v)->v==null?1:v+1);
    while(cnt.size() > k) {
      cnt.compute(arr[kl], (key,v)->v-1);  // given kl is moved to the lastest arr[kl] with cnt update, continue to use kl here;
      if(cnt.get(arr[kl])==0) { cnt.remove(arr[kl]); }
      kl++; l=kl; // set win left to kl when new window is started !!
    }
    if(cnt.size() == k) { // set kl to the last ocurrence of arr[kl];
      while(kl < r && cnt.get(arr[kl]) > 1) { // mov kl to the last occurrence with reducing its cnt;
        cnt.compute(arr[kl], (key,v)->v-1);
        kl++;
      }
      tot += kl-l+1;
    }
  }
  return tot;
}

// if maxdup stays, l will be moved along with r; (r-l+1-maxdups > w), no maxlen increase.
// until maxdups is recomputed with correct dup and increased, maxlen is then increased !
int characterReplacement(String s, int k) {
  // track max cnts of arr/i; r-l-maxdups shall < k
  Map<Character, Integer> charCount = new HashMap<>();
  int l=0,r=0,maxlen=0,maxdups=0;
  for(int r=0;r<arr.length;r++) {  // while(r<s.length()) {
    charCount.compute(s.charAt(r), (k,v) -> {return v==null ? 1 : v+1;});
    maxdups = Math.max(maxdups, charCount.get(s.charAt(r)));
    if(r-l+1-maxdups > k) {  // do not mov l when == k !
      charCount.compute(s.charAt(l), (key,cnt) -> cnt-= 1);
      l++;  // if l is maxdups, why not update maxdups ? cur maxdups already triggers the mov of l;
    }
    maxlen = Math.max(maxlen, r-l+1);
  }
  return maxlen;
}

// track avail/needed. when pop, needed++; result shall be the smallest in lexi order among all possible results.
// 用stack来先入再后悔。from left -> rite, if stk top > a[i] and count[stk.top] > 0, pop stk.top as still have chance to add stk.top later.
static String removeDupLetters(String s) {
  Map<Character, Integer> counts = new HashMap<>();
  Set<Character> keepSet = new HashSet<>();  // LoopInvar: track unique element.
  Stack<Character> keepStk = new Stack<>();

  for (int i=0;i<s.length();i++) { counts.compute(s.charAt(i), (k,v)-> v==null ? 1 : v+1) }
  for (int i=0; i<s.length(); i++) {  
    char c = s.charAt(i);
    if (!keepSet.contains(c)) {  // *bug*, did not keep invariant of counts when skipping
      while (!keepStk.empty() && keepStk.peek() > c && counts.get(keepStk.peek()) > 0 ) {
        keepSet.remove(keepStk.peek());
        keepStk.pop();   // no keep of keep stk top
      }
      keepStk.push(c);   // push current into keep stk.
      keepSet.add(c);
      counts.put(c, counts.get(c)-1);  // used one, reduce
    }
  }
  StringBuilder sb = new StringBuilder();
  while(!keepStk.empty()) { sb.append(keepStk.pop()); }
  return sb.toString();
}
// keep chars in win sorted in stk; the choosen chars, pop stk when consuming smaller char.
static String RemoveDup(String s) {
  int[] counter = new int[26]; 
  int[] kept_chars = new int[26];
  for(int i=0;i<s.length();++i) { counter[s.charAt(i)-'a']++; kept_chars[s.charAt(i) - 'a'] = 0;}
  Deque<Character> stk = new ArrayDeque<>();  // laster smaller char pop larger chars in stk
  stk.addLast(s.charAt(0)); counter[s.charAt(0)-'a']--; kept_chars[s.charAt(0)-'a']++;
  for(int i=1;i<s.length();++i) {
    char cur = s.charAt(i);
    if (kept_chars[cur-'a'] > 0) { counter[cur-'a']--; continue; } // no dup in stk.
    // later smaller char cancels large chars in stk. 
    while(!stk.isEmpty() && cur <= stk.peekLast() && counter[stk.peekLast() - 'a'] > 0) { 
      kept_chars[stk.peekLast()-'a']--; stk.removeLast(); // when poping, must has a dup later.
    } // after loop, a/i > top, or top can not be removed, no dup, or empty.
    stk.addLast(cur); counter[cur-'a']--; kept_chars[cur-'a']++; // bcdbc, bca
  }
  StringBuilder sb = new StringBuilder();
  while(!stk.isEmpty()) { sb.append(stk.pollFirst()); }
  return sb.toString();
}
System.out.println(removeDupLetters("dcdcad"));

// widx start from 1. compare against widx-1 and widx. Allow at most 2.
static void removeDup(int[] arr) {
  int l = 2;       // loop start from 3rd, widx as allows 2
  int widx = 1;    // widx 1, allow two dups
  for (int i=l; i<arr.length; i++) {
    if (arr[i] == arr[widx] && arr[i] == arr[widx-1]) { continue; } // skip current item when it is a dup
    widx += 1;
    arr[widx] = arr[i];   // overwrite the dup as widx is a dup.
  }
}
removeDup(new int[]{1,3,3,3,3,5,5,5,7});

// split into segments and find maxlen of each segment.
int longestSubstring2(String s, int k) {
  char[] arr = s.toCharArray();
  Map<Character, List<Integer>> map = new HashMap<>();
  Set<Character> missing = new HashSet<>();
  for(int r=0;r<arr.length;r++) {
    missing.add(arr[r]);
    map.putIfAbsent(arr[r], new ArrayList<>());
    List<Integer> list = map.get(arr[r]);
    list.add(r);
    if(list.size()>=k) { missing.remove(arr[r]);}
  }
  ArrayList<Integer> splits = new ArrayList<>();
  for(char c : missing) {  splits.addAll(map.get(c)); }
  if(splits.size() == 0) return arr.length;
  splits.sort((a,b) -> a-b);
  int l=0, maxw=0;
  Deque<int[]> segs = new ArrayDeque<>();
  for(Integer i : splits) {
    segs.addLast(new int[]{l, i-1});
    l=i+1;
  }
  segs.addLast(new int[]{l, arr.length-1});
  while(segs.size() > 0) {
    int[] seg = segs.removeFirst();
    if(seg[0] > seg[1]) continue;
    maxw = Math.max(maxw, segK(arr, seg[0], seg[1], k, map, segs));
  }
  return maxw;
}

// track the uniq chars within a wind; Why track uniq chars ? 
// To shrink win correct. Multi passes of each uniq win must contains a correct one.
int longestSubstring(String s, int k) {
  char[] arr = s.toCharArray();
  Map<Character, List<Integer>> map = new HashMap<>();
  for(int r=0;r<arr.length;r++) {
    map.putIfAbsent(arr[r], new ArrayList<>());
    map.get(arr[r]).add(r);
  }
  int maxw=0, totuniqchars = map.size();
  for(int uniqcw=1;uniqcw<=totuniqchars;uniqcw++) {
    int l=0,r=l, chars=0, kchars=0; 
    int[] cnt = new int[26];  // reset
    for(;r<arr.length;r++) {
      cnt[arr[r]-'a']++;
      if(cnt[arr[r]-'a']>=k) {
        if(cnt[arr[r]-'a']==k) kchars++; // inc only when from k-1 to k
        if(kchars==uniqcw) { maxw=Math.max(maxw, r-l+1); }
      }
      if(cnt[arr[r]-'a']==1) { 
        chars++;
        while(chars > uniqcw) {
          // update, shrink, exclude r
          cnt[arr[l]-'a']--;
          if(cnt[arr[l]-'a']==k-1) kchars--;  // reduced from k to k-1;
          if(cnt[arr[l]-'a']==0) chars--;  // reduced to 0
          l++;
        }
      }
    }
  }
  return maxw;
}

// LC 1425: Constrained Subsequence Sum: sort win k isum. max_isum_r-k + arr[r];
// Sort [isum,i]; isum is max sum at i; Later bigger isum cancels prev smaller jsum.
// max isum takes prev max in win_k(i-k...isum), if prev > 0; else arr_i; 
int constrainedSubsetSum(int[] arr, int k) {
  Deque<int[]> stk = new ArrayDeque<>(); // sorted store [isum, i], max at head.
  int isum=0,maxsum=-(int)1e9;
  for(int r=0;r<arr.length;r++) { // slide in r, incl r. skip r is done in prev r-1;
    while(stk.size() > 0 && r-k > stk.peekFirst()[1]) { // r-l<=k left must within k;
      stk.removeFirst(); // even if r not included, r-l 
    } // incl arr[r], take max jsum in win_k in only when it>0(it+arr_r>arr_r), or arr[r] is a new grp when it is bigger.
    isum = Math.max(arr[r], stk.size() > 0 ? stk.peekFirst()[0] + arr[r] : arr[r]);
    maxsum = Math.max(maxsum, isum); // if arr_r < 0, rsum will not on top, hence skipped.
    while(stk.size() > 0 && isum >= stk.peekLast()[0]) { stk.removeLast(); } // bigger sum with arr[r] cancel prev smaller sum.
    stk.addLast(new int[]{isum, r});
  }
  return maxsum;
}

// LC 1438: subary max-min<=K; LC 2597 Beautiful Subset: no pair=K vs. LC 2563 Fair Pairs;
// track subary win min/max 2 Deq/Heap; grp mod k; sort, bisect(0,i,arr_i, upper)-bisect(lower);
int longestSubarray(int[] arr, int limit) {
  Deque<Integer> minstk = new ArrayDeque<>(); // stk ascending order
  Deque<Integer> maxstk = new ArrayDeque<>(); // stk descending order
  int l=0,r=l,maxlen=0;
  for(r=0;r<arr.length;r++) {
    int nxtl=l;  // clean up before inserting arr_r;
    while(minstk.size() > 0 && Math.abs(arr[minstk.peekFirst()]-arr[r]) > limit) { // 
      nxtl = Math.max(nxtl, 1+minstk.removeFirst()); // minstk head too far,
    }
    while(minstk.size() > 0 && arr[minstk.peekLast()] > arr[r]) { mindeq.removeLast(); }
    minstk.addLast(r); // append cur to the minstk tail, cancel prev bigger tail.

    while(maxstk.size() > 0 && Math.abs(arr[maxstk.peekFirst()]-arr[r]) > limit) {
      nxtl = Math.max(nxtl, 1+maxstk.removeFirst());  // pop max deq head
    }
    while(maxstk.size() > 0 && arr[maxstk.peekLast()] < arr[r]) { maxstk.removeLast(); }
    maxstk.addLast(r);  // append cur to maxstk tail. cancel prev smaller;
    
    l=nxtl;
    maxlen = Math.max(maxlen, r-l+1);
  }
  return maxlen;
}
// LC 2563: sort, two ptrs, fix arr[r], bisect l, pairs+=(r-l+1);
long countFairPairs(int[] nums, int lower, int upper) {
  Arrays.sort(nums);
  bisect(int[] nums, int target) { // find tot pairs sum < target;
    int left = 0, right = nums.length - 1, tot=0;
    while (left < right) { if(arr[l]+arr[r] < target) { tot += r-l; l++} else r--;}
  }
  return bisect(arr, upper+1) - bisect(arr, low);
}

class Prefix { // LC 862. sorted prefix stk increasing. Later smaller prefix cancels prev bigger ones;
  long sum; int edidx; Prefix(long sum, int idx) {this.sum=sum; this.edidx=idx;}
}
int shortestSubarray(int[] arr, int k) {
  int minw=(int)1e9; // win left shrink only while prefix >= k; win.left=prefix.edidx;
  long prefixsum = 0;
  Deque<Prefix> stk = new ArrayDeque<>();  // sorted prefix; later smaller prefix cancel prev larger prefix b/c shorter win larger diff;
  for(int r=0;r<arr.length;r++) {
    prefixsum += arr[r];
    if(prefixsum >= k) { minw = Math.min(minw, r+1); }
    while(stk.size() > 0) { // pop min stk head;
      if(prefixsum - stk.peekFirst().sum >= k) { // shrink the win only when prefix(l..r) >= k.
        minw = Math.min(minw, r-stk.peekFirst().edidx); // global watermark l is replaced by each preifx end idx;
        stk.removeFirst();
      } else { break; }
    }
    // prefixsum stk is mono-incr, Prefix(r)-stkFirst, a later smaller prefix produces bigger diff shorter win; 
    while(stk.size() > 0 && stk.peekLast().sum >= prefixsum) { stk.removeLast(); } // cancels prev bigger prefix;
    stk.addLast(new Prefix(prefixsum, r)); // enq cur to stk;
  }
  if(minw==(int)1e9) return -1;
  return minw;
}
// LC 1574: remove shortest subary in middle. prefix ary and suffix ary are sorted. binary search suffx for each prefix;
// or two pointers. l=0; r=first arr_r > arr_r+1; advance l until arr_l > arr_r; then advance r;
int findLengthOfShortestSubarray(int[] arr) {
  int l=0;r=arr.length-1;
  for(;r>=0;r--) { if(arr[r]<arr[r+1]) r--; else break;}
  while(l<r && (l==0 || arr[l-1] < arr[l])) {
    while(r<arr.length && arr[l] > arr[r]) r++; // advance r to ensure arr[l] <= arr[r]
    maxlen=Math.max(maxlen, arr.length-(r-l-1));
    l++;
  }
  return maxlen;
}

// Subset sel; sort by dim1; maxheap of dim2; No inter-dep among dims.
// LC 857(Min cost to Hire), 502(IPO); sort+maxheap so high dim1 low dim2 can win;
class Worker {
  int id_; int q_; int w_; double r_;  // each item has a value/quality at a cost/weight;
  Worker(int id, int q, int w) {this.id_=id;this.q_=q;this.w_=w;this.r_=(double)w/q;}
  String toString() { return String.format("id %d q %d w %d r %f", id_, q_, w_, r_);}
}
// Subset sel of high ratio quality. Sort by ratio, a max heap win to sift big quality with low ratio.
double mincostToHireWorkers(int[] quality, int[] wage, int k) {
  double tot=0, kqualities=0, mintot=Double.MAX_VALUE;
  Worker[] workers = new Worker[wage.length];
  for(int i=0;i<wage.length;i++) { workers[i] = new Worker(i, quality[i], wage[i]); }
  Arrays.sort(workers, (a,b)->{if(a.r_ <= b.r_) return -1; else return 1;}); // sort workers by incr ratio.
  // a max heap to track bad actors(high quality, high ratio); replace with high ratio low qualities;
  PriorityQueue<Worker> maxquaHeap = new PriorityQueue<>((a,b)->b.q_-a.q_);
  for(int i=0;i<=k-1;i++) { // seed maxheap with first k low ratio workers;
    maxquaHeap.offer(workers[i]); 
    kqualities += workers[i].q_;
  }
  mintot = workers[k-1].r_*kqualities;
  for(int i=k;i<wage.length;i++) { // workers sorted by ratios. low ratio workers in heap now; 
    if(workers[i].q_ <= maxquaHeap.peek().q_) { // remove big quality low ratio workers from heap top;
      Worker topw = maxquaHeap.poll();
      kqualities -= (topw.q_-workers[i].q_); 
      mintot = Math.min(mintot, workers[i].r_*kqualities);
      maxquaHeap.offer(workers[i]);
    }
  }
  return mintot;
}
// LC 502. maxheap(profit), sort/minheap(capacity).
int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
  Proj[] projs = new Proj[profits.length];
  for (int i = 0; i < profits.length; i++) projs[i] = new Proj(profits[i], capital[i]);
  Arrays.sort(projs, (a, b) -> a.cap - b.cap); // sort cost, maxheap profit. 
  PriorityQueue<Proj> profitmaxheap = new PriorityQueue<>((a,b)->b.pro-a.pro); // profit maxheap;
  for(i=0;i<projs.length;i++) { // into maxheap all affordable projs;
    if(projs[i].cap <= w) { profitmaxheap.offer(projs[i]);} else { break;} // only enq cap<=W
  }
  int hired = 0;
  while(profitmaxheap.size() > 0 && hired < k) {
    Proj top = profitmaxheap.poll();
    w += top.pro; hired++; // take max pro. increase cap w;
    while(i<projs.length) { // after captial increase, continue add more project; 
      if(projs[i].cap <= w) { profitmaxheap.offer(projs[i]); i++;} // proj is sorted by increasing cap.
      else break;
    }
  }
  return w;
}

static int scheduleWithCoolingInterval(char[] tasks, int coolInterval) {
  int sz = tasks.length, timespan=0; // tot time for all intervals;
  int[] taskcounts = new int[26]; for (int i=0;i<sz;i++) { taskcounts[tasks[i]-'A'] += 1; }
  PriorityQueue<Integer> taskcountMaxheap = new PriorityQueue<Integer>(26, Collections.reverseOrder());
  for (int i=0;i<26;i++) { if (taskcounts[i] > 0) { taskcountMaxheap.offer(taskcounts[i]); } } 
  while (!taskcountMaxheap.isEmpty()) {
    List<Integer> unfinishedTasks = new ArrayList<Integer>();
    int intv = 0;                   // now I need a counter to break while loop.
    while (intv < coolInterval) {   // each cool interval
      if (!taskcountMaxheap.isEmpty()) {
        int task = taskcountMaxheap.poll(); // task with largest pending #;
        if (task > 1) { unfinishedTasks.add(task-1); }  // scheduled, -1
      }
      if (taskcountMaxheap.isEmpty() && unfinishedTasks.size() == 0) { break; }
      timespan += 1;
      intv += 1;       // update counter per iteration.
    }
    // re-add unfinished task into q 
    for (int t : unfinishedTasks) { taskcountMaxheap.add(taskcounts[t]); }
  }
  return timespan;
}

// LC 1235. Max Profit. F[iend_tm] = Max(F[bisect_prev_job_i_start_tm]+profit[i]); skip i, F[i]=F[i-1]; skip i;
int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
  Job[] jobs = new Job[profit.length];
  for(int i=0;i<profit.length;++i) { jobs[i] = new Job(startTime[i], endTime[i], profit[i]); }
  Arrays.sort(jobs, (a, b) -> a.ed - b.ed); // sort job by end time; leverage start < end.
  int bisectPreJobByEndTm(Job[] jobs, int start) {
    int l=0,r=jobs.length-1;
    while(l<=r) {
      int m = l+(r-l)/2;
      if(jobs[m].ed <= /* <= */ start) { l=m+1; }  else { r=m-1; }
    }
    return l-1;  //  l parked at first job whose ed across start; 
  }
  int[] dp = new int[jobs.length];  // F[i]=max(F[l]+A[i]) and F[i-1];
  // int[] F = new int[Math.max(endTime)]; // no need bisect if we can populate F[end_time];
  for(int i=0;i<jobs.length;++i) {
    int prejob = bisectPreJobByEndTm(jobs, jobs[i].st);
    int preprofit = prejob==-1 ? 0 : dp[prejob];
    // the profit at job i is the max of all prev jobs.
    dp[i] = Math.max(dp[i], preprofit+jobs[i].profit);
    if(i>0) { dp[i] = Math.max(dp[i], dp[i-1]);}  // carry over dp[i-1] to dp[i];
  }
  return dp[dp.length-1];
}

class Solution { // consume each task, update MaxHeap.
  class Task implements Comparable<Task> {
    char c_; int cnt_; int next_;
    Task(char c, int cnt) { this.c_ = c; this.cnt_ = cnt; this.next_=0;}
    @Override int compareTo(Task other) {
      if(this.cnt_ > other.cnt_) return -1; // return < 0 if this shall ahead other; >0 if this behind other. 
      if(this.cnt_ == other.cnt_) { return this.next_ - other.next_;} // smaller next_ ahead
      else return 1;
    }
    String toString() { return this.c_ + ":" + this.cnt_ + " next : " + this.next_;}
  }
  int leastInterval(char[] tasks, int n) {
    Map<Character, Integer> map = new HashMap<>(); // key is task, val is # of tasks;
    for(int i=0;i<tasks.length;++i) { map.compute(tasks[i], (k,v) -> v==null ? 1 : v+1); }
    PriorityQueue<Task> maxCountHeap = new PriorityQueue<>(); // use task compareTo sorting.
    for(Map.Entry<Character, Integer> e : map.entrySet()) { q.add(new Task(e.getKey(), e.getValue())); }
    int cycles = 0, scheduled = 0;
    List<Task> l = new ArrayList<>();
    while(q.size() > 0) {
      l.clear(); scheduled = 0;
      for(int k=0;k<=n;++k) { // consider all tasks every n+1 step.
        while(maxCountHeap.size() > 0) {
          Task c = maxCountHeap.poll();
          if(c.next_ > cycles) { l.add(c); continue; }
          if(c.next_ <= cycles) {
            c.next_ = cycles+1+n;
            c.cnt_--; // if used, reduce cnt.
            scheduled++;
            if(c.cnt_ > 0){ l.add(c); }
            break;
          }
        }
        cycles++; // along with k++
      }
      // last round, remove all rest non scheduled.
      if(l.size() == 0) { cycles -= (n+1-scheduled);} 
      // after every n+1 intervals, restart with all tasks.
      for(Task t : l) { maxCountHeap.add(t); }
    }
    return cycles;
  }
}

// Track incoming block's X with PQ's X while consuming each block.
List<List<Integer>> getSkyline(int[][] arr) {
  PriorityQueue<int[]> maxHtHeap = new PriorityQueue<>((a,b) -> b[1]-a[1]);
  List<List<Integer>> xys = new ArrayList<>();
  int bidx=0,lx=0,rx=0,ht=0;
  while(bidx < arr.length || maxHtHeap.size() > 0) {  // upon each block, update the state data structure.
    if(bidx < arr.length) {lx=arr[bidx][0];rx=arr[bidx][1];ht=arr[bidx][2];}
    // compare block start with maxHtHeap top block end, consume and enq bidx when overlap, or pop maxHtHeap top If not overlap. Loop back.
    if(maxHtHeap.size()==0 || bidx < arr.length && lx <= maxHtHeap.peek()[0]) {
      while(bidx < arr.length && arr[bidx][0]==lx) { // dedup blocks started with the same X.
        ht=Math.max(ht, arr[bidx][2]);
        maxHtHeap.offer(new int[]{arr[bidx][1], arr[bidx][2]}); // enq bidx(x, y), inc bidx
        bidx++;  // consume bidx, update maxHtHeap.
      }
      if(xys.size()==0 || xys.get(xys.size()-1).get(1) < ht) { // enq same lx, max ht.
        xys.add(Arrays.asList(lx, ht));
      } 
    } else { // pop maxHtHeap top if bidx not overlap with maxHtHeap top. pop it, loop back and re-eval bidx.
      int[] top = maxHtHeap.poll();
      int top_end = top[0];
      while(maxHtHeap.size()>0 && maxHtHeap.peek()[0] <= top_end) { maxHtHeap.poll();} // keep pop top when top's end shadowed by cur top.
      ht = maxHtHeap.size() == 0 ? 0 : maxHtHeap.peek()[1];
      if(xys.size()==0 || xys.get(xys.size()-1).get(1) != ht) {
        xys.add(Arrays.asList(top_end, ht));
      }
    }
    // after popping top, loop back to re-eval bidx as bidx only inc-ed after it's enq.
  }
  return xys;
}

//--------------------------------------------------------------------- 
// Dynamic Programming: F[i][dim1]；Each constraint a F[dim]; branch and collect;  l=0..i, max(F[l]+A[l..i]);
// DP n^2 vs. Sort(dim1) + MaxHeap(dim2) nlgn vs. slide win stk, vs. loop wd[i], loop last visited 26 chars,
// Subproblem states, absent#, painted_walls ? F[n][prev_path][total_absent][strides]; F[walls][paint_walls];F[left][rite]
// O(2^n), decision forms a uniq tree path to leaf, repr a subset from N. recur(i+1, incl/excl, state1, state2); leaf ret parent aggregate.
// Dfs/Sliding top down, update and forward prefix to i+1 till Leaf boundary; leaf return and parent aggregate all uniq incl/skip. dfs(root, prefix);
// O(n^2): F[i,j] = F[i-1, k..j], expand i sum up i-1; At each Wt, Profit, sum up j=W-cost_i, pro_j+pro_i; F[i,w,p]: A dim an Ary. Boundary Zero value is critical; 
// Ary conti Group: try i to each grp, i to cur grp, or new grp of i; recur i+1; F[cuts][l,r] = min(F[cut-1][l,k] + F[cut-1][k,r] + mx_kr);
// Knapsack: subset sel incl/skip i; dp[N,W,V]: aggr(i,w,p)<maxW >minP; cell[i,w]: agg from dp[i][w]=max(dp[i-1][w-w_i]+v_i, dp[i-1][w])
// For divide into K, 1. dfs(i+k, w): split a new grp vs. no split. 2. min(F(i+1,k-1), F(i+1, k); F[i,j]=F[i-1,0..j]+ilen;
// dfs(i,k,pre,pre_cnt, path_max): min of del i, merge i, new split i; return max from path leaf; recur ret min of all paths;
// for each op in seq, F[left][rite]; sum(dfs(0..op-1) + dfs(op+1..N)), recur on left and rite again;
//---------------------------------------------------------------------
int CalPackRob(int[] nums) { // max of incl/skip;
  int pre=0,prepre=0, incl=0, excl=0; // boundary;
  for(int i=0;i<arr.length;++i) { // max at i deps on max at i-1;
    incl = prepre+arr[i];
    excl = pre; // accumulate pre into i when excl i;
    prepre = pre;
    pre = Math.max(incl, excl); // max of inc/exl i; set as pre for next i+1;
  }
  return pre;
}
int decodeWays(int[] arr) {
  int sz = arr.length, cur=0;
  int prepre=1, pre=1;  // prepre = 1, as when 0, it is 0-2, the first 2 digits, count as 1.
  for (int i=1; i<sz; i++) {  // f[i] += f[i-1];
    cur = pre;
    if (arr[i] == 0) { prepre = 1; }
    if (arr[i-1] == 1 || (arr[i-1] == 2 && arr[i] <= 6)) { cur += prepre; }
    prepre = pre;  pre = cur;
  }
  return cur;
}
decodeWays(new int[]{2,2,6});

// decode with star, DP tab[i] = tab[i-1]*9
// 1. ways(i) -> that gives the number of ways of decoding a single character
// 2. ways(i, j) -> that gives the number of ways of decoding the two character string formed by i and j.
// tab(i) = ( tab(i-1) * ways(i) ) + ( tab(i-2) *ways(i-1, i) )
int ways(int ch) {}
int ways(char ch1, char ch2) {}
int decodeWays(String s) {
  long[] dp = new long[2];
  dp[0] = ways(s.charAt(0)); // F[i] = F[i-2] + F[i]; 
  if (s.length() < 2) return (int)dp[0];
  // dp[1] is nways from [0,1], dp[i] = dp[i-1]*ways(a[i]) + dp[i-2]
  dp[1] = dp[0] * ways(s.charAt(1)) + ways(s.charAt(0), s.charAt(1));
  for(int j=2; j<s.length(); j++) { // DP tab bottom up from string of 2 chars to N chars.
    long temp = dp[1];
    dp[1] = (dp[1] * ways(s.charAt(j)) + dp[0] * ways(s.charAt(j-1), s.charAt(j))) % 1000000007;
    dp[0] = temp;
  }
  return (int)dp[1];
}

boolean decompose(String s, Set<String> set) {
  int[] dp = new int[s.length()+1];
  dp[0] = 1; // String[0..i) is a composite word
  for (int i = 1; i <= length; ++i) {
    for(int j=0;j<i;j++){
      if(i==length && j==0) continue; // not composite, leaf word;
      if(dp[j]==1 && set.contains(s.substring(j, i))) dp[i]=1;
    }
  }
  return dp[length]==1;
}
// LC 474 at most m 0s and n 1s. iter each word, incl wd, m->0, n->0, f(m,n)=f(i-w0,j-w1)+1; 
int findMaxFormOpt(String[] strs, int m, int n) {
  int[][] dp = new int[m + 1][n + 1];
  for (String str : strs) {
    int cntZero = 0, cntOne = 0;
    for (char ch : str.toCharArray()) { if (ch == '0') cntZero++; else cntOne++; }
    for (int i = m; i >= cntZero; i--) {
      for (int j = n; j >= cntOne; j--) {
          dp[i][j] = Math.max(dp[i][j], dp[i - cntZero][j - cntOne] + 1);
      }
    }
  }
  return dp[m][n];
}

// LC 2140: f[i]: max from i..N; f[i] = max(f[i+1], f[i+k]+A_i);
long mostPoints(int[][] q) {
  long[] f = new long[arr.length];  // max pts from i..N;
  for(int i=arr.length-1;i>=0;i--){ // backwards. incl i, take f[i+k]+a_i; excl, f[i+1];
    int pts = arr[i][0], next = arr[i][1];
    if(i+next+1 < arr.length) {
      f[i] = Math.max(f[i+1], pts + f[i+next+1]); // skip i, result is the same as i+1; incl i, affect i+next;
    } else if(i+1<arr.length){
      f[i] = Math.max(pts, f[i+1]);
    } else { f[i]=pts; }
  }
  return f[0];
}
int dfs(int off, int[][] q, int[] cache) {
  if(off>=q.length) { return 0; }
  if(cache[off] != 0) return cache[off];
  int next=off+q[off][1]+1, val=q[off][0];
  cache[off] = Math.max(
    val + dfs(next, q, cache),  // agg on return
    dfs(off+1, q, cache));
  return cache[off];
}
// LC 474: max subset selection. MaxHeap no help. O(n2)  
// dp[m][n] = Max(dp[m-word_zeros][n-word_ones]+1) when count word[i] incr zeros/ones
int findMaxForm(String[] arr, int m, int n) {
  int[][] dp = new int[m+1][n+1];
  for(int k=0;k<arr.length;k++) {
    int[] zos = zeros(arr[k]);
    for(int i=m;i>=zos[0];i--) {
      for(int j=n;j>=zos[1];j--) {
        dp[i][j] = Math.max(dp[i][j], dp[i-zos[0]][j-zos[1]]+1);
      }
    }
  }
  return dp[m][n];
}
// 494. Target Sum; F[i][target]=F[i-1][t-a_i]
class Solution {
  // recur add/minus i from pathsum; ret 1 when recur is a valid path; aggre all sub paths at each parent node;
  int dfs(int i, int pathsum, int target, int[] arr, Map<String, Integer> cache) {
    if(i==arr.length) {if(pathsum==target) return 1; return 0; } // one recur path is one unique way of incl/skip i from root to leaf;
    int tot=0;
    String add_key = String.format("%d:%d", i, pathsum+arr[i]);
    if(cache.containsKey(add_key)) { tot=cache.get(add_key); }
    else {tot = dfs(i+1, pathsum+arr[i], target, arr, cache); cache.put(add_key, tot); }
    String minus_key = String.format("%d:%d", i, pathsum-arr[i]);
    if(cache.containsKey(minus_key)) { tot+=cache.get(minus_key);}
    else { int minus_tot = dfs(i+1, pathsum-arr[i], target, arr, cache); cache.put(minus_key, minus_tot); tot += minus_tot;}
    return tot;
  }
  int dp(int[] arr, int target) { // subset sum with variation arr_i can be negative;
    int sum = Arrays.stream(arr).sum();
    if(sum < Math.abs(target)) return 0;
    int[][] dp = new int[arr.length][2*sum+1]; // f[i,sum] to iter each i and each sum;
    dp[0][arr[0]+sum]=1; dp[0][sum-arr[0]]+=1; // note the +=, arr[0]=0, +0,-0 count as 2;
    for(int i=1;i<arr.length;i++) { // iter each i, nested iter each value;
      for(int v=-sum;v<=sum;v++) {  // dp[i][v] right shifted sum to avoid negative target value;
        if(sum+v-arr[i] >= 0) { 
          dp[i][sum+v] += dp[i-1][sum+v-arr[i]]; // for each f[i,v], collect from f[i-1, v-arr[i]]
          // dp[i][sum+v+arr[i]] += dp[i-1][v+sum]; // for each f[i,v], add/sub arr[i], project to f[i+1, v+-arr[i]];
        }
        if(sum+v+arr[i] <=2*sum) {
          dp[i][sum+v] += dp[i-1][sum+v+arr[i]];
          // dp[i][sum+v-arr[i]] += dp[i-1][v+sum];  // f(i,v+a/i)+=f(i-1,v); f(i,v-a/i)+=f(i-1,v);
        }
      }
    }
    return dp[arr.length-1][sum+target];
  }
  int findTargetSumWays(int[] nums, int target) {
    int[] dp = new int[2001]; // dp[v]: # ways sum to v; shift 1000 to accomodate neg, [0..1000];
    dp[nums[0] + 1000] = 1;  dp[-nums[0] + 1000] += 1;
    for (int i = 1; i < nums.length; i++) {   // iterate each a.i
      int[] nextdp = new int[2001];   // allocate nextdp, range [-1000...1000]
      for (int val = -1000; val <= 1000; val++) {  // at each slot, iter all vals.
        if (dp[val + 1000] > 0) {
          nextdp[1000 + val + nums[i]] += dp[val + 1000];
          nextdp[1000 + val - nums[i]] += dp[val + 1000];
        }
      }
      dp = nextdp;
    }
    return S > 1000 ? 0 : dp[target + 1000];
  }

  // LC 514: n^3: F[left][rite]: an extra loop all N to find to prev start point.
  int dfs(int si, int ki, String ring, String key, Map<String, Integer> cache) {
    if(ki==key.length()) return 0;
    String siki = String.format("%s:%s", si, ki);
    if(cache.containsKey(siki)) return cache.get(siki);
    int res = (int)1e9;
    for(int i=0;i<ring.length();i++) {
      if(ring.charAt(i)==key.charAt(ki)) {
        int dist = Math.min(Math.abs(si-i), ring.length()-(Math.abs(si-i))); // dist on ring circle; 
        res = Math.min(res, dist+1+dfs(i, ki+1, ring, key, cache));
      }
    }
    cache.put(siki, res);
    return res;
  }
}
// 2560. House Robber IV, reucr pass down path max; leaf echo back pathmax of a valid path, parent agg and ret the min of all paths;
// MinMax always use binary search lo hi bound is defined.
int dfsMinMax(int pos, int k, int pathmax, int[] arr) {
  if(pos>=arr.length) { 
    if(k>0) return (int)1e9;  // not k grps, invalid path.
    return pathmax; // Each path leaf echos back its path max for parent to agg min; 
  }
  int incl = (int)1e9;  // incl pos, recur pos+2/k-1;
  int skip = dfsMinMax(pos+1, k, pathmax, arr); // skip arr/i, recur pos+1/k;
  if(k>0){incl = dfsMinMax(pos+2, k-1, Math.max(pathmax, arr[pos]), arr); }
  return Math.min(incl, skip); // path max is passed down and echoed back; take min of all path on return;
}
// LC 2560, 2226, 2587, Binary search for MaxMin, or binsearch for min total time.
int minCapability(int[] arr, int k) {
  // int[] sortedIdx = new int[arr.length];
  // for(int i=0;i<arr.length;i++) sortedIdx[i] = i;
  while (minReward < maxReward) { // bisect within min max lo/hi bound;
    int minMax = (minReward + maxReward) / 2;
    int robs = 0, i=0;
    while(i<arr.length) { // each search, scan the entire ary;
      if (nums[i] <= minReward) { robs+=1; i+=2; }// increase robs and skip the next non-adjacent
      else { i+=1; }
    }
    if (robs >= k) maxReward = minMax;  // match robs;
    else minReward = minMax + 1;
  }
  return minReward;
}

// dp[item][cuts] = min(dp[i-k][cuts-1] + pathmax = max(cutmax, grpsum+arr[k]))
int shipWithinDays(int[] arr, int days) { 
    // MinMax, pass down both path sum and path max and echo back from end; recur return the min of all paths;
  int dfs(int pos, int remain_grps, int grpsum, int pathmax, int[] arr) {
    pathmax = Math.max(pathmax, grpsum);
    if(pos==arr.length) {
      if(remain_grps==1 || remain_grps==0 ) {return pathmax; } // return the passed down pathmax at leaf;
      else return (int)1e9;
    }
    if(remain_grps==0) return (int)1e9;
    // pos merged to cur grp; recur to pos+1 with cutmax; pos rets together when the cur grp rets.
    int nosplit = dfs(pos+1, remain_grps, grpsum+arr[pos], pathmax, arr); 
    // pos merged to cur grp; pos+1 starts new grp with 0 cutmax and pathmax.
    int split = dfs(pos+1, remain_grps-1, /*new grp sum*/0, Math.max(pathmax, grpsum+arr[pos]), arr);
    // path max is passed down and echoed back. On recur ret, ret the min of all paths;
    return Math.min(split, nosplit); 
  }
  return dfs(arr, 0, days, 0, 0); 
}
int shipWithinDays(int[] arr, int days) { // MinMax always use binary search.
  boolean canShipByDays(int[] arr, int days, int cap) { // fixed grps, cuts;
    int shipsum=0, needdays=1;
    for(int i=0;i<arr.length;i++) {
      if(shipsum + arr[i] <= cap) { shipsum += arr[i];} // arr[i] packed into current group;
      else { needday++; shipsum = arr[i]; } // add arr[i] overflow. restart a new grp with it;
    }
    return needdays <= days;
  }
  int left=0,rite=0; // search space, min is max of arr[i], max is tot;
  for(int i=0;i<arr.length;i++) {
    left = Math.max(left, arr[i]);
    rite += arr[i];
  }
  int mxcap=(int)1e9;
  while(left <= rite) {
    int cap = left+(rite-left)/2;
    if(canShip(arr, days, cap)) { 
      rite = cap-1; 
      mxcap = Math.min(mxcap, cap);
    } else { left=cap+1; }
  }
  return mxcap;
}

class Solution {  
  // 2597. Beautiful Subsets; grp[a_i_modk]={i,}. within a grp, g[i]-g[i-1]=k; can not rob adjacent.
  int dfs(int i, int k, Set<Integer> path_num_set, int[] sorted) { // powerset at i, pass down prefix constraints;
    if(i==sorted.length) return 0; // one of total 2^h recur ends at leaf; ret 0 and parent accumulate each path by +1.
    int tot = dfs(i+1, k, path_num_set, sorted); // aggre paths exclude i; 
    if (!path_num_set.contains(sorted[i]-k)) { // path set does not have arr[i]-k; add arr_i; dfs;
      tot += 1; // arr_i itself a set
      path_num_set.add(sorted[i]);
      tot += dfs(i+1, k, path_num_set, sorted); // dfs incl num_i, 1+dfs(i+1);
      path_num_set.remove(sorted[iminstk]);
    }
    return tot; // sum of incl and skip arr_i;
  }
  // for(n:nums) {map.compute(n%k, (key,v)->v.add(n);}
  // 368. Largest Divisible Subset; from root 0, incl/skip, add i to cur prefix set;
  int dfs(int i, char prefix_tail, char[] arr, int k) { // recur pass down last bit as prefix constraints.
    if(i==arr.length) return 0;
    int keep=0, skip=0;
    skip = dfs(i+1, prefix_tail, arr, k); // not include i into cur path subset. 
    if(prefix_tail == '0' || Math.abs(arr[i]-prefix_tail)<=k) {
      keep = 1 + dfs(i+1, arr[i], arr, k); // select arr_i into cur path, add 1 to sum up aggr recur tail=arr[i]
    }
    return Math.max(keep, skip);
    
    // Brutal Force. sort, n^2; inner try each 0..I find the longest mod J;
    Arrays.sort(arr);
    List<Integer>[] f = new ArrayList[arr.length];
    f[0] = new ArrayList<>(); f[0].add(arr[0]); // F[i][stride].addAll(F[i-1][stride]);
    List<Integer> maxl = f[0];
    for(int i=1;i<arr.length;i++) { // iter each i;
      List<Integer> il = new ArrayList(); il.add(arr[i]);
      for(int j=i-1;j>=0;j--) {  // try each prev j; brutal force.
        if(arr[i] % arr[j] == 0 && f[j].size()+1 >il.size()) {
          il = new ArrayList<>(f[j]); il.add(arr[i]);
          if(maxl==null || il.size() > maxl.size()) maxl=il;
        }
      }
      f[i] = il; // F[i]=F[j]+arr[i];
    }
    return maxl;
  }
  int numberOfArithmeticSlices(int[] nums) { // LC 446 M*N; dp[N][Diff] all pairs(j,i) diff(j,i)
    int tot=0, sz=nums.length;
    Arrays.sort(nums);
    int[][] f = new int[sz][nums[sz-1]-nums[0]+1]; // F[index][stride]; or F[left][rite];
    for(int i=1;i<nums.length;i++) { // iter sorted each i
      for(int j=0;j<i;j++) { // sum up all segments of [0..i]
        int stride=nums[i]-nums[j];
        f[i][stride] += 1+f[j][stride];
        tot += 1+f[j][stride];
      }
    }
    return tot - (sz*(sz-1)/2); // minus sequence with 2 ele.
  }

  // 2370. Longest Subseq; F[i]=max(F[l], F[r])+1; track char's last idx; n^2 to loop 26 prev chars;
  int longestIdealString(String s, int k) { // diff(adj) < k; not any pair < k; 
    char[] arr=s.toCharArray();
    int[] f = new int[arr.length]; // longest of substr [0..i]; 
    int[] char_last_idx = new int[26]; for(int i=0;i<26;i++) char_last_idx[i]=-1;
    f[0]=1; char_last_idx[arr[0]-'a'] = 0;
    int maxlen=1, win_maxlen=0;
    for(int i=1;i<arr.length;i++) {
      f[i] = 1;
      for(int j=0;j<26;j++) { // extra const loop, within k win.
        if(Math.abs(arr[i]-'a'-j)<=k && char_last_idx[j] >= 0) {
          f[i]=Math.max(f[i], f[char_last_idx[j]]+1); }}
      char_last_idx[arr[i]-'a']=i;
      maxlen=Math.max(maxlen, f[i]);
      
      int[] dp = new int[26]; dp[arr[0]-'a']=1;  // maxlen of string ending at each char. only 26 chars.
      for(int j=Math.max(0, arr[i]-'a'-k);j<=Math.min(arr[i]-'a'+k, 25);j++) {
        win_maxlen = Math.max(win_maxlen, dp[j]);} // max of all chars within k, and plus 1;
      dp[arr[i]-'a']=Math.max(dp[arr[i]-'a'], 1+win_maxlen); // of all in scope char, take max of len, +1;
      maxlen = Math.max(maxlen, dp[arr[i]-'a']);
    }
    return maxlen;
  }
}
// LC 1011. Capacity To Ship in D Days. Min of d grp's max; each grp track Group Max;
class Solution { 
  // LC 2218. Maximum K Coins from piles; subset select; F[pi][k]=F[pi-1][k-pi_coins]+A[pi][pi_coins] 
  int dfs(int pidx, int remain, List<List<Integer>> piles, int[][] cache) { // f(pile,remain)
    int psum=0, mxsum=0;
    if(remain==0) return 0;
    if(cache[pidx][remain] != 0) return cache[pidx][remain];
    if(pidx==piles.size()-1) { // end state; sum all remain coin;
      for(int i=0;i<Math.min(remain, piles.get(pidx).size());i++) {mxsum+=piles.get(pidx).get(i);}
      cache[pidx][remain]=mxsum; 
      return mxsum;
    }
    mxsum = dfs(pidx+1, k, piles, cache); // skip pile pidx totally 
    for(int pc=1;pc<Math.min(remain, piles.get(pidx).size());pc++) { // no maxheap, max the sum up of each piece sequentially.
      psum += piles.get(pidx).get(pc); // each pile take at most remaining k coins;
      mxsum = Math.max(mxsum, psum + dfs(pidx+1, remain-pc, piles, cache)); // sum up with next pile;
    }
    cache[pidx][k]=mxsum; 
    return mxsum;
  }
  int maxValueOfCoins(List<List<Integer>> piles, int k) {
    int[][] cache = new int[piles.size()][k+1];
    return dfs(0, k, piles, cache);
    // F(n,k) = max(f(n-1,k), // pick k from previous n-1 piles
		// 	max(f(n-1, k-j-1)+sum(0 to j) for j = 0 to min(k,p[n-1].size())) // pick j+1 from current pile and k-j-1 from previous n-1 piles
		// )
    int[][] dp = new int[n+1][K+1]; // F[i][k]=F[i-1][k-icoins]+A[i][icoins];
    for(int i = 1; i <= n; i ++) {
      for(int k = 1; k <= K; k ++) {
        dp[i][k] = dp[i-1][k];
        int sum = 0, picked=1;
        for(int picked=1;picked<k;picked++) {
          sum += piles.get(i-1).get(picked);
          dp[i][k] = Math.max(dp[i][k], dp[i-1][k-picked]+sum);
        }
      }
    }
    return dp[n][K];
  }

  // iter different new end cut conti ary[pos..end] as a new group;
  int dfs(int pos, int remain_cut, int[] arr) { // F[i][cuts] = F[l][cuts-1]+A[l..i];
    int seg = 0, min_of_max_seg = (int)1e9;
    if(pos>=arr.length) {
      if(remain_cut==0) return 0;  else return (int)1e9; // ret min upon valid path. ret max val upon invalid;
    }
    if(remain_cut==0) return (int)1e9; // invalid, no remain cuts but more days left.
    // pos forms a new cut, vs. pos join parent group; advance one cut with itering span [pos...k], dfs(k+1);
    // dfs(pos+1, remain_cut, pathsum+arr[pos]);
    for(int cut_end=pos;cut_end<=arr.length-remain_cut;cut_end++) {
      seg += arr[cut_end];
      min_of_max_seg = Math.min(min_of_max_seg, Math.max(seg, dfs(cut_end+1, remain_cut-1, arr)));
    }
    if(min_of_max_seg==(int)1e9) return -1;
    return min_of_max_seg;
  }
}
// LC 1335. Minimum Difficulty of a Jobs; F[i][grps]=F[l][grp-1]+A(l..i);
class Solution { 
  int dfs(int pos, int remain_grps, int[] arr, Map<String, Integer> map) { // each job, start a new grp or join parent recursion;
    if(pos>=arr.length) {if(remain_grps==0) return 0; else return (int)1e9;}
    if(remain_grps==0) { return (int)1e9; }
    String key = String.format("%d:%d", pos, remain_grps);
    if(map.containsKey(key)) { return map.get(key); }
    
    int grpmax=0, minsum=(int)1e9;
    for(int i=pos;i<=arr.length-remain_grps;i++) { // a cut starts from pos, with different spans;
      grpmax=Math.max(grpmax, arr[i]);
      minsum = Math.min(minsum, grpmax + dfs(i+1, remain_grps-1, arr, map)); // recur d-1 with the new recur from i+1;
    }
    if(minsum==(int)1e9) return -1;
    map.put(key, minsum);
    return minsum;
  }
  // k grps over arr; at i, min of either (new grp from i, cut-1) or (merge i to parent grp); cuts-1 & pos+1 a new cut vs. no cut merge to cur cut; after pos;
  int dfs(int i, int remain_grps, int parent_grp_max, int[] arr, Map<String, Integer> map) {
    if(i >= arr.length) { if(remain_grps==0) { return /* no remain grps*/0; } else return (int)1e9; }  // invalid cut.
    if(remain_grps==0) return (int)1e9; // invalid cut when arr NOT done but d=0
    parent_grp_max = Math.max(parent_grp_max, arr[i]); // consume arr_i into parent_grp_max;
    String key = String.format("%d:%d:%d", i, remain_grps, parent_grp_max);
    if(map.containsKey(key)) { return map.get(key); }
    int minsum = Math.min( // take min of all paths when recur rets;
      parent_grp_max + dfs(i+1, remain_grps-1, 0, arr, map), // close parent grp, sum its parent_grp_max and new cut from i+1,
      dfs(i+1, remain_grps, parent_grp_max, arr, map)); // i merged to cur grp, carry parent_grp_max to i+1; i rets along w/ cur grp rets later.
    map.put(key, minsum);
    return minsum;
  }
  int minDifficulty(int[] jobs, int d) {return dfs(jobs, 0, d, 0, new HashMap<>());}

  // F[job][grps]: grpMax(I, grps): an extra loop try k..i as a new grp. F(i,g) = F(l, g-1)+ max(arr[l..i]);
  int minDifficulty(int[] jobs, int grps) { // F[jobs][grps] = F[i-l][grp-1]+(l..i)
    if(jobs.length < grps) return -1;
    int[][] dp = new int[jobs.length][grps]; // min of partial [0..i][0..grps];
    int grp_max=0, kr_grp_max=0; // track max of [0..r]
    for(int r=0;r<jobs.length;r++) { // Iter each cut sequentially. NO maxheap.
      for(int g=0;g<grps;g++) { // Iter N grps so dp[i][g-1] carry over to dp[i][g];
        dp[0][g] = (int)1e9;  // arr[0] can not take cut>0, set to max. Note max int + 1 overflow to min int.
        kr_grp_max = Math.max(kr_grp_max, jobs[r]);
        if(g==0) { dp[i][0] = kr_grp_max; continue; }  
        kr_grp_max = 0;
        for(int l=r;l>0;l--) { // try [1..k..r] any seg [k..r] as a new grp;
          lr_grp_mx = Math.max(lr_grp_mx, jobs[l]); // max of the new grp
          dp[r][g] = dp[r][g] > 0 ? Math.min(dp[r][g], dp[l-1][g-1] + lr_grp_mx) : (dp[l-1][g-1] + max_kr);
        }
      }
    }
    if(dp[jobs.length-1][grps-1]==(int)1e9) return 0;
    return dp[jobs.length-1][grps-1];
  }
  // LC 2147. divide corridor; pre seg with 2 seats aggregate; 
  // F[left][rite] = F[left][m]*A[m..k]*F[k][rite], or F[end, stride]
  int dfs(int i, int seats, String corridor, Map<Pair<Integer, Integer>, Integer> cache) {
    int tot = 0;
    if(i==corridor.length()) return seats==2 ? 1 : 0;
    if(seats==2) {
      if (corridor.charAt(i) == 'S') {
        tot = dfs(i + 1, 1, corridor, cache); // seats=3, cut here;
      } else { // when arr/i=P and seat=2, sum cut at i and no cut at i;
        tot = (dfs(i+1,0,corridor,cache) + dfs(i+1,2,corridor,cache))%MOD;
      }
    } else { // seats<2, not cut, recur;
      tot = dfs(i+1, seats+(corridor.charAt(i)=='S'?1:0), corridor, cache);
    }
    cache.put(new Pair<>(i, seats), tot);
    return tot;
    // int[][] count = new int[corridor.length() + 1][3]; // 
    // count[corridor.length()][2] = 1;
    // for (int index = corridor.length() - 1; index >= 0; index--) {
    //  if (corridor.charAt(index) == 'S') {
    //    count[index][0] = count[index + 1][1]; // cut before idx, cur s to next grp;
    //    count[index][1] = count[index + 1][2];
    //    count[index][2] = count[index + 1][1];
    //  } else {
    //    count[index][0] = count[index + 1][0];
    //    count[index][1] = count[index + 1][1];
    //    count[index][2] = (count[index + 1][0] + count[index + 1][2]) % MOD;
    //  }
    // }
  }
  int numberOfWays(String corridor) { // sequentially check each seg(2S), aggreg by product.
    int l=-1,r=0,seats=0,plants_after_2S=0, tot=1; // track seats and plants_after_2S after matching seat=2;
    for(;r<arr.length;r++) {
      if(arr[r]=='S') { seats++;
        if(seats==1) continue;
        if(seats==2){ plants_after_2S=0; continue; } // track valid plants_after_2S; only after 2 SS;
        else if(seats > 2) { // saw 3 S, lseek l=r; S=1. 
          if(plants_after_2S > 0) { tot = tot*(plants_after_2S+1) % MOD; } // 3 SSS, cut at each plant.
          l=r; seats=1; continue; // lseek l=r, seats=1;
        }
      } else if(seats==2) { plants_after_2S++; } // only count plants_after_2S after 2 Seats;
    }
    if(seats<2) return 0; return (int)tot;
  }
  // LC 1547. Cut Stick; F[left, rite]=F[left,m]+A[m]+F[m+1,rite]; vs. F[end, stride] also fine;
  int minCostCutStick(int n, int[] cuts) { // grps not fixed, variable grps;
    Arrays.sort(cuts); // f(l,m)+f(m+1,r)+(r-l)
    int[] htcuts = new int[cuts.length+2]; // add head and tail
    System.arraycopy(cuts, 0, htcuts, 1, cuts.length); htcuts[0]=0; htcuts[cuts.length+1]=n;
    int[][] f = new int[n+1][n+1]; // F[left][rite]: min cost of range(l,r)=f(l,m)+f(m+1,r)+(r-l)
    Map<String, Integer> fmap = new HashMap<>();
    for(int left=1;left<=htcuts.length-2;left++) { // iter # of grps from 1..to N;
      for(int lcut=0,rite=lcut+left+1; rite<htcuts.length; lcut++,rite++) {
        for(int mid=lcut+1;mid<rite;mid++) {
          int lidx=htcuts[lcut], ridx=htcuts[rite], cutidx=htcuts[mid];
          f[lidx][ridx] = Math.min(
            f[lidx][ridx]==0 ? (int)1e9 : f[lidx][ridx], 
            ridx-lidx+f[lidx][cutidx]+f[cutidx][ridx]);
          fmap.put(lidx+":"+ridx, Math.min(fmap.getOrDefault(lidx+":"+ridx, (int)1e9), 
            ridx-lidx+fmap.getOrDefault(lidx+":"+cutidx, 0)+fmap.getOrDefault(cutidx+":"+ridx, 0)));
        }
      }
    } 
    return f[0][n]; return fmap.get(0+":"+n);
  }
  // LC 552. Student Attendance; 2 dims, N days, total abs, late strides;
  int checkRecord(int n) {
    int[][][] F = new int[n+1][2][3]; // 2 absents, 3 lates strides;
    F[0][0][0] =1;
    for(int len=1;len<=n;len++) { // len vs. index ? fill F[idx] collect from F[idx-1]; F[len]+append update F[len+1];
      for(int total_abs=0;total_abs<=1;total_abs++) { // no 2+ absents
        for(int late_stride=0;late_stride<=2;late_stride++) { // no 3+ lates
          F[len+1][total_abs][0] += F[len][total_abs][late_stride]; //  present, contribute to i+1
          if(total_abs < 1) { // append A will incr len and total_abs
            F[len+1][total_abs+1][0] += F[len][total_abs][late_stride]; // i is Absent,    
          }
          if(late_stride < 2) {
            F[len+1][total_abs][late_stride+1] += F[len][total_abs][late_stride];
          }
        }
      }
    }
    int count = 0;
    for(int total_abs=0;total_abs<=1;total_abs++) { // no 2+ absents
      for(int late_stride=0;late_stride<=2;late_stride++) {
        count += F[n][total_abs][late_stride];
      }
    }
    return count;
  }

  // LC 920. Number of Music Playlists. F[lens,uniqs]=F[len-1, uniqs]+F[len-1,uniqs+1] sum(add a new song, add a dup);
  int dfs(int listsz, int uniqsongs, int n, int goal, int k ) {
    if(listsz >= goal) { if(uniqsongs==n) return 1; else return 0;}
    int cnt = (n-uniqsongs) * dfs(listsz+1, uniqsongs+1, n, goal, k);
    if(uniqsongs > k) cnt += (uniqsongs-k) * dfs(listsz+1, uniqsongs, n, goal, k);
    return cnt;
  }
  public int numMusicPlaylists(int n, int goal, int k) { // F[playlist_len, uniq_songs]
    // return dfs(0, 0, n, goal, k);
    long[][] F = new long[goal+1][n+1]; // [playlist_len][uniq_songs]
    F[0][0]=1; // empty playlist count as boundary 1;
    for(int i=1;i<=goal;i++) { // at each intermediate step to goal, 2 choices.
      for(int uniqsongs=1;uniqsongs<=Math.min(i,n);uniqsongs++) {
        // add a new song. choices is n-(uniqsongs-1)
        f[i][uniqsongs] = (n-uniqsongs+1)*f[i-1][uniqsongs-1] % (int)(1e9+7);
        // add an old song, choice is uniqsongs-k
        if(uniqsongs>k) f[i][uniqsongs] += (uniqsongs-k)*f[i-1][uniqsongs];
        f[i][uniqsongs] %= (int)(1e9+7);
      }
    }
    return (int)f[goal][n];
  }
  
  // LC 2466, f[len+k] += f[len]; The count of F(len) contri to F(len+k); vs. 1+f(i) 
  int countGoodStrings(int low, int high, int zero, int one){ // no sort, no heap to leverage;
    // return dfs(new ArrayList<>(), low, high, zero, one);
    int[] f = new int[high+Math.max(zero, one)];
    for(int i=-1;i<high;i++) { // for(i=high-1..1) f[i] = f[i+zero];
      f[i+zero] += (i==-1 ? 1 : f[i]); f[i+zero] %= (int)1e9+7;
      f[i+one] += (i==-1 ? 1 : f[i]); f[i+one] %= (int)1e9+7;
    }
    int cnt=0;
    for(int i=low-1;i<high;i++) { cnt += f[i]; cnt %= (int)1e9+7; }
    return cnt;
  }
}
// LC 2742: painting the walls. F[n_walls][painted_walls]: min cost; n_paint=(pay+no_pay);
class Solution { 
  int dfs(int i, int paidtm, int freetm, int[] cost, int[] time){ // recur 3 dims,  
    if(i>=cost.length) {
      if(paidtm >= freetm) return 0; // constraints(paidtm > freetm) satisfied; ret 0 as min cost to parent;
      return (int)1e9; // invalid, ret max as ret path takes min;
    }
    int min_cost_skip = dfs(i+1, paidtm, freetm+1, cost, time); // excl i to pay;
    int min_cost_incl = cost[i] + dfs(i+1, paidtm+time[i], freetm, cost, time); // incl i to pay
    return Math.min(min_cost_incl, min_cost_skip);
  }
  int dfs(int i, int remain, int[] cost, int[] time) {
    if(remain<=0) return 0;
    if(i==sz) return (int)1e9; // invalid dfs, iter all options not paying enough to reduce remain;
    return Math.min(
      dfs(i+1, remain, cost, time),  // do not pay for wall i, no
      cost[i] + dfs(i+1, remain-1-time[i], cost, time)
    );
  }
  // dp[wall][paint]: incr N walls, incr painted(pay+no_pay) walls; no pay/no_pay dim; no pay no paint; F[i][paint]=F[i-1][paint]
  int n = cost.length;
  int[][] F = new int[n+1][n+1]; // min cost of n walls, 
  for(int i=1;i<=n;i++) F[0][i] = (int)1e9; // F[0][i] is invalid, set max value;
  for(int i=1;i<=n;i++) { // no sort, no heap to reduce to lgn; at wall i, either pay or not pay;
    for(int paint_walls=1;paint_walls<=n;paint_walls++) { // enum painted wall values;
      int not_pay = F[i-1][paint_walls]; // no pay, no advance of paint_walls;
      int pay = cost[i-1] + F[i-1][Math.max(0, paint_walls-1-time[i-1])]; // 1+free_painter_walls;
      F[i][paint_walls] = Math.min(pay, not_pay); // pay i, add cost[i];
    }
  }
  return F[n][n]; // over N walls, paint_walls N walls;
}

// 1531. String Compression: dp[i, dels]: compress arr_i, k dels of pre arr_i-1; minlen; 
// n^3; extra loop to compress [l..i][dels] so F[i,d]=F[l-1,d-li_diffs]+1+li_dups;
int getLengthOfOptimalCompression(String s, int k) {
  // IF s_i==prev,no del s_i; ELSE min(del s_i OR no del s_i); no del s_i, s_i is new prev;
  dfs(i, dels, prev, prev_cnt) { 
    if(i==s.length()) { if(dels==k) return 0; }
    if(s[i]==prev) { minlen = dfs(i+1,dels,prev,prev_cnt+1) + prev_cnt==[1,9,99] ? 1 :0; } 
    else { return min(dfs(i+1,dels-1,prev,prev_cnt), 1 + dfs(i+1, dels, s[i], 1));} 
  }
  char[] arr = s.toCharArray();
  int[][] F = new int[s.length()+1][k+1]; // N chars, k dels; F[i,d]=F[l,d-li_diffs]+1+li_dups;
  for(int i=1;i<=arr.length;i++) { // F[i][d] = F[l-1][d-li_diffs], to avoid i-1 OOB, F[0][0]=0;
    for(int d=0;d<=Math.min(k, i);d++) { // enum each dels at arr_i
      int li_dups=0,li_diffs=0,duplen=0;
      F[i][d] = (int)1e9;
      if(d>0) F[i][d] = F[i-1][d-1]; // carry over when skip i;
      for(int l=i;l>0;l--) {  // an extra loop to compress arr[l..i];
        if(arr[l-1]==arr[i-1]) li_dups++; else li_diffs++; // del prev chars != arr[i] 
        if(li_dups > 99) duplen=3; else if(li_dups>9) duplen=2; else if(li_dups>1) duplen=1; else duplen=0;
        // if(d==0) F[i][d]=F[l-1][d]+1+duplen; // wrong when arr[l-1] != arr[i] 
        if(li_diffs <= d) { // == to include d=0; d=0, F[i]=F[i-1]+aN?
          F[i][d] = Math.min(F[i][d], F[l-1][d-li_diffs]+1+duplen); 
        } // when li_diffs>0, can not collect from F[l-1] even d==0; 
    }}}
  }
  return F[arr.length][k];
}
// LC 879. Profitable Schemes, N ary subsets count: totW <= maxW, totP >= minP
// each constrait is a dim, +ary. F[ary_i][totW][totP]=F[i-1][w-wi][p-pi];
// LC 502 IPO. profit with capital; not n2; sort and heap, nlgn;
class Solution { 
  int dfs(int idx, int w, int p, int maxW, int minP, int[] weight, int[] profit, int[][][] cache) {
    int cnt=0, MOD=(int)1e9+7;
    if(w > maxW) return 0;
    if(p >= minP) p=minP; // cur recursion profit capped by minP;
    if(cache[idx][w][p] > 0) return cache[idx][w][p];

    if(idx==weight.length-1) { // recur done.
      if(w <= maxW && p >= minP) cnt++; // excl arr[N]
      if(w + weight[idx] <= maxW && p+profit[idx] >= minP) cnt++; // incl arr[N]
      cache[idx][w][p] = cnt;
      return cnt;
    }
    cnt = dfs(idx+1, w, p, maxW, minP, weight, profit, cache) % MOD; // skip arr_i;
    if(w + weight[idx] <= maxW) { // can incl arr_i; recur new wt and profit;
      cnt %= cnt + dfs(idx+1, w+weight[idx], p+profit[idx], maxW, minP, weight, profit, cache);
    }
    cache[idx][w][p] = cnt;
    return cnt;
  }
  int profitableSchemes(int maxW, int minP, int[] weight, int[] profit) {
    // int[][][] cache = new int[weight.length][n+1][minP+1];
    // return dfs(weight, profit, n, minP, 0, 0, 0, cache);
    
    // totW < maxW AND totP > minP; enum
    int[][][] dp = new int[weight.length][maxW+1][minP+1]; // f[i,w,p] += f[i-1, w-wi, p-pi], sum from prev excl Wi,Pi;
    for(int i=0;i<weight.length;i++) { // expand each item, enum each W and Profit, collect all prev w-cost/i;
      for(int w=0;w<=maxW;w++) { // iter each wt value and each profit values;
        for(int p=0;p<=minP;p++) { // incl boundary p==0; dp[i][w][0] += dp[i-1][w][0];
          if(weight[i] <= w && profit[i] >= p) { dp[i][w][p] = 1; } // single arr/i is a subset;
          if(i>0) { // the solution num at i,w,p is the same as i-1,w,p;
            dp[i][w][p] += dp[i-1][w][p]; // skip item i, inherit i-1;
            if(weight[i] <= w) { // incl i, sum all prev satisfying i-1 w-cost/i;
              dp[i][w][p] += dp[i-1][w-weight[i]][Math.max(0, p-profit[i])]; // all big profits fall into bucket 0;
            }
          }
          dp[i][w][p] %= (int)1e9+7;
        }
      } 
    }
    return dp[weight.length-1][maxW][minP] + (minP==0 ? 1 : 0);
  }
  // LC 474. subset at most m 0s and 1s.
  int findMaxForm(String[] arr, int mzeros, int nones) {
   int[][] dp = new int[m+1][n+1]; // Not F[arr.length][m][n];
   for(int k=0;k<arr.length;k++) {
    int[] zos = zeros(arr[k]);
    for(int i=m;i>=zos[0];i--) {
      for(int j=n;j>=zos[1];j--) {
        dp[i][j] = Math.max(dp[i][j], dp[i-zos[0]][j-zos[1]]+1);
    }}}
    return dp[m][n];
  }
}
// LC 1626. Best Team With No Conflicts: sort by age, score when age the same;
// LC 502 IPO, two dims, no constrains on both dims. MaxHeap Subset selection.
int bestTeamScore(int[] scores, int[] ages) {
  int[][] f=new int[ages.length][100];  // max of i with wmk j; wmk mean min score to count. f[i,j<wmk]=0;
  for(int j=players[0].score;j<100;j++) { f[0][j] = players[0].score; }
  for(int age=1;age<ages.length;age++) {
    f[i][players[age].score] = f[age-1][players[i].score-1]+players[i].score;
    for(int score=players[age].score+1;j<100;j++) { // dp is n^2 
      f[age][score] = Math.max(f[i][j-1], f[i-1][j]);
    }
  }
  Arrays.sort(players, (a,b)-> { if(a.age!=b.age) return a.age-b.age; return a.score-b.score;});
  int[] f=new int[ages.length]; // F[i] = F[l] + max(l..i);
  PriorityQueue<Player> scoreMaxheap = new PriorityQueue<>():
  for(int i=0;i<ages.length;i++) { // sort by age
    f[i] = players[i].score; max=Math.max(max, f[i]);
    // for(int j=i-1;j>=0;j--){ // decision tree 2^n to n^2.
    //   if(players[j].score <= players[i].score) {
    //     f[i] = Math.max(f[i], f[j]+players[i].score);
    //     max=Math.max(max, f[i]);
    //   }
    // }
    f[i] = Math.max(f[i], f[scoreMaxheap.peek().idx]+players[i].score);
    scoreMaxheap.offer(players[i]);
  }
}

// LC 629. K Inverse Pairs Array; add 4 to 1,2,3 can gen 1 pair(1243), 2pairs(1423), 3pairs(4123). Sum.
// f(i,k) = f(i-1, k-0) + f(i-1, k-1) + f(i-1, k-i); loop all possible inv pairs formed from i;
int kInversePairs(int n, int k) {
  int[][] f = new int[n+1][k+1]; // n items, tot k inver pairs;
  for(int i=1;i<n+1;i++) { f[i][0] = 1; // for each I, counts rev pairs it can create. 
    for(int pairs=1;pairs<k+1;pairs++) { // incremental build up to k inv pairs;
      // when add 4 to [1,2,3,4], 4 gen 4 rev seq, (4,3 | 4,2,3 | 4,1,2,3|  
      for(int ipairs=0;ipairs<i;ipairs++) { // at i, it can form i-1 rev pairs by putting i to any prev slots;
        if(pairs >= ipairs) { // only count in pairs formed by i that is less than this recur's max; 
          f[i][pairs] += f[i-1][pairs-ipairs]; f[i][pairs] %= (int)1e9+7;
        }
      }
    }
  }
  return f[n][k];
}
// LC 1639. Number of Ways to Form a Target; i/wmk agg from i-1/[0..wmk];
class Solution {
  // dfs expand target char with watermark constraint. Enum each char, each watermark;
  int dfs(int ti, int watermark, String[] words, String target, Map<String, Integer> cache) {
    if(ti==target.length()) { return 1; }
    String key = String.format("%d:%d", ti, watermark); int tot = 0;
    if(cache.containsKey(key)) return cache.get(key);
    for(int i=0;i<words.length;i++) { // inside dfs of ti, iter each word
      for(int j=watermark; j<words[i].length();j++) { // iter each char after watermark
        if(target.charAt(ti)==words[i].charAt(j)) {
          tot += dfs(ti+1, j+1, words, target, cache);  // sum up dfs recur next char in target.
        }
      }
    }
    cache.put(key, tot); return tot;
  }
  // return dfs(0, 0, words, target, new HashMap<String, Integer>());
  int numWays(String[] words, String target) {
    // F[i][wmk]: # till target[i][wmk], sum each word[wmk]==target[wmk]; +=dp[i-1][k<wmk];
    int[][] dp = new int[target.length()+1][words[0].length()+1];
    int wdlen=words[0].length(), tot=0, MOD=(int)1e9+7;
    for(int i=0;i<target.length();i++) {
      for(int wmk=i;wmk<wdlen;wmk++) { // iter each wmk;
        if(wmk>0) dp[i][wmk] = dp[i][wmk-1]; // accumulate dp(i-1)
        for(int j=0;j<words.length;j++) { // only collect i-1 smaller watermark;
          if(target.charAt(i)==words[j].charAt(wmk)) {
            if(i==0) { dp[0][wmk] += 1;}
            if(i>0 && wmk>0) {
              dp[i][wmk] += dp[i-1][wmk-1]; dp[i][wmk] %= MOD; }
          }
        }}}
    return dp[target.length()-1][wdlen-1];
  }
}

boolean canPartition(int[] arr) {
  int sum = Arrays.stream(arr).sum();
  if(sum % 2 == 1) return false;
  int[][] dp = new int[arr.length][sum/2+1]; // F[i,v]: exist subset sum to v;
  dp[0][0] = 1;
  if(arr[0] <= sum/2) dp[0][arr[0]] = 1;
  for(int v=1;v<=sum/2;++v) { // iter each target value,
    for(int i=1;i<arr.length;++i) { // iter each arr_i
      dp[i][v] = dp[i-1][v]; // skip i; f[i,v] the same as f[i-1,v]
      if(arr[i] > v) break;
      if(dp[i-1][v-arr[i]] == 1) dp[i][v] = 1;
    }
  }
  return dp[arr.length-1][sum/2] == 1;
}
// Partition to K groups, Equal Sum Subsets, Permutation search Recur. 
// Each i, dfs join parent or start new grp. The end: grp nums/sums and tot eles matched.
boolean canPartitionKSubsets(int[] arr, int k) {
  boolean dfs(int pos, int[] grpsum, int[] arr, int k, int target) {// grps sum as prefix;
    if(pos==arr.length) {for(int i=0;i<k;i++){ if(grpsum[i] != target) return false; } return true; }
    for(int i=0;i<k;i++) { // try dfs adding arr[pos] to each of k grps. 
      if(grpsum[i]+arr[pos] <= target) {
        grpsum[i] += arr[pos];
        if(dfs(pos+1, grpsum, arr, k, target)) return true;
        grpsum[i] -= arr[pos];
      }
    }
    return false;
  }
  return dfs(0, new int[arr.length], arr, k, Arrays.sum(arr)/k);
  
  // F[grp_mask]: repr grp with its mask as int; group mask with bit i repr arr/i; F[grp]: sum of grp 
  // F[mask]: sum of grp items. dp["1101"]: sum(arr[0], arr[1], arr[2]);
  int[] dp = new int[(1<<arr.length)];
  for(int i=0;i<dp.length;i++) dp[i]=-1; dp[0]=0;
  for(int mask=0; mask<(1<<arr.length); mask++) { // iter each grps by its mask, a grp is repred by its mask;
    if(dp[mask]==-1) continue;
    for(int i=0;i<arr.length;i++) { // if arr[i] not chosen in this grp, add it not exceed target.
      if((mask&(1<<i))==0 && dp[mask] + arr[i]<=target) { // arr_i not in grp, 
        dp[mask|(1<<i)] = (dp[mask] + arr[i]) % target; // add arr[i] into the grp.
      }
    }
  }
  return dp[(1<<arr.length)-1] == 0;
}

// divide into k groups; F[i,k] = F[l,k-1]+max(avg(l..i))
static int maxAvgSumKgroups(int[] arr, int k) {
  int sz = arr.length;
  int[] prefix_sum = new int[sz]; prefix_sum[0] = arr[0];
  for (int i=1;i<sz;i++) { prefix_sum[i] = prefix_sum[i-1] + arr[i]; }

  int[][] dp = new int[sz][k+1];
  for (int i=0;i<sz;i++) { dp[i][1] = prefix_sum[i]/(i+1); } // boundary case, just 1 grp
  for (int grp=2;grp<=k;grp++) { // iter incr grps from 2..k
    int[] nxtDp = new int[sz];
    for (int i=0; i<sz; i++) {  // for each grp, iter each arr_i; 
      for (int j=0; j<i; j++) { // try each prefix [0..j][k-1] grps, and [j..i] as a new grp;
        dp[i][grp] = Math.max(dp[i][grp], dp[j][grp-1] + (prefix_sum[i]-prefix_sum[j])/(i-j));
      }
    }
  }
  System.out.println("max k subary avg is " + dp[sz-1][k]);
  return dp[sz-1][k];
}
maxAvgSumKGroups(new int[]{9,1,2,3,9}, 3);

// split subary into k grps, min max sum among those k groups.
// binary search the max subgrp sum used to divide the ary. from l=max ele(n grps) to r=all sums.
int splitbysum(int[] prefix, int subsum) {
  int l=-1,r=0, grps=0;
  while(r<prefix.length) {
    if(l == -1 && prefix[r] < subsum || l>=0 && prefix[r]-prefix[l] < subsum) {
      r++; continue; // keep sliding r until grt than subsum;
    }
    l=r-1; grps++;
  }
  return ++grps;
}
int splitArray(int[] arr, int k) {
  int[] prefix = new int[arr.length];
  int maxv=0,tot=maxv;
  for(int i=0;i<arr.length;i++) {
    maxv = Math.max(maxv, arr[i]);
    prefix[i] = i==0 ? arr[i] : prefix[i-1] + arr[i];
    tot += arr[i];
  }
  int lo_bound=maxv,hi_bound=tot;
  while(lo_bound<=hi_bound) {
    int m=lo_bound+(hi_bound-lo_bound)/2;
    // treat m as the max subary sum, we want minimize it.
    // cut the ary with each subary <=sum. if subarys > k, increase subsum;
    int subarys = splitbysum(arr, m);
    if(subarys == k) return m;
    if(subarys > k) { 
      // too small subary sum leads to too many subarys
      l = m+1;
    } else { r=m-1; }
  }
  return l;
}

// Keep/Remove arr/i, not a decision tree dfs aggregation;
int minimumDeletions(String s) {
  int dfs(int l, int r, String s) {
    int la = l, lb=r, dels=0;
    if(l>=r) return 0;
    while(la < s.length() && s.charAt(la)=='a') la++;
    if(la==r+1) return 0;
    while(lb >= l && s.charAt(lb)=='b') lb--;
    if(lb==l-1) return 0;
    dels = 1 + Math.min(dfs(la+1, lb, s), dfs(la, lb-1, s));
    return dels;
  }
  return dfs(0, s.length()-1, s);

  int[][] f = new int[s.length()][2]; // 1 is keep, 0 is remov;
  f[i][1] = Math.min(f[prea][1]+(i-prea-1), f[preb][1]+i-preb);
  for(int i=0;i<s.length();i++) {
    if(s.charAt(i)=='a') ia++; else ib++;
    dels = (i+1-ia) + (sz-i-1)-(nb-ib);
    mindels = Math.min(mindels, dels);
  }
  return Math.min(mindels, Math.min(na, nb));
}

int maximalSquare(char[][] mat) {
  int maxsq = 0; // F[i,j] aggregate min from left, up and left_up.
  int[][] F = new int[mat.length][mat[0].length];
  for(int i=0;i<mat.length;++i) {
    for(int j=0;j<mat[0].length;++j) {
      if(mat[i][j] == '0') { F[i][j] = 0; continue;} 
      // take the min from left, above and above left.
      int above = i > 0 ? F[i-1][j] : 0;
      int left = j > 0 ? F[i][j-1] : 0;
      int aboveleft = i>0&&j>0 ? F[i-1][j-1] : 0;
      F[i][j] = Math.min(Math.min(above, left), aboveleft)+1;
        maxsq = Math.max(maxsq, F[i][j]);
      }
    }
  }
  return maxsq*maxsq;
}

// fence partition, F[end, k] = min(max(F[end-l,k-1]+sum[l..i])) for i in [0..n]
// subset sum, dp[i, v] = dp[i-1,v] + dp[i-1, v-A.i]
def integer_partition(A, n, k):   //  dp[i, v] = dp[i-1,v] + dp[i-1, v-A.i]
  m = [[0 for i in xrange(n)] for j in xrange(k)]  # min max value
  d = [[0 for i in xrange(n)] for j in xrange(k)]  # where divider is
  # calculate prefix sum
  for i in xrange(A): psum[i] = psum[i-1] + A[i]
  # fix and initial boundary, one divide, and one element
  for i in xrange(A): m[i][1] = psum[i]
  for j in xrange(k): m[1][j] = A[1]
  for i in xrange(2, n):
      for j in xrange(2, k):
          m[i][j] = sys.maxint
          for x in xrange(1, i):
              cost = max(m[x][j-1], psum[i]-psum[x])
              if m[i][j] > cost:
                  m[i][j] = cost
                  d[i][j] = x     # divide at pos x
  
// Two branches/States every day, hod or sod. Track each state.
// Profit from txn k is cumulated to next txn's bot cost.
// Profit at txn k is computed by the same txn bot.
// Correctness depends on the boundary values.
// verify with [2,3,2,4,1,6,8]
今天我没有持有股票，sod value is the max of( the same as sod i-1, or sell my hod/i-1 from the same txn )
Sod[day_i][k_txn] = max(Sod[i-1][k],  Bot[i-1][_k_] + prices[i])
                    max(  选择 rest,       选择 sell      )

Bot value is the max of (1. max_of_bot/i-1, or buy a/i with profit from sod txn k-1)
Bot[day_i][k_txn] = max(Bot[i-1][k], Sod[i-1][_k-1_] - prices[i])
                    max(  选择 rest  ,           选择 buy         )
// the max_hod/i carries max_hod/i-1, so at i, only need to look i-1, loop(0..i).
class Main {
  int MaxStockProfitTop2(int[] arr) {
    PriorityQueue toptxn = new PriorityQueue<Integer>((l, r) -> Integer.compare(r,l));
    int min_price = arr[0], cost =0;
    int state = 0; // 0 is not hold, 1 is bot;
    for(int i=1; i<arr.length; i++) {
        // if i-1 > i, if state=bot, sell(i-1); if state=sold, continue.
        // if i-1 < i, if state=bot, continue; if state=sold, buy(i-1);
        if (arr[i-1] < arr[i]) {  // nature order
            if (state == 0) { cost = arr[i-1]; state = 1; } 
            else { continue; }
        } else {  
            if (state == 0) { continue; } 
            else { toptxn.add(arr[i-1] - cost); state = 0; cost = 0; }
        }
    }
    if (state == 1 ) {  // Loop exit Final clean up after
        toptxn.add(arr[arr.length-1] - cost);
    }
    int max_profit = 0;
    while (!toptxn.isEmpty()) { max_profit += (int)toptxn.poll(); }
    return max_profit;
  };
  
  // static int MaxProfitDP(int[] arr, int txns) {
  //   boolean debug = false;
  //   int alen = arr.length; 
  //   int[][] hold = new int[txns][alen], sold = new int[txns][alen];
  //   hold[0][0] = -arr[0]; sold[0][0] = 0;
  //   for(int i=1;i<alen;++i) {
  //     hold[0][i] = Math.max(hold[0][i-1], -arr[i]);
  //     sold[0][i] = Math.max(arr[i]+hold[0][i-1], sold[0][i-1]);
  //   }
  //   // previous round sold[t-1][i-1]-arr/i carries the accumulated profit to next round hold. 
  //   for(int t=1;t<txns;++t) {
  //     int t_start = 2*t;  // this is to show correctness based on boundary values. 
  //     hold[t][t_start-1] = hold[t-1][t_start-1]; sold[t][t_start-1] = sold[t-1][t_start-1];
  //     hold[t][0] = hold[t-1][0]; sold[t][0] = sold[t-1][0]; // start at 1 is ok, just some redundant.
  //     for(int i=1/*t_start*/;i<alen;++i) { 
  //       hold[t][i] = Math.max(hold[t][i-1], sold[t-1][i-1]-arr[i]);
  //       sold[t][i] = Math.max(sold[t][i-1], arr[i] + hold[t][i-1]); 
  //     }
  //     if(debug) System.out.println("t="+ t + " sold = " + Arrays.toString(sold[t]));
  //   }
  //   return sold[txns-1][alen-1];
  // }
  int MaxProfitDP(int[] arr, int txns) {
    int[][] bot = new int[txns+1][arr.length];
    int[][] sod = new int[txns+1][arr.length];
    
    for(int t=1;t<=txns;++t) {
      bot[t][0]=-arr[0];sod[t][0]=0;  // as bot[t][0] is used for eval sod, can not be 0. must be -arr[i];
      for(int i=1;i<arr.length;++i) {
        bot[t][i] = Math.max(bot[t][i-1], sod[t-1][i-1]-arr[i]);
        sod[t][i] = Math.max(bot[t][i-1] + arr[i], sod[t-1][i-1]);
      }
      System.out.println(String.format("txn %d bot %s sod %s", t, Arrays.toString(bot[t]), Arrays.toString(sod[t])));
    }
    return sod[txns][arr.length-1];
  }

  int MaxProfitWithRest(int[] arr, int txns, int gap) {
    int alen = arr.length;
    int[][] hold = new int[txns][alen], sold = new int[txns][alen], rest = new int[txns][alen];
    hold[0][0] = -arr[0]; sold[0][0] = 0; rest[0][0] = 0;
    for(int i=1;i<alen;++i) {
      hold[0][i] = Math.max(hold[0][i-1], -arr[i]);
      sold[0][i] = Math.max(sold[0][i-1], hold[0][i-1] + arr[i]);
      if (i >= gap) { rest[0][i] = Math.max(rest[0][i-1], sold[0][i-gap]); } else { rest[0][i] = rest[0][i-1]; }
    }
    for(int t=1;t<txns;++t) {
      hold[t][0] = hold[t-1][0]; sold[t][0] = sold[t-1][0]; rest[t][0] = rest[t-1][0];
      for(int i=1;i<alen;++i) {
        if (i >= gap) { 
          hold[t][i] = Math.max(hold[t][i-1], rest[t-1][i-gap] - arr[i]);
          sold[t][i] = Math.max(sold[t][i-1], hold[t][i-1] + arr[i]);
          rest[t][i] = Math.max(rest[t][i-1], sold[t][i-gap]); 
        } else { 
          hold[t][i] = hold[t][i-1];
          sold[t][i] = Math.max(sold[t][i-1], hold[t][i-1] + arr[i]);
          rest[t][i] = rest[t][i-1];
        }
      }
    }
    return sold[txns-1][alen-1];
  }
  
  static void main(String[] args) {
      int[] arr = {2, 4, 1, 3, 5, 10};
      arr = new int[]{2, 4, 1, 5, 5, 6};
      System.out.println(MaxStockProfitDP(arr, 1) + " = 5");
      System.out.println(MaxStockProfitDP(arr, 2) + " = 7");
      System.out.println(MaxStockProfitDP(arr, 3) + " = 7");
      
      System.out.println("---------------------");
      arr = new int[]{6, 5, 2, 5, 4, 7};
      System.out.println(MaxStockProfitDP(arr, 1) + " = 5");
      System.out.println(MaxStockProfitDP(arr, 2) + " = 6");
      System.out.println(MaxStockProfitDP(arr, 3) + " = 6");

      System.out.println("---------------------");
      arr = new int[]{2, 3, 2, 4, 1, 6, 9}; 
      System.out.println(MaxStockProfitDP(arr, 1) + " = 8");
      System.out.println(MaxStockProfitDP(arr, 2) + " = 10");
      System.out.println(MaxStockProfitDP(arr, 3) + " = 11");
      
      System.out.println("---------------------");
      arr = new int[]{2, 4, 1, 2, 3, 6};
      System.out.println(MaxStockProfitWithRest(arr, 1, 1) + " = 6");
      System.out.println(MaxStockProfitDP(arr, 2) + " = 7");
      System.out.println(MaxStockProfitWithRest(arr, 2, 1) + " = 6");
      System.out.println(MaxStockProfitWithRest(arr, 2, 2) + " = 5");
  }
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Combination Permutation Powerset, SubsetSum: 
// outer loop coins, inner values, No Dup; Outer loop values, each coin contributes, DUP.
// dp[slots][target+1]; recur(k+1, sum[i..k], );
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Loop invoked from two recursions, instead of iter/for.
def combination(arr, prefix, remain, pos, result):
  if remain == 0: return result.append(prefix)  # add to result only when remain
  if pos >= len(arr): return result.append(prefix)
  incl_pos = prefix[:]  // deep copy local prefix to avoid interference among recursions.
  incl_pos.append(arr[pos])
  //recur next with incl and excl pos. Remember to guard offset < arr.length.
  recur(arr, incl_pos, remain-1, pos+1, result)
  recur(arr, prefix,   remain, pos+1, result)
  return

def comb(arr, hdpos, prefixlist, reslist):
  if hdpos == arr.length:  reslist.append(prefixlist)
  // construct candidates with rest everybody as header, Loop each candidate and recur.
  for i in xrange(hdpos, arr.length):
    curPrefixlist = prefixlist[:]
    swap(arr, hdpos, i) // swap, so hd can still be included in the rest candidates.
    curPrefixlist.append(arr[hdpos])
    comb(arr, hdpos+1, curPrefixlist, reslist) // <-- next is hdpos+1
    swap(arr, i, hdpos)
res=[];comb([1,2,3,4],0,[],res);print res;

// when dup allowed and order matters, recur each coin, not off in params. Recur is slow, use dp !
void combRecur(int[] arr, int prefix, int target, int[] res) {
  // on each recursion, all coins are considered to next recur(dup allowed and order matters).
  for(int i=0;i<arr.length;++i) {
    if(prefix+arr[i] == target) res[0]++;
    else if(prefix+arr[i] < target) {
      combRecur(arr, prefix+arr[i], target, res);
    }
  }
}

// recur without prefix; collect each arr/i in the ret path;
Set<Set<Integer>> powerset(int pos, Set<Integer> path, int[] arr) {
  Set<Set<Integer>> res = new HashSet<>();  
  if (pos == arr.length) { return res; } // no prefix; ret a empty; or ret the passed down path.
  Set<Set<Integer>> exclude = powerset(pos+1, arr); 
  res.addAll(exclude);  // add all child subset exclude arr/i
  for(Set<Integer> l : exclude) { // include arr/i by append arr/i to each subset;
    Set<Integer> include = new HashSet<>(l);
    include.add(arr[pos]);
    res.add(include);
  }
  HashSet<Integer> me = new HashSet<>();
  me.add(arr[pos]);
  res.add(me);
  return res;
}

// dp[i][v] = dp[i-1][v-slot_face(1..6)]; n slots. fill partial result slot i with 6 val; max N slots.
// as each slot sampled once, prefix slot/i/face5 is a different prefix than slot/i/face3. hence two dim dp[i][v]
static int combsumTotalWays(int slots, int target) {
  int[][] dp = new int[slots][target+1];
  for (int i=0;i<n;i++) { Arrays.fill(dp[i], 0); }
  for (int i=0;i<n;i++) {    // loop each slots
    for (int f=0;f<10;f++) { // one face value at slot i;
      for (int v=Math.max(1,f);v<target;v++) { // update sum to v.
        if (f == v) { dp[i][v] += 1;}
        dp[i][v] += dp[i-1][v-f];
      }
    }
  }
  return dp[n-1][target];
}
System.out.println(combsumWays(3,7));

// tab[i,v] = tot non descreasing at ith digit, end with value v
// dp[i,v]=sum(dp[i-1, v-k]) for k in 1..10;
// Total count = ∑ count(n-1, d) where d varies from 0 to n-1
def nonDecreasingCount(slots):
  tab = [[0]*10 for i in xrange(slots)]
  for v in xrange(10): tab[0][v] = 1
  for i in xrange(1, slots):
    for val in xrange(10):  // single digit value 0-9
      for k in xrange(val+1):  // k is capped by prev digit v
        tab[i][val] += tab[i-1][k]
  tot = 0
  for v in xrange(10):
    tot += tab[n-1][v]  // collect all 0-9 digits
  return tot
assert(nonDecreasingCount(2), 55)
assert(nonDecreasingCount(3), 220)

// count tot num of N digit with sum of even digits 1 more than sum of odds
// n=2, [10,21,32,43...] n=3,[100,111,122,...980]
// track odd[i,v] and even[i,v] use 2 tables
def digitSumDiffOne(n):
  odd = [[0]*18 for i in xrange(n)]
  even = [[0]*18 for i in xrange(n)]
  for l in xrange(n):
    for k in xrange(2*9):
      for d in xrange(9):
        if l%2 == 0:
          even[l][k] += odd[l-1][k-d]
        else:
          odd[l][k] += even[l-1][k-d]

// odd[i,s] is up to ith odd digit, total sum of 1,3,ith digit sum to s.
// for digit xyzw, odd digits, y,w; sum to 4 has[1,3;2,2;3,1], even digit, sum to 5 is
// [1,4;2,3;3,2;4,1], so tot 3*4=12, [1143, 2133, ...]
// so diff by one is odd[i][s]*even[i][s+1]
def diffone(n):
  odd = [[0]*10*n for i in xrange(n)]
  even = [[0]*10*n for i in xrange(n)]
  for k in xrange(1,10):
    even[0][k] = 1
    odd[1][k] = 1
  odd[1][0] = 1
  for i in xrange(2,n):
    for s in xrange((i+1)*10):
      for v in xrange(min(s,10)):
        if i%2 == 0:
          even[i][s] += even[i-2][s-v]
        if i%2 == 1:
          odd[i][s] += odd[i-2][s-v]
  cnt = 0
  for s in xrange((n-1)*10):
    if (n-1)%2 == 0:
      cnt += odd[n-2][s] * even[n-1][s+1]
    else:
      cnt += odd[n-1][s] * even[n-2][s+1]
  return cnt
print diffone(3)

// Min coins, not tot # of ways; Loop value first or coin first is the same.
int coinChangeMin(int[] coins, int target) {
  Arrays.sort(coins);
  int[] dp = new int[target+1];
  for(int i=1;i<=target;++i) { dp[i] = Integer.MAX_VALUE;} // track Min, not total. sentinel.
  dp[0] = 0;  // <-- dp[v-c=0] + 1;
  {
    for(int v=1;v<=target;++v) {
      for(int i=0;i<coins.length;++i) {
        if(v >= coins[i] && dp[v-coins[i]] != Integer.MAX_VALUE) {
            dp[v] = Math.min(dp[v], dp[v-coins[i]] + 1);
        }
      }
    }
  }
  {
    for(int i=0;i<coins.length;++i) {
      for(int v=coins[i];v<=target;++v) {
        if(dp[v-coins[i]] != Integer.MAX_VALUE) {
          dp[v] = Math.min(dp[v], dp[v-coins[i]]+1);
        }
      }
    }
  }
  return dp[target] == Integer.MAX_VALUE ? -1 : dp[amt];
}

// outer loop coins, inner loop value, [2,1] not possible. order does not matter.
// outer loop values, inner loop coins, dup, [1,2] != [2,1];
int combsumTotWays(int[] arr, int target) {
  // dp[i][v] = dp[i-1][v-slot_face(1..6)]; each slot i can only once with 6 face values; max N slots.
  int[] dp = new int[target+1];   // each option can have k dups that generate different prefix values.
  dp[0] = 1;   // when slot value = target, count single item 1.
  { // outer loop exhaust coin ordered one by one, combination, not possible produce both [1,2] and [2,1]
    for (int i=0;i<arr.length;i++) { // for each coin a/i, loop all values, hence k copies of a coin is considered.
      for (int v=arr[i]; v<=target; v++) { // v start from arr_i;
        // no need loop k. K dups is carried over built up from ((v-a/i)-a/i)-a/i
        dp[v] += dp[v-arr[i]]; // can take multiple a/i, += sum up. = will over-write.
      }
    }
  }
  { // outer loop values, at each v, every coin is considered again. can produce both [1,2] != [1,2]
    for(int v=1;v<=target;++v) {
      for(int i=0;i<nums.length;++i) {
          if(v >= arr[i]) {
              dp[v] += dp[v-arr[i]];
          }
      }
    }  
  }
  return dp[target];
}
combsumTotWays(new int[]{1,2,3}, 4);
combsumTotWays(new int[]{2,3,6,7}, 9);

void combSumRecur(int[] arr, int coinidx, int remain, int[] prefix, List<List<Integer>> gret) {
  if(remain == 0) {
    List<Integer> l = new ArrayList<>();
    for(int i=0;i<prefix.length;++i) {
      for(int j=0;j<prefix[i];++j) { l.add(arr[i]); }
    }
    gret.add(l);
    return;
  }
  if(coinidx >= arr.length) return;  // after checking remain == 0
  for(int k=0;k<=remain/arr[coinidx];++k) {  // k <= floor div.
    prefix[coinidx] += k;  // exhaust all possible of coin i before moving next.
    combSumRecur(arr, coinidx+1, remain-arr[coinidx]*k, prefix, gret);
    prefix[coinidx] -= k;
  }
  return;
}
List<List<Integer>> combinationSum(int[] arr, int target) {
  List<List<Integer>> ret = new ArrayList<>();
  Arrays.sort(arr);
  int[] prefix = new int[arr.length];
  search(arr, 0, target, prefix, ret);
  return ret;
}

// from root, fan out; multiways with diff prefix to pos, for each suffix, append head to prefix, recur suffix [i+1..];
// shall use dp, dp is for aggregation.
List<List<Integer>> combinSumTotalWays(int[] arr, int pos, int remain, List<Integer> prefix, List<List<Integer>> res) {
  if (remain == 0) { result.add(prefix.stream().collect(Collectors.toList())); return result; }
  { // caller single call jump start from 0; Loop each suffix[pos+1..], recur i+1, not pos+1;
    for (int i=pos;i<arr.length;i++) { // process each suffix with different prefix; recur i+1 means [0..i] done.
      for(int k=1;k<=remain/arr[i];++k) {  // incl arr[i] with at least 1+,  
        for(int j=0;j<k;j++) { prefix.add(arr[i]); } // <-- prefix is changed when recur down. restore it after recursion.
        recur(arr, /* next of coin i, not pos*/i+1, remain-k*arr[i], prefix, res); // <-- recur i+1 when outer loop over all coins.
        for(int j=0;j<k;j++) { prefix.remove(prefix.size()-1); }
      }   
    }
  }
  { // exhaust/consume head, incl/excl, recur head+1. start with k=0, exclude arr[pos]
    for(int k=0;k<=remain/arr[pos];++k) { // k=0 to recur excl arr/i, as there is no outer loop over all coins.
      for(int j=0;j<k;j++) { prefix.add(arr[i]); } // All dups of a/i as prefix before recur pass a/i
      recur(arr, pos+1, remain-k*arr[i], prefix, res); // <-- recur pos+1 after head.
      for(int j=0;j<k;j++) { prefix.remove(prefix.size()-1); }
    }
  }
  return result;
}
List<Integer> path = new List<>();
List<List<Integer>> result = new List<>();
System.out.println(combinSumTotalWays(new int[]{2,3,6,7}, 9, 0, path, result));

// F[i,v]=list of all subsets sum to v in arr[0..i]; f[i,v]=addAll(f[i-1,v-arr_i]+arr_i);
// Loop coin first, no perm dups. [1,2], not possible [2,1], hence no dup.
List<List<Integer>> combinationSum(int[] arr, int target) {
  List<List<Integer>>[] dp = new ArrayList[target+1];
  // Arrays.sort(arr);
  dp[0] = new ArrayList<>();
  dp[0].add(new ArrayList<>());
  // for(int v=1;v<=target,v++) { for c : coins {}}  // if outer looper value first, [2,2,3] != [2,3,2]
  for(int i=0;i<arr.length;++i) { // outer loop all coins, to avoid [2,3,2]; when arr/i is processed, no arr/i-1 appears.
    for(int v=arr[i];v<=target;++v) {
      if(dp[v-arr[i]] != null) {  // current v is built from prev v-arr[i]
        for(List<Integer> l : dp[v-arr[i]]) {
          List<Integer> vl = new ArrayList<>(l);
          vl.add(arr[i]);
          if (dp[v] == null) { dp[v] = new ArrayList<>();}
          dp[v].add(vl);
        }
      }
    }
  }
  return dp[target] == null ? new ArrayList<>() : dp[target];
}
// F[i,v]=list of all subsets sum to v in arr[0..i]; f[i,v]=addAll(f[i-1,v-arr_i]+arr_i);
List<List<Integer>>[] subsetsum(int[] arr, int target) {
  List<List<Integer>>[] pre = new ArrayList[target+1];
  for(int i=0;i<arr.length;i++) { // outer loop coins
    List<List<Integer>>[] cur = new ArrayList[target+1];
    for(int v=1;v<=target;v++) { // inner iter each values;
      if(pre[v]!=null) cur[v] = new ArrayList<>(pre[v]); // cumulate i-1 to i; not incl i, f[i,v]=f[i-1,v];
      if(v==arr[i]) { // arr/i == v, add list with single ele arr_i 
        if(cur[v]==null) {cur[v]=new ArrayList<>();}
        cur[v].add(Arrays.asList(i));
        continue;
      }
      if(i > 0 && arr[i] < v && pre[v-arr[i]] != null) { // prev only i > 0
        if(cur[v]==null) {cur[v]=new ArrayList<>();}
        for(List<Integer> l : pre[v-arr[i]]) {
          List<Integer> ll = new ArrayList<>(l);
          ll.add(i);
          cur[v].add(ll);
        }
      }
    }
    pre = Arrays.copyOf(cur, target+1);
  }
  return pre;
}

void PermSkipDup(int[] arr, int pos, List<Integer> prefix, List<List<Integer>> res) {
  if(pos == arr.length) { res.add(new ArrayList<>(prefix)); return res;}
  for(int i=pos;i<arr.length;++i) {
    if(i != pos && arr[i] == arr[pos]) continue;  // A.A,B = A,A.B
    if(i-1 > pos && arr[i] == arr[i-1]) continue; // ABB only needs swap BAB, as BBA is the next of BAB
    swap(arr, pos, i);
    prefix.add(arr[pos]);
      PermSkipDup(arr, pos+1, prefix, res);
    prefix.remove(prefix.size()-1);
    swap(arr, pos, i);
  }
}

void recur(String s, int off, Set<String> set, int[] res) {
  if(off == s.length()) { // update global only when reaching leaf
    res[0] = Math.max(res[0], set.size());
    return;
  }
  for(int i=off+1;i<=s.length();i++) {
    if(!set.contains(s.substring(off, i))) {
      set.add(s.substring(off, i));
      recur(s, i, set, res);
      set.remove(s.substring(off, i));
    }
  }
}
// when recur to me(pos), what substr can I +1. Not join or skip the parent prefix. 
int recur(String s, int off, Set<String> prefix) {
  int res = 0;
  if(off == s.length()) return res;
  for(int i=off+1;i<=s.length();i++) {
    if(!prefix.contains(s.substring(off, i))) {  
      prefix.add(s.substring(off,i));
      res = Math.max(res, 1+recur(s, i, prefix));
      prefix.remove(s.substring(off, i));
    }
  }
  return res;
}
maxUniqueSplit(s, 0, set);

//  recursive call this on each 0xFF '''
def sortdigit(l, startoffset, endoffset):
    l = sorted(l, key=lambda k: int(k[startoffset:endoffset]) if len(k) > endoffset else 0)

//  map lambda str(x) convert to string for at each offset for comparsion 
def radixsort(arr):
  // count sort each digit position from least pos to most significant pos.
  def countsort(arr, keyidx):
    out = [0]*len(arr)
    count = [0]*10
    strarr = map(lambda x: str(x), arr)
    karr = map(lambda x: x[len(x)-1-keyidx] if keyidx < len(x) else '0', strarr)
    for i in xrange(len(karr)):     count[int(karr[i])] += 1
    for i in xrange(1,10):          count[i] += count[i-1]
    // iterate original ary for rank for each, must go from end to start,
    for i in xrange(len(karr)-1, -1, -1):
      ranki = count[int(karr[i])]  # use key to get rank, and set val with origin ary
      // ranki starts from 1, -1 to convert to index
      out[ranki-1] = int(strarr[i])
      count[int(karr[i])] -= 1
    return out
  for kidx in xrange(3):
    arr = countsort(arr, kidx)
  return arr
print radixsort([170, 45, 75, 90, 802, 24, 2, 66])

// exp is 10^i where i is current digit offset, (v/exp)%10, the offset value
def radixsort(arr):
  def getMax(arr):
    mx = arr[0]
    for i in xrange(1,len(arr)):
      mx = max(mx, arr[i])
    return mx
  // exp is 10^i where i is current digit number
  def countsort(arr, exp):
    count = [0]*10
    output =[0]*len(arr)
    for i in xrange(len(arr)):
      count[(arr[i]/exp)%10] += 1
    for i in xrange(1,10):
      count[i] += count[i-1]
    for i in xrange(len(arr)-1,-1,-1):
      d = (arr[i]/exp)%10
      rank = count[d]
      output[rank-1] = arr[i]
      count[d] -= 1
    return output
  mx = getMax(arr)
  exp = 1
  while mx/exp > 0:
    arr = countsort(arr, exp)
    exp *= 10
  return arr
print radixsort([170, 45, 75, 90, 802, 24, 2, 66])

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Overlap Interval: sort by start, unroll [start,end] to [start, cnt+1], [end, cnt-1], sort;
// Sort by End. Bisect prev end time with start time. track global minHt, minEd. 
// Range is Interval Array, TreeMap<StartIdx, EndIdx> sorted, bisect TreeSet<End> with start time.
// Operation on intv[l..r] convert to diff ary diff[l]+1, diff[r]-1; diff_ary_prefix[i] is net op on i;
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
class Solution {
  int[] leftrite(int[][] arr, int start, int end) { // arr[l].end < start ... end > arr[r].start;
    // bisect arr[i].end with start, arr[l].end < start; no eq; [0..l] can be discarded safely.
    // bisect arr[r].start with end; end < arr[r].start; no eq; [r..n] can be discarded safely.
    int[] lr = new int[2];
    int l=0,r=arr.length-1;
    while(l<=r) { // update l even a single entry.
      int m = l+(r-l)/2;
      if(arr[m][1] < start) { l = m+1;} // no lift l when arr[l].end = start; as l will be merged with start. [0..l-1].end strictly < start;
      else { r=m-1;}
    }
    lr[0] = l-1; // [0..l-1], can be discarded safely.
    
    l=0; r=arr.length-1;
    while(l<=r) { // update l even a single entry.
      int m =l+(r-l)/2;
      if(arr[m][0] <= end) { l = m+1;} // lift l even end=arr[l].start so [l..N] is good arr[l].start is strictly > end;
      else { r=m-1;}
    }
    lr[1] = l; // [l..n] arr[l].start > end; include l;
    // if(lr[1] < lr[0]) { }
    return lr;
  }
  int[][] insert(int[][] arr, int[] newiv) {
    List<int[]> res = new ArrayList<>();
    int[] lr = leftrite(arr, newiv[0], newiv[1]);
  
    for(int i=0;i<=lr[0];i++) { res.add(arr[i]);} // left and rite untouch, merge the middiile 
  
    int mergedl = newiv[0], mergedr = newiv[1];
    if(lr[0]+1 < arr.length) { mergedl = Math.min(mergedl, arr[lr[0]+1][0]); }
    if(lr[1]-1 >= 0) { mergedr = Math.max(mergedr, arr[lr[1]-1][1]);}
    res.add(new int[]{mergedl, mergedr});
  
    for(int i=lr[1];i<arr.length;i++) {res.add(arr[i]);}
    return res.stream().toArray(int[][]::new);
  }
}
// unroll [start, end] to (start,1), (end, -1); max grp is the max overlapping;
int minGroups(int[][] arr) {
  int[][] events = new int[arr.length*2][2];
  for(int i=0,j=0;i<arr.length;i++,j+=2) {
    events[j][0]=arr[i][0]; events[j][1]=1;
    events[j+1][0]=arr[i][1]; events[j+1][1]=-1;
  }
  // to avoid sort, using counting in bucket[maxend];
  Arrays.sort(events, (a,b) -> {
    if(a[0] == b[0]) { return b[1] - a[1]; }
    return a[0]-b[0];
  });
  int maxoverlap=0, overlaps = 0;
  for(int i=0;i<arr.length*2;i++) {
    overlaps += events[i][1];  // exit event value is -1;
    maxgrps = Math.max(maxgrps, overlaps);
  }
  return maxgrps;
}

// cnt+1 when interval starts, -1 when interval ends. scan each point, ongoing += v.
class MyCalendar {
  TreeMap<Integer, Integer> tmlineMap = new TreeMap<>(); // TreeMap saves explicit sort;
  int book(int start, int end) {
    tmlineMap.put(start, tmlineMap.getOrDefault(start, 0) + 1); // 1 new event will be starting at [s]
    tmlineMap.put(end, tmlineMap.getOrDefault(end, 0) - 1); // 1 new event will be ending at [e];
    int overlap = 0, maxgrps = 0;         // overlap counter track loop status
    for (int v : tmlineMap.values()) {    // iterate all intervals.
      overlap += v;     // value is 1 when event starts and -1 when event ends
      maxgrps = Math.max(maxgrps, overlap);         // when v is -1, reduce actives length.
    }
    return maxgrps;
  }
}
// 2402. Meeting Rooms III
// unroll meetings to (start,1),(end,-1), take top availableRoomMinHeap when start, add to room heap when ends;

// interval overlap, track min_end; if st < min_ed, overlap++; else need a new arrow; update min_end;  
int findMinArrowBurstBallon(int[][] points) {
  if (points == null || points.length == 0)   return 0;
  Arrays.sort(points, (a, b) -> a[0] - b[0]);   // sort intv start, track minimal End of all intervals
  int minEnd = (int)1e9, count = 0;
  for (int i = 0; i < points.length; ++i) { //  minEnd tracks the min end of all pre intv; min_end avoid double counting overlap++; 
    if (points[i][0] > minEnd) {   // cur ballon start is bigger, no overlap      
      count++;                     // no overlap, one more arrow to clear prev all.
      minEnd = points[i][1]; // advance minEnd to point[i] when no overlap.
    } else {   // cur start overlaps min, if cur end is less, shrink minEnd to it;
      minEnd = Math.min(minEnd, points[i][1]); // overlap, possibly shrink minEnd
    }
  }
  return count + 1;
}
// F[l..r], iterate k between l..r, where k is the LAST to burst. Matrix Production.
// matrix product dp[l,r] = dp[l,k] + dp[k+1,r] + p[l-1]*p[k]*p[r].
// dp[l,r] = max(dp[l,k] + dp[k+1,r] + arr[l]*arr[k]*arr[r])

// F(i, j, left, right): max coins from bursting nums[i, j] with boundary left, [i, j], right. 
// The start T(0, n - 1, 1, 1) and the termination is T(i, i, left, right) = left * right * nums[i].
// T(i, j, left, right) = for k in i..j as the last one to burst.
//    max(left * nums[k] * right + T(i, k - 1, left, nums[k]) + T(k + 1, j, nums[k], right))
static int burstBallonMaxCoins(int[] arr) {
  int sz = arr.length;
  int[] nums = new int[sz+2];
  for (int i=0;i<sz;i++) { nums[i++] = arr[i]; }
  nums[0] = nums[sz] = 1;

  int[][] dp = new int[sz][sz];  // F[left,rite]=F[left,m]+A[m]+F[m+1,rite];
  for (int gap=2; gap<sz; gap++) {
    for (int l=0;l<sz-gap;l++) {
      int r = l + gap;
      for (int k=l+1; k<r; k++) {
        dp[l][r] = Math.max( dp[l][r], nums[l]*nums[k]*nums[r] + dp[l][k] + dp[k][r]);
      }
    }
  }
  return dp[0][sz-1];
}

// sort by fin. track global min; If new one is overlap, discard it as its end time is larger.
int eraseOverlapIntervals(Interval[] intervals) {
  if (intervals.length == 0)  return 0;
  Arrays.sort(intervals, (a, b) -> a[1] - b[1]); // by fin to leverage start < fin.
  int dels = 0, minEnd = intervals[0][1];
  for(int i=1;i<intervals.length;++i) {
    if(interval[i][0] >= minEnd) { minEnd = interval[i][1]; continue;}
    else {
      // overlap, discard one. if sort by start, reduce minEnd is new interval fin is smaller. del
      // if sort by fin, no need to reduce minEnd.
      if(arr[i][1] <= minEnd) {minEnd = arr[i][1];} 
      dels++;
    }
  }
  return dels;
}

// longest length of interval chain, sort interval by END time. interval scheduling.
static int longestIntervalChain(int[][] arr) {
  Arrays.sort(arr, (a, b) -> a[1] - b[1]);  // sort by end time.
  List<Integer> dp = new ArrayList<>();
  for (int i=0; i<arr.length; i++) {
    int istart,iend = arr[i][0][1];
    int idx = Arrays.binarySearch(dp, istart);  // bisect closest end time with this start time.
    if (idx == dp.length) {  dp.add(iend); }  // append END time to dp end
    else if (iend < dp[idx]) { dp.set(idx, iend); }  // overlap interval, shrink idx end with smaller.
  }
  return dp.size() - 1;
}
// The other way is stk, sort by end time, and stk thru it.
if stk.top().end > curIntv.end: stk.top().end = curInterv.end;

// LC 2381 shift letter. op on intv[l..r] converted to diff ary(dff[start]++, diff[end]--). 
// To increase intv[l..r] by 1: diff[l]+1; diff[r]-1; prefix[i] is sum of range ops overlap i;
// diff[i]: all prefix sum[j>=i] increased by diff[i]; prefix[i+1]=prefix[i]+diff[i];
String shiftingLetters(String s, int[][] shifts) {
  Arrays.sort(shifts, (a,b)->a[0]-b[0]);
  int[] diff = new int[s.length()];
  for(int i=0;i<shifts.length;i++) {
    int[] cur=shifts[i];
    if(cur[2]==1) { 
      diff[cur[0]]++;
      if(cur[1]+1 < s.length()) diff[cur[1]+1]--;
    } else {
      diff[cur[0]]--;
      if(cur[1]+1 < s.length()) diff[cur[1]+1]++;
    }
  }
  int[] prefix = new int[s.length()];
  prefix[0]=diff[0];
  for(int i=1;i<s.length();i++) { prefix[i]=prefix[i-1]+diff[i]; }
  StringBuilder sb = new StringBuilder();
  for(int i=0;i<s.length();i++) {
    int ord = s.charAt(i)-'a'; ord = (ord+prefix[i]) % 26; if(ord < 0) ord+=26;
    sb.append((char)('a'+ord));
  }
  return sb.toString();
}

// Range is Interval Array, TreeMap<StartIdx, EndIdx> represents intervals. on overlap intervals exist in the map.
class RangeModule {
  TreeMap<Integer, Integer> map;  // key is start, val is end.
  RangeModule() { map = new TreeMap<>(); }
  
  void addRange(int left, int rite) {
    if (rite <= left) return;
    Integer startKey = map.floorKey(left), endKey = map.floorKey(rite); // floorKey of the rite edge. (greatest key that <= rite, might eqs startkey
    if (startKey == null && endKey == null) {  map.put(left, rite); }
    else if (startKey != null && map.get(startKey) >= left) {   // end > left, startkey overlap new range, merge with max endkey[0] and rite
        map.put(startKey, Math.max(map.get(endKey), Math.max(map.get(startKey), rite)));
    } else {  map.put(left, Math.max(map.get(endKey), rite)); }
    
    // clean up overlapped intermediate intervals
    Set<Integer> set = new HashSet(map.subMap(left, false, rite, true).keySet());
    map.keySet().removeAll(set);
  }  
  boolean queryRange(int left, int rite) {
    Integer startKey = map.floorKey(left);
    if (startKey == null) return false;
    return map.get(startKey) >= rite;
  }
  void removeRange(int left, int rite) {
    if (rite <= left) return;
    Integer startKey = map.floorKey(left);
    Integer endKey = map.floorKey(rite);
    if (endKey != null && map.get(endKey) > rite) {
      map.put(rite, map.get(endKey));
    }
    if (startKey != null && map.get(startKey) > left) {
      map.put(startKey, left);
    }
    // clean up intermediate intervals
    Map<Integer, Integer> subMap = map.subMap(left, true, rite, false);
    Set<Integer> set = new HashSet(subMap.keySet());
    map.keySet().removeAll(set); 
  }
}

// exist pair(i,j) abs(i,j)<=win, abs(arr[i],arr[j])<=diff; 
// slide win with TreeSet to sort val in win; find least/max ele within arr_i +_ diff;
// TreeSet/TreeMap to balance height by rotating when adding arr[r]/remove arr[l] sliding win. 
boolean containsNearbyAlmostDuplicate(int[] arr, int win, int diff) {
  int l=0,r=l+1;
  TreeSet<Integer> winset = new TreeSet<>((a, b) -> a-b);  // Treeset to bisect lo/hi bound.
  winset.add(arr[l]);
  while(r<arr.length) { // expand r, bisect lowerbound in winset;
    int lowerbound=arr[r]-diff, upperbound=arr[r]+diff;
    Integer ceiling = winset.ceiling(lowerbound); // least element > lower bound
    if(ceiling != null && ceiling.intValue() <= upperbound) return true;
    if(r-l==win) { winset.remove(arr[l]); l++; }
    winset.add(arr[r]);
    r++;
  }
  return false;
}

// triplet inversion. LoopInvar: track loop i's inversion(bigger on left, smaller on rite)
// TreeMap as avl tree, headMap / tailMap / subMap.
long maxInversions(List<Integer> prices) {
  TreeMap</*price=*/Integer, /*cnt=*/Integer> tree = new TreeMap<>();
  int[] leftBig = new int[prices.size()],riteSmall = new int[prices.size()];
  for (int i=0; i<prices.size(); i++) {
      Map<Integer, Integer> larger = tree.tailMap(prices.get(i), false); // map keys are > key;
      // include duplicates.
      for (int v : larger.values()) { leftBig[i] += v; }
      tree.put(prices.get(i), tree.getOrDefault(prices.get(i), 0) + 1);
  }
  tree.clear();
  for (int i=prices.size()-1;i>=0;i--) {
      Map<Integer, Integer> smaller = tree.headMap(prices.get(i), false); // map keys are < key;
      for (int v : smaller.values()) { riteSmall[i] += v; }
      tree.put(prices.get(i), tree.getOrDefault(prices.get(i), 0) + 1);
  }
  int tot = 0;
  for(int i=1;i<prices.size()-1;i++) { tot += leftBig[i] * riteSmall[i]; } 
  return tot;
}
// prices =  [15,10,1,7,8]
// tot = (15, 10, 1), (15, 10, 7), and (15, 10, 8)

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// AVL Tree and TreeMap, sorted after Add/Removal. RedBlack tree Rebalance. 
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
class Node {
  int val, rank;
  Node left, rite;
  Node(int val) { this.val = val; this.rank = 1;}
}
AVLTree(object):
  def __init__(self,key):
    self.key = key; self.size = 1;  self.height = 1; self.left = self.rite = None;
  // # recurisve to child, rotate and update subtree stats in each recur return new root;
  def insert(self, key):
    if self.key == key: return self
    if key < self.key:
      if not self.left: self.left = AVLTree(key)
      else:             self.left = self.left.insert(key)
    else:
      if not self.rite: self.rite = AVLTree(key)
      else:             self.rite = self.rite.insert(key)
    self.updateHeight()  # rotation may changed left. re-calculate
    newself = self.rotate(key)  ''' current node changed after rotate '''
    print "insert ", key, " new root after rotated ", newself
    return newself  // ret rotated new root
  
  // delete a node.
  def delete(self,key):
    if key < self.key:   self.left = self.left.delete(key)
    elif key > self.key: self.rite = self.rite.delete(key)
    else:
      if not self.left or not self.rite:
        node = self.left
        if not node:  node = self.rite
        if not node:  return None
        else:  self = node
      else:
        node = self
        while node.left: node = node.left
        self.key = node.key
        self.rite = self.rite.delete(node.key)  // recursion
    self.updateHeight()
    subtree = self.rotate(key)
    return subtree

  // update current node stats by asking your direct children. The children also do the same.
  def updateHeight(self):
    self.height,self.size = 0
    if self.left:
      self.height = self.left.height
      self.size = self.left.size
    if self.rite:
      self.height = max(self.height, self.rite.height)
      self.size += self.rite.size
    self.height += 1
    self.size += 1

  # left heavy, pos balance, right heavy, neg balance
  # rite subtree left heavy, >, LR rotation. left subtree rite heavy, <, RL.
  def balance(self):    # balance subtree root is this node.
    if not self.left and not self.rite:    return 0
    if self.left and self.rite:            return self.left.height - self.rite.height
    if self.left:                          return self.left.height
    else:                                  return self.rite.height
  
  def rotate(self, key):
    bal = self.balance()
    if bal > 1 and self.left and key < self.left.key:       return self.riteRotate()
    elif bal < -1 and self.rite and key > self.rite.key:    return self.leftRotate()
    elif bal > 1 and self.left and key > self.left.key:   # < , left rotate left subtree.
        self.left = self.left.leftRotate()
        return self.riteRotate()  # then rite rotate after /
    elif bal < -1 and self.rite and key < self.rite.key: # >, rite rotate rite subtree.
        self.rite = self.rite.riteRotate()
        return self.leftRotate()  # left rotate after \
    return self  # no rotation
  ''' bend / left heavy subtree to ^ subtree by promote left child as new root'''
  def riteRotate(self):
    newroot = self.left  newrite = newroot.rite
    newroot.rite = self
    self.left = newrite

    self.updateHeightSize()
    newroot.updateHeightSize()
    return newroot   # ret the new root of subtree. parent point needs to update.
  ''' bend \ subtree to ^ subtree by promote rite child as new root'''
  def leftRotate(self):
    newroot = self.rite  newleft = newroot.left
    newroot.left = self
    self.rite = newleft
    
    self.updateHeightSize()
    newroot.updateHeightSize()
    return newroot

  def searchkey(self, key):
    if self.key == key:  return True, self
    elif key < self.key: return self.left.search(key) if self.left else False, self
    else: return self.rite.search(key) if self.rite else False, self 
  ''' recurisve, find subtree node that is max smaller than the key '''
  def maxSmaller(self, key):
    if key < self.key:
      if not self.left:   return None   # no smaller in this sub tree.
      else:               return self.left.maxSmaller(key)
    if key > self.key:
      if not self.rite:   return self    # cur is largest smaller
      else:               return max(self, self.rite.maxSmaller(key))
    return self
  ''' rank is num of node smaller than key, track each subtree size '''
  def getRank(self, key):
    if key == self.key:
      if self.left:   return self.left.size + 1
      else:           return 1
    elif self.key > key: 
      if self.left:   return self.left.getRank(key)
      else:           return 0
    else:   // root key < key
      r = 1
      if self.left:   r += self.left.size
      if self.rite:   r += self.rite.getRank(key)
      return r
  
// rite Smaller, and left smaller, the same.
def riteSmaller(arr):
    sz = len(arr)
    root = AVLTree(arr[-1])
    riteSmaller = [0]*sz
    for i in xrange(sz-2, -1, -1):
        v = arr[i]
        rank = root.getRank(v)
        root = root.insert(v)    // will update subtree size.
        riteSmaller[i] = rank
    return riteSmaller

assert(riteSmaller([12, 1, 2, 3, 0, 11, 4]) == [6, 1, 1, 1, 0, 1, 0] )
assert(riteSmaller([5, 4, 3, 2, 1]) == [5, 4, 3, 2, 1])

// - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - 
// Binary Index Tree: Prefix Sum query(arr[i]), allow update(arr[i], 1);
// Fenwick Tree must be 1-indexed; Segment tree as heap queue; each node covers a segment/range; root:0..N, left:0..m;
// the LSB bit, update all parents by adding LSB to the current index. query all parents by removing LSB;
// Fenwick Prefix Range Sum or Max; tree[arr[i]]; LIS can be solved by Fenwick tree; 
// - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - 
int tripletMaxProd(int[] arr) {
  int sz = arr.length;
  int[] RMax = new int[sz]; // LMin[i] arr[i]<cur; RMax[i] arr[i]>cur;
  RMax[sz-1] = arr[sz-1];
  for (int i=sz-2;i>=0;i--) { RMax[i] = Math.max(RMax[i+1], arr[i]); }
  AVLTree root = createAVLTree(arr);
  for (int i=0;i<sz;i++) {
    int leftmax = root.getLeftMax(arr[i]);
    tripletmax = Math.max(tripletmax, leftmax * arr[i] * RMax[i]);
  }
  return tripletmax;
}
print triplet_maxprod([7, 6, 8, 1, 2, 3, 9, 10])
print triplet_maxprod([5,4,3,10,2,7,8])

// Fenwick BinIndexTree, Prefix Sum or Max, add arr[i] to flipped tree bit to 1;
// Least significant bit, n&(-n), to move up and down segments in the tree:
// update: move from i->N add LSB to the current index: index += index & (-index).
// query: move from i->0 subtract LSB from the current index: index -= index & (-index).
// sum up range 0..13=(1101)₂, sum val from 13=(1101)₂, 12=(1100)₂, 8=(1000)₂.
// https://medium.com/carpanese/a-visual-introduction-to-fenwick-tree-89b82cac5b3c
int LSB(int idx) return (n & (-n)); // (n & n-1) turn off the LSB at the end;
vector<int> arr;  // original arr.
vector<int> fenwick_range_sum_tree = new int[arr.length+1]; // 1-indexed rang sum tree store partial sum [0..idx];
// the sum of first n element;
int query(int idx, int[] fenwick_range_sum_tree) { // sum 
  int sum = 0;
  while (idx > 0) { // iter each segment by -LSB, sum up value from range sum tree at each LSB index;
    sum += fenwick_range_sum_tree[idx]; // sum up [0..idx] from range sum_tree; 
    idx -= lsb(n);  // turn off LSB, reduce idx to 0;
  }
  return sum;
}
// update arr[idx] to newval; ent [idx] and update each segment on our way up.
void update(int idx, int newval, int[] fenwick_range_sum_tree) {
  int delta = newval-arr[idx];
  while (idx+1 < fenwick_range_sum_tree.size()) { // range sum tree is 1-indexed;
    fenwick[idx] += delta;  // every parent segment adds up delta in the range sum tree;
    idx += lsb(idx); // turn on LSB, idx up to N;
  }
  v[idx]=newval;
}

// LC 1395: mono ascending Triplet teams, arr[i]<arr[j]<arr[k];
// iter each i, left_smaller*rite_bigger; AVL tree, left tree children when adding arr/i 
// Binary Index Tree[maxrating+1]: rank_value as BIT ary index and set value to 1; 
// update(rank, 1): mark fenwick[arr[i]] as 1, sum all set bits is the # of ele with range ranges;
// query(rank_val1, val2): range(rank1..rank2) sum(eles) between two ranks; 
int numTeams(int[] ratings) {
  class Soldier {int rating; int idx;} Arrays.sort(soldiers, (a,b)-> a.rating - b.rating);
  int[] sorted = new int[ratings.length];
  for(int i=0;i<arr.length;i++) sorted[i] = i;
  mergesort(arr, 0, arr.length-1, sorted);
  int[] rank = new int[rating.length];
  for(int i=0;i<arr.length;i++) { rank[sorted[i]] = i;} // smallest idx is sorted[0], rank of it is 0;
  return countTeams(rank, maxratings, true) + countTeams(rank, arr.length, false);

  int countTeams(int[] rank, int maxrating, boolean ascending) {
    int teams = 0;  
    int[] range_sum_tree = new int[maxrating+1];
    
    for (int i = 0; i < n; i++) { // consum arr[i], update fenwick tree each; 
      int irank = rank[i] + 1; // BIT ary is indexed by rank; range_sum_tree[i] is ith rank, f[i]-f[j] is diff between rank j and rank i; 
      int left = ascending ? getSum(range_sum_tree, irank-1) : getSum(range_sum_tree, maxratings) - getSum(range_sum_tree, irank);
      int right = ascending ? maxratings - irank - (getSum(range_sum_tree, maxratings) - getSum(range_sum_tree, irank)) : irank-1- getSum(range_sum_tree, irank-1);
      teams += left * right;
      update(range_sum_tree, irank, 1); // value is always updated from 0 to 1 to indicate i is consumed. 
    } 
    return teams;
  }
  // fenwick range sum tree[i] stores freq of rank value i; query(l,r) is # of items with l < rank value < r; 
  void update(int[] range_sum_tree, int index, int val) {
    while (index < range_sum_tree.length) {
      range_sum_tree[index] += val;
      index += index & (-index); // update parent index [idx..N]
  }
  int getSum(int[] range_sum_tree, int index) { // get sum of items with rank < index
    int sum = 0;
    while (index > 0) { // collect range_sum/prefix from [0..index]
      sum += range_sum_tree[index]; // not necessarily prefix sum, can be range max;
      index -= index & (-index);
    }
    return sum;
  }
}
class BIT{
  vector<int> bit;
  public:
    BIT(int n) : bit(n+1) { }
    void update(int i, int val) {
      for(;i <= size(bit); i += i & -i) bit[i] += val;
    }
    int query(int i) {  // returns number of elements in [0...i]
      int res = 0;
      for(;i > 0; i -= i & -i) res += bit[i];
      return res;
    }
    vector<int> countSmaller(vector<int>& nums) {
      const int MAX = 1e4+1;   // max range
      for_each(begin(nums), end(nums), [](int& n){n += MAX;});    // converting range from [-10^4,10^4] to [0,2*10^4]
      BIT T(2*MAX);  // fenwick ary with arr[i] as index and val=1 for arr[i];
      for(int i = size(nums)-1; ~i; i--) 
        T.update(nums[i], 1); // arr[i] as fenwick ary index; val is 1;
        nums[i] = T.query(nums[i] - 1);  // range sum all 1s in range [0..arr[i]];
      return nums;
    }
}
// LC 1964. LIS; or Fenwick tree query range max;
int[] longestObstacleCourseAtEachPosition(int[] obstacles) {        
  tree = new int[10000001];
  int[] res = new int[n];
  for (int i = 0; i < obstacles.length; i++) {
    int len = query(obstacles[i]);  // Find the length of LIS for 'obstacles[i]'         
    res[i] = len + 1; // Update the result
    add(obstacles[i], len + 1); // Update the fenwick tree.
  }
  return res;
}    
// update the Fenwick tree with new length 'x' for all elements >= id
void add(int height, int len) {
  for (int i = height; i < tree.length; i += i & (-i)) { // all tree[larger_ht] bumps up; 
    tree[i] = Math.max(tree[i], len);
  }
}
public int query(int height) {
  int res = 0;
  for (int i = height; i > 0; i -= i & (-i)) {
    res = Math.max(tree[i], res);  // Range max, not prefix sum.
  }
  return res;
}

// segment tree by heap q. q[lchild]=parent*2+1; heap[0]=min(arr[0..n]), heap[1]=min(arr[0..mid]), heap[1]=min(arr[mid+1..n]);
// search starts from q[0] root; check each q node coverage[arr_st, arr_ed]; return min in the ret path. 
class RMQ {  # RMQ has heap ary and tree. Tree repr by root Node.
  class QNode { int q_idx; int arr_st_idx; int arr_ed_idx; int val; }
  heap = [None]*pow(2, 2*int(math.log(self.size, 2))+1)
  // seg tree repr by root Node with heap_q_idx 0, left heap_q_idx 1, rite heap_q_idx 2
  root = self.build(0, self.size-1, 0)[0]
  def build(self, arr_st_idx, arr_ed_idx, q_idx):  // heap q_idx, lc/rc of q_idx
    if arr_st_idx == arr_ed_idx: heap[q_idx] = RMQ.Node(q_idx, arr_st_idx, arr_ed_idx, arr[arr_st_idx]);
    mid = (arr_st_idx+arr_ed_idx)/2
    left,lidx = self.build(arr_st_idx, mid, 2*q_idx+1)  # lc of q_idx
    rite,ridx = self.build(mid+1, arr_ed_idx, 2*q_idx+2)  # rc of q_idx
    minval = min(left.minval, rite.minval)
    n = RMQ.Node(arr_st_idx, arr_ed_idx, minval, minidx, q_idx)
    n.left = left   n.rite = rite
    self.heap[q_idx] = n
    return [n,minidx]
  // search always start from queue head/tree root; search queue nodes cover the range.
  def rangeMin(self, root, arr_st_idx, arr_ed_idx):
    // ret max if requested range outside current node's subtree range. 
    if arr_st_idx > root.arr_ed_idx or arr_ed_idx < root.arr_st_idx:
      return sys.maxint, -1
    // range bigger than current node's range, return tracked min. 
    if arr_st_idx <= root.arr_st_idx and root.arr_ed_idx <= arr_ed_idx:
      return root.minval, root.minvalidx
    lmin,rmin = root.minval, root.minval
    // ret min of both left/rite.
    if root.left:   lmin,lidx = self.rangeMin(root.left, arr_st_idx, arr_ed_idx)
    if root.rite:   rmin,ridx = self.rangeMin(root.rite, arr_st_idx, arr_ed_idx)
    minidx = lidx if lmin < rmin else ridx
    return min(lmin,rmin), minidx
  
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Graph Travsal, Strong conn comp(SCC), Grouping UnionFind. Edge types, In-degrees.
// The variations: DFS/BFS. PQ, visited_time[], update one level per epoch. update min/max/UF
// dfs loop all options, max subtree. leaf ret back sum; parent update Map/Heap across all children; 
// Dijkstra: dist[nb]={cost, pathlen, obstacles}. cost[r,c]=hdcost+path(hd,r,c);  
// Shortest root->node is first time deq ndoe; kth dist is deque node minHeap kth times;
// BFS itself is shortest path when all edges wt is 1; LC 2045;
// All pairs shortest path: for(l,r,m) F[l,r]=F[l,m]+F[m,r]+?; map(s->d), next nbs, prereq are adj edges.
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// r,c has two paths to it. aggregate them; backwards [m,n] to 0; 
// f(r,c) is the min health needed when reaching r,c; f(r,c) + grid(r,c) >= f(r,c+1);
// grid(r,c) has 4 but need 6 to rite, hence f(r,c)= f(r,c+1)-grid(r,c);  
// f[i,j] = max(1, min(f[i+1,j], f[i,j+1])-grid[i,j])
int dungeon(int[][] arr) {
  int rows=arr.length,cols=arr[0].length;
  int[][] dp = new int[rows+1][cols+1]; Arrays.fill(dp, (int)1e9);
  dp[n][m - 1] = 1; dp[n - 1][m] = 1;
  for(int i=rows-1;i>=0;--i) { // f[i+1=rows] still inbound, 1e9;
    for(int j=cols-1;j>=0;--j) {
      int need = min(dp[i + 1][j], dp[i][j + 1]) - grid[i][j];                  
      dp[i][j] = need <= 0 ? 1 : need;
    }
  }
  return dp[0][0];
}
// loop robotA's options, loop robotB's options,
int cherryPickup(int[][] grid) {
  int m = grid.length, n = grid[0].length, ans=0;
  int[][][] dp = new int[m][n][n];
  for(int i=0;i<m;i++) for(int j=0;j<n;j++) Arrays.fill(dp[i][j], -1);
  dp[0][0][n-1] = grid[0][0] + grid[0][n-1];

  for(int i=1;i<m;i++) {
    for(int j=0;j<n;j++) { // loop robotA's options
      for(int k=j+1;k<n;k++) { // loop robotB's options
        int max = -1;
        for(int x=-1;x<=1;x++) { // loop A's directions
          for(int y=-1;y<=1;y++) { // loop B's directions
            if(j+x >= 0 && j+x < n && k+y >= 0 && k+y < n)
              max = Math.max(max, dp[i-1][j+x][k+y]);
        }}
        if(max != -1) dp[i][j][k] = max + grid[i][j] + grid[i][k];
        if(ans < dp[i][j][k]) ans = dp[i][j][k];
  }}}
  return ans;
}

int minFallingPathSum(int[][] A) {
  for (int i = 1; i < A.length; ++i)
    for (int j = 0; j < A.length; ++j) // cur cell aggregate state from prev step.
      A[i][j] += Math.min(A[i - 1][j], Math.min(A[i - 1][Math.max(0, j - 1)], A[i - 1][Math.min(A.length - 1, j + 1)]));
  return Arrays.stream(A[A.length - 1]).min().getAsInt(); // ret min of end row.
}
// LC 1289 Minimum Falling Path Sum II: n^2, loop each col(0..n) max(F[co]);
// max of all options at (r,c)=max(r,k) for k in cols; 
int minFallingPathSum(int[][] grid) {
  int[][] memo = new int[grid.length][grid.length];
  for (int col = 0; col < grid.length; col++) {
    memo[grid.length - 1][col] = grid[grid.length - 1][col];
  }
  for (int row = grid.length - 2; row >= 0; row--) {
    for (int col = 0; col < grid.length; col++) {
      int nextMinimum = (int)1e9; // min from valid cells of next row
      for (int nxtcol = 0; nxtcol < grid.length; nxtcol++) {
        if (nxtcol != col) {
          nextMinimum = Math.min(nextMinimum, memo[row + 1][nxtcol]);
        }
      }
      memo[row][col] = grid[row][col] + nextMinimum;
    }
  }
  int answer = Integer.MAX_VALUE;
  for (int col = 0; col < grid.length; col++) {
    answer = Math.min(answer, memo[0][col]);
  }
}
int findPaths(int m, int n, int N, int x, int y) { // within N moves, count paths;
  int count=0; M = (int)1e9 + 7;
  int dp[][] = new int[m][n];
  dp[x][y] = 1;
  for (int moves=1;moves<=N;moves++) { // within N moves, f(mov,r,c) each move loop all r,c
    int[][] temp = new int[m][n]; // each mov explore all options, f[mov]=f[mov-1];
    for (int i = 0; i < m; i++) { // loop all options of rows, cells;
      for (int j = 0; j < n; j++) { // dp[mov] carries dp[mov-1]
        if (i == 0 || i== m - 1) count = (count + dp[i][j]) % M; // reach the boundary, count.
        if (j ==0 || == n - 1) count = (count + dp[i][j]) % M;
        temp[i][j] = ( // cur cell aggregate prev path state 
          ((i > 0 ? dp[i - 1][j] : 0) + (i < m - 1 ? dp[i + 1][j] : 0)) % M +
          ((j > 0 ? dp[i][j - 1] : 0) + (j < n - 1 ? dp[i][j + 1] : 0)) % M
        ) % M;
      }
    }
    dp = temp;
  }
  return count;
}        
int countPaths(int n, int[][] edges) {
  Map</*node*/Integer, List<Path>> edgeMap = build(n, edges);
  if(!edgeMap.containsKey(0)) return 1;
  int[] src_paths = new int[n]; src_paths[0] = 1;
  long[] src_dist = new long[n]; // min dist[n] to src
  for(int i=1;i<n;i++) { src_dist[i]=Long.MAX_VALUE;}
  PriorityQueue<Path> minheap = new PriorityQueue<>((pa, pb) -> Long.compare(pa.dist, pb.dist));
  for(Path p : edgeMap.get(0)) { minheap.offer(p); } // all edges into heap
  while(minheap.size() > 0) {
    Path minpath = minheap.poll();
    if(minpath.dist > src_dist[minpath.to]) continue; // skip this path, not the shortest
    if(minpath.dist==src_dist[minpath.to]) { 
      src_paths[minpath.to] = add(src_paths[minpath.to], src_paths[minpath.from]); }
    if(minpath.dist < src_dist[minpath.to]) { // update dist[to]
      src_dist[minpath.to] = minpath.dist; 
      src_paths[minpath.to] = src_paths[minpath.from]; // update to node path
      if(minpath.to != n-1) {
        for(Path nbpath : edgeMap.get(minpath.to)) { // add explored new nbs.
          if(nbpath.to != minpath.from && src_dist[nbpath.to] > minpath.dist+nbpath.dist) 
            minheap.offer(new Path(nbpath.from, nbpath.to, src_dist[nbpath.from]+nbpath.dist));
        }
      }
    }
  }
  return src_paths[n-1];
}
// A Leaf-Root Tree, all nodes in the same component aggregate up to parent root.
class UnionFind { // fixed # of nodes
  int[] parent;   // parent[idx] = idx;
  int[] state;  // track component status at root. Longest Path, # of nodes, etc;
  Map<Integer, Integer> parentMap = new HashMap<>();
  UnionFind(int n) {
    parent = new int[n]; childrens = new int[n]; state = new int[n];
    for(int i=0; i<n; i++) { parent[i] = -1; chidrens[i]=0; state[i]=0;}
  }
  int setParent(int i, int p) { parent[i]=p; childrens[p]+=1; return p;}
  int findParent(int i) {
    if(parentMap.get(i) != i) { parentMap.put(i, findParent(parentMap.get(i))); return parentMap.get(i); }
    if (parent[i] != i) {  // recursion down to root where root is returned and node in path is updated to root;
      parent[i] = findParent(parent[i]); // recur in stk updated to root; path compression.
    }
    return parent[i]; // recur done to root where parent[root] is root; ret root;
  }
  boolean same_component(int i, int j) { return findParent(i) == findParent(j); }
  boolean union(int x, int y) { // union 2 grps, update state to grp root;
    int xr = findParent(x), yr = findParent(y);
    if(xr != yr) {
      if(childrens[xr] >= childrens[yr]) {
        setParent(y, xr); // parent[yr] = xr;  // next findParent will flatten the children of yr.
        childrens[xr] += childrens[yr];
        state[xr] = state[yr];
      } else {
        setParent(x, yr);//  parent[xr] = yr;
        childrens[yr] += childrens[xr];
        state[yr] = state[xr];
      }
      return true;
    }
    return false;
  }
  // returns the maxium size of union
  int maxUnion(){ // O(n)
    int[] count = new int[parent.length];
    int max = 0;
    for(int i=0; i<parent.length; i++){
      count[root(i)] ++;
      max = Math.max(max, count[root(i)]);
    }
    return max;
  }
}

// Longest Consecutive Sequence. longest consecutive elements sequence.
// bucket, union to merge. use the index of num as index to 
int longestConsecutive(int[] nums) {
  UnionFind uf = new UnionFind(nums.length); // use a parent ary to track and merge slots of nums[i].
  Map<Integer,Integer> map = new HashMap<Integer,Integer>(); // <value,index>
  for(int i=0; i<nums.length; i++) {
    if (map.containsKey(nums[i])){ continue; }
    map.put(nums[i], i); // merge nums_i +- 1 nb;
    if(map.containsKey(nums[i]+1)) { uf.union(i, map.get(nums[i]+1)); } // union index i and index of nums[i]+1;  
    if(map.containsKey(nums[i]-1)) { uf.union(i, map.get(nums[i]-1)); } // 
  }
  return uf.maxUnion(); // for(int v=1;v<maxv;v++) uf.union(v, v+1);
}
// LC 2493. group Nodes into the max Groups: BFS Longest path within each component;
// merge connected nodes to components; longest path within each component via BFS flooding.
int magnificentSets(int n, int[][] edges) {
  UF uf=new UF(n);
  int totgrps = 0;
  for(int[] e : edges) { uf.union(e[0], e[1]); } // connected component by edges;
  for(int i=0;i<n;i++) { // bfs from each node as root to get the longest path.
    int idepth = bfs(i); // update longest path from i to parent(i);
    map.compute(parent[i], (k,v) -> idepth > v ? idepth :v); // longest path within each component;
  }
  for(int grps : map.values()) { totgrps += grps; }
  return totgrps;
}

// LC 2503 max points from grid. adj nodes value < X form a component.
// find the size(# of node) of each component;
while (cellId < totalCells && sortedCells[cellIndex].value < query.value) {
  for (int direction = 0; direction < 4; direction++) {
    if (grid[adjRow][adjCol] < query.value) {
      uf.union(cellId, adjCellId);
    }
  }
}

// LC 2127: DAG Strong connected component with 2+ nodes vs 2 nodes;
// 2+ nodes cycle length; 2 nodes: path length of the connected component;

// - - - - - - - - - - - -// - - - - - - - - - - - -// - - - - - - - - - - - -
// From src -> dst, min path dp[src][1111111]  bitmask repr subset.
//  1. DFS(root, prefix); From root, for each child, if child is dst, result.append(path)
//   dfs(r,c, visited): for(nb): if nb_ok: ret=min(grid[r,c]+dfs(nb, visited))
//  2. BFS(PQ); PQ(sort by weight), poll PQ head, for each child of hd, relax by offering child.
//  3. DP[src,dst] = DP[src,k] + DP[k, dst]; In Graph/Matrix/Forest, all pair, loop gap/mid, loop src/dst, src,mid,dst
//      dp[i,j] = min(grid[i,j] + min(dp[i,k] + dp[k,j], dp[i,j]);
//      dp[i,j] = min(dp[i,j-1], dp[i-1,j]) + G[i,j];
//  4. Topsort(G, src, stk): map(src).add(dfs(child)); vs. BFS remove edge enq leaves; map(parent).add(leaf);
//  5. Edge_Type: TREE:parent[to]==from; BACK:disc[to] && !processed[to]; FORWARD: processed[to] && entry_time[to >(later) from]; 
//  6. node in path repr as a bit. END path is (111). Now enum all path values, and check from each head, the min.
// - - - - - - - - - - - -// - - - - - - - - - - - -// - - - - - - - - - - - -
// For DAG, enum each intermediate node, otherwise, topsort, or tab[i][j][step]
def minpathDFS(G, src, dst, cost, tab):
  mincost = cost[src][dst]
  // for each intermediate node
  for v in G.vertices():
    if v != src and v != dst:
      mincost = min(mincost, minpathDFS(G,src,v) + minpathDFS(G,v,dst))
      tab[src][dst] = mincost
  return tab[src][dst]

// Matrix min path from 0,0 -> rows,cols
int minPathSum(G) {
  int rows = G.size(), cols = G[0].size();
  int[] dp = new dp[cols];  // dp table ary width is cols.
  // boundary cols, add first row.
  for (int i=1;i<cols;i++) { dp[i] = dp[i-1] + G[0][i]; }
  for (int i=0;i<rows;i++) {
    dp[0] += G[i][0];
    for (int j=0;j<cols;j++) {
      dp[i] = Math.min(dp[i-1], dp[i]) + G[i][j];
    }
  }
  return dp[cols-1];
}

// mobile pad k edge, BFS search each direction and track min with dp[src][dst][k] '''
def minpath(G, src, dst, k):
  for gap in xrange(k):
    for s in G.vertices():
      for d in G.vertices():
        dp[s][d][gap] = 0 // subproblem deps on src, dst, gap;
        if gap == 1 and G[s][d] != sys.maxint:
          dp[s][d][gap] = G[s][d]
        elif s == d and gap == 0:
          dp[s][d][gap] = 0
        if s != d and gap > 1:
          for k in src.neighbor():
            dp[s][d][gap] = min(dp[k][d][gap-1]+G[src][k])
  return dp[src][dst][k]

// first step to use topology sort prune, tab[i] = min cost to reach dst from i'''
def minpath(G, src, dst):
  def topsort(G, src, disc, visited, memorize, stk):  // dfs from root to collect deps.
    def dfs(src,G,stk):
      disc[src] = 1
      for to in neighbor(G,src):
        if(edge(src,to)) == BACK: return false;
        if not disc[to]:
          process_edge(from, to)
          dfs(to, G, stk)
        if(directed_graph) process_edge(from, to)
        if visited[to]:
          stk.apend(memorized.get(to))
      visited[src] = true
      stk.append(src) // all children of src visited, add src also.
      return stk;
    
  for v in G: // each node as root to dfs;
    dfs(v,G,stk)

  toplist = topsort(G, src, stk)
  for i in g.vertices(): tab[i] = sys.maxint
  // after topsort, leave at the end of the list.
  dp[d] = 0   // dp[i] = dist 
  for nb in d.neighbor():   dp[nb] = cost[nb][d]
  for i in xrange(len(toplist)-1,-1,-1):
    for j in i.neighbor():
      dp[j] = min(dp[i]+cost[i][j], dp[j])
  return dp[src]

// BFS PQ as heap. Poll, for each nb, update cost, then offer(nb)
int cutOffTree(List<List<Integer>> forest, int sr, int sc, int tr, int tc) {
  int R = forest.size(), C = forest.get(0).size();
  HashMap<Integer, Integer> cost = new HashMap();
  cost.put(sr * C + sc, 0);
  PriorityQueue<int[]> heap = new PriorityQueue<int[]>( (a, b) -> Integer.compare(a[0], b[0]));
  heap.offer(new int[]{0, 0, sr, sc});
  while (!heap.isEmpty()) {
    int[] cur = heap.poll();
    int g = cur[1], r = cur[2], c = cur[3];
    if (r == tr && c == tc) return g;
    for (int di = 0; di < 4; ++di) {
      int nr = r + dr[di], nc = c + dc[di];
      if (0 <= nr && nr < R && 0 <= nc && nc < C && forest.get(nr).get(nc) > 0) {
        int ncost = g + 1 + Math.abs(nr-tr) + Math.abs(nc-tr);
        if (ncost < cost.getOrDefault(nr * C + nc, 9999)) {
          cost.put(nr * C + nc, ncost);
          heap.offer(new int[]{ncost, g+1, nr, nc});
        }
      }
    }
  }
  return -1;
}

// minheap edges dist. poll, add nb, relax with reduced stops. DP[to][k]=min(dp[from][k-1]+(from,to))
int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
  int[][] dp = new int[n][k+1];
  int[][] prices = new int[n][n];
  PriorityQueue<int[]> minheap = new PriorityQueue<>((a, b) -> a[0]-b[0]);
  for(int i=0;i<flights.length;++i) {
    int[] f = flights[i];
    prices[f[0]][f[1]] = f[2];
    if(f[0]==src) { minheap.offer(new int[]{f[2], f[1], 0}); }
  }
  while(minheap.size() > 0) {
    int[] e = minheap.poll();
    int pathsum=e[0], cur=e[1], hops=e[2]; //, pathsum=e[3];
    dp[cur][hops] = dp[cur][hops] > 0 ? Math.min(dp[cur][hops], pathsum) : pathsum;
    if(cur==dst) return pathsum;
    if(hops == k || pathsum > dp[cur][hops]) continue;
    for(int i=0;i<n;++i) { // for each children of to
      if(prices[cur][i] > 0 && i != src && (dp[i][hops] == 0 || prices[cur][i] < dp[i][hops])) {
        minheap.offer(new int[]{pathsum+prices[cur][i], i, hops+1}); 
      }
    }
  }
  int mincost = Integer.MAX_VALUE;
  for(int i=0;i<=k;i++) {if(dp[dst][i] > 0) mincost = Math.min(mincost, dp[dst][i]);
  }
  return mincost == Integer.MAX_VALUE ? -1 : mincost;
}

// minHeap(dist_src) with src, edges repr by G Map<Integer, List<int[]>>, for each nb edge, relax dist
int networkDelayTime(int[][] times, int N, int K) {
  Map<Integer, List<int[]>> adjacentMap = new HashMap();   // vertex --> [edge pair].
  for (int[] edge: times) { adjacentMap.computeIfAbsent(edge[0], new ArrayList<int[]>()).add(new int[]{edge[1], edge[2]}); }
  Map<Integer, Integer> distMap = new HashMap(); // src->dst mindist[dst] updated during exploring.
  PriorityQueue<int[]> minheap = new PriorityQueue<int[]>((info1, info2) -> info1[0] - info2[0]);    
  minheap.offer(new int[]{0, K}); // dist[src] is 0;
  while (!minheap.isEmpty()) {
    int[] info = minheap.poll();
    int d = info[0], node = info[1];
    if (distMap.containsKey(node))  continue; // already seen.
    distMap.put(node, d);
    if (adjacentMap.containsKey(node)) {
      for (int[] edge: adjacentMap.get(node)) {     // for each edge, relax distMap
        int nei = edge[0], d2 = edge[1];
        if (!distMap.containsKey(nei))   // only exlore unseen neighbors
          minheap.offer(new int[]{d+d2, nei}); // accumulate dist to nei;
      }
    }
  }
  if (distMap.size() != N) return -1;
  int ans = 0;
  for (int cand: distMap.values()) ans = Math.max(ans, cand);
  return ans;
}

// node is path as bit, the the path covers all node has (11111)
// enum all path masks, from any head. DP[head][path] = min(DP[nb][path-1])
int shortestPathToAllNode(int[][] graph) {
  int N = graph.length;
  for (int[] row: dist) Arrays.fill(row, N*N);
  int[][] dist = new int[1<<N][N]; // dist[path][node]
  for (int x = 0; x < N; ++x) dist[1<<x][x] = 0; // path to itself is 0;
  // iterate on all possible path values. At each path value, fill all header's value.
  for (int path=0; path< 1<<N; path++) { // iter each possible path with diff nodes
    boolean repeat = true;
    while (repeat) {
      repeat = false;
      for (int curnode = 0; curnode < N; ++curnode) { // iter each node under a path
        int d = dist[path][curnode];
        for (int nb : graph[curnode]) {
          int nxtpath = path | (1 << nb); // incl nb into path by turn it on;
          if (dist[nxtpath][nb] > dist[path][curnode] + 1) {
            dist[nxtPath][nb] = d+1; // dist to nb by path is 1 more;
            if (path == nxtpath) repeat = true;
          }
        }
      }
    }
  }
  int ans = N*N;
  for (int node: dist[(1<<N) - 1]) // path with all nodes set;
    ans = Math.min(node, ans);
  return ans;
}
// min vertex set, dfs, when process vertex late, i.e., recursion of 
// all its next done and back to cur node, use greedy, if exist
// children not in min vertex set, make cur node into min set. return
// recursion back to bottom up.
def min_vertex(root):
  if root.leaf        # for leaf node, greedy use parent to cover the edge.
    return false
  for c in root.children:
    s = min_vertex(c)
    if not s:
      root.selected = true
  return root.selected

// the same as Largest Indep Set.
def vertexCover(root):
  if root == None or root.left == None and root.rite == None:
    return 0
  return root.vc if root.vc

  incl = 1 + vertexCover(root.left) + vertexCover(root.rite)
  excl = 1 + vertexCover(root.left.left) + vertexCover(root.left.rite)
  excl += 1 + vertexCover(root.rite.left) + vertexCover(root.rite.rite)
  root.vc = min(incl, excl)
  return root.vc

def liss(root):
  root.liss = max(liss(root.left)+liss(root.rite), 1+liss(root.[left,rite].[left,rite]))


// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// DFS branches + DP
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
class IpSeg {
  int v; int nextOff; int seq; IpSeg parent;
  IpSeg(int v, int nextOff, int seq) { this.v = v; this.nextOff = nextOff; this.seq = seq; }
}
// Backtrack all intermediate result with a queue.
int restoreIP(String s) { // F[i][4] = sum(F[i-1][3], F[i-2][3], ...);
  int sz = s.length(), off=0, tot=0;
  Deque<IpSeg> q = new ArrayDeque<>();
  for (int k=0;k<3;k++) {
    int v = Integer.parseInt(s.substring(off, off + k + 1));
    if (v <= 255) { q.offerLast(new IpSeg(v, off+k+1, 1)); }
  }
  while (q.size() > 0) {
    IpSeg cur = q.pollFirst();
    if (cur.seq == 4 && cur.nextOff == sz) {tot += 1; continue; }
    if ( cur.seq >= 4) continue;
    for (int k=0;k<3;k++) {
      if (cur.nextOff + k < sz) {
        int v = Integer.parseInt(s.substring(cur.nextOff, cur.nextOfff + k +1));
        if (v <= 255) { q.offerLast(new IpSeg(v, cur.nextOff+k+1, cur.seq+1)); }
      }
    }
  }
  return tot;
}

// adding + in between. [1,2,0,6,9] and target 81.
// dp[i][val] += dp[i-l][val-A[l..i]]; iter all values; 
// F[left,rite] = F[left,m] + A[m..n] + F[n, rite];
def plusBetween(arr, target):
  def toInt(arr, st, ed): return int("".join(map(lambda x:str(x), arr[st:ed+1])))
  dp = [[False]*(target+1) for i in xrange(len(arr))]
  for i in xrange(len(arr)):
    for j in xrange(i,-1,-1):  # one loop j:i to coerce current val
      val = toInt(arr,j,i)
      if j == 0 and val <= target:
        dp[i][val] = True   # dp[i,val], not dp[i,target]
      for v in xrange(val, target+1): // enum all values between val to target, for dp[i,s]
        dp[i][v] |= dp[j-1][v-val]
  return dp[len(arr)-1][target]
print plusBetween([1,2,0,6,9], 81)

int eval(char[] arr, int l, int r) {
  for(int i=l;i<r;++i) {
    if(arr[c]=='+,-,*,/') {
      recur(arr, l, i);
      recur(arr, i+1, r);
    }
  }
}

// DFS for ary, no incl/excl biz. carry offset. branch at offset, recur dfs expands 4 ways on 
// each segment moving idx, carry prepre, pre, and calulate this cur in next iteration
def addOperator(arr, target):
  def recur(a, exp, t, prepre, pre, sign):
    out = []
    preval = sign ? prepre + pre : prepre - pre
    if len(a) == 0:
      if preval == t:
        out.append(exp)
        return
    for i in xrange(1, len(a)):
      if i >= 2 and a[i] == 0: continue
      cur = a[:i+1]
      recur(a[i:], "{}+{}".format(exp, cur), t, preval, cur, True)
      recur(a[i:], "{}-{}".format(exp, cur), t, preval, cur, False)
      recur(a[i:], "{}*{}".format(exp, cur), t, prepre, pre*cur, sign)
    return out
  
  # [l..k..r] both left seg and rite seg needs recur.
  for i in xrange(len(arr)):
    if i >= 2 and arr[i] == 0: continue
    # At top root level, branch out at each position. Forest VS. Tree.
    recur(arr[i:], arr[:i+1], target, 0, int(arr[:i+1]), True)

// divide half, recur, memoize, and merge.
// calculate dp[l,r,v] directly with gap, i, i+gap.
def addOperator(arr, l, r, target):   # when recur, need fix left/rite edges
  def recur(arr, l, r):
    if not dp[l][r]:      dp[l][r] = defaultdict(list)
    else:                 return dp[l][r]     // return memoized  
    out = []
    if l == r:            return [[arr[l],{}.format(arr[l])]]
    // arr[l:r] as one num, no divide
    out.append([int(''.join(map(str,arr[l:r+1]))),  "{}".format(''.join(map(str,arr[l:r+1])))])
    
    for i in xrange(l,r):      // divide into left / rite partition
      for lv, lexp in recur(arr, l, i):
        for rv, rexp in recur(arr, i+1, r):
          out.append([lv+rv, "{}+{}".format(lexp,rexp)])
          out.append([lv*rv, "{}*{}".format(lexp,rexp)])
          out.append([lv-rv, "{}-{}".format(lexp,rexp)])
    dp[l][r] = out   // memoize l,r result
    return out

  tab = [[None]*(l+1) for i in xrange(r+1)]
  out = []
  for v, exp in recur(arr, l, r):
    if v == target:
      out.append(exp)
  return out
print addOperator([1,2,3],0,2,6)
print addOperator([1,0,5],0,2,5)

// DP[l,r] = dp[l,k] + dp[k,r]; standard 3 loop, gap=1..n, i=1..n-gap, k=l..r
def addOperator(arr, l, r):
  for gap in xrange(r-l):
    for i in xrange(l, r-l-gap):
      j = i + gap
      dp[i][i+1] = arr[l];
      for k in xrange(i,j):
        for lv, lexpr in dp[i][k]:
          for rv,rexp in dp[k+1][j]:
            out.append([lv+rv, "{}+{}".format(lexp,rexp)])
            out.append([lv*rv, "{}*{}".format(lexp,rexp)])
            out.append([lv-rv, "{}-{}".format(lexp,rexp)])
    dp[i][j] = out
  return dp[l][r]

List<String> wordBreakDfs(int l, String path, String s, Set<String> dict) {
  List<String> ret = new ArrayList<>();
  if(l==s.length()) { ret.add(path); return ret; } // echo back valid path string at leaf.
  for(int k=l;k<s.length();k++) {
    String w = s.substring(l,k+1);
    if(dict.contains(w)) {
      String wpath = path == null ? w : path + " " + w;
      List<String> restl = wordBreakDfs(k+1, wpath, s, dict);
      if(restl.size() > 0) { ret.addAll(restl); }
    }
  }
  return ret;
}
List<String> wordBreak(String s, List<String> wordDict) {
  List<String>[] f = new List<String>[s.length];
  for (int startIdx = s.length(); startIdx >= 0; startIdx--) {
    List<String> validSentences = new ArrayList<>();
    for (int endIdx = startIdx; endIdx < s.length(); endIdx++) {
      if(isWordInDict(s.substring(startIdx, endIdx+1), wordDict)){
        for(String s : f[endIdx]) {
          validSentences.add(s.substring(startIdx, endIdx+1) + " " + s);
      }
    }
    f[startIdx] = validSentences; // F[i] = F[k] + sub(i..k); for k=i..N;
  }
  return f[0];
}

// stk left, pop stk upon rite; track valid_start upon first (;
int longestValidParenth(String s) {
  Deque<Integer> stk = new ArrayDeque<>();
  for(int i=0;i<s.length();i++) {
    if(s.charAt(i)='(') stk.addLast(i);
    else if(stk.size() == 1) { maxlen=Math.max(maxlen, i-valid_start);}
    else if(stk.size() > 1) { 
      maxlen = Math.max(maxlen, i-stk.removeLast()); 
    } else { valid_start = i+1;}
  }
  return maxlen;
}
// LC 1249: Two passes, stash matched valid LR;, skip unmatched rite ). backward remove extra lefts;
int minRemoveToMakeValid(String s) {
  List<Character> matched, res; 
  int lefts = 0;
  for(char c : s.toCharArray()) {
    if(c=='(') { lefts++; matched.append(c); } // count in all valid ( and ) in first pass. skip unmatched ).
    else if(c==')' && lefts>0) { lefts--; matched.append(c); } // skip unmatched ), but reduce lefts;
    else if(c != ')') { matched.append(c); }
  }
  for(int i=matched.size()-1; i>=0;i--) { // backward remove extra left until remain 0;
    if(matched.get(i)=='(' and lefts > 0) lefts--; // filter out extra (;
    else res.add(matched.get(i));
  }
  return String.reverse(res.toString());
}
// (( ))) ())
// when first unbalance, make it balance, as the prefix, (prefix + rest), as a new string to balance; 
// balance is trigger upon first ), branch each prev ), form the new prefix, and recur rest with diff prefix;
void removeInvalidParenDfs(String s, Set<String> res) {
  int i=0, j=0, lefts = 0;
  for(i=0;i<s.length();i++) {
    if(s.charAt(i) != '(' || s.charAt(i) != ')') continue; // skip non ( and )
    if(s.charAt(i)=='(') lefts++; else lefts--; // count unmatched lefts;
    if(lefts < 0) { // when lefts < 0, just need remove a single prev ); i is consumed and is )
      while(j<=i) { // lefts < 0 b/c s[i] is a ')'; loop j= 0..i
        while(j<=i && s.charAt(j) != ')') { j++; } // j starts prev cuts(0), lseek to first );
        while(j<=i && s.charAt(j) == ')') j++; // skip the continuous ))));
        String prefix=s.substring(0,j-1)+s.substring(j,i+1);
        removeInvalidParenDfs(prefix+s.substring(i+1), res); // dfs after making it valid;
      }
    }
    return;  // first unbalance falls into here and no recur anymore.
  }
  if(lr==0) {res.add(s); return;}
  for(String ss : removeInvalidParenDfs(s)) { res.add(ss); }
}
// LC 2116. valid parenth with changeable parenth. Changeable means wildcard.
// stash aside wildcard. use it to save invalid;
boolean canBeValid(String s, String locked) {
  char[] arr = s.toCharArray(); char[] lock = locked.toCharArray();
  Deque<Integer> wildcard = new ArrayDeque<>();
  Deque<Integer> stk = new ArrayDeque<>();
  for(int i=0;i<arr.length;i++) {
    if(lock[i]=='0') { wildcard.add(i); continue;} // stash aside wildcard.
    if(arr[i]=='(') { stk.addLast(i); continue;} // stk left;
    if(stk.size() > 0) {stk.removeLast(); continue; } // a rite cancel an in stk left;
    if(wildcard.size() > 0) {wildcard.removeLast();} else return false;} // no more left in stk, cancel a wildcard. 
  }
  while(stk.size() > 0) {
    if(wildcard.size() <= 0 || stk.removeLast() > wildcard.removeLast()) return false;
  }
  if(wildcard.size()%2 == 0) return true; return false;
}

// BFS LinkedList. lseek to first diff, swap each off, j, like permutation, Recur with new state.
int kSimilarity(String A, String B) {
  if (A.equals(B)) return 0;
  Queue<String> q = new LinkedList<>();  # LinkedList Queue
  Set<String> seen = new HashSet<>();
  q.add(A);       // seed q with A start
  seen.add(A);
  int res=0;
  while(!q.isEmpty()){
    res++;
    for (int sz=q.size(); sz>0; sz--) {   // each level
      String s = q.poll();
      int i=0;
      while (s.charAt(i)==B.charAt(i)) i++;  # lseek to first not equal.
      for (int j=i; j<s.length(); j++){      # into not equal chars
        if (s.charAt(j) == B.charAt(j) || s.charAt(i) != B.charAt(j) ) continue; // lseek
        String temp= swap(s, i, j);        # swap each i, j
        if (temp.equals(B)) return res;
        if (seen.add(temp)) { q.add(temp); }    // new state, recur.
      }
    }
  }
  return res;
}

String longestDupSubstring(String s) {
  int sublenWin(String s, int winlen, int start) {
    Set<String> set = new HashSet<>();
    for(int i=start;i<=s.length()-winlen;i++) {
      String ss = s.substring(i, i+winlen);
      if(set.contains(ss)) { return i; } 
      else { set.add(ss); }
    }
    return -1;
  }
  int l=0,r=s.length()-1, maxlen=0, maxi=0;
  while(l<=r) {
    int m=l+(r-l)/2;
    int pos = sublenWin(s, m+1, 0); // bin search different len of substring;
    if(pos >= 0) {
      maxlen=m+1; maxi=pos;
      l=m+1;
    } else {
      r=m-1;
    }
  }
  if(maxlen==0) return "";
  return s.substring(maxi, maxi+maxlen);
}

// In matrix, The carryover from row-1 to row, col-1 to col, vs Not carryover col-1
// F[target_chars][src_chars], match each char in pattern string again src string.
// Not eq, skip src char, carry over src i-1 from left; If eqs, 1+[i-1/j-1], 1+upleft;
//  \ a  b   a   b  c
//  a  1  1   2   2  2
//  b  0 0+1  1  1+2 3
//  c  0  0   0   0  3
//
int DistinctSubseq(String s, String t) {  // F[i]+=F[up_left] or F[left];
  char[] sa = s.toCharArray(), ta = t.toCharArray();
  int[] dp = new int[sa.length];
  for(int i=0;i<sa.length;i++) {
    dp[i] = i > 0 ? dp[i-1] : 0;
    if(sa[i]==ta[0]) dp[i] += 1;
  }
  int upleft=0, stash=0;
  for(int ti=1;ti<ta.length;++ti) {
    upleft = dp[ti-1];
    for(int i=ti;i<sa.length;i++) {
      stash = dp[i];
      // if longest common seq, incl/excl ti, if eq, upleft+1, noteq, max(up, left); 
      dp[i] = i==ti ? 0 : dp[i-1]; // adding new pattern char, start from 0; carry over prev dp[i-1];
      if(ta[ti] == sa[i]) { // for tot subseqs, must incl ti; if exclue ti, take left, If eq, + upleft; 
        dp[i] += upleft;
      }
      upleft=stash;
    }
  }
  return dp[s.length()-1];
}
print distinct("bbbc", "bbc")
print distinct("abbbc", "bc")

int longestCommonSeq(char[] text, char[] p) {
  int[] dp = new int[text.length];
  for(int pi=0;pi<p.length;pi++) {
    int upleft = 0;  // init to 0, not dp[0] !
    for(int ti=0;ti<text.length;ti++) {
      int stage = dp[ti];  // stage prev state before mutation, staged prev as upleft for next.
      dp[ti] = ti == 0 ? dp[ti] : Math.max(dp[ti], dp[ti-1]); // Max of ti, ti-1 !!
      if(text[ti] == p[pi]) {  dp[ti] = upleft + 1; }
      upleft = stage;
    }
  }
  return dp[text.lenght-1];
}
int[][] longestCommonSeq(char[] text, char[] p) {
  // to reconstruct sequence, must use 2-d array. if only to find length, 1-d array is ok.
  int[][] dp = new int[p.length+1][text.length+1];
  for(int pi=1;pi<=p.length;pi++) {
    for(int ti=1;ti<=text.length;ti++) {  // two corners i==0, j==0, start from 1 is better
      if(p[pi-1]==text[ti-1]) { dp[pi][ti] = dp[pi-1][ti-1] + 1; } 
      else { dp[pi][ti] = Math.max(dp[pi-1][ti], dp[pi][ti-1]); } // LCS: max of [i-1/j], [i,j-1], not just i-1;
    }
  }
  return dp;
}
int maxUncrossedLines(int[] nums1, int[] nums2) {
  int[][] f = new int[nums1.length+1][nums2.length+1];
  for(int i=1;i<=nums1.length;i++) {
    for(int j=1;j<=nums2.length;j++) {
      if(nums1[i-1]==nums2[j-1]) {
        f[i][j] = 1 + f[i-1][j-1];
      }else{ 
        f[i][j] = Math.max(f[i-1][j], f[i][j-1]); 
      }
    }
  }
  return f[nums1.length][nums2.length];
}

String shortestCommonSupersequence(String str1, String str2) {
  StringBuilder sb = new StringBuilder();
  char[] text = str1.toCharArray(); char[] pat = str2.toCharArray();
  // cur[i] is the shortest common super between pat suffix and text suffix[i..N];
  String[] cur = new String[text.length+1]; String[] pre = new String[text.length+1]; 
  // no need prefill border lines; loop start from i=length to include init border lines;
  for(int pi=pat.length;pi>=0;pi--) { // for each pat suffix[pi..N]
    for(int ti=text.length;ti>=0;ti--) { // for each text suffix[ti...N]
      if(pi==pat.length) { // fill the border line when pat string is empty, common super is substr(ti..N)
        cur[ti] = ti==str1.length() ? "" : str1.substring(ti);
      } else {
        if(ti==text.length) { cur[ti] = str2.substring(pi); } // borderline empty text, common super is substr(pi..N)
        else if(text[ti]==pat[pi]) { cur[ti] = str1.charAt(ti)+pre[ti+1]; } // match, take pre processed i+1
        else if(pre[ti].length() <= cur[ti+1].length()) { // down vs. rite, take the shorter.
          cur[ti] = pat[pi] + pre[ti];
        } else {
          cur[ti] = str1.charAt(ti) + cur[ti+1]; // preappend text[ti] to common super. 
        }
      }
    }
    pre = Arrays.copyOf(cur, text.length+1);
  }
  return cur[0];
}


// long incr seq; still NxM. still 3 way merge;
String shortestCommonSupersequence(String str1, String str2) {
  // if(str1.length() >= str2.length()) { return shortest(str1, str2);} else return shortest(str2, str1);
  StringBuilder sb = new StringBuilder();
  char[] text = str1.toCharArray(); char[] pat = str2.toCharArray();
  int[][] dp = longestCommonSeqDP(text, pat);
  int i=pat.length,j=text.length;
  while(i>=1 && j>=1) {
    if(pat[i-1]==text[j-1]) { sb.append(text[j-1]); i--;j--; } 
    else if(dp[i-1][j] > dp[i][j-1]) { sb.append(pat[i-1]); i--; } 
    else { sb.append(text[j-1]); j--;}
  }
  while(i>=1) { sb.append(pat[i-1]); i--; } // append the remaining pat 
  while(j>=1) { sb.append(text[j-1]); j--; } // append the remaing text
  return sb.reverse().toString();
}

// the idea is to use rolling hash, A/C/G/T maps to [00,01,10,11]
// or use last 3 bits of the char, c%7. 3 bits, 10 chars, 30bits, hash stored in last 30 bits
from collections import defaultdict
def dna(arr):
  cache = defaultdict(int)
  mask = (1 << 30) - 1  # hash stores in the last 30 bits
  hashv,out = 0,[]
  for i in xrange(len(arr)):
    v = arr[i]
    vh = ord(v)&7
    hashv = ((hashv << 3) & mask) | vh
    if i > 9 and cache[hashv] > 0:
      out.append(arr[i-9:i+1])
    cache[hashv] += 1
  return out
print dna("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT")


// min insertion into str to convert to palindrom. 
// tab[i,j]=min ins to convert str[i:j], Increasing gap, [i, i+gap]
// tab[i][j] = tab[i+1,j-1] or min(tab[i,j-1], tab[i+1,j])
def minInsertPalindrom(arr):
  dp = [[0]*len(arr) for i in xrange(len(arr))]
  for gap in xrange(2,len(arr)+1):  # need +1 to make gap = len(arr)
    for i in xrange(len(arr)-gap):
      j = i + gap - 1
      if arr[i] == arr[j]:
        dp[i][j] = dp[i+1][j-1]
      else:
        dp[i][j] = min(dp[i][j-1], dp[i+1][j]) + 1
  return dp[0][len(arr)-1]


// convert recursion Fn call to DP tab. F[i] is min cut of S[i:n], solve leaf, aggregate at top.
// F[i] = min(F[k] for k in i..n). To aggregate at F[i], we need to recursive to leaf F[i+1].
// For DP tabular lookup, we need to start from the end, which is minimal case, and expand
// to the begining filling table from n -> 0.
def palindromMincut(s):
  n = len(s)
  dp[n-1] = 1  # one char is palindrom
  for i in xrange(n-2, 0, -1):  # start from end that is minimal case, expand to full.
      for j in xrange(i, n-1):
          if s[i] == s[j] and palin[i+1, j-1] is True:   # palin[i,j] = palin[i+1,j-1]
              palin[i,j] = True
              dp[i] = min(dp[i], 1+dp[j+1])
  # iterate from dp[n] to dp[0] expand to full string.
  return dp[0]

// DP[i] = mincut of A[0..i], = min(DP[i-1]+1 iff A[j]=A[i] and Pal[j+1,i-1], DP[i]) 
// for each j in i->0 where s[j,i] is palin)
def palindromMincut(s):
  n = len(s)
  tab[1] = 1  # tab[i], the mincut of string s[0:i]
  for i in xrange(n):
    for j in xrange(i, 0, -1):
      if s[i] == s[j] and palin[j+1, i-1] is True:
        palin[j,i] = True
        tab[i] = min(tab[j+1] + 1, tab[i])
  # expand to full string by the end
  return tab[n-1]

// enum all prefix and suffix segs, prefix[0..i],suffix[i+1..N]; if prefix isPal, lookup rev of suffix in dict, 
// prefix pair(j,i); suffix pair(i,j)
boolean isPalin(char[] arr, int l, int r) {
  if(l==0 && l==r) return true;  // empty at head is Pal;
  if(l==arr.length) return false; // empty at end is not Pal;
  if(l+1==r) return true; // single is Pal.
  while(l<r) {
    if(arr[l] != arr[r-1]) return false;
    l++;r--;
  }
  return true;
}
List<List<Integer>> PrefixSuffixPal(String w, int widx, Map<String, Integer> map) {
  List<List<Integer>> res = new ArrayList<>();
  StringBuilder sb = new StringBuilder(w); sb.reverse();
  String rw = sb.toString();
  char[] arr = w.toCharArray();
  for(int i=0;i<=arr.length;i++) { // 
    String sufrev = rw.substring(0, arr.length-i); // i=0, sufrev is the entire string.
    String prerev = rw.substring(arr.length-i); // start from len-i
    if(isPalin(arr, 0, i) && map.containsKey(sufrev) && map.get(sufrev) != widx) { // start from i=0, empty prefix;
      List<Integer> pair = new ArrayList<>(); 
      pair.add(map.get(sufrev)); pair.add(widx); res.add(pair);
    }
    if(isPalin(arr, i, arr.length) && map.containsKey(prerev) && map.get(prerev) != widx) {
      List<Integer> pair = new ArrayList<>(); 
      pair.add(widx); pair.add(map.get(prerev)); res.add(pair);
    }
  }
  return res;
}

// palindrom partition. First, find all palindroms.
// From head, DFS offset+1 find all palindrom, add res to global result set.
List<List<String>> partition(String s) {
    List<List<String>> res = new ArrayList<>();
    boolean[][] dp = new boolean[s.length()][s.length()];
    for (int gap=2;gap<s.length;gap++) {
      for (int i=0; i<s.length-gap; i++) {
        int j = i+gap-1;
        if (s.charAt(i) == s.charAt(j) && (gap <= 2 || dp[i+1][j-1]) {
          dp[i][j] = true;
        }
      }
    }
    recurDFS(res, new ArrayList<>(), dp, s, 0);
    return res;
  }
  void recurDFS(List<List<String>> res, List<String> prefix, boolean[][] dp, String s, int off) {
    if(off == s.length()) {
        res.add(new ArrayList<>(prefix));
        return;
    }
    for(int i = off; i < s.length(); i++) {
        if(dp[off][i]) {
            prefix.add(s.substring(off,i+1));    // consume off->i, recur i+1;
              recurDFS(res, prefix, dp, s, i+1); // partial result add to global result set. 
            prefix.remove(prefix.size()-1);
        }
    }
  }
}

// dp[i]: cost of word wrap [0:i]. foreach k=[i..k..j], w[j] = min(w[i..k] + lc[k,j])
def wordWrap(words, m):
  sz = len(words)
  // extras[i][j] extra spaces if words i to j are in a single line
  extras = [[0]*sz for i in xrange(sz)]
  dp = [[0]*sz for i in xrange(sz)] # cost of line with word i:j
  for linewords in xrange(sz/m):
    for st in xrange(sz-linewords):
      ed = st + linewords
      extras[st][ed] = abs(m-sum(words, st, ed))
  for i in xrange(sz):
    for j in xrange(i):
      dp[i] = min(dp[i] + extras[j][i])
  return dp[sz-1]


// 2 string, lenth,ia,ib; 3 dim, in each len, test sublen k/len-k. tab[0..k][ia][ib];
// normally, one dim ary with each tab[lenth,i], j=i+len,
// here, vary all 3; tab[len,i,j], break lenth in sublen, and i,j with sublen.
// tab[l,i,j] = 1 iff (tab[k,i,j] and tab[l-k,i+k,j+k]
def isScramble(a, b):
  dp = [[[0]*len(a) for i in xrange(len(b))] for j in xrange(len(a))]
  for gap in xrange(len(a)):
    for ia in xrange(len(a)-gap):
      for ib in xrange(len(b)-gap):
        if gap == 0:
          dp[gap][ia][ib] = a[ia] == b[ib]
        for k in xrange(gap):
          if dp[k][ia][ib] and dp[gap-k][ia+k][ib+k]:
            dp[gap][ia][ib] = True
          elif dp[k][ia][ib+gap-k] and dp[gap-k][ia+k][ib]:
            dp[gap][i][j] = True
  return dp[len(a)-1][0][0]


// min number left after remove triplet. track min left within [i..j]
// [lo, i, j] is a k-spaced triplet [lo+1..i-1],[i+1..j-1]
// [2, 3, 4, 5, 6, 4], ret 0, first remove 456, remove left 234.
// dp[i,j] track the min left in subary i..j = min(recur(i+1, j))
def minLeft(arr, k):
  dp = [[0]*len(arr) for i in len(arr)]
  def minLeftSub(arr, lo, hi, k):
    if dp[lo][hi] != 0:
        return dp[lo][hi]
    if hi-lo+1 < 3:  # can not remove ele less than triplet
        return hi-lo+1
    ret = 1 + minLeftSub(arr, lo+1, hi, k)

  for i in xrange(lo+1, hi-1):
    for j in xrange(i+1, hi):
      # lo, i, j is k-spaced triplet, and [lo+1..i-1],[i+1..j-1] can be removed
      if arr[i] == arr[lo] + k and arr[j] == arr[lo] + 2*k and
        minLeftSub(arr, lo+1, i-1, k) == 0 and 
        minLeftSub(arr, i+1, j-1, k) == 0:
          # now triplet lo,i,j also can be removed
          ret = min(ret, minLeftSub(arr, j+1, hi, k))
    dp[lo][hi] = ret
    return ret

// Max Flow Min cut for Max Matchings.
int path_vol(g, st, ed, parent) {
  if st == ed return 0;
  // one recur remove the last segment, parent[ed] -> ed;
  edge = g[parent[ed]][ed];
  if parent[ed] == st {
    return edge->residual;
  }
  return min(recur(g, st, parent[ed], parents),
      edge->residual)

}
netflow(g, src, sink) {
  add_residual_edges(g);
  loop {
    bfs(src, sink, parent);
    vol = path_vol(g, src, sink, parent);
    if (vol < 0) break;
    augment_path(g, src, sink, parent, vol);
  }
}

"""
http://www.geeksforgeeks.org/find-the-largest-rectangle-of-1s-with-swapping-of-columns-allowed/
"""
if __name__ == '__main__':
  pass

