// Drain Pipeline: [arr | stk/queue/priority | aggregate]. while(idx<n || pq.size() > 0) {}
// 1. Abstract Invariants to tracked state, data structure(PQ, TreeMap). At I, what's given, what's answer ? 
// 2. Keep result of A/i for A/i+1 ! Ensure processing A/i leverages prev aggregations !!. Boundary/exit checks.
// 2. Bisect: l: first bigger(>target) for safely discard the left smaller; bucket. arr[i] vs arr[arr[i]];
// 4. Sliding Min win sum K: keep prefix sorted, counted !! subary(l,r) or dist(l,r). Global L watermark trap. 
// 3. Two pointers ary: Invariants. Update conditions. move smaller when <=; while(l<r){while(arr/l <= minH){l++;r--} minH=min(l,r)};
// 4. Recursion(root, parent, prefix, collector); prune repeated state, edge class; leaf recur done update collector and parent; parent aggregates all children and update up again.
// 5. Coin change order matters, outer loop v, inner loop each coins. dp[v]+=dp[v-a/i]; order not matter, outer loop each coin, inner loop v. Recur(coins, prefix+a/pos) not scale.
// 6. K lists processing. merge, priority q. reasoning by add one upon simple A=[min/max], B=[min, min+1, min+k...] 
// 7. Tree: recur(root, parent, prefix, collector) || while(cur!=null||stk.emptry), update state upon popping. parent aggreg all children and updates up.
// 8. DP carry over aggregations: tot/min, word break/palind, DP stores aggregation and pass forward. Boundary rows!!
// 9. Search: DFS/BFS vs. Recur(off, prefix, coll), DP[i-1] carry aggregation state down.
// 10. BFS/DFS not scalable with edges. use PriorityQueue to sort min and prune.
// 11. MergeSort: process each pair, turn N^2 to NlgN. all pairs(i, j) comp skipping some when merge sorted halves. Count of Range Sum within.
// 12. i-j: dist(# of eles) between two ptrs. (i, j] exclude i; when i==j; len=1 ele no diff; len-i: # of suffix eles. [i..len)
// 13. Recur(arr,off,prefix) {for(i=off;i++){recur(i+1, prefix+arr[i]}}; outer loop over all coins, recur i+1 prefix into sub-recur; i was incl/excl in outer i iter.
// 14. TreeMap/TreeSet(balance rotate) to sort a dynamic win of arr updated add r remove l; MergeSort when arr is fixed.   

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
// Backtracking: DFS recur each next candidates, branch(incl/excl); Recur(i+1, prefix state, Collector). Restore prefix after recursion!! 
// Dfs Done End state Aggregate max.
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

// knapsack: one bin, iter add item_i once a time, loop each w -> W, max of incl/excl item_i. dp[i][w]=max(dp[i-1][w-w_i]+v_i, dp[i-1][w])
// binpacking: NP hard. multiple bin. First fit, first unfit, best fit. etc.

// 1. subsetSum, combinSum, coinChange, dup allowed, reduce to dp[v] += dp[v-vi]
// dup not allowed, track 2 rows, dp[i][v] = (dp[i-1][v] + dp[i-1][v-vi])

// 1. Tree, recur with prefix context, or iterate with root and stk. Restore Prefix after subtree recursion.
// 2. Ary, recur cur+1, prepare next candidates.
// 4. loop gap=1,2.., start = i, end = i+gap.
// 5. dp[i,j,val] = dp[i-1,j-1,val=0..v]
// 6. BFS with priorityqueue, topsort

// Tree build Recur(RMQ, kd). recur(): root.left = recur(); root.rite=recur(); ret root;
// Merge sort and AVL tree, riteSmaller, Stack.push(TreeMap for sort String, index, ) 
// Recursive, pass in the parent node and fill null inside recur. left = insert(left, val);
// https://leetcode.com/problems/reverse-pairs/discuss/97268/General-principles-behind-problems-similar-to-%22Reverse-Pairs%22
// Recursion(prefix) fan out dup path to leaves. avoid double counting in each node, see PathSum. Restore Prefix each subtree recursion.

// BFS/DFS recur search, permutation swap to get N^2. K-group, K-similar, etc.
// Merge sort to Reduce to N2 comparsions or lgN, break into two segs, recur in each seg.
// 0. MAP to track a[i] where, how many, aggregates, counts, prefix sum, radix rank, position/index. pos[a[i]] = i;
// 1. Bucket sort / bucket merge. A[i] stores value = index value+1
// 2. Merge/quick sort, partition, divide and constant merge.
// 3. Union set to merge list in a bucket
// 4. Stack push / pop when bigger than top. mono inc/desc.    
//      Deque<Integer> stack = new ArrayDeque<Integer>();
//      stack.isEmpty(), peekLast(), offerFirst(e), offerLast(e), pollFirst(), pollLast()
//    Stk can store current workset map, or index of ary. refer to origin ary for value a[i]
//    Stack<Map<String,Integer>> stack= new Stack<>();   // workset map, index of ary, value
// 5. Ary divide into two parts, up[i]/down[i], wiggly sort, or longest ^ mountain.
// 5. BFS heap / priorityqueue
// 1. sort and hash, binary search.
// 2. prefix sum.
// 3. RMQ  
// 4. Formalize the problem into DP recursion. when two input array.
//   for (work set value i = 0 .. n)
//     for (candidate j 0..i )
//       dp[i] = max(dp[i-k]+1, dp[i])
//   DP is layer by layer zoom in, first, 1 layer/gap/grp/sublen, then 2, 3, K layer, etc.
//   At each layer, you can re-use k-1 layer at k, reduce DP[i,k] to DP[i]

// https://leetcode.com/problems/student-attendance-record-ii/discuss/101643/Share-my-O(n)-C++-DP-solution-with-thinking-process-and-explanation
// """

// """ Intervals scheduling, TreeMap, Segment(st,ed) is interval.
// Sort by start, use stk to filter overlap. Greedy can work.
// sort by finish(start<finish). no stk. track global end. end=max(iv.end).
// Interval Array, add/remove range all needs consolidate new range.
// TreeMap to record each st/ed, slide in start, val+=1, slide out end, val-=1. sum. find max overlap.
// Interval Tree, find all overlapping intervals. floorKey/ceilingKey/submap/remove on add and delete.

// Trie and Suffix tree. https://leetcode.com/articles/short-encoding-of-words/
// Node is a map of edges { Map<headChar,Edge> edges; rootNode} Edge{startidx,endidx, endNode}; Edge stores label of substr index.
// Split edge by inserting new nodes. 

// """ Graph: Adj edge list: Map<Node, LinkedList<Set<Node>>, recur thru all edges visiting nodes.
// DFS(root, prefix, collector): Recur only __undisc__. DFS done, all child visited, memorize path[cur]. 
// DFS Edge: Tree: !discovered, Back: !processed anecestors in stk. Forward: already found by. Cross: Directed to already traversed component.
// Edge exact once: dfs(parent,me), dfs(me, !discovered); Back:(in stk !processed, excl parent). processed_me is done when node is processed.  
// Strong Connected Component(SCC): DFS Back-edge earliest, or two-DFS: label vertex by completion. DFS reversed G from highest unvisited. each dfs run finds a component.
// List All paths: DFS(v, prefix, collector). ret paths or collect at leaves. BFS(v,prefix): no enqueue traverse when nb in prefix. 
// - - - - -
// Graph Types: Directed/Acyclic/Wegithed(negative). Edge Types: Tree, Back, Forward, Cross(back_to_dfs-ed_component) 
// a) DFS/BFS un-weighted, b) Relaxation Greedy for weighted, c) DP[i,k,j] all pairs paths.
// Dijkstra/MST_prim: pick a root, PQ<[sort_by_dist, node, ...]>; shortest path to all. Kruskal. UnionFind shortest edges.
// Dijkstra no negative edges. dist[parent] no longer min if (parent-child) < 0 reduce weights. 
// Single src Shortest path to all dst, Bellman: loop |v| times { for each edge(u,v,e), d[v] = min(d[u] + e)}
// DAG: topsort. linear dijkstra. DFS+memorize all subpaths, parent is frozen by order. parents min before expanding subgraph.   
// MST(union-find): relaxation.. Shortest path changes when each weight+delta as the path is multihops.
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

// Integer Programming: Combinatorial Optimization, Bin packing
3. MailBox in K-V store. Push / Pull / Hybrid.
  https://www.slideshare.net/nkallen/q-con-3770885

1. https://github.com/checkcheckzz/system-design-interview
2. Graph Search, index ranking, typeAhead, table both forward/reverse relation. Vertical Aggregator.
  https://www.facebook.com/notes/facebook-engineering/under-the-hood-building-out-the-infrastructure-for-graph-search/10151347573598920
  https://www.facebook.com/notes/facebook-engineering/under-the-hood-indexing-and-ranking-in-graph-search/10151361720763920
3. Feed Generation, MailBox and Cron.
  https://www.slideshare.net/danmckinley/etsy-activity-feeds-architecture/


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
FROM transactions t1 LEFT OUTER JOIN transactions t2 ON (t1.acct_id = t2.acct_id AND t1.trans_date < t2.trans_date)
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

select p1.id, p1.salary, count(*) from people p1 
left join people p2 on p1.id <> p2.id and p1.salary < p2.salary
where p2.id is not null
group by p1.id
having count(*) = 2

SELECT t1.*
FROM `Table` AS t1 LEFT OUTER JOIN `Table` AS t2 ON ( t1.GroupId = t2.GroupId AND 
    (t1.OrderField < t2.OrderField OR (t1.OrderField = t2.OrderField AND t1.Id < t2.Id))
WHERE t2.GroupId IS NUL

// group by has dedup.
select count(*)
from insurance i1 inner join (select tiv_2011 from insurance group by tiv_2011 having count(*) > 1) i2 on i1.tiv_2011 = i2.tiv_2011

select count(distinct i1.pid)
from insurance i1 inner join insurance i2 on (i1.tiv_2011 = i2.tiv_2011 and i1.pid != i2.pid)

// salary count be 0, so select null.
SELECT Salary FROM Employee GROUP BY Salary 
UNION ALL 
(SELECT null AS Salary)
ORDER BY Salary DESC LIMIT 1 OFFSET 1

// id +/- 1
SELECT s1.* FROM stadium AS s1, stadium AS s2, stadium as s3
WHERE  ((s1.id + 1 = s2.id AND s1.id + 2 = s3.id) OR (s1.id - 1 = s2.id AND s1.id + 1 = s3.id) OR (s1.id - 2 = s2.id AND s1.id - 1 = s3.id)
AND s1.people>=100 
AND s2.people>=100
AND s3.people>=100
GROUP BY s1.id

// TOP N in group. use <=, itself 1, <= another one.
SELECT a.person, a.group, a.age, count(*)
FROM person a LEFT OUTER JOIN person b on (a.group = b.group and a.age <= b.age)
  #where b.age is NULL   // with a.age < b.age, so max entry has null which is greater than it.
group by a.person 
having count(*) <= 2
ORDER BY a.group ASC, a.age DESC
GO

select t.Request_at Day, round(sum(case when t.Status like 'cancelled_%' then 1 else 0 end)/count(*),2) Rate
from Trips t inner join Users u on t.Client_Id = u.Users_Id and u.Banned='No'
where t.Request_at between '2013-10-01' and '2013-10-03'
group by t.Request_at
  
SELECT Score, (SELECT count(distinct Score) FROM Scores WHERE Score >= s.Score) Rank
FROM Scores s
ORDER BY Score desc

SELECT Score, (SELECT count(*) FROM (SELECT distinct Score s FROM Scores) tmp WHERE s >= Score) Rank
FROM Scores
ORDER BY Score desc

select if(id < (select count(*) from seat), if(id mod 2=0, id-1, id+1), if(id mod 2=0, id-1, id)) as id, student
from seat
order by id asc;

select id,
case 
    when id%2 = 0 then (select student from seat where id = (i.id-1) )  
    when id%2 != 0 and id<(select count(student) from seat) then (select student from seat where id = (i.id+1) )  
    else student
end as student
from seat i 

select d.name, sum(case when s.dept_id is null then 0 else 1 end) cnt
from department d left outer join student s on d.id = s.dept_id
group by d.id, d.name
order by cnt desc, d.name;

// subquery to create a table to join.
SELECT salesperson_id, Salesperson.Name, Number AS OrderNumber, Amount
FROM Orders JOIN Salesperson ON Salesperson.ID = Orders.salesperson_id
JOIN (SELECT salesperson_id, MAX( Amount ) AS MaxOrder 
      FROM Orders
      GROUP BY salesperson_id
) AS TopOrderAmountsPerSalesperson
USING ( salesperson_id ) 
WHERE Amount = MaxOrder
GROUP BY salesperson_id, Amount

select cast(sum(i1.tiv_2012) as decimal(10, 2))
from insurance i1 inner join (select tiv_2011 from insurance group by tiv_2011 having count(*) > 1) i2
on i1.tiv_2011 = i2.tiv_2011
inner join (select lat, lon
    from insurance
    group by lat, lon
    having count(*) = 1) i3
on i1.lat = i3.lat and i1.lon = i3.lon;

// pascal triangle, 
// http://www.cforcoding.com/2012/01/interview-programming-problems-done.html

// recur with boundary, recur(head, tail)
def list2BST(head, tail):
    mid = len(tail-head)/2
    mid.prev = list2BST(head, mid.prev)
    mid.next = list2BST(mid.next, tail)
    return mid

Node build(List<Object> arr, int depth) {
  int dim = depth % DIMS;
  sort(arr, dim);
  median = arr.get(arr.size()/2);
  Node root = new Node(median);
  root.left = build(arr[0, sz/2], depth+1);
  root.rite = build(arr[sz/2:], depth+1);
  return root;
}
Nodee query(Node root, query, best_node, best_dist) {
}

// sort the list by x dim. recur left, right.  
def KdTree(pointList, height):
  // Node {point, left, rite}
  cur_dim = height % DIMS
  sort(pointList, (a, b) {a[cur_dim] < b[cur_dim]} )
  split_median = median(pointList)
  root = Node(split_median)
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
// linear
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
public double myPower(double x, int n) {
  double v = x;
  double tot = 1.0;
  while (n > 0) {
    if((n & 1) == 1) { tot *= v;}
    n >>>= 1;  // binary search, 1,2,4,8,16. leverage 8421 double every bit.
    v *= v;
  }
  return tot;
}

static int kadan(int[] arr) {
  int spanStart = 0, spanEnd = 0, spanSum, gmax;
  // Loop Invar: max till I. 
  for (int i = 0; i < arr.length; i++ ) {
    spanSum = Math.max(spanSum + arr[i], arr[i]);
    gmax = Math.max(gmax, spanSum);
  return gmax;
}
// # HIdx: h = max_i(min(i, citations[i])): max of (min(idx, citation[idx])) for all papers.
// aggregate # of papers whose citations larger than their indexes,.
// 
int n = citations.size(), h = 0;
  int* citation_counts = new int[n + 1]();
  for (int c : citations)  citation_counts[min(c, n)]++; // cap to n.
  for (int i = n; i; i--) {
    h += citation_counts[i];
    if (h >= i) return i; // when aggrated citations_counts at idx > idx;
  } 
  return h;
}
//  binsect hidx with citations[hidx].
int hIndex2(vector<int>& citations) {
  int n = citations.size(), l = 0, r = n - 1;
  while (l <= r) {
    int m = l + (r - l) / 2;
    if (citations[m] >= n - m) r = m - 1;
    else l = m + 1;
  }
  return n - r - 1;
}
assertEqual(3, hIndex(new int[]{3,0,6,1,5}));

// for each i, either belong to prev seq, or starts its own.
// s[i] = max(s[i-1]+A.i, A.i), either expand sum, or start new win from cur '''
// arr = [-1, 40, -14, 7, 6, 5, -4, -1]
def maxSumWrap(arr):
    maxsum = kadan(arr)
    for i in xrange(len(arr)):
        wrapsum += arr[i]
        arr[i] = -arr[i]
    maxWrap = wrapsum - kadan(arr)
    return max(maxsum, maxWrap)

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
public int findMaxLength(int[] arr) {
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

int subarraySumToK(int[] nums, int k) {
  int prefixSum = 0, tot = 0;
  Map</*prefixSum=*/Integer, /*cnt=*/Integer> prefixSumMap = new HashMap<>();
  prefixSumMap.put(0, 1);
  for (int i = 0; i < nums.length; i++) {
    prefixSum += nums[i];
    tot += prefixSumMap.getOrDefault(prefixSum - k, 0); 
    prefixSumMap.compute(prefixSum, (k,v) -> v == null ? 1 : v+1);
  }
  return tot;
}
System.out.println(subarraySumToK(new int[]{0,0,0}, 0) + " = 6");

// segment sum(i..j) = PathSum[0..j]-[0..i];
public int PathSum(TreeNode root, int target, long prefix, Map<Long, Integer> prefixSumMap) {
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

''' max prod exclude a[i], keep track of pre prod excl cur '''
def productExcl(arr):
  prod = [1]*len(arr)
  tot = 1      // seed as 1.
  for i in xrange(len(arr)):
    prod[i] = tot
    tot *= arr[i]
  tot = 1
  for i in xrange(len(arr)-1,-1,-1):
    prod[i] *= tot
    tot *= arr[i]
  return prod

''' track max[i]/min[i] up to a/i, swap max/min when a/i < 0; this handles 0 gracefully.
always include a[i] to decide max/min at each idx; either it starts a new seq, or it join prev sum
'''
int maxProduct(int[] arr) {
  int curmax=arr[0], curmin=arr[0], gmax=curmax;
  {
    for(int i=1;i<arr.length;i++) {
      if(arr[i] > 0) {
        curmax = Math.max(curmax*arr[i], arr[i]);
        // minP = Math.min(minP*arr[i], arr[i]);
        curmin = curmin*arr[i];
      } else {
        int tmp = curmax;
        curmax = Math.max(curmin*arr[i], arr[i]);
        curmin = Math.min(tmp*arr[i], arr[i]);
      }
      gmax = Math.max(gmax, curmax);
    }
    return gmax;
  }
  {
    for(int i=1;i<arr.length;++i){
      if(arr[i] < 0) {  // just swap max min upon negative.
        int tmp = curmax;  curmax = curmin;   curmin = tmp;
      }
      curmax = Math.max(curmax*arr[i], arr[i]); // always include arr/i as it can be a max ending at i;
      curmin = Math.min(curmin*arr[i], arr[i]);
      gmax=Math.max(gmax, curmax);
    }
  }
  return gmax;
}
print maxProduct([-2, -3, -4])
print maxProduct([2,0,-4, 3, 2])

""" Lmin[i] contains idx in the left arr[i] < cur, or -1
    Rmax[i] contains idx in the rite cur < arr[i]. or -1
    then loop both Lmin/Rmax, at i when Lmin[i] > Rmax[i], done
    
left < cur < rite, for each i, max smaller on left and max rite.
reduce to problem to pair, so left <- rite. find rmax  riteMax[i] = arr[i]*rmax
for left max smaller, left -> rite, build AVL tree to get leftMax, or use a stk, push when bigger than top.
maxSmaller recur: if root.rite.maxSmalle(k) == None, return cur. if k < root and no left, return None.
max = max(max[i]*lmax)
"""
static int tripletMaxProd(int[] arr) {
    int sz = arr.length;
    int[] RMax = new int[sz];
    RMax[sz-1] = arr[sz-1];
    for (int i=sz-2;i>=0;i--) {
        RMax[i] = Math.max(RMax[i+1], arr[i]);
    }
    AVLTree root = createAVLTree(arr);
    for (int i=0;i<sz;i++) {
        int leftmax = root.getLeftMax(arr[i]);
        tripletmax = Math.max(tripletmax, leftmax * arr[i] * RMax[i]);
    }
    return tripletmax;
}
print triplet_maxprod([7, 6, 8, 1, 2, 3, 9, 10])
print triplet_maxprod([5,4,3,10,2,7,8])

// l ptr to the last eq/first larger(<= target), or first eq(< target); 
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

""" Loop Invar: arr[l..r] sorted and contains v; 
 only check arr/m with target and discard the half if it is sorted and target not in it!!
 target can be in either left/rite, deps on whether m is peak or valley, ensure [l,m] or[m,r] are sorted.
"""
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
public int rotatedMin(int[] arr) {
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
""" find min in rotated. / / with dup, advance mid, and check on each mov """
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

""" find the max in a ^ ary,  m < m+1 """
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

"""
max distance between two items where arr/left < arr/right. LIS.
http://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
leftmin[0..n] store min 0..n, desc to valley.
ritemax[n-1..0] climb to max from end. so ritemax[0..n] desc from max.
two ptrs, decr rmax unilt not, decr lmin until < rmax.
"""
static int maxDist(int[] arr) {
  int sz = arr.length;
  int[] leftmin = new int[arr.length], ritemax = new int[arr.length];
  int valley = 999, peak = 0;
  for (int i=0; i<arr.length; i++) {
    valley = Math.min(valley, arr[i]);
    leftmin[i] = valley;
    peak = Math.max(peak, arr[arr.length-1-i]);
    ritemax[arr.length-1-i] = peak;
  }
  int l = 0, r = 0, maxdist = 0;   // two counters to break while loop.
  // stretch r as r from left to rite is mono decrease.
  while (l < sz && r < sz) {
    if (ritemax[r] >= leftmin[l]) {
        maxdist = Math.max(maxdist, r-l+1);  // calculate first, mov left second.
        r += 1;
    } else {
        l += 1;
    }
  }
  return maxdist;
}
maxDist(new int[]{34, 8, 10, 3, 2, 80, 30, 33, 1});

// container most water: track each bin **first** left higher. not left max higher. 
// how left/rite ptrs avoid bisect of stk ? upon each bin, need find the first bigger.
// left ptr incrs until bigger. because r from the end, it captures the length.
public int maxArea(int[] arr) {
  int l = 0, r=arr.length-1, maxw = 0;
  while(l<r) {
    gmax = Math.max(gmax, Math.min(arr[l], arr[r])*(r-l));
    // stretch L or R to bigger area.
    if(arr[l] <= arr[r]) l++; else r--;
  }
  return maxw;
}

public int largestRectangleArea(int[] arr) {
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
  while(stk.size() > 0) {
      riteedge[stk.removeFirst()] = arr.length;
  }
  int gmax = 0;
  for(int i=0;i<arr.length;++i) {
      gmax = Math.max(gmax, arr[i]*(riteedge[i]-leftedge[i]-1));
  }
  return gmax;
}
// stk to store left edge. rite edge is when arr[i] < stk.peekLast();
public int largestRectangleArea(int[] arr) {
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

public int trapWater(int[] arr) {
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

""" put 0 at head, 2 at tail, 1 in the middle, wr+1/wb-1 """
void swap(int[] arr, int i, int j) { int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; }
public int[] sortColors(int[] arr) {
  int lidx=0,ridx=arr.length-1,i=0;
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
      a[mid] = a[low];
      a[low] = cur;
      low++;
    } else if (a[mid] > 1) {
      a[mid] = a[high];
      a[high] = cur;
      high--; 
    }
}}

public void sortColors(int[] arr) {
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

public void rainbowSort(int[] colors, int left, int rite, int startColor, int endColor) {
  if (colorFrom == colorTo) { return; }
  if (left >= right) { return; }
  int colorMid = (colorFrom + colorTo) / 2;
  int l = left, r = right;
  while (l < r) {
    while (l < r && colors[l] <= colorMid) { l++;  }
    while (l < r && colors[r] > colorMid) { r--;  }
    if (l < r) {
      int temp = colors[l]; colors[l] = colors[r]; colors[r] = temp;
      l++;
      r--;
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
      if (!stk.isEmpty() && stk.peek() < arr[r]) {   // after pop, if stk not empty, and its top < arr[r]
        return true;
      }
      stk.push(arr[r]);
    }
  }
}

public void stackSorting(Stack<Integer> stack) {
  Stack<Integer> helpStack = new Stack<Integer>();
  while (!stack.isEmpty()) {
      int element = stack.pop();
      while (!helpStack.isEmpty() && helpStack.peek() < element) {
          stack.push(helpStack.pop());
      }
      helpStack.push(element);
  }
  while (!helpStack.isEmpty()) {
      stack.push(helpStack.pop());
  }
}

// # subary is conti, and simple than subsequence
// # while sliding, count up first, then count down, then add up+down. Reset counters at turning point.
int longestMountainSubary(int[] arr) {
  int maxlen =0, up = 0, down = 0;
  { for (int i=1;i<arr.length;i++) {
    // down > 0, means we are in down phase, when switching to up phase, reset.
    if (down > 0 && arr[i-1] < arr[i] || arr[i-1] == arr[i]) {   // turning point. reset counters.
      up = down = 0;    // reset when encouter turning points
    }
    if (arr[i-1] < arr[i]) up++;
    if (arr[i-1] > arr[i]) down++;
    if (up > 0 && down > 0 && up + down + 1 > maxlen) { maxlen = up + down + 1;}
    }
    return maxlen;
  }
  {
    int lstart=-1, rend=-1, maxw=0; 
    for(int i=1;i<arr.length;i++) {
      if(arr[i] > arr[i-1]) {
        if(lstart == -1) lstart=i-1;
        else if(rend != -1) { lstart = i-1; rend=-1;} // reset when \ chaged to /
      } else if(arr[i] < arr[i-1]) {
        if(lstart != -1) { rend = i; maxw = Math.max(maxw, rend-lstart+1);
        }
      } else {lstart = -1; rend=-1; }
    }
    return maxw;
  }
}

// Longest mountain = LUp[i] + RDown[i]; both are Long Increasing Seq. Simple stk not working.
public int insertPos(List<Integer> arr, int target) {
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
public int minimumMountainRemovals(int[] arr) {
  int[] lup = new int[arr.length];
  int[] rdown = new int[arr.length];
  List<Integer> list = new ArrayList<>(); 
  list.add(arr[0]); lup[0]=0;
  for(int i=1;i<arr.length;i++) {
    lup[i] = insertPos(list, arr[i]);
  }
  list.clear(); list.add(arr[arr.length-1]);
  for(int i=arr.length-2;i>=0;i--) {
    rdown[i] = insertPos(list, arr[i]);
  }
  int maxlen = 0;
  for(int i=1;i<arr.length-1;i++) {
    if(lup[i] > 0 && rdown[i] > 0) {
      maxlen = Math.max(maxlen, (lup[i]+rdown[i]+1));
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

""" dp[i,j] : LAP of arr[i:j] with i is the anchor, and k is the left and j is the rite. 
Try each i anchor.  k <- i -> j,  dp[i,j] = dp[k,i]+1, iff arr[k]+arr[j]=2*arr[i] """
def longestArithmathProgress(arr):
mx = 0
dp = [[0]*len(arr) for i in xrange(len(arr))]
//  boundary, dp[i][n] = 2, [arr[i], arr[n]] is an AP with len 2
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
public List<List<Integer>> threeSum(int[] num) {
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

// iterate each cur of arr, bisect each ele into dp[] sorted array.
// cur bigger than all in dp, append to dp end, increase lislen.
// enhancement is use parent[] to track the parent for each node in the ary.
int lengthOfLIS(int[] arr) {
  int[] dp = new int[arr.length];  // LoopInvar: track lis[i]; Pre: dp[i] = 0; post: dp[n] = max
  int lislen = 0;   // dp[] is empty initially.
  for (int cur : arr) {
      int lo = 0, hi = lislen;     // lislen is 0 for now.
      while (lo != hi) {  
          int m = (lo + hi) / 2;
          if (dp[m] < cur)  lo = m + 1;
          else              hi = m;
      }
      dp[lo] = cur;   // will not outofbound as hi=m, and after binsearch, a[lo] bound to > than k
      if (lo == lislen) { ++lislen; }  // bisect indicate append to DP end. increase size.
  }
  return lislen;
}


// return -1 if a shall precedes b, ab > ba; otherwise 1, ba > ab;
public int compare(int a, int b) {
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

public int parent(int idx) { return (idx-1)/2;}
public int[] children(int idx) { return new int[]{2*idx+1, 2*idx+2};}
public void swap(int[] arr, int i, int j) { int tmp = arr[i]; arr[i]=arr[j];arr[j]=tmp;}
public int maxof(int[] arr, int i, int j) { 
  if(compare(arr[i], arr[j]) > 0) { return j; } else {return i;}
}
public void siftup(int[] arr, int idx) {
  while(idx > 0) {
      int parent = parent(idx);
      if(maxof(arr, parent, idx) == idx) {
          swap(arr, parent, idx);
          idx = parent;
      } else { break;}
  }
}
public void siftdown(int[] arr, int idx, int end) {
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
public void heapify(int[] arr) {
  int m = arr.length/2;
  while(m >=0) {
    siftdown(arr, m, arr.length-1);
    m--;
  }
}
public String largestNumber(int[] arr) { // arr is heapsort by heapify to a max heap.
  heapify(arr);
  int k = 0;
  StringBuilder sb = new StringBuilder();
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

''' http://stackoverflow.com/questions/5212037/find-the-recur-largest-sum-in-two-arrays
    The idea is, take a pair(i,j), we need to consider both [(i+1, j),(i,j+1)] as next val
    in turn, when considering (i-1,j), we need to consider (i-2,j)(i-1,j-1), recursively.
    By this way, we have isolate subproblem (i,j) that can be recursively checked.
'''
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
 
  Arrays.sort(envelopes, new comparator<int[]>( (a,b) -> {
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

""" the max num can be formed from k digits from a, preserve the order in ary.
key point is drop digits along the way.
Stack to save the branch(drop/not-drop) presere the order. 用来后悔 track each branch point.
"""
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

public String removeKdigits(String num, int k) {
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

""" max k digit num can be formed from a, b, preserve the order in each ary 
idea is i digits from a, k-i digits from b, then merge.
"""
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

// -----------------------------------------------------------
// sliding win: track pre state with map of needed and seen.
// keep prefix sorted !! so finding dist(i, j) or subary(i,j)
// To be able to determinstically move l, fix uniq chars in each win. Multi pass to expand uniq char window. 
// -----------------------------------------------------------
static int[] slidingWinMax(int[] arr, int w) {
  Deque<Integer> maxInWin = new ArrayDeque<>();
  int[] res = new int[arr.length-w+1];
  int l=0,r=0;
  for(;r<arr.length;++r) {
    // slide in r, consume it means check whether it's the largest.
    while(maxInWin.size() > 0 && arr[maxInWin.peekLast()] <= arr[r]) {
        maxInWin.removeLast();
    }
    maxInWin.addLast(r); // store the index
    if(r-l+1 == w) {  // l..r length
      res[l] = arr[maxInWin.peekFirst()];
      if(l == maxInWin.peekFirst()) {
          maxInWin.removeFirst();
      }
      l++;
    }
  }
  return res;
}
print maxWin([5,3,4,6,9,7,2],2)
print maxWin([5,3,4,6,9,7,2],3)

// longest non repeat substring. sliding window(locate left/rite). when match, need to mov l = pre+1.
static String maxNoRepeat(String s) {
  if (s == null || s.length() == 0) { return s;  }
  int maxlen = 0, maxstart = 0, maxend = 0;
  Map<Character, Integer> pos = new HashMap<>();  // LoopInvar: track arr[pos]'s max/last idx.
  for(int l=0, r=0; r<s.length(); r++) {
    char ch = s.charAt(r);
    if (pos.containsKey(ch)) {
      int ridx = pos.get(ch);
      if (l <= ridx) { l = ridx + 1;}  // <= as we need to advance l even when equals.
    }
    pos.put(ch, r);
    if (r - l + 1 > maxlen) {   // global LoopInvar track. [...] winsz = r-l+1;
      maxstart = l; maxend = r; maxlen = r - l + 1;
    }
  }
  System.out.println("non repeat substring - > " + s.substring(maxstart, maxend+1));
  return s.substring(maxstart, maxend+1);
}

""" next greater rite circular: idea is to go from rite, pop stk top when top is less than cur.
to overcome circular, do 2 passes, as when second pass, stk has the greater rite stk from prev
"""
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

// track expected count and winmap cnt for each char as LoopInvar.
static int minWinString(String s, String t) {
  int[] expects = new int[26];
  int[] winmap = new int[26];
  for (int i=0;i<t.length();i++) {  expects[t.charAt[i] - 'a'] += 1; }

  int winsize = 0, minwin = 999;
  for (int l = 0, r = 0; r < s.length(); ++r) {   // mov right edge, half open
    char cur = s.charAt(r);  // process current r; move r to next at the end;
    int cidx = cur - 'a';
    winmap[cidx] += 1;
    if (winmap[cidx] <= expects[cidx]) { winsize += 1; }
    if (winsize == t.length()) {
      while (l < r) {     // while loop to squeeze left edge until l,r the same.
        int lcharidx = s.charAt(l) - 'a';
        if (winmap[lcharidx] > expects[lcharidx]) {
            winmap[lcharidx] -= 1;   // reduce winmap cnt when sliding out.
            l += 1;
        } else {
            break;    // can not squeeze left anymore, break.
        }
      }
      minwin = Math.min(minwin, r-l+1);  // r is processed, so win size need to +1.
    }
  }
  System.out.println("min win of " + s + " : " + t + " = " + minwin);
  return minwin;
}
print minwin("azcaaxbb", "aab")
print minwin("ADOBECODEBANC", "ABC")

public String minWindow(String s, String t) {
  Map<Character, Integer> needMap = new HashMap<>();
  for(int i=0;i<t.length();++i) {
    needMap.compute(t.charAt(i), (k,v) -> {return v==null ? 1 : v+1;});
  }
  boolean allseen = false;
  Map<Character, Integer> seenMap = new HashMap<>();
  int l=0, r=0, minl=0, minr=0, minw=Integer.MAX_VALUE;
  while(r<s.length()) {
    Character rchar = s.charAt(r);
    seenMap.compute(rchar, (k,v) -> { return v==null ? 1 : v+1;});
    if(!allseen) {
        int i = 0;
        for(;i<t.length();++i) { 
          if(!seenMap.containsKey(t.charAt(i)) || 
              seenMap.get(t.charAt(i)) < needMap.get(t.charAt(i))) 
              break;
        }
        if(i == t.length()) allseen = true; 
        else { r++; continue; }
    }
    while(!needMap.containsKey(s.charAt(l)) || 
            seenMap.get(s.charAt(l)) > needMap.get(s.charAt(l))) {
        seenMap.compute(s.charAt(l), (k, v) -> v-=1);
        l++;
    }
    
    if(r-l+1 < minw) {
        minw = Math.min(minw, r-l+1);
        minl=l; minr=r;
    }
    r++;
  }
  if(minw != Integer.MAX_VALUE)
      return s.substring(minl, minr+1);
  else return "";
}
public List<String> findRepeatedDnaSequences(String s) {
  Set<Integer> words = new HashSet<>();
  Set<Integer> doubleWords = new HashSet<>();
  List<String> rv = new ArrayList<>();
  char[] map = new char[26];
  map['A' - 'A'] = 0; map['C' - 'A'] = 1; map['G' - 'A'] = 2; map['T' - 'A'] = 3;

  Set<Integer> temp = new HashSet<Integer>();
	Set<Integer> added = new HashSet<Integer>();
 
	for (int hash = 0, i = 0; i < len; i++) {
		if (i < 9) {
			//each ACGT fit 2 bits, so left shift 2
			hash = (hash << 2) + map.get(s.charAt(i)); 
		} else {
			hash = (hash << 2) + map.get(s.charAt(i));
			// make length of hash to be 20
			hash = hash &  (1 << 20) - 1; 
 
			if (temp.contains(hash) && !added.contains(hash)) {
				result.add(s.substring(i - 9, i + 1));
				added.add(hash); //track added
			} else {
				temp.add(hash);
			}
		}
	}
  return result;
}

// always consume a.i first, adjust state after that.
static int getSpecialSubstring(String s, int k, String charValue) {
  int normal = 0; int maxlen = 0; int l = 0;
  
  for(int i = 0; i < s.length(); ++i) {
    char ichar = charValue.charAt(s.charAt(i)-'a');
    if (ichar == '0') { normal++; }
    if (normal > k) {   // mov l when > k.
      while (charValue.charAt(s.charAt(l)-'a') == '1') { l++; }  // break out and stop at first != 1
      l++;
      normal--;
    }
    // always upbeat on new i.
    maxlen = Math.max(maxlen, i-l+1);
  }
  return maxlen;
}

// Top N : a priority heap to store and sort top k,
List<String> topKFrequent(String[] words, int k) {
    Map<String, Integer> wordCount = new HashMap();
    for (String word: words) {    // wd, and its counts
      wordCount.put(word, wordCount.getOrDefault(word, 0) + 1); }
    PriorityQueue<String> minHeap = new PriorityQueue<String>(
        (w1, w2) -> wordCount.get(w1).equals(wordCount.get(w2)) ?
        w2.compareTo(w1) : wordCount.get(w1) - wordCount.get(w2) );

    for (String word: wordCount.keySet()) {   // add all words to PQ
        minHeap.offer(word);
        if (minHeap.size() > k) minHeap.poll();     // cap the minHeap size.
    }
    List<String> ans = new ArrayList();
    while (!minHeap.isEmpty())  {  ans.add(minHeap.poll()); }
    Collections.reverse(ans);
    return ans;
}
// if maxdup stays, l will be moved along with r; (r-l+1-maxdups > w), no maxlen increase.
// until maxdups is recomputed with correct dup and increased, maxlen is then increased !
public int characterReplacement(String s, int k) {
  // track max cnts of arr/i; r-l-maxdups shall < k
  Map<Character, Integer> map = new HashMap<>();
  int l=0,r=0,maxlen=0,maxdups=0;
  for(int r=0;r<arr.length;r++) {  // while(r<s.length()) {
    map.compute(s.charAt(r), (k,v) -> {return v==null ? 1 : v+1;});
    maxdups = Math.max(maxdups, map.get(s.charAt(r)));
    if(r-l+1-maxdups > k) {  // do not mov l when == k !
      map.compute(s.charAt(l), (key,cnt) -> cnt-= 1);
      l++;  // if l is maxdups, why not update maxdups ? cur maxdups already triggers the mov of l;
    }
    maxlen = Math.max(maxlen, r-l+1);
  }
  return maxlen;
}

// slide r, mov l when a.r is not needed. 
static String anagram(String arr, String p) {
  Map<Character, Integer> winMap = new HashMap<>();
  Map<Character, Integer> patternMap = new HashMap<>();
  for (int i=0;i<p.length();i++) {
    patternMap.put(p.charAt(i), patternMap.getOrDefault(p.charAt(i), 0)+1);
  }
  for(int l = 0, r=0; r<arr.length(); r++){
    char ic = arr.charAt(r);
    if (!patternMap.containsKey(ic)) {
      l = r+1;
      winMap.clear();
      continue;
    }
    winMap.put(ic, winMap.getOrDefault(ic, 0)+1);
    if (winMap.get(ic) == patternMap.get(ic)) {
      if (r-l+1 == p.length()) {
        return arr.substring(l);
      }
    }
    if (winMap.get(ic) > patternMap.get(ic)) {
      while (arr.charAt(l) != ic) {
        winMap.put(arr.charAt(l), winMap.get(arr.charAt(l)-1));
        l++;
      }
      l++;
      winMap.put(ic, winMap.get(ic)-1);
    }
  }
  return null;
}
System.out.println(anagram("hello", "lol"));

// track avail/needed. when pop, needed++; result shall be the smallest in lexi order among all possible results.
// 用stack来先入再后悔。from left -> rite, if stk top > a[i] and count[stk.top] > 0, pop stk.top as still have chance to add stk.top later.
static String removeDupLetters(String s) {
  Map<Character, Integer> counts = new HashMap<>();
  Set<Character> keepSet = new HashSet<>();  // LoopInvar: track unique element.
  Stack<Character> keepStk = new Stack<>();

  for (int i=0;i<s.length();i++) { counts.put(s.charAt(i), counts.getOrDefault(s.charAt(i), 0) + 1); }
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
// the key is to keep the chars in order. stk the choosen chars, pop stk when consuming smaller char.
public static String RemoveDup(String s) {
  int slen = s.length();
  int[] counter = new int[26];int[] choosen = new int[26];
  for(int i=0;i<slen;++i) { counter[s.charAt(i)-'a']++; choosen[s.charAt(i) - 'a'] = 0;}
  Deque<Character> deq = new ArrayDeque<>();
  deq.addLast(s.charAt(0)); counter[s.charAt(0)-'a']--; choosen[s.charAt(0)-'a']++;
  // invariant of deq: sorted up to no dup char. 
  for(int i=1;i<slen;++i) {
      char cur = s.charAt(i);
      if (choosen[cur-'a'] > 0) { counter[cur-'a']--; continue; } // no dup in deq.
      // pop stk if cur is smaller. not choosen after poping. No poping if no char later.
      while(!deq.isEmpty() && cur <= deq.peekLast() && counter[deq.peekLast() - 'a'] > 0) { 
        choosen[deq.peekLast() - 'a']--; deq.removeLast(); // when poping, must has a dup later.
      } // after loop, a/i > top, or top can not be removed as no dup, or empty.
      deq.addLast(cur); counter[cur-'a']--; choosen[cur-'a']++; // bcdbc, bca
  }
  StringBuilder sb = new StringBuilder();
  while(!deq.isEmpty()) { sb.append(deq.pollFirst()); }
  return sb.toString();
}
System.out.println(removeDupLetters("dcdcad"));

// widx start from 1. compare against widx-1 and widx. Allow at most 2.
static void removeDup(int[] arr) {
    int l = 2;       // loop start from 3rd, widx as allows 2
    int widx = 1;    // widx 1, allow two dups
    for (int i=l; i<arr.length; i++) {
        if (arr[i] == arr[widx] && arr[i] == arr[widx-1]) {
            continue;  // skip current item when it is a dup
        }
        widx += 1;
        arr[widx] = arr[i];   // overwrite the dup as widx is a dup.
    }
    System.out.println(" " + Arrays.toString(arr));
}
removeDup(new int[]{1,3,3,3,3,5,5,5,7});


// Track incoming block's X with PQ's X while consuming each block.
public List<List<Integer>> getSkyline(int[][] arr) {
  PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> b[1]-a[1]);
  List<List<Integer>> xys = new ArrayList<>();
  int bidx=0,lx=0,rx=0,ht=0;
  // each incoming block either add to pq or pop top from pq.
  while(bidx < arr.length || pq.size() > 0) {  // upon each block, update the state data structure.
    if(bidx < arr.length) {lx=arr[bidx][0];rx=arr[bidx][1];ht=arr[bidx][2];}
    // compare block start with pq top block end, consume and enq bidx when overlap, or pop pq top If not overlap. Loop back.
    if(pq.size()==0 || bidx < arr.length && lx <= pq.peek()[0]) {
      while(bidx < arr.length && arr[bidx][0]==lx) { // dedup blocks started with the same X.
        ht=Math.max(ht, arr[bidx][2]);
        pq.offer(new int[]{arr[bidx][1], arr[bidx][2]}); // enq bidx(x, y), inc bidx
        bidx++;  // consume bidx, update pq.
      }
      if(xys.size()==0 || xys.get(xys.size()-1).get(1) < ht) { // enq same lx, max ht.
        xys.add(Arrays.asList(lx, ht));
      } 
    } else { // pop pq top if bidx not overlap with pq top. pop it, loop back and re-eval bidx.
      int[] top = pq.poll();
      int top_end = top[0];
      while(pq.size()>0 && pq.peek()[0] <= top_end) { pq.poll();} // keep pop top when top's end shadowed by cur top.
      ht = pq.size() == 0 ? 0 : pq.peek()[1];
      if(xys.size()==0 || xys.get(xys.size()-1).get(1) != ht) {
        xys.add(Arrays.asList(top_end, ht));
      }
    }
    // after popping top, loop back to re-eval bidx as bidx only inc-ed after it's enq.
  }
  return xys;
}

//
// Shortest subary with sum >= K;  // Keep prefix sum ordered !! 
// Must find what min ary[L..R] at each R !! which [0..L] to remove ? check each L ? sort l by prefixsum. 
// 
public int shortestSubarray(int[] arr, int k) {
  Deque<Integer> q = new ArrayDeque<>();
  int l=0,r=0,minr=0,minw=Integer.MAX_VALUE;
  long lrsum=0;
  boolean neg_mode = false;
  long[] prefix = new long[arr.length];
  for(int i=0;i<arr.length;i++) { prefix[i] = arr[i] + (i==0 ? 0 : prefix[i-1]); }
  while(r<arr.length) {
    if(arr[r] <= 0) neg_mode=true;
    else if(neg_mode) { neg_mode=false; q.addLast(r); }
    lrsum += arr[r];
    if(lrsum < 0) { l=r+1; r=l; lrsum=0; continue; } // l always ptred to positive.
    // move R without checking prefix/L is problematic. Must find what win ary at each R !!
    if(lrsum < k) { r++; continue; } 
    
    // lrsum >= k, squeeze l, adjust r
    minw=Math.min(minw, r-l+1); 
    for(int i=r-1;i>=l;i--) {
      if(prefix[r] - prefix[i] >= k ) { 
        minw=Math.min(minw, r-i); l=i+1; 
        break;
      }
    }
    // now two ptrs, mov l or r; shall mov l to the first pos that contributes
    l++; 
    while(l <= r && arr[l] <= 0) l++; // lseek l to positive;
    for(int i=r;i>=l;i--) {
      if(prefix[i]-prefix[l-1] <= 0) { l=i+1; break;} // exclude [l..i] which is neg !
    }
    lrsum = prefix[r]-prefix[l-1];
    r++;
  }
  if(minw==Integer.MAX_VALUE) return -1;
  return minw;
}

// The key is to keep prefix/l sorted to discard upperbound l prefix/l < prefix/r - K; No global watermark L !!
// As win is min, later i with smaller prefix/i shall cancel prev j whose prefix/j is larger.
public class PosSeg {
  int l; int r;  public PosSeg(int i, int j) {this.l=i; this.r=j;}
}
public class Prefix {
  long sum; int edidx;
  public Prefix(long sum, int idx) {this.sum=sum; this.edidx=idx;}
}
public int shortestSubarray(int[] arr, int k) {
  Deque<Prefix> q = new ArrayDeque<>();
  int minw=Integer.MAX_VALUE; // no global l watermark, replaced by each prefix end idx;
  long prefixsum = 0;
  for(int r=0;r<arr.length;r++) {//   while(r<arr.length) {
    prefixsum += arr[r];
    if(prefixsum >= k) { minw = Math.min(minw, r+1); }
    while(q.size() > 0) {
      if(prefixsum - q.peekFirst().sum >= k) {
        Prefix p = q.removeFirst();
        minw = Math.min(minw, r-p.edidx); // global watermark l is replaced by each preifx end idx;
      } else { break; }
    }
    while(q.size() > 0 && q.peekLast().sum > prefixsum) { q.removeLast(); }
    q.addLast(new Prefix(prefixsum, r));
  }
  if(minw==Integer.MAX_VALUE) return -1;
  return minw;
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
public int segK(char[] arr, int l, int r, int k, Map<Character, List<Integer>> map, Deque<int[]> segs) {
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
// split into segments and find maxlen of each segment.
public int longestSubstring2(String s, int k) {
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
public int longestSubstring(String s, int k) {
  char[] arr = s.toCharArray();
  Map<Character, List<Integer>> map = new HashMap<>();
  for(int r=0;r<arr.length;r++) {
    map.putIfAbsent(arr[r], new ArrayList<>());
    List<Integer> list = map.get(arr[r]);
    list.add(r);
  }
  int maxw=0, totuniqchars = map.size();
  for(int uniqcw=1;uniqcw<=totuniqchars;uniqcw++) {
    int l=0,r=l, chars=0, kchars=0; 
    int[] cnt = new int[26];  // reset
    for(;r<arr.length;r++) {
      cnt[arr[r]-'a']++;
      if(cnt[arr[r]-'a']>=k) {
        if(cnt[arr[r]-'a']==k) kchars++; // inc only when from k-1 to k
        if(kchars==uniqcw) { 
          maxw=Math.max(maxw, r-l+1);
        }
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

// track prepre, pre, cur idx of zeros '''
def maxones(arr):
  mx,mxpos = 0,0
  prepre,pre,cur = -1,-1,-1  # LoopInvar: precondition 
  for i in xrange(len(arr)):
    if arr[i] == 1:    continue
    cur = i           // mark i as cur
    if cur-prepre-1 > mx:
        mx = cur-prepre-1
        mxpos = pre
    prepre,pre = pre,cur
  if i-prepre > mx:  mxpos = pre
  return mxpos
print maxones([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
print maxones([1, 1, 1, 1, 0])

""" Longest substring with at most 2 distinct chars 
prepre to track first cliff, pre to track second.
sliding win, i -> win start, j -> the second char in win.
slide k, K++ if A[k] = A[k-1]
case 1: A[k] == A[j], still 2 char, but need to let j = k-1 so i will jmp to j+1.
case 2: A[k] != A[j], i,j are 2 chars, upbeat, and set i = j-1 and update j=k-1.
so i reset to the first char(j=k-1), as last time when (A[k] != A[k-1] and A[k] == A[j]), 
"""
static int maxWinTwoChars(int[] arr) {
  int prepre = 0;
  int pre = -1;  // no second yet
  int sz = arr.length;
  int maxlen = 0;
  for(int i=1; i<sz; i++) {
      if (arr[i] == arr[i-1]) {  continue; }       // simplest, cur is the same as pre
      else if (pre == -1) { pre = i;  continue; }  // first time pre, track and continue
      else if (arr[i] == arr[prepre]) { pre = i - 1; }  // cur != pre, but =prepre, !!! relocate pre at the start of non continue gap
      else {
          maxlen = Math.max(maxlen, i-prepre);
          prepre = pre + 1;  // + 1
      }
  }
  System.out.println("max len is " + maxlen);
  return maxlen;
}
maxWinTwoChars(new int[]{2,1,3,1,1,2,2,1});

// - - - - - - - - - - - - - - - - - - - - -  - - -
// Bucket sort: bucket width, # of bucket. 
// sort eles into buckets so buckets are partially sorted. 
// 1+ ele per bucket. inside buckets tracks aggregation of eles(min/max).
// - - - - - - - - - - - - - - - - - - - - -  - - -
static void swap(int[] arr, int i, int j) { int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; }
static int[] buckesort(int[] arr) {
  int l = 0; int sz = arr.length; int r = sz -1;
  while (l < sz) {
    while (arr[l] != l+1 && arr[l] >= 0) {
      int lv = arr[l];
      if (arr[lv-1] == -999) {
          arr[lv-1] = -1;
          arr[l] = -999;
      }
      else if (arr[lv-1] < 0 ) {
          arr[lv-1] -= 1;
          arr[l] = -999;
      } else {
          swap(arr, l, lv-1);
          arr[lv-1] = -1;
      }
    }
    if (arr[l] == l+1) {
        arr[l] = -1;
    }
    l += 1;
  }
  System.out.println("arr -> " + Arrays.toString(arr));
  return arr;
}

public int firstMissingPositive(int[] arr) {  // arr[i] == i+1, arr[arr[i]-1]=arr[i]-1
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
public int findDuplicate1(int[] nums) {
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
public int findDuplicate(int[] arr){
  for(int i=0;i<arr.length;i++){
    if(arr[Math.abs(arr[i])-1] < 0) return Math.abs(arr[i])
    arr[Math.abs(arr[i])-1] *= -1;
  }
  return -1;
}

""" max repetition num, bucket sort. change val to -1, dec on every reptition.
    bucket sort, when arr[i] is neg, means it has correct pos. when hit again, inc neg.
    when dup processed, set it to max num to avoid process it again.
"""
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
  n = len(nums)
  d = {}
  w = t + 1
  for i in xrange(n):
      m = nums[i] / w
      if m in d:   return True
      if m - 1 in d and abs(nums[i] - d[m - 1]) < w:  return True
      if m + 1 in d and abs(nums[i] - d[m + 1]) < w:  return True
      d[m] = nums[i]
      if i >= k: del d[nums[i - k] / w]
  return False

class Solution {
  public class Bucket {
    int min; int max; int left; int rite; int idx;
  }
  public int maximumGap(int[] arr) {
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
    if (j == pat.length || (j+2 == pat.length && pat[j+1] == '*')) { 
      return true;  
    }
    return false;
  } else if ( j == pat.length) {
    return false;
  }
  // both i and j are valid
  if (j+1 < pat.length && pat[j+1] == '*') {
    if (pat[j] == '.' || arr[i] == pat[j]) {
      int k = i;
      for(; k < arr.length && arr[i] == arr[k]; k++) {
        if (RegMatch(arr, k, pat, j+2))
          return true;
      }
      // when out, k == arr.length or arr[k] != arr[i]
      return RegMatch(arr, k, pat, j+2);
    }
    return RegMatch(arr, i, pat, j+2);
  } else {
    if (pat[j] == '.' || arr[i] == pat[j]) {
      return RegMatch(arr, i+1, pat, j+1);
    } else {
      return false;
    }
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

//https://github.com/labuladong/fucking-algorithm/blob/master/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%B3%BB%E5%88%97/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E4%B9%8B%E6%AD%A3%E5%88%99%E8%A1%A8%E8%BE%BE.md
bool dp(string& s, int i, string& p, int j) {
  if (s[i] == p[j] || p[j] == '.') {  // 匹配，检查 j+1 ?= *
    if (j < p.size() - 1 && p[j + 1] == '*') {
        // 1.1 通配符匹配 0 次或多次
        return dp(s, i, p, j + 2)
            || dp(s, i + 1, p, j);
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

""" LoopInvar: track of 2 values for each ele, max of exclude this ele, and the
max of no care of whether incl or excl cur ele. """
public int CalPackRob(int[] nums) {
  int pre=0,prepre=0;  // boundary condition
  for(int i=0;i<arr.length;++i) {
    int incl = prepre+arr[i];
    int excl = pre; // Math.max(prepre, pre);
    prepre = pre;
    pre = Math.max(incl, excl); // max of inc/exl i; set as pre for next i+1;
  }
  return pre;
}

public boolean canJump(int[] arr) {
  int r=0,maxr=0;
  while(r<=maxr) {
    maxr = Math.max(maxr, r+arr[r]);
    if(maxr+1 >= arr.length) return true;
    r++;
  }
  return false;
}

def minjp(arr):  // use single var to track, no need to have ary.
  mjp = 1;
  curmaxRite = arr[0]
  nextMaxRite = arr[0]   // seed with value of first.
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
  posJumps.put(0, new HashSet<Integer>());
  posJumps.get(0).add(1);  // first jmp to first pos
  for (int i = 1; i < stones.length; i++) {
    posJumps.put(stones[i], new HashSet<Integer>() );   // hashset store how many prev jmps when reaching this stone.
  }
  
  for (int i = 0; i < stones.length - 1; i++) {
    int iStop = stones[i];
    for (int step : posJump.get(iStop)) {
      int nextReach = iStop + step;
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

public int canCompleteCircuit(int[] gas, int[] cost) {
  int[] net = new int[gas.length];
  int sum = 0, start=0;
  for(int i=0;i<gas.length;++i) {
    net[i] = gas[i] - cost[i];
    sum += net[i];
  }
  if(sum < 0) return -1; 
  sum = 0;
  for(int i=0;i<gas.length;++i) {
    sum += net[i];
    if(sum < 0) {
      start = -1;
      sum = 0;
    } else if(start < 0) {
      start = i;
    }
  }
  return start;
}

// two errors: 1. did not max subtree incl/excl. 2. shall max incl/excl of the subtree, max(l.[incl|excl])+max(r.[incl/excl]) vs. max([l|r].incl, [l|r].excl)
public class NodeMax {
  int incl_root; // incl root, child must excl root; +child.excl_root
  int excl_root; // excl root, regardless child incl/excl root; take max of (child.incl_root, child.excl_root);
  
  int exclrootmax;  // definitely not incl root, to add to parent incl_root;
  int overallmax;  // regardless incl/excl, max of(incl root, exclroot); 
}
public NodeMax dfs(TreeNode root, TreeNode parent) {
  NodeMax s = new NodeMax();
  if(root==null) return s;
  NodeMax lchild = dfs(root.left, root);
  NodeMax rchild = dfs(root.right, root);
  s.incl_root = root.val + lchild.excl_root + rchild.excl_root;
  // Difference !! max(l.incl + r.incl) vs. max(l.incl, l.excl) + max(r.incl, r.excl);
  // s.excl_root = Math.max(lchild.incl_root + rchild.incl_root, lchild.excl_root + rchild.excl_root);
  s.excl_root = Math.max(lchild.incl_root, lchild.excl_root) + Math.max(rchild.incl_root, rchild.excl_root);
  s.overallmax = Math.max(lchild.overallmax+rchild.overallmax, root.val + lchild.exclrootmax + rchild.exclrootmax);
  s.exclrootmax = lchild.overallmax+rchild.overallmax;
  return s;
}
public int rob(TreeNode root) {
  NodeMax max = dfs(root, null);
  // return Math.max(max.incl_root, max.excl_root);
  return max.overallmax;
}

""" if different coins count the same as 1 coin only, the same as diff ways to stair """
def nways(steps, n):
  ways[i] = steps[i]
  for i in xrange(steps[1], n):
    for step in steps:
      ways[i] += ways[i-step]    // ways inheritated from prev round
  return ways[n-1]


// dep carry on aggregated state from above + left;
public int maximalSquare(char[][] mat) {
  int maxsq = 0;
  int[][] square = new int[mat.length][mat[0].length];
  for(int i=0;i<mat.length;++i) {
    for(int j=0;j<mat[0].length;++j) {
      if(mat[i][j] == '0') { square[i][j] = 0; continue;} 
      // take the min from left, above and above left.
      int above = i > 0 ? square[i-1][j] : 0;
      int left = j > 0 ? square[i][j-1] : 0;
      int aboveleft = i>0&&j>0 ? square[i-1][j-1] : 0;
      square[i][j] = Math.min(Math.min(above, left), aboveleft)+1;
        maxsq = Math.max(maxsq, square[i][j]);
      }
    }
  }
  return maxsq*maxsq;
}

static int decodeWays(int[] arr) {
  int sz = arr.length, cur=0;
  int prepre = 1, pre = 1;  // prepre = 1, as when 0, it is 0-2, the first 2 digits, count as 1.
  for (int i = 1; i<sz; i++) {  // start from 1
      cur = pre;
      if (arr[i] == 0) { prepre = 1; }
      if (arr[i-1] == 1 || (arr[i-1] == 2 && arr[i] <= 6)) { cur += prepre; }
      prepre = pre;
      pre = cur;
  }
  System.out.println("decode ways " + cur);
  return cur;
}
decodeWays(new int[]{2,2,6});
  
//  decode with star, DP tab[i] = tab[i-1]*9
// https://leetcode.com/problems/decode-ways-ii/discuss/105258/Java-O(N)-by-General-Solution-for-all-DP-problems
//  1. ways(i) -> that gives the number of ways of decoding a single character
//  2. ways(i, j) -> that gives the number of ways of decoding the two character string formed by i and j.
//  tab(i) = ( tab(i-1) * ways(i) ) + ( tab(i-2) *ways(i-1, i) )
int ways(int ch) {
    if(ch == '*') return 9;
    if(ch == '0') return 0;
    return 1;
}
int ways(char ch1, char ch2) {
    String key = "" + ch1 + "" + ch2;  // memoize
    if(ch1 != '*' && ch2 != '*') {     // no wild card
      if (Integer.parseInt(key) >= 10 && Integer.parseInt(key) <= 26) 
        return 1;
    } 
    else if (ch1 == '*' && ch2 == '*') { return 15; }  // both wildcard
    else if (ch1 == '*') {   // one wildcard
        if (Integer.parseInt(""+ch2) >= 0 && Integer.parseInt(""+ch2) <= 6) return 2;
        else return 1;
    } else {
        if (Integer.parseInt(""+ch1) == 1 ) { return 9;} 
        else if(Integer.parseInt(""+ch1) == 2 ) { return 6; } 
    }
    return 0;
}
// # dp[i] = dp[i-2] + dp[i]; 
static int decodeWays(String s) {
    long[] dp = new long[2];
    dp[0] = ways(s.charAt(0));
    if (s.length() < 2) return (int)dp[0];
    // dp[1] is nways from [0,1], dp[i] = dp[i-1]*ways(a[i]) + dp[i-2]
    dp[1] = dp[0] * ways(s.charAt(1)) + ways(s.charAt(0), s.charAt(1));
    // DP tab bottom up from string of 2 chars to N chars.
    for(int j = 2; j < s.length(); j++) {  // start from j=2
        long temp = dp[1];
        dp[1] = (dp[1] * ways(s.charAt(j)) + dp[0] * ways(s.charAt(j-1), s.charAt(j))) % 1000000007;
        dp[0] = temp;
    }
    return (int)dp[1];
}


""" 
  Partial sort by partition to find Kth element. partition sort is O(n). Find Median. Be careful of pivot.
  http://www.ardendertat.com/2011/10/27/programming-interview-questions-10-kth-largest-element-in-array/
"""
public int[] partition(int[] arr, int l, int r) {
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
public int findKth(int[] arr, int l, int r, int k) {
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

"""
find kth smallest in a union of two sorted list
"""
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
public int[] bisectMedian(int[] arra, int al, int[] arrb, int bl, int discard) {
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

public double findMedianSortedArrays(int[] arra, int[] arrb) {
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

// - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - -- - - 
// Merge sort to convert N^2 to Nlgn. all pair(i,j) comp, skipping some during merge.
// - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - -- - - 
""" inverse count(right smaller) with merge sort. sort A, instead of sorted another indirect lookup, 
Rank[i] stored idx of Arr, an indirection layer. not the actual A[i]. max =/= A[n] but A[rank[n]]
"""
public class MergeInverse {
  static void MergeInverse(int[] arr, int left, int rite, int sorted[], int[] inverse) {
      // iterative, recur, dp.
      // exit boundary check
      if (left == rite) { return; }
      // loop body, enter setup, exit cleanup.
      int mid = left + (rite-left)/2;
      MergeInverse(arr, left, mid, sorted, inverse);
      MergeInverse(arr, mid+1, rite, sorted, inverse);

      int[] interval_sorted = new int[rite-left+1];
      int lidx = left, ridx = mid+1, k = 0;
      // 3 while loop is better
      while(lidx <= mid && ridx <= rite) {
      }
      while(lidx <= mid) {}
      while(ridx <= rite) {}
      //
      while(lidx < mid+1 || ridx < rite+1) {
          if (lidx == mid+1) {
              interval_sorted[k++] = ridx++;
          } else if (ridx == rite+1) {
            // when moving left, the [mid+1..ridx] all smaller than arr[sorted[lidx]];
            inverse[sorted[lidx]] += ridx-(mid+1);  // for [9, 5], sorted[0]=1, loop sorted[0..1], inv[sorte[0..1]], not inv[0..1]
            interval_sorted[k++] = lidx++;
          }
          // we are iterating thru sorted idx, not nature index of the original ary.
          // sorted[0] sorted[1]
          else if (arr[sorted[lidx]] < arr[sorted[ridx]]) {
              inverse[sorted[lidx]] += ridx-(mid+1);
              interval_sorted[k++] = lidx++;
          } else {  // left interval > rite_interval
              interval_sorted[k++] = ridx++;
          }
      }
      for(int i=left;i<=rite;++i){
          sorted[i] = interval_sorted[i-left];
      }
  }
  static void MergeInverse2(int[] arr, int left, int rite, int[] sorted, int[] inverse) {
    if (left == rite) { return; }
    int mid = left + (rite-left)/2;
    int leftlen = mid-left+1, ritelen=rite-mid;
    int[] segment_sorted = new int[leftlen+ritelen];
    
    MergeInverse2(arr, left, mid, sorted, inverse);
    MergeInverse2(arr, mid+1, rite, sorted, inverse);
    // each left rite subary is sorted, iterate via sorted[0..leftlen], use l,r from 0;
    // inverse[] must on sorted order, hence inverse[sorted[l]]. left sub[9,5] is itered as [5,9]
    for(int l=0,r=0; l<leftlen || r<ritelen;) {
      if (l==leftlen) {
        segment_sorted[l+r] = mid+1+r;
        r++;
      } else if (r==ritelen) {
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
  public static void TestMergeInverse(String[] args) {
      System.out.println("------ TestMergeInverse ---------------"); 
      // int[] arr = new int[]{4,5,2,1,9,8,3};
      int[] arr = new int[]{9,5,6};
      int[] sorted = new int[arr.length];
      int[] inv = new int[arr.length];
      for (int i=0;i<arr.length;++i) {
          inv[i] = 0;
          sorted[i] = i;
      }
      MergeInverse(arr, 0, arr.length-1, sorted, inv);
      // arr[sorted[0]] < arr[sorted[1]]
      // inv[i]: sum(arr[i] > arr[k] k = [i+1...n])
      System.out.println("arr : " + Arrays.toString(arr) + " inv " + Arrays.toString(inv));
      System.out.println("------ END --------------"); 
  }
}

// count the number of range sums of an ary that lie in [lower, upper] inclusive.
// 1. map prefixsum[i] to its counts. from n..0, find all prefix[i]-[lower..upper].
// 2. any prefix pair(i,j) is a range. count all pairs within low, upper. N^2.
// 3. merge sort to compare any pair. some pairs are skipped during merge sorted halves.
public void merge(long[] prefix, int la, int lb, int rb, long[] staging) {
  int ra=lb-1, si=0;
  while(la <= ra && lb <= rb) {
    if(prefix[la] <= prefix[lb]) {staging[si++]=prefix[la++];}
    else {staging[si++]=prefix[lb++];}
  }
  while(la<=ra) {staging[si++]=prefix[la++];}
  while(lb<=rb) {staging[si++]=prefix[lb++];}
}
public int mergeSort(long[] prefix, int l, int r, int lo, int up) {
  int tot=0;
  if(l==r) { if(lo <= prefix[l] && prefix[l] <= up) return 1; else return 0;}
  long[] staging = new long[r-l+1];
  int m=l+(r-l)/2;
  int ltot = mergeSort(prefix, l, m, lo, up);
  int rtot = mergeSort(prefix, m+1, r, lo, up);
  tot += (ltot+rtot);
  // System.out.println(String.format("prefix %s lr %d..%d ltot %d rtot %d Tot %d ", Arrays.toString(prefix), l, r, ltot,rtot, tot));
  
  int rl=m+1,rr=m+1;
  for(int i=l;i<=m;i++) { // compare pair(l, rl..rr); count matched pair(l,r)
    while(rl<=r && prefix[rl] - prefix[i] < lo) rl++;  // rl parked to the first bigger than lo, inclusive.
    while(rr<=r && prefix[rr] - prefix[i] <= up) rr++; // rr parked to the first bigger than up, exclusive.
    tot += rr-rl; // after loop break, rr parked over up, pointed to the end, not inclusive.
    // System.out.println(String.format("prefix %s i %d rl:rr %d %d Tot %d ", Arrays.toString(prefix), i, rl, rr, tot));
  }
  merge(prefix, l, m+1, r, staging); // shall populate staging during the merge loop.
  for(int i=l;i<=r;i++) { prefix[i] = staging[i-l]; }
  return tot;
}
public int countRangeSum(int[] arr, int lo, int up) {
  long[] prefix = new long[arr.length];
  for(int i=0;i<arr.length;++i) {
    prefix[i] = (i==0 ? 0 : prefix[i-1]) + arr[i];
  }
  return mergeSort(prefix, 0, arr.length-1, lo, up);
}  

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - -------
// Tree recur(root, parent, prefix, collector); or Tree iter while(cur != null || !stk.empty())
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - -------
public boolean isValidBST(TreeNode cur, long lo, long hi) { // lo/hi replace the need to detect cur is left/rite.
  if(cur==null) return true;
  if(cur.val >= hi || cur.val <= lo) return false; // cur must between lo < cur < hi
  // reduce hi when recur left, expand lo when recur right.
  return isValidBST(cur.left, cur, lo, cur.val) && isValidBST(cur.right, cur, cur.val, hi);
}
public boolean isValidBST(TreeNode root) {
  return recur(root.left, root, Long.MIN_VALUE, root.val) && recur(root.right, root, root.val, Long.MAX_VALUE);
}
public String serialize(TreeNode root) {
  StringBuilder sb = new StringBuilder();
  if(root == null) {sb.append("$,"); return sb.toString();}
  sb.append(root.val + ",");
  sb.append(serialize(root.left));
  sb.append(serialize(root.right));
  return sb.toString();
}
public NodeIdx recur(String[] arr, int idx) {
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

class Node {
  Node left, rite;
  int value, rank;
}
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
// recur done collects # of nodes in subtrees. 
public int[] kthSmallestRecur(TreeNode root, int k) {
  if(root == null) return new int[]{0, -1};
  int[] left = recur(root.left, k);
  if(left[1] >= 0) return left; // found
  if(left[0]+1 == k) return new int[]{left[0]+1, root.val};
  int[] rite = recur(root.right, k-left[0]-1);
  if(rite[1] >= 0 ) return rite;
  else return new int[]{left[0]+rite[0]+1, -1}; // return pair [num_children, found]
}
public int kthSmallestLoop(TreeNode root, int k) {
  Deque<TreeNode> stk = new ArrayDeque<>();
  TreeNode cur = root;
  while(cur != null || stk.size() > 0) {
    while(cur != null) { stk.addLast(cur); cur = cur.left; continue;}
    // no more left, pop stk and visit cur;
    cur = stk.removeLast();
    --k;
    if(k==0) break;
    cur = cur.right; // after visit, move to right.
  }
  return cur.val;
}

// given a root, and an ArrayDeque (initial empty). iterative. root is not pushed to stack at the begining.
public List<Integer> preorderTraversal(TreeNode root) {
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
public List<Integer> postorderTraversal(TreeNode root) {
  Deque<Integer> result = new ArrayDeque<>();
  Deque<TreeNode> stack = new ArrayDeque<>();
  TreeNode cur = root;
  while(!stack.isEmpty() || cur != null) {
      while(cur != null) {
          stack.offerLast(cur);
          result.offerFirst(cur.val);  
          cur = cur.right;               // go right first. Reverse the process of preorder
      } 
      TreeNode node = stack.pop();
      cur = node.left;            // after poping from stack, go left. Reverse the process of preorder
  }
  return result;
}

""" two branches, left, rite. Stash rite, make rite to point to left.
"""
static Node flatten(Node root) {
  Stack<Node> stk = new Stack<>();
  // LoopInvar: stk left, rite <= left, mov to rite; if no rite, pop stk.
  while (root != null || stk.size() > 0 ) {
    if (root != null) {
      if (root.rite != null) { stk.add(root.rite); }   // set aside rite to stk, will come back later.
      root.rite = root.left;
      root = root.rite;           // descend
    } 
    else {  root = stk.pop(); }
  }
  return root;
}
// Nested loop version. Inner loop to pop stk until cur.rite not null.
static Node flatten(Node root) {
  while(root != null) {
    if (root.left) { stk.add(root); root = root.left;}
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

// recursive this fn. iterate within the body.
def popNext(root):
  nexthd, nextpre = None,None
  cur = root
  while cur:
    for c in cur.children:
      if not nexthd:
        nexthd = c
        nextpre = c
      else:
        nextpre.next = c
      nextpre = c
    cur = cur.next
  nextpre.next = None
  // recur on nexthd, or you can another while to avoid recur.
  popNext(nexthd)

// --------------------------------------------------
// Tree DFS vs. while loop
// --------------------------------------------------
/**
 * 1. a graph, start from a root.
 * 2. a queue for each level.
 * 3. head, for each child,
 * 4. a) not visited, visit edge, set parent. enqueue
 *    b) in stack, visit edge. not parent, no enqueue.
 *    c) visited, must be directed graph, otherwise, cycle detected, not a DAG.
 */
bfs(graph *g, int start) {
  Queue<int[]> q = new LinkedList<int[]>();
  enqueue(&q,start);
  discovered[start] = TRUE;

  while (empty_queue(&q) == FALSE) {
    v = dequeue(&q);         // deq to current
    process_vertex_enter(v);
    processed[v] = TRUE;     // mark
    edges = g->edges[v];
    while (edges != NULL) {
      for each edge to child e, ( v -> e):
        if ((processed[e] == FALSE) || g->directed )
          process_edge(v,e);
        if (discovered[y] == FALSE) {
          parent[y] = v;
          discovered[y] = TRUE;
          enqueue(&q,y);
        }
      }
    process_vertex_exit(v);
  }
}

def two_color(G, start):
  def process_edge(from, to):
    if color[from] == color[to]:
      bipartite = false
      return
    color[to] = complement(color[from])

/**
 * 1. recursive from current root
 * 2. visit order matters, (parent > child), topology sort
 * 3. process_node_enter/exit, track invariant on enter/exit. discved/visited
 * 4. process edge: tree_edge, back_edge, cross_edge;
 *    a) not , process edge, set parent. recursion
 *    b) in stack, process edge. not parent, no recursion.
 *    c) visited, must be directed graph, otherwise, cycle detected, not a DAG.
 * 5. variant: discoved/visited, earliest_back;
 */
dfs(root, parent, colector):
  if not root: return
  process_vertex_dfs_enter(v) // { disved = true; entry_time=now}
  rank.root = gRanks++   // set rank global track.
  for c in root.children:
    if not discved[c]:
      discved[c]=true; parent[c] = root; 
      process_edge(root, c, tree_edge)  // first, process edge forward edge
      dfs(c, parent=root)    // then recur dfs
    else if !visited[c] or directed:
      process_edge(root, c, back_edge)  // back edge
    else if visited[c]:      // must be DAG, otherwise, not DAG.
      process_edge(root, c, cross_edge) // 
  visited[root] = true
  process_vertex_dfs_exit(v);

def topology_sort(G):
  sorted = []
  for v in g where visited[v] == false:
    dfs(v)

  def process_vertex_post_children_recursion_done(v):
    sorted.append(v)

def strong_component(g):
  // for each edge from x -> y, fan out, get min from back edges.
  def process_edge(x, y):
    class = edge_classification(x,y)
    if (class == TREE)
      tree_out_degree[x] = tree_out_degree[x] + 1
    if ((class == BACK) && (parent[x] != y))
      if (entry_time[y] < entry_time[ reachable_ancestor[x]]
        reachable_ancestor[x] = y;

  // all eddges done, recursion returns. update parent's states.
  def process_vertex_dfs_exit(v):
    int time_v;    // earliest reachable time for v;
    int time_parent;  // earliest reachable time for parent[v]
    if (parent[v] < 1)   // no need post/late processing for root.
      if (tree_out_degree[v] > 1)
        return;
    root = (parent[parent[v]] < 1); /* is parent[v] the root? */
    if ((reachable_ancestor[v] == parent[v]) && (!root))
      printf("parent articulation vertex: %d \n",parent[v]);
    if (reachable_ancestor[v] == v)
      printf("bridge articulation vertex: %d \n",parent[v]);
      if (tree_out_degree[v] > 0) /* test if v is not a leaf */
        printf("bridge articulation vertex: %d \n",v);
    // either in parent for-loop all children, or in each children update parent.
    time_v = entry_time[reachable_ancestor[v]];
    time_parent = entry_time[reachable_ancestor[parent[v]]];
    if (time_v < time_parent)    // update reachable_anecestor of my parent based on mine.
      reachable_ancestor[parent[v]] = reachable_ancestor[v];

public void dfsClone(Node node, Node parent, Map<Node, Node> clone) {
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

public List<Integer> rightSideView(TreeNode root) {
  List<Integer> view = new ArrayList<>();
  if (root == null) return view;
  Deque<TreeNode> deq = new ArrayDeque();
  TreeNode level_head = root; // set to null when pop. reset when enq if it is null.
  deq.addLast(level_head);
  TreeNode last = null;  // invariance: updated every pop. add to view when level head polled.
  while(!deq.isEmpty()) {
      TreeNode cur = deq.pollFirst();
      if (cur == level_head) {
          level_head = null;
          if (last != null) view.add(last.val);
      }
      last = cur;
      if (cur.left != null) { deq.addLast(cur.left); }
      if (cur.right!= null) { deq.addLast(cur.right);}
      if (level_head == null) { if (cur.left != null) level_head = cur.left; else level_head=cur.right;}
  }
  view.add(last.val);
  return view;
}

// find the root of a Min Heigth Tree. Remove leaves layer by layer.
// MHT can only have at most 2 roots. 3 roots folds to 1 root.
public List<Integer> MinHeightTreeBfsRemoveLeaves(Map<Integer, Node> nodes) {
  Deque<Node> q = new ArrayDeque<>();
  int remains = nodes.size();
  while(remains > 2) { // MinHeightTree can only have at most 2 root. 3 roots will fold to 1 root.
    for(Node n : nodes.values()) {
      if(!n.visited && n.removedChildren.size()+1==n.children.size()) {
        q.addLast(n);
      }
    }
    while(q.size() > 0) {
      Node hd = q.removeFirst();
      hd.visited = true;
      remains--;
      for(Node c : hd.children) {
        if(!c.visited) {
          c.removedChildren.add(hd);
        }
      }
    }
  }
  List<Integer> roots = new ArrayList<>();
  for(Node n : nodes.values()) {
    if(!n.visited) roots.add(n.id);
  }
  return roots;
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// TrieNode, The root TrieNode only has childrens. The leaf has the word.
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
class TrieNode {
    TrieNode[] children;    # children[i] => new TrieNode
    String word;            # leaf word
    TrieNode() {  children = new TrieNode[26]; }
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
      cur.word = root;     // word set at leaf level;
  }

  StringBuilder ans = new StringBuilder();
  for (String word: sentence.split("\\s+")) {
      if (ans.length() > 0) ans.append(" ");

      TrieNode cur = trieRoot;
      for (char letter: word.toCharArray()) {
          if (cur.children[letter - 'a'] == null || cur.word != null)  // this leaf
              break;
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
    for (String root: roots) {
        TrieNode cur = trieRoot;
        for (char letter: root.toCharArray()) {
            if (cur.children[letter - 'a'] == null) cur.children[letter - 'a'] = new TrieNode();
            cur = cur.children[letter - 'a'];
        }
        cur.word = root;
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
  public int sum(String prefix) {
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

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Pratt parser, while() { recur()}, vs. graph dfs. if recur
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// // 2 + 3 * 4 - 5 /（6 + 7）/ 8 - 9; recur stop arg is __parent_min_bp__. recur ret when cur op < parent_op, bind to parent.
// fn Pratt(lexer, loop_stop_cond_from_parent) -> S { // set to min of all op, will consum all lexer tokens. 
//   // steps: { Prep parent, peek op, loop until op.lbp < min}
//   lhs = match(lexer.next());
//   op = lexer.peek();  // look ahead peek the next op
//   while (op.lbp > loop_stop_cond_from_parent) {  // loop + recursion consume child ops till an op's lbp is less.
//     lexer.pop();  // if recur, consume head
//     child = recur(loop_stop_cond_from_parent = op.lbp);  // child recur ret, parent dfs done.
//     lhs = cons(op, lhs, child); // child dfs done as a new TreeNode.
//     op = lexer.peek();  // look ahead prep loop condition.
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
  op = src[idx];
  while (op.lbp < parent_op.rbp)) { /* loop stop when top op < parent_op, then all prefix can bind to lhs */
    op = l.pop();
    rhs,nextidx = parseInternal(src, idx, op);
    lhs = (op, lhs, rhs); // update lhs to contain consumed parts.
    op = src[nextidx];
  }
  return lhs;
}

class Solution {
  public class Expr {
    int val; int idx;  // idx to the op just after number token.
    public Expr(int val, int idx) { this.val=val;this.idx=idx;}
  }
  public int skipEmpty(char[] arr, int idx) {
    while(idx < arr.length && arr[idx] == ' ') idx++;
    return idx;
  }
  public int getInt(char c) {
    if(c >= '0' && c <= '9') { return c-'0';}
    return -1;
  }
  public int[] getNumAndNextOpIdx(char[] arr, int idx) {
    int num = 0;
    while(idx < arr.length && getInt(arr[idx]) != -1) { 
      num = num*10+getInt(arr[idx]); idx++; 
    }
    return new int[]{num, skipEmpty(arr, idx)};
  }
  public int eval(int lhs, char op, int rhs) {
      switch (op) {
          case '+': return lhs+rhs;
          case '-': return lhs-rhs;
          case '*': return lhs*rhs;
          case '/': return lhs/rhs;
      }
      return -1;
  }
  // how power the op binds to a number to the right next recur vs. to the left parent.
  public int bp(char op) { 
      switch (op) {
        case '(', ')': return 1;
        case '+', '-': return 2;
        case '*', '/': return 3;
      }
      return -1;
  }
  // when entering recur, pos points to the start number of lhs, can be a num or an (; 
  // return lhs and next operator pos.
  public Expr pratt(char[] arr, int pos, int parentOpBp) {
    int[] numIdx = getNumAndNextOpIdx(arr, pos);
    int lhs = numIdx[0], opidx=numIdx[1];
    while(opidx < arr.length && opbp(arr[opidx]) > parentOpBp) {
      Expr rhs = pratt(arr, opidx+1, opbp(arr[opidx]));
      lhs = eval(lhs, arr[opidx], rhs.val);
      opidx = rhs.idx;
    }
    return new Expr(lhs, opidx);
  }
  public int calculate(String s) {
      char[] arr = s.toCharArray();
      return pratt(arr, 0, -1).val;
  }
  public int calculate("(3-(5-(8)-(2+(9-(0-(8-(2))))-(4))))");
}

class Solution {
  public class ExpRes {
    String str_;
    int edidx;
    public ExpRes(String s, int i) { this.str_ = s; this.edidx=i;}
  }
  public int getNum(char i) {
    if(i >= '0' && i<='9') { return i-'0';}
    return -1;
  }
  public boolean isLetter(char c) {
    return c >= 'a' && c <= 'z';
  }
  // Expand recursively. a top level n[] entry that begin with a num, recursion ends with ].
  public ExpRes expand(char[] arr, int numidx) {
    ExpRes res = null;
    int i=numidx, repn=0;
    while(i<arr.length && getNum(arr[i]) >= 0) {
      repn = repn*10 + getNum(arr[i]);
      i++;
    }
    i++; // skip [
    String reps = "";
    StringBuilder sb = new StringBuilder();
    // recursion start with a `num[`, recursion until hitting the end ]
    while(arr[i] != ']') {
      while(i<arr.length && isLetter(arr[i])) {
        sb.append(arr[i]); i++;
      }
      if(getNum(arr[i]) >= 0) {
        res = expand(arr, i);  // recur child inside outer while loop.
        sb.append(res.str_);
        i = res.edidx + 1;
      }
    }
    StringBuilder retsb = new StringBuilder();
    for(int k=0;k<repn;k++) {  retsb.append(sb); }
    // out of loop, i parked at post ]
    return new ExpRes(retsb.toString(), i);
  }

  public String decodeString(String s) {
    char[] arr = s.toCharArray();
    ExpRes res = null;
    int i = 0;
    StringBuilder sb = new StringBuilder();
    // Main loop consumes token stream. when seeing n[], expand it and append res;
    while(i < arr.length) {
      while(i<arr.length && isLetter(arr[i])) {
        sb.append(arr[i]); i++;
      }
      if(i<arr.length && getNum(arr[i]) >= 0) {
        res = expand(arr, i);  // expand one top [] entry. 
        sb.append(res.str_);
        i = res.edidx + 1;
      }
    }
    return sb.toString();
  }
  public String decodeString("2[abc]3[cd]ef");
}
class Solution {
  // one Expansion is the expansion of {}, union items separated by comma. 
  public class Expansion {
    int start; int end;
    List<Expansion> childexps;
    boolean combination;
    List<String> expanded;
    Expansion parent;

    public Expansion(int start) { 
      this.childexps=new ArrayList<>(); this.expanded=new ArrayList<>(); 
      this.start=start;this.combination=true;
    }
    public void add(Expansion sube) { this.childexps.add(sube);}
    public List<String> expand() {
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
    public void expand(int off, String prefix, List<String> out) {
      if(off >= this.childexps.size()) return;
      List<String> curexp = this.childexps.get(off).expand();
      if(off==this.childexps.size()-1) {
        for(String s : curexp) { out.add(prefix+s);  }
        return;
      }
      for(String s : curexp) { expand(off+1, prefix+s, out); }
    }
    public String toString() { return String.format("Expansion from %d %d subexps %d comb ? %b expanded %s", start, end, childexps.size(), combination, String.join(",", expanded));}
  }
  
  // Each token belongs to the current expansion; A child exp starts at { or letter, ends at , or }; 
  // recur {}; stk child exps, {}a{}e; merge child exp upon , or };
  public int parse(char[] arr, int off, Expansion curexp) { // curexp starts at offset. root starts at -1 to include {}{} 
    Deque<Expansion> stk = new ArrayDeque<>();  // stk stores child expansion of curexp;
    int idx = off+1;  // consume the next token of curexp.
    while(idx < arr.length) {
      if(arr[idx] == '{') {  // child expansion with recursion {};
        Expansion parenexp = new Expansion(idx);
        idx = parse(arr, idx, parenexp); // recur child exp of curexp, start at {  till the closing }.
        stk.addLast(parenexp); // child recursion done, stash to the curexp's stk until closing } to trigger consolidate.
        continue;
      } 
      // upon closing symbols, merge expansion in stk as a single expansion to parent. 
      if(arr[idx] == '}' || arr[idx] == ',') { // comma trigger consolidate. { trigger recursion.
        if(stk.size() > 1) { // {a{},{}e}
          // A new parent expansion to merge in stk exps. The new parent is a child of the current expansion that the , } belongs to.
          Expansion merge = new Expansion(stk.peekFirst().start); 
          while(stk.size()>0) { merge.add(stk.pollFirst()); }
          merge.end = idx;
          merge.expand();
          curexp.add(merge);
        } else {
          curexp.add(stk.pollFirst()); // add child expansion in stk to cur exp as , } are tokens of cur exp;
        }
        curexp.end = idx;
        if(arr[idx] == '}') { // close child {}, recursion done; return;
          return idx+1; 
        } else {   // cur expansion not ended. continue consuming inner expansion of the current expansion this comma beongs to.
          curexp.combination = false;
          idx++; continue;
        }
      }
      // child expansion with letters;
      Expansion letterexp = new Expansion(idx);
      String s = "";
      while(idx < arr.length && arr[idx] >='a' && arr[idx]<='z') { s += arr[idx++];}
      letterexp.expanded.add(s);
      stk.addLast(letterexp);
    }
    while(stk.size()>0) { curexp.add(stk.pollFirst());}
    return idx+1;
  }
  public List<String> braceExpansionII(String expression) {
    char[] arr = expression.toCharArray();
    Expansion exp = new Expansion(-1);
    // the expansion rules(comb, union) is the same for outer and inner, hence consolidate into one parse fn.
    parse(arr, -1, exp);
    return exp.expand();
  }
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// recursion example, Flattern nested lists.
// A nested list has a iter. the current nested list's iter recursively call into child's hasNext
// with recursive iter for nested child, no need stk.
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
public class NestedIterator implements Iterator<Integer> {
  int pos = 0; // cursor to the current item in the list 
  int expandedpos = -1; // not yet expanded.  
  NestedIterator childiter; // recursive
  List<NestedInteger> list;
  // Deque<NestedIterator> stk;  // no need stk for recursion.

  public NestedIterator(List<NestedInteger> nestedList) {
    pos = 0; expandedpos=-1; // not expanded on construction. Lazy expansion;
    list = nestedList;
  }
  // idempotent; will only expand once when pos is a list;
  public void expandList(int pos) {  // expand only when pos not yet expanded. 
    if(expandedpos < pos && !list.get(pos).isInteger()) {
      childiter = new NestedIterator(list.get(pos).getList()); // empty list ok;
      expandedpos = pos;  // pos is expanded if it is a nested list;
    } 
  }
  @Override
  public Integer next() {
    // hasNext already checked pos != end; as no next when pos == end ! 
    if(list.get(pos).isInteger()) {
      Integer ret = list.get(pos).getInteger();
      pos++;
      return ret;
    }
    // hasNext already expanded the list when pos is a list;
    return childiter.next();
  }

  @Override
  public boolean hasNext() {
    // loop until next available iter has next.
    while(true) {  // loop to skip empty list !
      if(pos>=list.size()) return false;
      if(list.get(pos).isInteger()) return true;  
      // when pos is a list, expand it and recur to child iterator's hasNext;
      expandList(pos);  
      if(childiter.hasNext()) return true;
      pos++; // loop to evaluate next pos hasNext();
    }
  }
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Sub Ary, SubSequence, K groups
// recur: fill the partial result exhaust arr/i for each of [i+1..n] coins, recur i+1, not pos+1 to next pos.
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Loop invoked from two recursions, instead of iter/for.
def combination(arr, prefix, remain, pos, result):
  if remain == 0:
    result.append(prefix)  # add to result only when remain
    return result
  if pos >= len(arr):
    result.append(prefix)
    return result
  incl_pos = prefix[:]  # deep copy local prefix to avoid interference among recursions.
  incl_pos.append(arr[pos])
  //recur next with incl and excl pos. Remember to guard offset < arr.length.
  recur(arr, incl_pos, remain-1, pos+1, result)
  recur(arr, prefix,   remain, pos+1, result)
  return

""" 
off is the coin index after which arr/i form the partial solution prefix into recusion.
recur with partial prefix, for loop each, as incl/excl 
"""
def comb(arr, off, remain, prefix, res):
  if remain == 0:  # stop recursion when remain = 1
    cp = prefix[:]
    res.append(cp)
    return res
  for i in range(off, len(arr)-remain+1): # each arr/i contributes partial solution recursion.
    prefix.append(arr[i])
    recur(arr, /*recur pass current i*/i+1, remain-1, prefix, res)  // <-- recur after coin i; no consider i any more.
    prefix.pop()   # deep copy prefix or pop
  return res
res=[];comb([1,2,3,4],0,2,[],res);print(res);

def comb(arr, hdpos, prefixlist, reslist):
  if hdpos == arr.length:
    reslist.append(prefixlist)
  // construct candidates with rest everybody as header, Loop each candidate and recur.
  for i in xrange(hdpos, arr.length):
    curPrefixlist = prefixlist[:]
    swap(arr, hdpos, i) // swap, so hd can still be included in the rest candidates.
    curPrefixlist.append(arr[hdpos])
    comb(arr, hdpos+1, curPrefixlist, reslist) // <-- next is hdpos+1
    swap(arr, i, hdpos)
res=[];comb([1,2,3,4],0,[],res);print res;

"""  dp[i][v] = dp[i-1][v-face_value]. means 1) Each slot i sampled only once with diff face values, different prefix at i-1/face_i. 
track each fan out recur prefix at each slot. hence dp[i][v]=dp[i-1][v-f]. can have 9 face values. hence
when each slot needs to be distinct, only append j when it is bigger
"""
def combinationSum(n, target, distinct=False):  # n slots, sum to sm.
  dp = [[None]*(max(10, target+1)) for i in range(n)]
  // each slot used only once, but can have 9 face values. populate boudnary slot 0
  for v in range(0, 10): # face value 0 means skip the slot.
    dp[0][v] = []
    dp[0][v].append([v])
  for i in range(1, n):        
    for f in range(0, 10): # face value 0, skip the slot
      for v in range(f, target+1):  # enum to sm for each slot, dp[i,sm]
        if not dp[i][v]: dp[i][v] = []
        for e in dp[i-1][v-f]:
          // if need to be distinct, append j only when it is bigger.
          if distinct and f <= e[-1]: continue
          vl = e[:]
          vl.append(f)  // <-- when f=0, slot_i is 0; s=s-0; no incr of s.
          dp[i][v].append(vl)
  return dp[n-1][target]
print(combinationSum(3,7,False))

// when dup allowed and order matters, recur each coin, not off in params. Recur is slow, use dp !
public void combRecur(int[] arr, int prefix, int target, int[] res) {
  // on each recursion, all coins are considered to next recur(dup allowed and order matters).
  for(int i=0;i<arr.length;++i) {
    if(prefix+arr[i] == target) res[0]++;
    else if(prefix+arr[i] < target) {
      combRecur(arr, prefix+arr[i], target, res);
    }
  }
}
public int combinationSum4(int[] nums, int target){
  int[] res = new int[1];
  combRecur(nums, 0, target, res);
  return res[0];
}

""" incl offset into path, recur. skip ith of this path, next 
recur fn params must be subproblem with offset and partial result.
# [a, ..] [b, ...], [c, ...]
# [a [ab [abc]] [ac]] , [b [bc]] , [c]
"""
def powerset(arr, offset, path, result):
  ''' subproblem with partial result, incl all interim results. '''
  result.append(path)
  processResult(path)   # optional process result for the path.
  // within cur subproblem, for loop to incl/excl for next level.
  for i in xrange(offset, len(arr)):
    # compare i to i-1 !!! not i to offset, as list is sorted.
    if i > offset and arr[i-1] == arr[i]:  continue # not arr[offset]
    l = path[:]
    l.append(arr[i])
    powerset(arr, i+1, l, result)   # recur with i+1, not offset+1
path=[];result=[];powerset([1,2,2], 0, path, result);print result;

public static void powerset(List<Integer> arr, int offset, Set<Set<Integer>> prefix, Set<Set<Integer>> result) {
  if (offset == arr.size()) { result.addAll(prefix); return; }
  Set<Set<Integer>> cpPrefix = new HashSet<Set<Integer>>();
  for (Set<Integer> e : prefix) {
    cpPrefix.add(new HashSet<>(e));
      e.add(arr.get(offset));  // local copy each pref result so form new set by adding cur to it.
    cpPrefix.add(e);
  }
  HashSet<Integer> me = new HashSet<>();
  me.add(cur);
  cpPrefix.add(me);
  powerset(arr, offset+1, cpPrefix, result);
}

""" m faces, n dices, total num of ways to get value target.
bottom up, 1 dice, 2 dices,..., 3 dices, n dices. Last dice take face value 1..m.
[[v1,v2,...], [v1,v2,...], d2, d3, ...], each v1 column is cnt of face 1..m
tab[d,v] += tab[d-1,v-[1..m]]. Outer loop is enum dices, so it's like incl/excl.
"""
def dice(m,n,target):
  dp[n, v] = [[0]*n for i in xrange(target+1)]
  // init bottom, one coin only
  for f in xrange(m):  dp[1][f] = 1
  for i in xrange(1, n):      # bottom up coins, 1 coin, 2 coin
    for v in xrange(target):  # bottom up each value to target
      for f in xrange(m,v):   # iterate each face value within each tab[dice, V]
        dp[i][v] += dp[i-1][v-f]  or dp[i][v] += dp[i][v-f]
  return dp[n,x]

""" from L <- R, find first a[i] < a[i+1] = pivot.
then swap with first rite larger, and reverse right part.
"""
def nextPermutation(txt):
  for i in xrange(len(txt)-2, -1, -1):
    if txt[i] < txt[i+1]:
      break
  if i < 0:
    return ''.join(reversed(txt))
  pivot = txt[i]
  for j in xrange(len(txt)-1, i, -1):
    if txt[j] > pivot:
      break
  txt[i],txt[j] = txt[j],txt[i]
  l,r = i+1, len(txt)-1
  while l < r:
    txt[l],txt[r] = txt[r],txt[l]
    l += 1
    r -= 1
  return txt

""" iter each head, rank += index(hd,a[i:])*(n-i-1)! 
the index calcu is based off smaller substr by removing current head
"""
from math import factorial
def permRank(arr):
  def index(c, arr):
    return sorted(arr).index(c)
  sz,rank = len(arr),1
  for i in xrange(sz):
    hd = arr[i]
    rank += index(hd, arr[i:]) * factorial(sz-i-1)
  return rank
print permRank("bacefd")


""" find the index of the char of each next subsize, 
rank/factorial(n) is the index of the char in the sorted arr """
from math import factorial
def nthPerm(arr, n):
  out = []
  sz = len(arr)
  cnt = factorial(sz)
  for i in xrange(sz):
    cnt = cnt/(sz-i)   # cnt = (n-1)! = n!/n
    idx = n/cnt
    n -= idx*cnt
    out.append(arr[idx])
    arr = arr[:idx] + arr[idx+1:]   # del arr[idx] after pick
  return "".join(out)
assert(nthPerm("1234", 15), "3241")

""" iter each off i from 0 <- n. for width w, w! perms.
At each i, there are (n-i)! perms. If there are k eles smaller than cur,
means cur rank += those k packs that is smaller than cur. so rank += k*(n-i)!
with dup, tot is (n-i)!/A!*B!, and we need sum up all k copies of dup ele to rank
"""
from collections import defaultdict
from math import factorial
def permRankDup(arr):
  ''' tot and divide dup! in the counter '''
  def permTotal(n, counter):
    tot = factorial(n)
    for k,v in counter.items():
      tot /= factorial(v)
    return tot
  rank = 1
  counter = defaultdict(int)
  for i in range(len(arr)-1,-1,-1):
    e = arr[i]
    counter[e] += 1
    // tot perms i..n. arr[i] rank
    tot = permTotal(len(arr)-i-1, counter)
    for k,v in counter.items():
      if e > k:  # arr[i] > k at rite, sum up v dups of k.
        rank += tot*v
  return rank
assert permRankDup('3241') == 16
print permRankDup('baa')
print permRankDup('BOOKKEEPER') == 10743
assert permRankDup("BCBAC") == 15

// Generate all possible sorted arrays from alternate eles of two sorted arrays
def alterCombRecur(a,b,ai,bi,froma,path,out):
  if froma:
    v = a[ai]
    found,bs = bisect(b, v)
    path.append(v)
    for j in xrange(bs,len(b)):
      l = path[:]  # deep cpoy path before each recursion
      alterCombRecur(a,b,ai,j,False,l,out)
  else:
    path.append(b[bi])
    out.append(path)
    found,sa = bisect(a,b[bi])
    for i in xrange(sa, len(a)):
      l = path[:] # deep copy path before each recursion
      alterCombRecur(a,b,i,bi,True,l,out)
def alterComb(a,b):
  out = []
  for ai in xrange(len(a)):  # a new recursion seq
    av = a[ai]
    found,bi = bisect(b, av)
    path = []  # empty new path for each recursion
    if bi < len(b):
      alterCombRecur(a,b,ai,bi,True,path,out)
  return out
print alterComb([10, 15, 25], [1,5,20,30])

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

""" tab[i,v] = tot non descreasing at ith digit, end with value v
dp[i,v]=sum(dp[i-1, v-k]) for k in 1..10;
Total count = ∑ count(n-1, d) where d varies from 0 to n-1
"""
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

"""
count tot num of N digit with sum of even digits 1 more than sum of odds
n=2, [10,21,32,43...] n=3,[100,111,122,...980]
track odd[i,v] and even[i,v] use 2 tables
"""
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

""" odd[i,s] is up to ith odd digit, total sum of 1,3,ith digit sum to s.
for digit xyzw, odd digits, y,w; sum to 4 has[1,3;2,2;3,1], even digit, sum to 5 is
[1,4;2,3;3,2;4,1], so tot 3*4=12, [1143, 2133, ...]
so diff by one is odd[i][s]*even[i][s+1]
"""
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
public int coinChangeMin(int[] coins, int amt) {
  Arrays.sort(coins);
  int[] dp = new int[amt+1];
  for(int i=1;i<=amt;++i) { dp[i] = Integer.MAX_VALUE;} // track Min, not total. sentinel.
  dp[0] = 0;  // <-- dp[v-c=0] + 1;
  {
    for(int v=1;v<=amt;++v) {
      for(int i=0;i<coins.length;++i) {
          if(v >= coins[i] && dp[v-coins[i]] != Integer.MAX_VALUE) {
              dp[v] = Math.min(dp[v], dp[v-coins[i]] + 1);
          }
      }
    }
  }
  {
    for(int i=0;i<coins.length;++i) {
      for(int v=coins[i];v<=amt;++v) {
          if(dp[v-coins[i]] != Integer.MAX_VALUE) {
              dp[v] = Math.min(dp[v], dp[v-coins[i]]+1);
          }
      }
    }
  }
  return dp[amt] == Integer.MAX_VALUE ? -1 : dp[amt];
}

// outer loop values, inner loop coins, [1,2] != [2,1];
// outer loop coins, inner loop value, [2,1] not possible. order does not matter.
static int combsumTotWays(int[] arr, int target) {
  // dp[i][v] = dp[i-1][v-slot_face(1..6)]; each slot i can only once with 6 face values; max N slots.
  int[] dp = new int[target+1];   // each option can have k dups that generate different prefix values.
  dp[0] = 1;   // when slot value = target, count single item 1.
  { // outer loop exhaust each coin ordered one by one, order not matter, [2,1] not possible, [1,2]
    for (int i=0;i<arr.length;i++) { // for each coin a/i, loop all values, hence k copies of a coin is considered.
      for (int v=arr[i]; v<=target; v++) {
        // no need loop k. K dups is carried over built up from ((v-a/i)-a/i)-a/i
        dp[v] += dp[v-arr[i]]; // can take multiple a/i, += sum up. = will over-write.
      }
    }
  }
  { // outer loop values, at each v, all coins are considered. order matters. [1,2] != [1,2]
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

public void combSumRecur(int[] arr, int coinidx, int remain, int[] prefix, List<List<Integer>> gret) {
  if(remain == 0) {
      List<Integer> l = new ArrayList<>();
      for(int i=0;i<prefix.length;++i) {
          for(int j=0;j<prefix[i];++j) {
              l.add(arr[i]);
          }
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
public List<List<Integer>> combinationSum(int[] arr, int target) {
  List<List<Integer>> ret = new ArrayList<>();
  Arrays.sort(arr);
  int[] prefix = new int[arr.length];
  search(arr, 0, target, prefix, ret);
  return ret;
}

// from root, fan out; multiways with diff prefix to pos, for each suffix, append head to prefix, recur suffix [i+1..];
// shall use dp, dp is for aggregation.
static List<List<Integer>> combinSumTotalWays(int[] arr, int pos, int remain, List<Integer> prefix, List<List<Integer>> res) {
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

// Loop coin first, no perm dups. [1,2], not possible [2,1], hence no dup.
public List<List<Integer>> combinationSum(int[] arr, int target) {
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


""" knapsack, track item list at each item and value iteration. 
when dup allowed, recur same row(item), tab[item][v-vi], else tab[item-1][v-vi].
so when dup allowed, the problem reduced to coin change, 1-dim array is good.
"""
def subsetSum(arr, t, dupAllowed=False):
  dp = [[None]*(t+1) for i in xrange(len(arr))]
  for i in xrange(len(arr)):
    for s in xrange(1, t+1): # iterate to target sum at each arr[i]
      if i == 0:   # boundary, init
        dp[i][s] = []
      else:        # not incl arr[i], some subset 0..pre has value s
        dp[i][s] = dp[i-1][s][:]  # copy prev val s subset, excl cur item.
      if s == arr[i]:
        dp[i][s].append([s])
      if s > arr[i]:   # incl arr[i]
        if len(dp[i][s-arr[i]]) > 0:  # there
          if dupAllowed: # incl arr[i] multiple times
            for e in dp[i][s-arr[i]]:
              p = e[:]
              p.append(arr[i])
              dp[i][s].append(p)
          elif i > 0:
            for e in dp[i-1][s-arr[i]]:
              p = e[:]
              p.append(arr[i])
              dp[i][s].append(p)
  return dp[len(arr)-1][t]
print subsetSum([2,3,6,7],9)
print subsetSum([2,3,6,7],7, True)
print subsetSum([2,3,6],8)
print subsetSum([2,3,6],8, True)

// Each item is a row. Each val is a col. Iter item one by one and enum all values.
// Regardless dup allowed or not, dp/i/k means a[0..i] has subset sum to k.
// dp/i/k = dp/i-k/k-iv || dp/i-k/k
// init dp boundary dp[0] = 1 when deduction to 0, i.e. empty subset sum to 0.
static int subsetSum(int[] arr, int sum) {
  int[] curRow = new int[sum/2+1], preRow = new int[sum/2+1];
  for(int i=0;i<sum/2+1;++i) { curRow[i] = 0; preRow[i] = 0;}
  for(int i=0;i<arr.length;++i) { // always loop coins first to avoid dup of [1,2], [2,1]
      if(arr[i] > sum/2) return false;
      // preRow = curRow;  // stage cur before modify
      System.arraycopy(curRow, 0, preRow, 0, sum/2+1);
      
      curRow[0] = 1; curRow[cur] = 1; // exclude cur, include cur
      for(int k=1;k<=sum/2;++k) {
          curRow[k] = Math.max(curRow[k], preRow[k]);  // dp/i,k = dp/i-1,k
          if(k >= arr[i]) {
              curRow[k] = Math.max(curRow[k], preRow[k-arr[i]]);
          }
      }
      //System.out.println("i: " + i + " a/i " + cur + " row: " + Arrays.toString(curRow));
  }
  return curRow[sum/2] == 1;
}
subsetSum(new int[]{2,3,6,7}, 9);

//  A good example to see difference between recur dfs memorize VS. dp
//  target sum, with two operations. add/sub, recur with each at header.
//  recur i+1 with add/sub
int targetSumRecurDFS(int[] nums, int offset, int sum, int S, int[][] memo) {
    if (offset == nums.length) { return sum == S ? 1 : 0;   } 
    if (memo[offset][sum + 1000] != Integer.MIN_VALUE) {    // cache hits
      return memo[offset][sum + 1000];
    }
    int add      = targetSumRecurDFS(nums, offset+1, sum + nums[offset], S, memo);
    int subtract = targetSumRecurDFS(nums, offset+1, sum - nums[offset], S, memo);
    memo[offset][sum + 1000] = add + subtract;
    return memo[offset][sum + 1000];
}
// dp[i][s] += dp[i-1][s +/- a[i]], two dp rows. for each outer loop, always allocate new one.
int findTargetSumWays(int[] nums, int S) {
    int[] dp = new int[2001];
    dp[nums[0] + 1000] = 1;
    dp[-nums[0] + 1000] += 1;
    for (int i = 1; i < nums.length; i++) {   // iterate each a.i
        int[] nextdp = new int[2001];         // allocate nextdp
        for (int val = -1000; val <= 1000; val++) {  // at each slot, enum all vals.
            if (dp[val + 1000] > 0) {
                nextdp[val + nums[i] + 1000] += dp[val + 1000];
                nextdp[val - nums[i] + 1000] += dp[val + 1000];
            }
        }
        dp = nextdp;
    }
    return S > 1000 ? 0 : dp[S + 1000];
}

// # https://leetcode.com/problems/target-sum/discuss/97334/Java-(15-ms)-C++-(3-ms)-O(ns)-iterative-DP-solution-using-subset-sum-with-explanation
// # find positive subset P, and negative subset N. prefixSum(P) - prefixSum(N) = target. P + N = all.
// # prefixSum(P) + prefixSum(N) + prefixSum(P) - prefixSum(N) = target + prefixSum(P) + prefixSum(N)
// # 2 * prefixSum(P) = target + prefixSum(all)
// # convert to find combinationSum(arr) = (tot + target)/2
int findTargetSumWays(int[] nums, int target) {
    int sum = 0;
    for (int n : nums) sum += n;
    return sum < target || (target + sum) % 2 > 0 ? 0 : combinationSum(nums, (target + sum) >> 1); 
}

// # Partition to K groups, Equal Sum Subsets, Permutation search Recur.
// # Permutation recur with k groups, For each ele, try it for each group, and recur, prune.
boolean kGroups(int[] arr, int k) {
  int sum = Arrays.stream(nums).sum();
  if (sum % k > 0) return false;
  int target = sum / k;
  Arrays.sort(nums, (a, b) -> {a-b});
  int row = nums.length - 1;
  if (nums[row] > target) return false;
  while (row >= 0 && nums[row] == target) {
    row--; k--;
  }
  return permutationDFS(new int[k], row, nums, target);   // start from global init root state
}

boolean permutationDFS(int arr, int[] groups, int row, int target) {
  if (row < 0) return true;
  int v = arr[row];
  // permutation on each group
  for (int grp=0; grp<k; grp++) {
    if (groups[grp] + v < target) {
      groups[grp] += v;   // dfs recur with new state, try all until success. or
      if (recur(arr, groups, row-1, target)) return true;  // try each new state
      groups[grp] -= v;   // restore before another try.
    }
    if (groups[i] == 0) break;
  }
  return false;
}

""" Maximum Sum of 3 Non-Overlapping Subarrays,
https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/discuss/108231/C++Java-DP-with-explanation-O(n)
enum all possible middle subary from [i,i+k] then we cal left and rite.
dp[i] is middle subary is i..i+k, max left subary 0..i, max rite subary i+k..n.
just enum all i as middle ary, max left and max rite is easy to find.
"""
static int maxSum3Subary(int[] arr, int subarylen) {
    int sz = arr.length;
    int[] dp = new int[sz];
    int[] psum = new int[sz];
    int[] leftMax = new int[sz];
    int[] riteMax = new int[sz];
    int leftmaxsofar = 0, ritemaxsofar = 0, allmax = 0;
    psum[0] = arr[0];
    for (int i=1;i<sz;i++) {  psum[i] += psum[i-1] + arr[i]; }

    leftmaxsofar = psum[subarylen-1];
    leftMax[subarylen-1] = leftmaxsofar;
    for (int i=subarylen; i<sz; i++) {
        int thisdiff = psum[i] - psum[i-subarylen];
        leftmaxsofar = Math.max(leftmaxsofar, thisdiff);
        leftMax[i] = leftmaxsofar;
    }

    // when mov from rite to left i, to include i, need i-1.
    ritemaxsofar = psum[sz-1] - psum[sz-1-subarylen];
    riteMax[sz-subarylen] = ritemaxsofar;
    for (int i=sz-1-subarylen; i>=subarylen; i--) {
        int thisdiff = psum[i+subarylen-1] - psum[i-1];  // to include i, need i-1
        ritemaxsofar = Math.max(ritemaxsofar, thisdiff);
        riteMax[i] = ritemaxsofar;
    }

    for (int i=subarylen; i<sz-subarylen; i++) {
        # to include i, need i-1
        dp[i] = Math.max(dp[i], leftMax[i-1] + (psum[i+subarylen-1] - psum[i-1]) + riteMax[i+subarylen]);
        allmax = Math.max(allmax, dp[i]);
    }
    System.out.println("allmax " + allmax + " leftmax " + Arrays.toString(leftMax));
    return allmax;
}
maxSum3Subary(new int[]{1,2,1,2,6,7,5,1}, 2);   // 23, [1, 2], [2, 6], [7, 5], no greedy [6,7]

// split ary into k grps, min max sum among those k groups.
// enum num grps from 1 to k, dp[i, k] = min(max(dp[j,k-1], arr[j..i]))
static int splitAryKgroupsMinMaxSum(int[] arr, int grps) {
  int sz = arr.length;
  int[] prefixsum = new int[sz];
  int[] dp = new int[sz];      # dp[i] store the max sum up to i, groups does not matter.
  // num of grps from 1 to grps
  for (int c=1; c<grps; c++) {
    for (int i=0; i<sz; i++) {
      dp[i] = Integer.MAX_VALUE;
      for (int j=0; j<i; j++) {
        int mx = Math.max(dp[j], prefixsum[i]-prefixsum[j]);
        if (mx < dp[i]) { dp[i] = mx; } 
        else { break; }
      }
    }
  }
  return dp[sz-1];
}

// https://leetcode.com/problems/largest-sum-of-averages/discuss/122739/C++JavaPython-Easy-Understood-Solution-with-Explanation
// divide into k conti subary, find the cut that yields max sum of avg of each subary
// dp[i][k] = max(dp[j][k-1] + avg(j..k))
static int maxAvgSumKgroups(int[] arr, int k) {
    int sz = arr.length;
    int[] psum = new int[sz];
    psum[0] = arr[0];
    for (int i=1;i<sz;i++) { psum[i] = psum[i-1] + arr[i]; }

    int[][] dp = new int[sz][k+1];
    // boundary case, just 1 grp
    for (int i=0;i<sz;i++) { dp[i][1] = psum[i]/(i+1); }
    // expand to multiple grps
    for (int grp=2;grp<=k;grp++) {
        int[] nxtDp = new int[sz];
        for (int i=0; i<sz; i++) {
            for (int j=0; j<i; j++) {
                dp[i][grp] = Math.max(dp[i][grp], dp[j][grp-1] + (psum[i]-psum[j])/(i-j));
            }
        }
    }
    System.out.println("max k subary avg is " + dp[sz-1][k]);
    return dp[sz-1][k];
}
maxAvgSumKGroups(new int[]{9,1,2,3,9}, 3);

""" 
https://leetcode.com/problems/count-of-range-sum/discuss/77990/Share-my-solution
All pairs sum[l:r] within range [mn,mx]. mergesort
as there are negative numbers, prefix sum not mono increase,
master therom: T(0, n-1) = T(0, m) + T(m+1, n-1) + C
if the merge part is Constant, it's less than n^2.
To make merge constant, sort the output.

count range sum. i,j so low < prefixsum(j)-prefixsum(i) < upper.
merge sort, divide to left, rite, To make merge constant, sort the out result in each merge.
"""
static int pairsCnt = 0;
static int[] allPairsRangeSumBetweenLowUpper(int psum[], int l, int r, int low, int upper) {
    int sz = psum.length;
    int[] out = new int[r-l+1];
    if (l == r) {
        out[0] = psum[l];
        if (psum[l] >= low && psum[l] <= upper) {
            pairsCnt += 1;
        }
        return out;
    }
    // divide, recur, and merge, when merge, make it sorted.
    int m = (l+r)/2;
    int[] left = allPairsRangeSumBetweenLowUpper(psum, l, m, low, upper);
    int[] rite = allPairsRangeSumBetweenLowUpper(psum, m+1, r, low, upper);
    // now merge
    int outidx = 0, li = 0, ri = 0, lsz = left.length, rsz = rite.length;
    while (li < lsz && ri < rsz) {
        int diff = rite[ri] - left[li];
        if (diff >= low &&  diff <= upper) { pairsCnt += 1; }
        if (left[li] <= rite[ri]) { out[outidx++] = left[li]; li += 1; } 
        else {                      out[outidx++] = rite[ri]; ri += 1; }
    }
    while (li < lsz) { out[outidx++] = left[li++];  }
    while (ri < rsz) { out[outidx++] = rite[ri++];  }
    System.out.println("range sum out " + Arrays.toString(out));
    return out;
}

int[] arr = new int[]{-2,5,-1};
int[] psum = new int[arr.length];
psum[0] = arr[0];
for (int i=1;i<arr.length;i++) { psum[i] += psum[i-1]+arr[i];}
allPairsRangeSumBetweenLowUpper(psum, 0, 2, -2, 2);
System.out.println("count of range sum is " + pairsCnt);


// count total pairs. For each growing i, lseek j=i+1, 
static int countPairsDistLessThan(int[] arr, int d) {
  int totPairs, j;
  for (int i=0; i < arr.length; i++) {
    j = i+1;
    while (j < arr.length && arr[j] - arr[i] <= d) j += 1; 
    totPairs += (j-i-1);
  }
  return totPairs;
}

// partition by median pivot distnace value (1/2 of max diff), Loop left/rite find cnt pairs dist less than it. 
// If cnt < k, mid is big, pull down hi.
// at each distance value, l=0; loop left / rite to find k th.
int kthSmallestPairDistance(int[] nums, int k) {
    Arrays.sort(nums);  // nums is sorted.
    int lo = 0, hi = nums[nums.length - 1] - nums[0];   // largest dist
    while (lo < hi) {
        int mi = (lo + hi) / 2;
        int count = 0, left = 0;   // reset count and left on each pivot.
        // enum each rite, lseek left, add up counts that great than mid.
        for (int right = 0; right < nums.length; ++right) {
            while (nums[right] - nums[left] > mi) left++;   // lseek left
            count += right - left;
        }
        // count = number of pairs with distance <= mi
        if (count >= k) hi = mi;
        else lo = mi + 1;
    }
    return lo;
}

# median of median of divide to n groups with each group has 5 items,
def MedianMedian(A, beg, end, k):
  def partition(A, l, r, pivot):
    widx = l
    for i in xrange(l, r+1):
      if A.i < pivot:
        A.widx, A.i = A.i, A.widx  // swap(widx, i)
        widx += 1
    // i will park at last item that is < pivot
    return widx-1   # left shift as we always right shift after swap.

  sz = end - beg + 1  # beg, end is idx
  if not 1<= k <= sz: return None     # out of range, None.
  # ret the median for this group of 5 items
  if end-beg <= 5:   return sorted(A[beg:end])[k-1]  # sort and return k-1th item
  # divide into groups of 5, ngroups, recur median of each group, pivot = median of median
  ngroups = sz/5
  medians = []
  for i in xranges(ngroups):
    gm = MedianMedian(A, beg+5*i, beg+5*(i+1)-1, 3)
    medians.append(gm)
  pivot = MedianMedian(medians, 0, len(medians)-1, len(medians)/2+1)
  pivotidx = partition(A, beg, end, pivot)  # scan from begining to find pivot idx.
  rank = pivotidx - beg + 1
  if k <= rank:  return MedianMedian(A, beg, pivotidx, k)
  else:          return MedianMedian(A, pivotidx+1, end, k-rank)


## Median of a data stream
# split data stream to 2 heap, max heap of 1..mid, min heap of mid..n.
# when add, add to min heap or max heap.
class MedianOfStream {
    // max queue is always larger or equal to min queue
    PriorityQueue<Integer> min = new PriorityQueue();
    PriorityQueue<Integer> max = new PriorityQueue(1000, Collections.reverseOrder());
    // Adds a number into the data structure.
    public void addNum(int num) {
        max.offer(num);
        min.offer(max.poll());
        if (max.size() < min.size()){
            max.offer(min.poll());
        }
    }
    // Returns the median of current data stream
    public double findMedian() {
        if (max.size() == min.size()) return (max.peek() + min.peek()) /  2.0;
        else return max.peek();
    }
};

public ListNode mergeKLists(ListNode[] lists) {
  ListNode head=null, tail=null;
  PriorityQueue<int[]> q = new PriorityQueue<>((a, b) -> a[0] - b[0]);
  for(int i=0;i<lists.length;++i) {
    if(lists[i] != null) { q.offer(new int[]{lists[i].val, i}); }
  }
  while(q.size() > 0) {
    int[] qhd = q.poll();
    int idx = qhd[1];
    ListNode cur = lists[idx];
    if(cur.next != null) {
      lists[idx] = cur.next;
      q.offer(new int[]{lists[idx].val, idx});
    }
    if(head==null) { head=cur; head.next = null; continue;}
    if(tail==null) { tail = cur; head.next = tail;}
    else { tail.next = cur; tail = cur; }
  }
  return head;       
}

// find the # of increasing subseq of size k, just like powerset, recur i+1 with the new path '''
def numOfIncrSeq(l, start, prefix, k):
  if k == 0:  print prefix  return
  if start + k > len(l):    return
  if len(prefix) > 0:       last = prefix[len(prefix)-1]
  else:                     last = -sys.maxint
  for i in xrange(start, len(l), 1):
    if l[i] > last:
      tmpsol = list(prefix)
      tmpsol.append(l[i])
      numOfIncrSeq(l, i+1, tmpsol, k-1)     # reduce k on each recur.


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

""" tab[n] = 1 + tab[n - closest_2_pow] """
def countBits(n):
  tab = [0]*n
  twosquare = 1
  for i in xrange(3,n):
    if i & i-1 == 0:  twosquare = i; tab[i] = 1
    else:             tab[i] = tab[i-twosquare]
  return tab

""" sub string by concat all same length keys 
continue mov r when each r+k is in dict. skip exceeding dups.
mov l to r and reset trackings when break out when r+k not in dict.
"""
from collections import defaultdict
def concatKeys(txt, keys):
  keydict = defaultdict(int)  # key count
  out = []
  for k in keys:
    keydict[k] += 1
  k,cnt = len(keys[0]),0
  l,r = 0,0
  curmap = defaultdict(int)
  while l < len(txt):
    r = l
    while r < len(txt):
      wd = txt[r:r+k]
      if wd in keydict:
        curmap[wd] += 1
        if curmap[wd] == keydict[wd]:
          cnt += 1
          if cnt == len(keys):
            while curmap[arr[l:l+3]] > keydict[arr[l:l+3]]:
              l += 3
            out.append(l)
        elif curmap[wd] > keydict[wd]:
          l = keydict[wd]
          continue
        else:
          continue
      else:
        break
    # after break, which r+k not dict word, reset, mov left edge
    l = r+k
    cnt = 0
    curmap.clear()
  return out
print concatKeys("barfoothebarfoobarfooman", ["foo", "bar", "foo"])
print concatKeys("barfoothebarfoobarfooman", ["foo", "bar", "bar"])

""" use expected[] and found[] to track the stats on each traverse """
from collections import defaultdict
def concatKeys(s, arr):
  l,r,k = 0,0,len(arr[0])
  out = []
  expect = defaultdict(int)
  count = defaultdict(int)
  seen = 0
  for w in arr:
    expect[w] += 1
  # keep mov rite edge, inner adjust left edge when skip dups
  while r < len(s):
    wd = s[r:r+k]
    if not wd in expect:  # concat must be conti, no intervene.
      r += 1
      l = r
      seen = 0
      count.clear()
      continue
    // calculate rite edge first, as r starts from 0
    count[wd] += 1
    seen += 1
    # appear more caused by dups in left, slide left
    while count[wd] > expect[wd]:
      prewd = s[l:l+k]
      seen -= 1
      count[prewd] -= 1
      l += k    # slide left edge
    if seen == len(arr):
      out.append([l,r])
    # slide rite edge
    r += k
  return out

// max gap of an between pairs when they are sorted.
// use bucket to partition number into bucket,
// Let gap = ceiling[(max - min ) / (N - 1)]. We divide all numbers in the array into n-1 buckets
// Each bucket store [min, max], scan the buckets for the max gap
// https://leetcode.com/problems/maximum-gap/discuss/50643/bucket-sort-JAVA-solution-with-explanation-O(N)-time-and-space

# Patch Ary so all number from 1..n can be formed from some combination sum.
# bottom up from 1..n, if n is missing, add it.
static int minPatch(int[] arr, int k) {
    int missing = 1;    # continue pushing up missing.
    int i = 0;
    int added = 0;
    while (missing <= k) {      // always update missing
      if (i < arr.length && arr[i] <= missing) {  missing += arr[i]; }
      else {                                      missing += missing; added += 1; }
    }
    return added;
}

# LoopInvar: for each loop var bar x, track its first smaller left, first smaller right,
# area(x) = first_right_min(x) - first_left_min(x) * X, largestRectangle([6, 2, 5, 4, 5, 1, 6])

# Every row in the matrix can be viewed as the ground. height is the count
# of consecutive 1s from the ground. Iterate each ground.
# htMatrix[row] = [2,4,3,0,...] the ht of each col if row is the ground
def LargestRectangle(matrix):
  # iterate each row, build matrix store height for each col. 
  rows, cols = len(matrix), len(matrix[0])
  htMatrix[0][c] = 1 if matrix[0][c] == 1 else 0 for c in xrange(cols)
  for r in xrange(1, rows):
    for c in xrange(cols):
      if matrix[r][c] == 1:  
        htMatrix[r][c] = htMatrix[r-1][c] + 1
  for rowHt in htMatrix:
    mx = max(mx, maxHistogram(rowHt))
  return mx

""" A[r,c] = (riteEdge[r,c] - leftEdge[r,c]) * ht. 
use max left when substract left edge.
left_edge[r,c] = max(left_edge[r-1,c], cur_left_edge), max of prev row's left edge.
rite_edge[r,c] = min(rite_edge[r-1,c], cur_rt_edge), min of prev row's rite edge
as we use max left edge, cur_left_edge = c+1, we use min rite edge, cur_rite_edge = c
"""
def maxRectangle(matrix):
  for r in xrange(len(matrix)):
    ht,left_edge,rite_edge = [0]*n, [0]*n, [n]*n
    cur_left_edge = 0, cur_rite_edge = n
    for c in xrange(len(matrix[0])):
      if matrix[r][c] == 1: ht[c] += 1
      else:                 htc[c] = 0
      if matrix[r][c] == 1: left_edge[c] = max(left_edge[c], cur_left_edge) # max left when substract
      else:                 left_edge[c] = 0      # reset when matrix 0
                            cur_left_edge = c+1   # mov left edge -->
      if matrix[r][c] == 1: rite_edge[c] = min(rite_edge[c], cur_rite_edge) # min rite to narrow
      else:                 rite_edge[c] = n    # defaut is n as we using min rite
                            cur_rite_edge = c   # shrink rite edge
    for c in xrange(n):
      maxA = max(maxA, (rite_edge[c] - left_edge[c])*ht[c])
  return maxA


""" largest Rect with swap of any pairs of cols 
first, build height col for each row. Then sort it.
"""
def largestRectWithSwapCol(mat):
  pass

"""
convert array into sorted array with min cost. reduce a bin by k with cost k, until 0 to delete it.
cost[i], at ary index i, the cost of sorted subary[0..i] with l[i] is the last.
l = [5,4,7,3] => 5-1, rm 3. cost[i+1], dp each 1<k<i, 
l[i+1] > l[k], cost[i+1] = cost[k] + del l[k] .. l[i+1]
l[i+1] < l[k], find the first l[j] < l[i+1], and cost[i+1] = min of make each l[j] to l[i+1], or del l[k]
"""
def convert_to_sort(l):
  dp = [sys.maxint for i in xrange(len(l))]
  # solve the cost of each subary [0..i]
  for i in xrange(len(l)):
    cur = l[i]
    for j in xrange(i-1, 0, -1):
        costi = 0
        if l[j] < cur:
            for k in xrange(j,i): delcost += l[k]
            costi = cost[i] + delcost
        else:
            # find first l[j] <= l[i]
            costi += delcost
            while l[j] > cur and j > 0:
                costi += l[j] - cur
                j -= 1
            costi += dp[j+1]

        # after each round of j, up-beat
        dp[i] = costi if dp[i] > costi else costi

  # as dp[i] is the cost of with l[i] as the last item.
  # global can be the one with last item bing deleted.
  gmin = sys.maxint
  for i in xrange(len(l)):
      costi = dp[i] + delcost[n-1]-delcost[i]
      gmin = min(costi, gmin)
  return gmin


""" fence partition, m[n, k] = min(max(m[i-1,k-1], sum[i..n])) for i in [0..n]
    boundary: m[1,k] = A.1, one item, k divider; m[n,1] = sum(A)
    subset sum, dp[i, v] = dp[i-1,v] + dp[i-1, v-A.i]
"""
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


// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Merge Interval
// Interval TreeMap Array and Interval Tree, segment
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
class Solution {
  public int[] leftrite(int[][] arr, int start, int end) {
    // use start, search arr[l].end, where arr[l].end < start; no eq; [0..l] can be discarded safely.
    // use end, search arr[r].start where arr[r].start > end, no eq; [r..n] can be discarded safely.
    int[] lr = new int[2];
    int l=0,r=arr.length-1;
    while(l<=r) { // update l even a single entry.
      int m = l+(r-l)/2;
      if(arr[m][1] < start) { l = m+1;} // not mov l when arr[l].end = start; so [l-1].end strictly < start;
      else { r=m-1;}
    }
    lr[0] = l-1; // [0..l-1], can be discarded safely.
    
    l=0; r=arr.length-1;
    while(l<=r) { // update l even a single entry.
      int m =l+(r-l)/2;
      if(arr[m][0] <= end) { l = m+1;} // mov l even end=arr[l].start so arr[l].start strictly> end;
      else { r=m-1;}
    }
    lr[1] = l; // [l..n] arr[l].start > end; include l;
    return lr;
  }
  public int[][] insert(int[][] arr, int[] newiv) {
    List<int[]> res = new ArrayList<>();
    int[] lr = leftrite(arr, newiv[0], newiv[1]);
  
    for(int i=0;i<=lr[0];i++) { res.add(arr[i]);}
  
    int mergedl = newiv[0], mergedr = newiv[1];
    if(lr[0]+1 < arr.length) { mergedl = Math.min(mergedl, arr[lr[0]+1][0]); }
    if(lr[1]-1 >= 0) { mergedr = Math.max(mergedr, arr[lr[1]-1][1]);}
    res.add(new int[]{mergedl, mergedr});
  
    for(int i=lr[1];i<arr.length;i++) {res.add(arr[i]);}
    return res.stream().toArray(int[][]::new);
  }
}

// cnt+1 when interval starts, -1 when interval ends. scan each point, ongoing += v.
class MyCalendar {
  TreeMap<Integer, Integer> tmlineMap = new TreeMap<>();
  int book(int s, int e) {
      tmlineMap.put(s, tmlineMap.getOrDefault(s, 0) + 1); // 1 new event will be starting at [s]
      tmlineMap.put(e, tmlineMap.getOrDefault(e, 0) - 1); // 1 new event will be ending at [e];
      int ongoing = 0, k = 0;               // ongoing counter track loop status
      for (int v : tmlineMap.values()) {    // iterate all intervals.
          ongoing += v;                     // value is negative when event ends
          k = Math.max(k, ongoing);         // when v is -1, reduce ongoing length.
      }
      return k;
  }
}

// # longest length of interval chain, sort interval by END time. interval scheduling.
// # https://leetcode.com/problems/maximum-length-of-pair-chain/discuss/105631/midBthon-Straightforward-with-Explanation-(N-log-N)
// #  dp[i] = the lowest END of the chain of length i
static int longestIntervalChain(int[][] arr) {
  Arrays.sort(arr, (a, b) -> a[1] - b[1]);   // sort by end time.
  List<Integer> dp = new ArrayList<>();
  for (int i=0; i<arr.length; i++) {
    int st,ed = arr[i][0][1];
    int idx = Arrays.binarySearch(dp, st);  // find start time among end time ary.
    if (idx == dp.length) {       dp.add(ed); }     // append to the end
    else if (ed < dp[idx]) {      dp.set(idx,ed); }  // update ix with current smaller value
  }
  return dp.size() - 1;
}
// The other way is stk, sort by end time, and stk thru it.
if stk.top().end > curIntv.end: stk.top().end = curInterv.end;

// sort by fin. If overlap, never increase fin.
// if sort by start, 
int eraseOverlapIntervals(Interval[] intervals) {
    if (intervals.length == 0)  return 0;
    Arrays.sort(intervals, (a, b) -> a[1] - b[1]); // by fin to leverage start < fin.
    int dels = 0, minfin = intervals[0][1];
    for(int i=1;i<intervals.length;++i) {
        if(interval[i][0] >= minfin) { minfin = interval[i][1]; continue;}
        else {
          // overlap, discard one. if sort by start, reduce minfin is new interval fin is smaller. del
          // if sort by fin, no need to reduce minfin.
          // if(arr[i][1] <= minfin) {minfin = arr[i][1];} 
          dels++;
        }
    }
    return dels;
}

// interval overlap, when adding a new interval, track min END edge of all overlap intervals.
// if curInterval.start < minEnd, means cur overlap. update minEnd with the smaller end minEnd = min(minEnd, curinterval.end), 
// else, no overlap minEnd = cur.end
static int findMinArrowBurstBallon(int[][] points) {
  if (points == null || points.length == 0)   return 0;
  Arrays.sort(points, (a, b) -> a[0] - b[0]);   // sort intv start, track minimal End of all intervals
  int minEnd = Integer.MAX_VALUE, count = 0;
  //  minEnd record the min end of previous intervals
  for (int i = 0; i < points.length; ++i) {
      // whenever current balloon's start is bigger than minEnd, 
      // no overlap, that means we need an arrow to clear all previous balloons
      if (minEnd < points[i][0]) {   // cur ballon start is bigger, not overlap      
          count++;                   // one more arrow to clear prev all.
          minEnd = points[i][1];
      } else {                       // overlap, if cur.rite less, update to minR.
          minEnd = Math.min(minEnd, points[i][1]);
      }
  }
  return count + 1;
}

// Range is Interval Array, TreeMap<StartIdx, EndIdx> represents intervals. on overlap intervals exist in the map.
class RangeModule {
  TreeMap<Integer, Integer> map;  // key is start, val is end.
  public RangeModule() { map = new TreeMap<>(); }
  
  # find floor of range left/rite
  public void addRange(int left, int rite) {
      if (rite <= left) return;
      Integer startKey = map.floorKey(left);
      Integer endKey   = map.floorKey(rite);  // floorKey of the rite edge. (greatest key that <= rite, might eqs startkey
      if (startKey == null && endKey == null) {  map.put(left, rite); }
      else if (startKey != null && map.get(startKey) >= left) {   // end > left, startkey overlap new range, merge with max endkey[0] and rite
          map.put(startKey, Math.max(map.get(endKey), Math.max(map.get(startKey), rite)));
      } else {  map.put(left, Math.max(map.get(endKey), rite)); }
    
    // clean up overlapped intermediate intervals
    Map<Integer, Integer> subMap = map.subMap(left, false, rite, true);
    Set<Integer> set = new HashSet(subMap.keySet());
    map.keySet().removeAll(set);
  }  
  public boolean queryRange(int left, int rite) {
      Integer startKey = map.floorKey(left);
      if (startKey == null) return false;
      return map.get(startKey) >= rite;
  }
  public void removeRange(int left, int rite) {
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

// --------------------------------------------------
// Scheduling, MaxProfit
// --------------------------------------------------
// # task scheduling. BFS with PQ to sort task counts. poll PQ with cooling interval. 
// # we want schedule largest task once cool constraint is satisfied.
static int scheduleWithCoolingInterval(char[] tasks, int coolInterval) {
  int[] counts = new int[26];
  int sz = tasks.length;
  for (int i=0;i<sz;i++) { counts[tasks[i]-'A'] += 1; }
  int timespan = 0;  // tot time for all intervals
  PriorityQueue<Integer> taskq = new PriorityQueue<Integer>(26, Collections.reverseOrder());
  // seed q with each tasks counts.
  for (int i=0;i<26;i++) { if (counts[i] > 0) { taskq.offer(counts[i]); } } 
  while (!taskq.isEmpty()) {
    List<Integer> unfinishedTasks = new ArrayList<Integer>();
    int intv = 0;                   // now I need a counter to break while loop.
    while (intv < coolInterval) {   // each cool interval
        if (!taskq.isEmpty()) {
            int task = pq.poll();
            if (task > 1) { unfinishedTasks.add(task-1); }  // scheduled, -1
        }
        if (taskq.isEmpty() && unfinishedTasks.size() == 0) {
            break;
        }
        timespan += 1;
        intv += 1;       // update counter per iteration.
    }
    // re-add unfinished task into q 
    for (int t : unfinishedTasks) { taskq.add(counts[t]); }
  }
  return timespan;
}

int bisect(Job[] jobs, int start) {
  int l=0,r=jobs.length-1;
  while(l<=r) {
    int m = l+(r-l)/2;
    if(jobs[m].ed <= /* <= */ start) { l=m+1; }  else { r=m-1; }
  }
  return l-1;  //  l parked at first job whose ed abs > st; l-1 ed <= st;
}
// dp[i] = Max(dp[i-1], dp[i], dp[prev]+profit);
public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
  Job[] jobs = new Job[profit.length];
  for(int i=0;i<profit.length;++i) { jobs[i] = new Job(startTime[i], endTime[i], profit[i]); }
  Arrays.sort(jobs, (a, b) -> a.ed - b.ed); // sort with job's end time; leverage start < end.
  int[] dp = new int[jobs.length];  // the max profit of each job.
  for(int i=0;i<jobs.length;++i) {
    int prejob = bisect(jobs, jobs[i].st);
    int prevprofit = prejob==-1 ? 0 : dp[prejob];
    // the profit at job i is the max of all prev jobs.
    dp[i] = Math.max(dp[i], prevprofit+jobs[i].profit);
    if(i>0) { dp[i] = Math.max(dp[i], dp[i-1]);}
  }
  return dp[dp.length-1];
}

class Solution { // Task Scheduler.
  public class Task implements Comparable<Task> {
    char c_; int cnt_; int next_;
    public Task(char c, int cnt) { this.c_ = c; this.cnt_ = cnt; this.next_=0;}
    // return < 0 if this shall ahead other; >0 if this behind other. 
    @Override public int compareTo(Task other) {
      if(this.cnt_ > other.cnt_) return -1;
      if(this.cnt_ == other.cnt_) { return this.next_ - other.next_;} // smaller next_ ahead
      else return 1;
    }
    public String toString() { return this.c_ + ":" + this.cnt_ + " next : " + this.next_;}
  }
  public int leastInterval(char[] tasks, int n) {
    Map<Character, Integer> map = new HashMap<>();
    for(int i=0;i<tasks.length;++i) {
      map.compute(tasks[i], (k,v) -> v==null ? 1 : v+1);
    }
    PriorityQueue<Task> q = new PriorityQueue<>();
    for(Map.Entry<Character, Integer> e : map.entrySet()) {
      q.add(new Task(e.getKey(), e.getValue()));
    }
    int cycles = 0, scheduled = 0;
    List<Task> l = new ArrayList<>();
    while(q.size() > 0) {
      l.clear();
      scheduled = 0;
      for(int k=0;k<n+1;++k) { // consider all tasks every n+1 step.
        while(q.size() > 0) {
          Task c = q.poll();
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
      // last round, remove the trailing non scheduled.
      if(l.size() == 0) { cycles -= (n+1-scheduled);} 
      // after every n+1 intervals, restart with all tasks.
      for(Task t : l) { q.add(t); }
    }
    return cycles;
  }
}

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
public class Main {
  static int MaxStockProfitTop2(int[] arr) {
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
  static int MaxProfitDP(int[] arr, int txns) {
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

  static int MaxProfitWithRest(int[] arr, int txns, int gap) {
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
  
  public static void main(String[] args) {
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


""" walk with f, where f is for leaf node. for list form, map(walk f ...)
map a curry fn that is recursive calls walk f to non-leaf node """
(defn walk
  [f form]
  (let [pf (partial walk f)]
    (if (coll? form)
      (into (empty form) (map pf form))
      (f form))))

""" outer fn is apply to leaf node. inner fn is for nested list forms.
map inner fn into list
inner is fn need to apply to nested form. outer is fn apply to non-nest form
"""
(defn walk
  "Traverses form, an arbitrary data structure.  inner and outer are
  functions.  Applies inner to each element of form, building up a
  data structure of the same type, then applies outer to the result.
  Recognizes all Clojure data structures. Consumes seqs as with doall."
  
  [fn-to-coll fn-to-leaf form]
  (cond
   (list? form) (fn-to-leaf (apply list (map fn-to-coll form)))
   (instance? clojure.lang.IMapEntry form) (fn-to-leaf (vec (map fn-to-coll form)))
   (seq? form) (fn-to-leaf (doall (map fn-to-coll form)))
   (instance? clojure.lang.IRecord form)
     (fn-to-leaf (reduce (fn [r x] (conj r (fn-to-coll x))) form form))
   (coll? form) (fn-to-leaf (into (empty form) (map fn-to-coll form)))
   :else (fn-to-leaf form)))

""" f is fn apply to leaf form. for non-leaf/list, recursive apply f into """
(defn postwalk
  "Performs a depth-first, post-order traversal of form.  Calls f on
  each sub-form, uses f's return value in place of the original.
  Recognizes all Clojure data structures. Consumes seqs as with doall."
  [f form]
  (walk (partial postwalk f) f form))

(defn prewalk
  "Like postwalk, but does pre-order traversal."
  {:added "1.1"}
  [f form]
  (walk (partial prewalk f) identity (f form)))


// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// AVL Tree and TreeMap, sorted after Add/Removal. RedBlack tree Rebalance. 
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
""" AVL tree with rank.
"""
class Node {
    int val, rank;
    Node left, rite;
    Node(int val) { this.val = val; this.rank = 1;}
}
""" 
AVL tree with rank, getRank ret num node smaller. left/rite rotate.
Ex: count smaller ele on the right, find bigger on the left, max(a[i]*a[j]*a[k])
"""
AVLTree(object):
  def __init__(self,key):
    self.key = key
    self.size = 1
    self.height = 1
    self.left = None   // children are AVLTree
    self.rite = None
  def __repr__(self):
      return str(self.key) + " size " + str(self.size) + " height " + str(self.height)
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

// exist pair(i,j) abs(i,j)<=win, abs(arr[i],arr[j])<=diff; 
// Least ele(ceiling) of lowerbound(cur-diff) shall within upper bound(floor) of arr+diff; 
// sorted AVL tree, find the least ele >= lowerbound(arr/i-diff) shall be bounded/within (arr/i+diff).
// TreeSet/TreeMap to balance height by rotating when adding arr[r]/remove arr[l] sliding win. 
public boolean containsNearbyAlmostDuplicate(int[] arr, int win, int diff) {
  int l=0,r=l+1;
  TreeSet<Integer> winset = new TreeSet<>((a, b) -> a-b);
  winset.add(arr[l]);
  while(r<arr.length) {
    int lowerbound=arr[r]-diff, upperbound=arr[r]+diff;
    Integer ceiling = winset.ceiling(lowerbound);
    if(ceiling != null && ceiling.intValue() <= upperbound) return true;
    if(r-l==win) { winset.remove(arr[l]); l++; }
    winset.add(arr[r]);
    r++;
  }
  return false;
}

// segment tree. Each node track the range[st, ed] and the range aggregation(min,max,sum)
// first segment 1 cover range[0,n], second segment covers [0, mid], seg 3, [mid+1,n]. 
// Segment num is btree, child[i] = 2*i,2*i+1.
// search always starts from seg 0, and found which segment [lo,hi] lies.
import math
import sys
class RMQ(object):  # RMQ has heap ary and tree. Tree repr by root Node.
  class Node():
    def __init__(self, lo=0, hi=0, minval=0, minvalidx=0, segidx=0):
      self.startIdx = startIdx  # each node tracks a range incl [st, ed]
      self.endIdx = endIdx
      self.segidx = segidx      # segment idx in rmq heap tree.
      self.minval = minval      # range minimal, can be range sum
      self.minvalidx = minvalidx  # idx in origin val arr
      self.left = self.rite = None  # Node childeren are Node
  
  def __init__(self, arr=[]):
    self.arr = arr  # original arr
    self.size = len(self.arr)
    # heap repr by ary. Node inside heap.
    self.heap = [None]*pow(2, 2*int(math.log(self.size, 2))+1)
    # seg tree repr by root Node with segidx 0, left segidx 1, rite segidx 2
    self.root = self.build(0, self.size-1, 0)[0]
  # 0th segment covers entire, 1th, left half, 2th, rite half
  def build(self, startIdx, endIdx, segRootIdx):  # root segRootIdx, lc/rc of segRootIdx
    if startIdx == endIdx:
      n = RMQ.Node(startIdx, endIdx, self.arr[startIdx], startIdx, segRootIdx)
      self.heap[segRootIdx] = n
      return [n, startIdx]
    mid = (startIdx+endIdx)/2
    left,lidx = self.build(startIdx, mid, 2*segRootIdx+1)  # lc of segRootIdx
    rite,ridx = self.build(mid+1, endIdx, 2*segRootIdx+2)  # rc of segRootIdx
    minval = min(left.minval, rite.minval)
    minidx = lidx if minval == left.minval else ridx
    n = RMQ.Node(startIdx, endIdx, minval, minidx, segRootIdx)
    n.left = left   n.rite = rite
    self.heap[segRootIdx] = n
    return [n,minidx]
  # segment tree in bst, search always from root.
  def rangeMin(self, root, startIdx, endIdx):
    // ret max if requested range outside current node's subtree range. 
    if startIdx > root.endIdx or endIdx < root.startIdx:
      return sys.maxint, -1
    // range bigger than current node's range, return tracked min. 
    if startIdx <= root.startIdx and root.endIdx <= endIdx:
      return root.minval, root.minvalidx
    lmin,rmin = root.minval, root.minval
    # ret min of both left/rite.
    if root.left:   lmin,lidx = self.rangeMin(root.left, startIdx, endIdx)
    if root.rite:   rmin,ridx = self.rangeMin(root.rite, startIdx, endIdx)
    minidx = lidx if lmin < rmin else ridx
    return min(lmin,rmin), minidx
  # segment tree in heap tree. always search from root, idx=0
  def queryRangeMinIdx(self, idx, lo, hi):
    if idx > self.heap.size(): return sys.maxint, -1
    node = self.heap[idx]
    if lo > node.hi or hi < node.lo:     return sys.maxint, -1
    if lo <= node.lo and hi >= node.hi:  return node.minval, node.minvalidx
    lmin,rmin = node.minval, node.minval
    if self.heap[2*idx+1]:   lmin,lidx = self.queryRangeMinIdx(2*idx+1, lo, hi)
    if self.heap[2*idx+2]:   rmin,ridx = self.queryRangeMinIdx(2*idx+2, lo, hi)
    minidx = lidx if lmin < rmin else ridx
    return min(lmin,rmin), minidx

def build(self, arr, lo, hi):
  if lo > hi:   return None
  node = RMQ.Node(lo, hi)
  if lo == hi:  node.sum = arr[lo]
  else:
    mid = (lo+hi)/2
    node.left = self.build(lo, mid)
    node.rite = self.build(mid+1, hi)
    node.sum = node.left.sum + node.rite.sum
  return node
def update(self, idx, value):
  if self.lo == self.hi:  self.sum = value
  else:
    m = (node.lo + node.hi)/2
    if idx <= m: update(self.left, idx, value)
    else:        update(self.rite, idx, value)
    self.sum = self.left.sum + self.rite.sum
def sumRange(self, lo, hi):
  if self.lo == lo and self.hi == hi: return self.sum
  m = (node.lo + node.hi)/2
  if hi <= m:  return sumRange(self.left, lo, hi)
  elif lo >= m+1:  return sumRange(self.rite, lo, hi)
  return sumRange(self.left, lo, m) + sumRange(self.rite, m+1, hi)

def test():
    r = RMQ([4,7,3,5,12,9])
    print r.rangeMin(r.root,0,5)
    print r.queryRangeMinIdx(0,0,5)

// find all words in the board, dfs, backtracking and Trie, with visited map. """
def wordSearch(board, words):
  out = []
  def buildTrie(words):
    root = Trie(None)
    for w in words:
      cur = root
      for c in w:
        if not cur.next[c]:  cur.next[c] = Trie(c)
        cur = cur.next[c]
      cur.leaf = True
      cur.word = w
  def dfs(board, i, j, node):
    c = board[i][j]
    nxtnode = node.next[c]
    if not nxtnode or c == "visited": return
    if nxtnode.leaf:  out.append(nxtnode.word) return
    board[i][j] = "visited"
    for nxti,nxtj in nextmove(i,j):
      dfs(board, nxti, nxtj, nextnode) # recur with next trie node
    board[i][j] = c
    board[i][j] = "un-visited"
  root = buildTrie(words)
  # for each cell, do dfs
  for i,j in board:
    dfs(board, i, j, root)
  return out


''' tab[i] = num of bst for ary 1..i, tab[i] += tab[k]*tab[i-k] '''
def numBST(n):
  tab = [0]*(n+1)
  tab[0] = tab[1] = 1
  for i in xrange(2,n):
    for k in xrange(1,i+1):
      tab[i] += tab[k-1]*tab[i-k]   # each lchild combine with each rchild. lxr
  return tab[n]

""" find pair node in bst tree sum to a given value, lgn.
first, lseek to lmin, rseek to rmax, stk along the way, then back one by one.
down to both ends(lstop/lcur), with stk to rem parent, then mov rite or left """
def pairNode(root, target):
  lstk,rstd = deque(),deque()  # use lgn stk to store tree path parent
  lstop,rstop = False,False
  lcur,rcur = root,root
  while True:
    while not lstop:
      if lcur:
        lstk.append(lcur)
        lcur = lcur.left
      else:
        if not len(lstk):
          lstop = 1
        else:
          lcur = lstk.pop()
          lval = lcur
          lcur = lcur.rite
          lstop = 1
      while not rstop:
        if rcur:
          rstk.append(rcur)
          rcur = rcur.rite
        else:
          if not len(rstk):
            rstop = 1
          else:
            rcur = lstk.pop()
            rval = rcur
            rcur = rcur.left
            rstop = 1
      if lval + rval == target:
        return lcur,rcur
      if lval + rval > target:
        rstop = False
      else:
        lstop = False
  return lcur,rcur


// insertion is a procedure of replacing null with new node !
def insertBST(root, node):
  if not root.left and node.val < root.val:   root.left = node
  if not root.right and node.val > root.val:  root.right = node
  if root.left and root.val > node.val:       insertBST(root.left, node)
  if root.right and root.val < node.val:      insertBST(root.right, node)

def common_parent(root, n1, n2):
  if not root: return None
  if root is n1 or root is n2: return root
  left = common_parent(root.left, n1, n2)
  rite = common_parent(root.rite, n1, n2)
  retur root if left and rite else return left if left else rite.

// convert bst to link list, isleft flag is true when root is left child of parent 
// test with BST2Linklist(root, False), so the left min is returned.
// recur with context, if is left child, return right most, else, return left min '''
def BST2Linklist(root, is_left_child):
  // leaf, return node itsef
  if not root.left and not root.rite: return root
  if root.left:
      lefthead = BST2Linklist(root.left, True)
      if lefthead:
          lefthead.next = root
          root.pre = lefthead
  if root.rite:
      ritehead = BST2Linklist(root.rite, False)
      if ritehead:
          root.next = ritehead
          ritehead.pre = root
  if is_left_child:
      // ret max rite
      maxrite = root
      while maxrite.rite: maxrite = maxrite.rite
      return maxrite
  else:
      // ret left min
      minleft = root
      while minleft.left: minleft = minleft.left 
      return minleft

// max of left subtree, or first parent whose right subtree has me
def findPrecessor(node):
  if node.left:                return findLeftMax(node.left)
  while root.val > node.val:   root = root.left  // smaller preced must be in left tree
  while findRightMin(root) != node:  root = root.right
  return root

def delBST(node):
  parent = findParent(root, node)
  if not node.left and not node.right:  parent.left = null
  # if node has left, max of left child
  if node.left:
      cur = findLeftMax(node)
      delBST(cur)
      parent.left = cur
      return
  if node.right:
      cur = findRightMin(node)
      delBST(cur)
      parent.left = cur
      return
def testIsBST():
    root = Node()
    isBST(root, -sys.maxint, sys.maxint)
def toBST(l):
    if not l: return None
    mid = len(l)/2
    midv = l[mid]
    root = Node(midv)
    root.prev = toBST(l[:mid])
    root.next = toBST(l[mid:])
    return root

// triplet inversion. LoopInvar: track loop i's inversion(bigger on left, smaller on rite)
// use TreeMap as avl tree, headMap / tailMap / subMap.
static long maxInversions(List<Integer> prices) {
  TreeMap</*price=*/Integer, /*cnt=*/Integer> tree = new TreeMap<>();
  int[] leftBig = new int[prices.size()];
  int[] riteSmall = new int[prices.size()];
  for (int i=0; i<prices.size(); i++) {
      Map<Integer, Integer> sub = tree.tailMap(prices.get(i), false);
      // include duplicates.
      for (int v : sub.values()) { leftBig[i] += v; }
      tree.put(prices.get(i), tree.getOrDefault(prices.get(i), 0) + 1);
  }
  tree.clear();
  for (int i=prices.size()-1;i>=0;i--) {
      Map<Integer, Integer> sub = tree.headMap(prices.get(i), false);
      for (int v : sub.values()) { riteSmall[i] += v; }
      tree.put(prices.get(i), tree.getOrDefault(prices.get(i), 0) + 1);
  }
  int tot = 0;
  for(int i=1;i<prices.size()-1;i++) {
      tot += leftBig[i] * riteSmall[i];
  } 
  return tot;
}
// prices =  [15,10,1,7,8]
// tot = (15, 10, 1), (15, 10, 7), and (15, 10, 8)

""" no same char shall next to each other """
from collections import defaultdict
def noAdj(text):
  cnt = defaultdict(lambda: 0)
  sz = len(text)
  for i in xrange(sz): cnt[text[i]] += 1
  l = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
  start = 0
  out = [None]*sz
  for e in l:
      gap = sz/e[1]
      for s in xrange(e[1]):
          pos = start + s*gap
          while out[pos]:
              pos += 1
              if pos >= sz:
              return False, out
          out[pos] = e[0]
      start += 1
  return True, out

print noAdj("abacbcdc")
print noAdj("aabbc")

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Graph, Edge types.
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 

// A Leaf-Root Tree, using parent to track same component. Leaves to single root, vs. single root to leaves. 
class UnionFind {
  private int[] parent;   // parent[idx] = idx;
  Map<Integer, Integer> parentMap = new HashMap<>();
  public UnionFind(int n) {
      parent = new int[n];
      childrens = new int[n];
      for(int i=0; i<n; i++) { parent[i] = -1; chidrens[i]=0;}
  }
  public int setParent(int i, int p) { parent[i]=p; childrens[p]+=1; return p;}
  public int findParent(int i) {
    {
      if(!parentMap.containsKey(i)) return null;
      if(parentMap.get(i) != i) {
        parentMap.put(i, findParent(parentMap.get(i)));
      }
      return parentMap.get(i);
    }
    if (parent[i] != i) {  // recursion not ret in the middle. no need while loop.
      // note parent[i] is used as the idx for recursion.
      parent[i] = findParent(parent[i]); // flatten parent[i] direct to parent[root]=root; reduce height;
    }
    return parent[i];
  }
  public boolean same_component(int i, int j) { return findParent(i) == findParent(j); }
  // union merge two groups into one if not the same. return true. If already same, no merge, ret false;
  public boolean union(int x, int y){ 
    int xr = findParent(x), yr = findParent(y);
    if(xr != yr) {
      if(childrens[xr] >= childrens[yr]) {
        setParent(y, xr); // parent[yr] = xr;  // next findParent will flatten the children of yr.
        childrens[xr] += childrens[yr];
      } else {
        setParent(x, yr);//  parent[xr] = yr;
        childrens[yr] += childrens[xr];
      }
      return true;
    }
    {
      int pi=findParent(i), pj=findParent(j);
      if(pi != pj) {
        if(pi > pj) { setParent(j, pi); }
        else { setParent(i, pj);}
      }
      return Math.max(pi, pj);
    }
    return false;
  }
  // returns the maxium size of union
  public int maxUnion(){ // O(n)
    int[] count = new int[parent.length];
    int max = 0;
    for(int i=0; i<parent.length; i++){
        count[root(i)] ++;
        max = Math.max(max, count[root(i)]);
    }
    return max;
  }
}

// # Longest Consecutive Sequence. unsorted array find the length of the longest consecutive elements sequence.
// # bucket, union to merge. use the index of num as index to 
public int longestConsecutive(int[] nums) {
  UnionFind uf = new UnionFind(nums.length); // use a parent ary to track and merge slots of nums[i].
  Map<Integer,Integer> map = new HashMap<Integer,Integer>(); // <value,index>
  for(int i=0; i<nums.length; i++) {
    if (map.containsKey(nums[i])){ continue; }
    map.put(nums[i], i); // <value, index>
    // looking for its neighbors, union i and x into one bucket
    if (map.containsKey(nums[i]+1)) { uf.union(i, map.get(nums[i]+1)); }  // union index i, and k
    if (map.containsKey(nums[i]-1)) { uf.union(i, map.get(nums[i]-1)); }
  }
  return uf.maxUnion();
}

// - - - - - - - - - - - -// - - - - - - - - - - - -// - - - - - - - - - - - -
// From src -> dst, min path dp[src][1111111]   bitmask as path.
//  1. DFS(root, prefix); From root, for each child, if child is dst, result.append(path)
//  2. BFS(PQ); PQ(sort by weight), poll PQ head, for each child of hd, relax by offering child.
//  3. DP[src,dst] = DP[src,k] + DP[k, dst]; In Graph/Matrix/Forest, from any to any, loop gap, loop start, loop start->k->dst.
//      dp[i,j] = min(dp[i,k] + dp[k,j] + G[i][j], dp[i,j]);
//      dp[i,j] = min(dp[i,j-1], dp[i-1,j]) + G[i,j];
//  4. Topsort(G, src, stk), back-edge. 
//  int edge_classification(int x, int y){
//    if (parent[y] == x) return(TREE);
//    if (discovered[y] && !processed[y]) return(BACK);
//    if (processed[y] && (entry_time[y]>entry_time[x])) return(FORWARD);
//    if (processed[y] && (entry_time[y]<entry_time[x])) return(CROSS);
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
  int rows = G.size();
  int cols = G[0].size();
  int[] dp = new dp[cols];  // dp table ary width is cols.
  // boundary cols, add first row.
  for (int i=1;i<cols;i++) {
    dp[i] = dp[i-1] + G[0][i];
  }
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
        dp[s][d][gap] = 0
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
  def topsort(G, src, stk):  # dfs all children, then visit root by push to stk top.
    def dfs(s,G,stk):
      for v in neighbor(G,s):
        if(edge(s,v)) == BACK: return false;
        if not visited[v]:
          dfs(v, G, stk)
      visited[v] = true
      stk.append(v)
    for v in G:
      if not visited[v]:
        dfs(v,G,stk)
    return stk

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
  PriorityQueue<int[]> heap = new PriorityQueue<int[]>( (a, b) -> Integer.compare(a[0], b[0]));
  heap.offer(new int[]{0, 0, sr, sc});

  HashMap<Integer, Integer> cost = new HashMap();
  cost.put(sr * C + sc, 0);

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

// PQ edges sorted by dist. poll, add nb, relax with reduced stops. DP[to][k]=min(dp[from][k-1]+(from,to))
public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
  int[][] dp = new int[n][k+1];
  int[][] prices = new int[n][n];
  PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0]-b[0]);
  for(int i=0;i<flights.length;++i) {
    int[] f = flights[i];
    prices[f[0]][f[1]] = f[2];
    if(f[0]==src) { pq.offer(new int[]{f[2], f[1], 0}); }
  }
  while(pq.size() > 0) {
    int[] e = pq.poll();
    int totprice=e[0], to=e[1], hops=e[2]; //, totprice=e[3];
    dp[to][hops] = dp[to][hops] > 0 ? Math.min(dp[to][hops], totprice) : totprice;
    if(to==dst) return totprice;
    if(hops == k || totprice > dp[to][hops]) continue;
    for(int i=0;i<n;++i) {
      if(prices[to][i] > 0 && i != src && (dp[i][hops] == 0 || prices[to][i] < dp[i][hops])) {
        pq.offer(new int[]{totprice+prices[to][i], i, hops+1}); 
      }
    }
  }
  int mincost = Integer.MAX_VALUE;
  for(int i=0;i<=k;i++) {
    if(dp[dst][i] > 0)
      mincost = Math.min(mincost, dp[dst][i]);
  }
  return mincost == Integer.MAX_VALUE ? -1 : mincost;
}

// seed pq with src, edges repr by G Map<Integer, List<int[]>>,  for each nb edge, relax dist
int networkDelayTime(int[][] times, int N, int K) {
    Map<Integer, List<int[]>> adjacentMap = new HashMap();   // vertex --> [edge pair].
    for (int[] edge: times) {
        adjacentMap.putIfAbsent(edge[0], new ArrayList<int[]>()).add(new int[]{edge[1], edge[2]});
    }
    // dist map from src-> [destination, dist]
    Map<Integer, Integer> distMap = new HashMap();

    PriorityQueue<int[]> heap = new PriorityQueue<int[]>((info1, info2) -> info1[0] - info2[0]);    
    heap.offer(new int[]{0, K});    // seed pq with src
    while (!heap.isEmpty()) {
        int[] info = heap.poll();
        int d = info[0], node = info[1];
        if (distMap.containsKey(node))  continue;      // already seen.
        distMap.put(node, d);
        if (adjacentMap.containsKey(node)) {
            for (int[] edge: adjacentMap.get(node)) {     // for each edge, relax distMap
                int nei = edge[0], d2 = edge[1];
                if (!distMap.containsKey(nei))           // only unseen neighbors
                    heap.offer(new int[]{d+d2, nei});
            }
        }
    }
    if (distMap.size() != N) return -1;
    int ans = 0;
    for (int cand: distMap.values())
        ans = Math.max(ans, cand);
    return ans;
  }
}

// node is path as bit, the the path covers all node has (11111)
// enum all values of path, from any head. DP[head][path] = min(DP[nb][path-1])
int shortestPathToAllNode(int[][] graph) {
  int N = graph.length;
  int dist[][] = new int[1<<N][N];   // path, hd

  for (int[] row: dist) Arrays.fill(row, N*N);
  for (int x = 0; x < N; ++x) dist[1<<x][x] = 0;

  // iterate on all possible path values. At each path value, fill all header's value.
  for (int path=0; path<1<<N; path++) {
    boolean repeat = true;
    while (repeat) {
        repeat = false;
        for (int head = 0; head < N; ++head) {
            int d = dist[path][head];
            for (int nb : graph[head]) {
                int nxtPath = path | (1 << nb);
                if (d + 1 < dist[nxtPath][nb]) {
                    dist[nxtPath][nb] = d+1;
                    if (path == nxtPath) repeat = true;
                }
            }
        }
    }
  }
  int ans = N*N;
  for (int cand: dist[(1<<N) - 1])
      ans = Math.min(cand, ans);
  return ans;
}

// use two pq, one is sorted by capacity, the other sort by profit. extract from capPQ and insert to proPQ
int findMaximizedCapital(int k, int W, int[] Profits, int[] Capital) {
    PriorityQueue<int[]> pqCap = new PriorityQueue<>((a, b) -> (a[0] - b[0]));
    PriorityQueue<int[]> pqPro = new PriorityQueue<>((a, b) -> (b[1] - a[1]));
    
    for (int i = 0; i < Profits.length; i++) { pqCap.add(new int[] {Capital[i], Profits[i]}); }
    
    for (int i = 0; i < k; i++) {  // loop k batches, each batch polling capacity and add to profit
        while (!pqCap.isEmpty() && pqCap.peek()[0] <= W) {  // one batch
            pqPro.add(pqCap.poll());
        }
        if (pqPro.isEmpty()) break;
        W += pqPro.poll()[1];
    }
    return W;
  }
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

""" """ """ """ """
Dynamic Programming, better than DFS/BFS.
bottom up each row, inside each row, iter each col.
[ [col1, col2, ...], [col1, col2, ...], row3, row4, ...]
At each row/col, recur incl/excl situation.
Boundary condition: check when i/j=0 can not i/j-1.
""" """ """ """ """

''' start from minimal, only item 0, loop items i=0..n, w loops weights at each item. 
v[i, w] is max value for each weight with items from 0 expand to item i, v[i,w] based on f(v[i-1, w]).
v[i, w] = max(v[i-1, w], v[i-1, w-W[i]]+V[i]) 
'''
def knapsack(warr, varr, totalW):
  prerow, row = [0]*(totalW+1), [0]*(totalW+1)
  for i in xrange(len(warr)):
    row = prerow[:]
    w_i,v_i = warr[i], varr[i]
    for w in xrange(w_i, totalW+1):
      if i == 0:     # init boundary, first 
        row[w] = v
      else:
        row[w] = max(prerow[w-w_i]+v_i, prerow[w])
    prerow = row[:]
  return row[W]
print knapsack([2, 3, 5, 7], [1, 5, 2, 9], 18)

// mem optimization. only 2 rows needed.
def backpack(arr, W):
  pre,row = [0]*(W+1), [0]*(W+1)
  for i in xrange(len(arr)):
    row = pre[:]
    for w in xrange(arr[i], W+1):      
      row[w] = max(row[w], pre[w-arr[i]]+arr[i])
    pre = row[:]
  return row[W]
print backpack([2, 3, 5, 7], 11)

static class IpSeg {
    int v;
    int nextOff;
    int seq;  // 0.0.0.0
    IpSeg parent;
    public IpSeg(int v, int nextOff, int seq) {
        this.v = v;
        this.nextOff = nextOff;
        this.seq = seq;
    }
}
// Backtrack all intermediate result with a queue.
static int restoreIP(String s) {
  int sz = s.length();
  Deque<IpSeg> ips = new ArrayDeque<>();
  int off = 0;
  for (int k=0;k<3;k++) {
      int v = Integer.parseInt(s.substring(off, off + k + 1));
      if (v <= 255) {
          ips.offerLast(new IpSeg(v, off+k+1, 1));
      }
  }
  int tot = 0;
  while (ips.size() > 0) {
    IpSeg cur = ips.pollFirst();
    if (cur.seq == 4 && cur.nextOff == sz) {
        tot += 1;
        continue;
    }
    if ( cur.seq >= 4) continue;

    off = cur.nextOff;
    for (int k=0;k<3;k++) {
        if (off + k < sz) {
          int v = Integer.parseInt(s.substring(off, off + k +1));
          if (v <= 255) {
              ips.offerLast(new IpSeg(v, off+k+1, cur.seq+1));
          }
        }
    }
  }
  System.out.println("total ips " + tot);
  return tot;
}

// adding + in between. [1,2,0,6,9] and target 81.
// dp[pos][val] = dp[pos-k][val-A[k..pos]]; enum all values, the value dim ary size could be large.
def plusBetween(arr, target):
  def toInt(arr, st, ed):
    return int("".join(map(lambda x:str(x), arr[st:ed+1])))
  dp = [[False]*(target+1) for i in xrange(len(arr))]
  for i in xrange(len(arr)):
    for j in xrange(i,-1,-1):  # one loop j:i to coerce current val
      val = toInt(arr,j,i)
      if j == 0 and val <= target:
        dp[i][val] = True   # dp[i,val], not dp[i,target]
      # enum all values between val to target, for dp[i,s]
      for v in xrange(val, target+1):
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

// BFS, search recursion carrying context, when reaching pos, carry path
from collections import deque
def genParenth(n):
  out = []
  q = deque()   
  q.append(["(", 1, 0])   // seed the bfs q from start, carry left / rite num of ()
  while len(q) > 0:
    [path,l,r] = q.popleft()
    if l == n:
      for i in xrange(2*n-l-r):
        path += ")"
      out.append(path)
    else:
      if l == r:
        p = str(path) + "("
        q.append([p, l+1, r])
      else:  # at this pos, we can append eith l/r, branch.
        p = str(path) + "("
        q.append([p,l+1,r])
        p = str(path) + ")"
        q.append([p,l,r+1])
  return out
print genParenth(3)

// recur, generize formula, with boundary condition first """
from collections import deque
def bfsInvalidParen(s):
  res = []
  q = deque()
  q.append([1, ["("], 1, 0])
  while len(q):
    pos,path,l,r = q.popleft()
    if pos == len(s):
      if l == r:
        res.append(path)
      continue
    if s[pos] == "(":
      p = path[:]
      p.append("(")
      q.append([pos+1, p, l+1, r])
    else:
      if l > r:
        p = path[:]
        p.append(")")
        q.append([pos+1, p, l, r+1])
      else:
        p = path[:]
        q.append([pos+1,p,l,r])

        pr = ''.join(map(lambda x:str(x),path)).rindex("(")
        for i in xrange(pr,0,-1):
          if path[i] == ")":
            p = path[0:i] + path[i+1:]
            p.append(")")
            q.append([pos+1, p, l, r])
  return res
print bfsInvalidParen("()())()")

// (( ))) ())
// when first unbalance, make it balance, as the prefix, (prefix + rest), as a new string to balance; 
// balance is trigger upon first ), remove each prev ), form the new prefix, and recur rest with diff prefix;
public void removeInvalidParen(String s, Set<String> res) {
  int i=0, j=0, lr = 0;
  for(i=0;i<s.length();i++) {
    if(s.charAt(i)=='(') lr++;
    else if (s.charAt(i)==')') lr--;
    else continue;
    if(lr >= 0) continue;
    // lr < 0, remove one of the prev ); i is consumed and is )
    while(j<=i) {
      if(s.charAt(j) != ')') { j++; continue; }
      while(j<=i && s.charAt(j)== ')') j++;
      String prefix=s.substring(0,j-1)+s.substring(j,i+1);
      removeInvalidParen(prefix+s.substring(i+1), res);
    }
    return;  // first unbalance falls into here and no recur anymore.
  }
  if(lr==0) res.add(s);
  else {
    for(String ss : removeInvalidParentheses(s)) {
      res.add(ss);
    }
  }
}
  
def rmParenth(s, res):
  path = []
  L,R = 0,0
  for i in xrange(s):   # calculate balance(l-r)
    if s[i] == "(":  L += 1
    else:
      if L != 0:
        L -= 1
      else:
        R += 1
  recurDFS(res, s, 0, L, R, 0, path)
  return res

  # Perm recur search, each pos, L,R is balance of l-r, either L=0 or R=0
  def recurDFS(res, s, pos, L, R, openL, path):
    if pos == len(s) and L == 0 and R == 0 and openL == 0:
      p = path[:]
      res.append(p)
      return
    if pos == len(s) or L < 0 or R < 0 or openL < 0:
      return

    if s[pos] == "(":
      recurDFS(res, s, pos+1, L,   R, openL,  path)  # skip cur L
      recurDFS(res, s, pos+1, L-1, R, openL+1,path.append("(")) # use cur, L-1
      path.pop()
    else if s[pos] == ")":
      recurDFS(res, s, pos+1, L, R-1, openL,   path) # skip cur R, R-1
      recurDFS(res, s, pos+1, L, R,   openL-1, path.append(")"))  # use cur R,
      path.pop()
    else:
      recurDFS(res, s, pos+1, L, R, openL, path.append(s[pos]))
    return

s="()())()";path=[];res=[];print rmParenth(s,0,path,0,0,res)
s="()())()";path=[];res=[];print rmParenth(s,0,path,0,0,res)

# at head, dfs all possibles. track L, R, no need to track offset.
static void parenthDFS(int sz, int l, int r, String presult) {
  if (l == sz) {
      for(int k=0;k<sz-r;k++) { presult += ")"; }
      System.out.println(presult);
  } else if (l > r) {
      parenthDFS(sz, l+1, r,   presult+"(");   // dfs recur with
      parenthDFS(sz, l,   r+1, presult+")");
  } else {
      parenthDFS(sz, l+1, r, presult+"(");
  }
  // memoize
  // cache[l,r] = presult;
}
parenthDFS(3, 0, 0, "");

// # https://pathcompression.com/sinya/articles/4
// # Let f(i) be the length of the longest valid substring starting from the i th character. 
// # Then the final solution is the largest of all possible f(i).
// # If the i th character is ), any substrings starting from the i-th character must be invalid. Thus f(i) = 0.
// # If i th character is (. First we need to find its corresponding closing parentheses. 
// # If a valid closing parentheses exists, the substring between them must be a valid parentheses string. 
// # Moreover, it must be the longest possible valid parentheses string starting from (i+1) th character. 
// # Thus we know that the closing parentheses must be at (i+1 + f(i+1)) th character. 
// # If this character is not ), we know that we can never find a closing parentheses and thus f(i) = 0.
// # If the (i+1 + f(i+1)) th character is indeed a closing parentheses. 
// # To find the longest possible valid substring, we need to keep going and see if we can 
// # concatenate more valid parenthesis substring if possible. 
// # The longest we can go is f(i+1 + f(i+1) + 1). Therefore we have
// # f(i) = f(i+1) + 2 + f(i + f(i+1) + 2)
static int longestValidParenth(String parenth) {
  int sz = parenth.length();
  int[] f = new int[sz];
  int maxlen = 0;
  for (int i=sz-1;i>=0;i--) {
      if (parenth.charAt(i) == ')') { f[i] == 0; continue;}
      int closingIdx = i+1 + f[i+1];
      if (closingIdx < sz && s.charAt(closingIdx) == ')') {
          f[i] = f[i+1] + 2 + f[closingIdx+1];
          maxlen = Math.max(maxlen, f[i]);
      }
  }
  return maxlen;
}

// stk stores the idx of '(' to compute length. We only push ( into stack.
static int longestValidParenth(String parenth) {
  int sz = parenth.length();
  int maxlen = 0;
  Stack<Integer> stk = new Stack<Integer>();
  for (int i=0;i<sz;i++) {
    if (parenth.charAt(i) == '(') { stk.add(/*stk_(_index*/i); }     // push in index
    else {
      if (stk.size() > 0) {
        stk.pop();
        int begIdx = stk.isEmpty() ? 0 : stk.peek();
        maxlen = Math.max(maxlen, i-begIdx);
      }                
    }
  }
  System.out.println("longest valid " + maxlen);
  return maxlen;
}
longestValidParenth(")()()");

""" DFS for ary, no incl/excl biz. carry offset. branch at offset, recur dfs expands 4 ways on 
each segment moving idx, carry prepre, pre, and calulate this cur in next iteration"""
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


""" add operators to exp forms. two way, recur and memoize
1. divide half, recur, memoize, and merge.
2. calculate dp[l,r,v] directly with gap, i, i+gap.
"""
def addOperator(arr, l, r, target):   # when recur, need fix left/rite edges
  '''recursion with divide to half, must use memoize dp'''
  def recur(arr, l, r):
    if not dp[l][r]:      dp[l][r] = defaultdict(list)
    else:                 return dp[l][r]     // return memoized
    
    out = []
    if l == r:            return [[arr[l],"{}".format(arr[l])]]
    # arr[l:r] as one num, no divide
    out.append([int(''.join(map(str,arr[l:r+1]))),  "{}".format(''.join(map(str,arr[l:r+1])))])
    
    for i in xrange(l,r):      // divide into left / rite partition
      L = recur(arr, l, i)     // merge sort, divide and merge
      R = recur(arr, i+1, r)
      # merge Left / Rite results to produce new results.
      for lv, lexp in L:
        for rv, rexp in R:
          out.append([lv+rv, "{}+{}".format(lexp,rexp)])
          out.append([lv*rv, "{}*{}".format(lexp,rexp)])
          out.append([lv-rv, "{}-{}".format(lexp,rexp)])
    dp[l][r] = out   # memoize l,r result
    return out

  tab = [[None]*(l+1) for i in xrange(r+1)]
  out = []
  for v, exp in recur(arr, l, r):
    if v == target:
      out.append(exp)
  return out
print addOperator([1,2,3],0,2,6)
print addOperator([1,0,5],0,2,5)

""" DP[l,r] = dp[l,k] + dp[k,r]; standard 3 loop, gap=1..n, i=1..n-gap, k=l..r
"""
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

public boolean wordBreak(String s, List<String> dict) {
  Map<String, Integer> map = build(dict);
  // dp[i] [0..i-1] breakable; dp[0]=1; dp[1]=[0..0], the first
  int[] dp = new int[s.length()];                     // dp = new int[s.length()+1];
  for(int i=0;i<s.length();++i) {                     // for(int i=1;i<s.length()+1;i++) {  // i start from 1
    for(int j=i;j>=0;--j) {                           //   for(int j=i-1;j>=0;--j) {        // j=start from i-1=0;
      if(map.containsKey(s.substring(j,i+1))) {       //      substring(j,i) { dp[i] = max(dp[i], dp[j]); }
        // Math.max avoid a matching being overridden by a later dp[j-1].
        dp[i] = Math.max(dp[i], (j == 0 ? 1 : dp[j-1])); // 
      }
    }
  }
  return dp[s.length()-1]==1;
}
print wordbreak("leetcodegeekforgeek",["leet", "code", "for","geek"])
print wordbreak("catsanddog",["cat", "cats", "and", "sand", "dog"])

public boolean recurMatch(String s, int start, Map<String, Integer> dict) {
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
wordbreak("aaaaab",["a", "aa", "aaa"])

""" dp[ia,ib] means the distance(difference) between a[0:ia] and b[0:ib] """
def editDistance(a, b):
  dp = [[0]*(len(b)+1) for i in xrange(len(a)+1)]   # sz = len+1 to avoid i-1/j-1 boundary check.
  for i in xrange(len(a)+1):
    for j in xrange(len(b)+1):
      # boundary condition, i=0/j=0 set in the loop, i
      if i == 0:      dp[0][j] = j    # distance is j
      elif j == 0:    dp[i][0] = i
      elif a[i-1] == b[j-1]:  dp[i][j] = dp[i-1][j-1]
      else:  dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])  // get min
    return dp[len(a)][len(b)]

// String length - Longest Common Sequence
int minDistance(String word1, String word2) {
    int dp[][] = new int[word1.length()+1][word2.length()+1];  # 1+len to avoid 
    # Longest common sequence.
    for(int i = 0; i <= word1.length(); i++) {
        for(int j = 0; j <= word2.length(); j++) {
            if(i == 0 || j == 0) dp[i][j] = 0;       # extra boundary row/col, set to 0.
            else dp[i][j] = (word1.charAt(i-1) == word2.charAt(j-1)) ? 
              dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);    # max
        }
    }
    int val = dp[word1.length()][word2.length()];
    return word1.length() - val + word2.length() - val;
}

// BFS LinkedList. lseek to first diff, swap each off, j, like permutation, Recur with new state.
int kSimilarity(String A, String B) {
    if (A.equals(B)) return 0;
    Set<String> seen = new HashSet<>();
    Queue<String> q = new LinkedList<>();  # LinkedList Queue
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
                if (seen.add(temp)) {
                  q.add(temp);    // new state, recur.
                }
            }
        }
    }
    return res;
}

"""
M[target_chars][src_chars], expand tgt string by itering each tgt char again src string.
two choices, Not eq, excl src char, dp[tidx][sidx]=dp[tidx][sidx-1]; When Eq, consume tidx. 
  \ a  b   a   b  c
 a  1  1   2   2  2
 b  0 0+1  1  1+2 3
 c  0  0   0   0  3
"""
static int DistinctSubseq(String s, String t) {
  int[] match = new int[s.length()];
  // t/i == s/j, tot = sum(incl_j, excl_j) = sum(match/i-1/j-1 + /*excl_s/j_the_same_as_s/j-1*/match/j-1)
  // t/i != s/j, tot = excl_j = the same as s/j-1, match/j-1
  // Boundary: first row t/0, no prev row, s/j = s/j-1 + 1 or s/j-1, culmulative.
  // row t/i, match/0..i-1/ = 0. s/i != t/i, match/i = 0; when match, tot = only incl_j as excl_j = 0. 
  // stage match/j before updating it when processing s/j, as match/j is the i-1/j-1 of s/j+1.
  for(int i=0;i<s.length();++i) {
      int same = s.charAt(i) == t.charAt(0) ? 1 : 0;
      match[i] = i == 0 ? same : match[i-1] + same;
  }
  int prev_left = match[0], staged_cur = 0;
  for(int i=1;i<tlen;++i) {  // outer loop consumes each target pattern, inner loop consumes src.
      for(int j=i;j<slen;j++) {
          boolean same = s.charAt(j) == t.charAt(i);
          staged_cur = match[j];
          match[j] = i == j ? 0 : match[j-1];
          match[j] += same ? prev_left : 0;
          prev_left = staged_cur;
      }
  }
  return match[s.length() - 1];
}
print distinct("bbbc", "bbc")
print distinct("abbbc", "bc")

// dp[m][n] = Max(dp[m-wi_zeros][n-wi_ones]+1) when include arr[i] with wi_zeros/ones
public int findMaxForm(String[] arr, int m, int n) {
  // sortbysmaller(arr, m, n);
  int[][] dp = new int[m+1][n+1];
  // Res[][] dp = new Res[m+1][n+1];
  // for(int i=0;i<=m;i++){ for(int j=0;j<=n;j++) { dp[i][j] = new Res(); }
  for(int k=0;k<arr.length;k++) {
    int[] zos = zeros(arr[k]);
    for(int i=m;i>=zos[0];i--) {
      for(int j=n;j>=zos[1];j--) {
        // dp[i][j].setlen = Math.max(dp[i][j].setlen, dp[i-zos[0]][j-zos[1]].setlen+1);
        dp[i][j] = Math.max(dp[i][j], dp[i-zos[0]][j-zos[1]]+1);
          // if(dp[i][j].l.size() <= dp[i-zos[0]][j-zos[1]].l.size()) {
          //   dp[i][j] = new Res(dp[i-zos[0]][j-zos[1]]);
          //   dp[i][j].add(k, zos[0], zos[1]);
          //   assert(dp[i][j].zeros <= i && dp[i][j].ones <= j);
          // }
      }
    }
  }
  return dp[m][n];
}

// backwards from [m,n] up left to dp[row-1][col], dp[row][col-1] to [0,0] 
// dp[i,j] = max(1, min(dp[i-1,j], dp[i,j-1])-arr[i,j])
public int dungeon(int[][] arr) {
  int rows=arr.length,cols=arr[0].length;
  int[][] dp = new int[rows][cols];
  // for(int row=rows-1;row>=0;--row) {
  //   for(int col=cols-1;col>=0;--col) { 
  //     dp[row][col] = Integer.MAX_VALUE; 
  // }}
  dp[rows-1][cols-1] = Math.max(1, 1-arr[rows-1][cols-1]);
  for(int i=rows-1;i>=0;--i) {
    for(int j=cols-1;j>=0;--j) {
      if(i > 0) {
        dp[i-1][j] = Math.max(1, dp[i-1][j] > 0 ? Math.min(dp[i-1][j], dp[i][j]-arr[i-1][j]) : dp[i][j]-arr[i-1][j]);
      } 
      if(j > 0) {
        dp[i][j-1] = Math.max(1, dp[i][j-1] > 0 ? Math.min(dp[i][j-1], dp[i][j]-arr[i][j-1]) : dp[i][j]-arr[i][j-1]);
      }
    }
  }
  return dp[0][0];
}

""" the idea is to use rolling hash, A/C/G/T maps to [00,01,10,11]
or use last 3 bits of the char, c%7. 3 bits, 10 chars, 30bits, hash stored in last 30 bits
"""
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


""" next palindrom num, cp left -> rite, inc mid, propagate carry left/rite is mid=9"""
def nextPalindrom(n):    
  s = list(str(n))
  sz = len(s)
  l,r,m = (sz-1)/2,sz/2,(sz-1)/2
  while l >= 0:
    if s[l] != s[r]:
      s[r] = s[l]
    l -= 1
    r += 1
  if int("".join(s)) <= n:
    s[m] = str(int(s[m])+1)
  if sz % 2 == 0:
    s[m+1] = s[m]
  if s[m] == "10":
    s[m] = "0"
    l,r = m-1, min(sz-1, m+1)
    if s[r] == "10":
        s[r] = "0"
        r += 1
    while s[l] == "9":
        s[l] = s[r] = "0"
        l -= 1
        r += 1
    if l >= 0:
        s[l] = str(int(s[l])+1)
        s[r] = s[l]
    else:
        del s[m]
        s.insert(0, "1")
        s.append("1")
  return int("".join(s))
assert nextPalindrom(9) == 11
assert nextPalindrom(99) == 101
assert nextPalindrom(1234) == 1331

""" min insertion into str to convert to palindrom. 
tab[i,j]=min ins to convert str[i:j], Increasing gap, [i, i+gap]
tab[i][j] = tab[i+1,j-1] or min(tab[i,j-1], tab[i+1,j])
"""
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


''' convert recursion Fn call to DP tab. F[i] is min cut of S[i:n], solve leaf, aggregate at top.
    F[i] = min(F[k] for k in i..n). To aggregate at F[i], we need to recursive to leaf F[i+1].
    For DP tabular lookup, we need to start from the end, which is minimal case, and expand
    to the begining filling table from n -> 0.
'''
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

''' DP[i] = mincut of A[0..i], = min(DP[i-1]+1 iff A[j]=A[i] and Pal[j+1,i-1], DP[i]) 
for each j in i->0 where s[j,i] is palin)
'''
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
public boolean isPalin(char[] arr, int l, int r) {
  if(l==0 && l==r) return true;  // empty at head is Pal;
  if(l==arr.length) return false; // empty at end is not Pal;
  if(l+1==r) return true; // single is Pal.
  while(l<r) {
    if(arr[l] != arr[r-1]) return false;
    l++;r--;
  }
  return true;
}
public List<List<Integer>> PrefixSuffixPal(String w, int widx, Map<String, Integer> map) {
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

""" next palindrom """
def nextPalindrom(v):
  # arr[0:li] -> arr[li+1:ri]
  def copyLeftToRite(arr, li, ri):
    left = arr[:li+1]
    left.reverse()
    arr[ri:] = left
  arr = list(str(v))
  sz = len(arr)
  leftSmaller = False
  if sz % 2:
    l,r = (sz-1)/2, sz/2
  else:
    l,r = (sz-1)/2, (sz+1)/2
  while l >= 0 and r < sz and arr[l] == arr[r]:
    l -= 1
    r += 1
  if l > 0 and arr[l] < arr[r]:
    leftSmaller = True
  copyLeftToRite(arr, l, r)
  if leftSmaller:
    if sz % 2 == 1:
      mid = (sz-1)/2
      arr[mid] = str(int(arr[mid])+1)
    else:
      ml,mr = (sz-1)/2, sz/2
      arr[ml] = str(int(arr[ml])+1)
      arr[mr] = str(int(arr[mr])+1)
  return int(''.join(arr))


''' bfs search on Matrix for min dist, put intermediate result[i,j] index in Q '''
from collections import deque
def matrixBFSk(G, m, n):
  q = deque()
  def init(G, m, n):
    for i in xrange(m):
      for j in xrange(n):
        if G[i,j] == 1:  # start from cell val 1
          G[i,j] = 0
          q.append([i,j])  # can do bfs from this cell
        else:
          G[i,j] = sys.maxint  # unknow space init dist to max
  while len(q) > 0:
    [i,j] = q.popleft()
    if G[i,j] < k:
      if G[i+1,j] == sys.maxint:  # cell not explored yet
        G[i+1,j] = G[i,j] + 1     # set dist+1
        q.append([i+1,j]) 
      if G[i-1,j] == sys.maxint:
        G[i-1,j] = G[i,j] + 1
        q.append([i+1,j])    
      if G[i,j+1] == sys.maxint:
        G[i,j+1] = G[i,j] + 1
        q.append([i,j+1])
      if G[i,j-1] == sys.maxint:
        G[i,j-1] = G[i,j] + 1
        q.append([i,j-1])
  for i in xrange(m):
    for j in xrange(n):
      if G[i,j] < k:
        G[i,j] = 1

""" BFS, enum 26 chars at each pos of each word for all possible words. recur not visited.
"""
from collections import deque
import string
def wordladder(start, end, dict):
  ''' given a word, for each pos of the wd, avoid already seen'''
  def getNext(wd, seen):
    nextwds = []
    for i in xrange(len(list(wd))):
      for c in list(string.ascii_lowercase):
        l = list(wd)
        l[i] = c
        nextwd = "".join(l)
        if nextwd == end or (nextwd in dict and not nextwd in seen):
          nextwds.append(nextwd)
    return nextwds
  def replace(wd):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(wd[:i],wd[i:]) for i in xrange(len(wd)+1)]
    replace = [a + c + b[1:] for a,b in splits for c in alphabet if b]
    nextwds = []
    for e in replace:
      nextwd = "".join(e)
      if nextwd == end or (nextwd in dict and not nextwd in seen):
        nextwds.append(nextwd)
        seen.add(nextwd)
    return nextwds
  ''' bfs search from start, ladderQ'''
  if start == end: return
  ladder = []
  seen = set()     # global visited map for all bfs iterations
  q = deque()
  q.append([start,[]])  // bfs search from start
  while len(q) > 0:
    [wd,path] = q.popleft()
    for nextwd in getNext(wd, seen):  // iterate each next word
      if nextwd == end:
        path.append(nextwd)
        ladder.append(path)
      else:
        seen.add(nextwd)
        npath = path[:]
        npath.append(wd)
        q.append([nextwd, npath])
  return ladder
start = "hit"
end = "cog"
dict = set(["hot","dot","dog","lot","log"])
print wordladder(start, end, dict)

""" consider every cell as start, dfs find all word in dict carrying seen map.
reset visited map for each dfs iteration.
Optimization: build Trie for all words, then backtrack bail out early
"""
def boggle(G, dictionary):
  def isValid(r,c,M,N,seen):
    if r >= 0 and r < M and c >= 0 and c < N and G[r][c] not in seen:
      return True
    return False
  ''' recur from r,c search for all cells '''
  def recur(G, dictionary, r, c, seen, path, out):
    if "".join(path) in dictionary:
      out.append("".join(path))
    ''' iterate each neighbor as child, if not visited, recur with partial '''
    for nr,nc in [[-1,-1],[-1,0],[-1,1]...]:
      if isValid(nr,nc,M,N,seen):
        path.append(G[nr][nc])
        seen.add(G[nr][nc])
        recur(G, dictionary, nr, nc, seen, path, out)
        path.pop()
        seen.remove(G[nr][nc])
      return out
  M,N = len(G),len(G[0])
  out, seen = [], set()
  for i in xrange(M):     # start from each cell
    for j in xrange(N):
      seen.add(G[i][j])
      path = []
      path.append(G[i][j])
      recur(G, dictionary, i, j, seen, path, out)
      path.pop()
      seen.remove(G[i][j])
  return out

""" 
dp[i]: cost of word wrap [0:i]. foreach k=[i..k..j], w[j] = min(w[i..k] + lc[k,j])
"""
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

""" m[i,j] = matrix(i,j) = min(m[i,k]+m[k+1,j]+A[i-1]*A[k]*A[j]) """
def matrix(arr):
  dp = [[0]*len(arr) for i in xrange(len(arr))]
  for gap in xrange(2,len(arr)):
    for i in xrange(1, len(arr)-gap+1):
      j = i + gap -1
      dp[i][j] = sys.maxint
      for k in xrange(i,j):
        dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + arr[i]*arr[k]*arr[j])
  return dp[1][len(arr)-1]
print matrix([1,2,3,4])


""" tab[i][j] whether a[0:i] is interleave b[0:j]; tab[i,j] = tab[i-1,j], etc"""
def isInterleave(a, b, c):
  tab = [[[0]*(len(b)+1) for i in xrange(len(a)+1)]]
  for i in xrange(len(s1)+1):
    for j in xrange(len(s2)+1):
      if i == 0 and j == 0:
        tab[i][j] = True
      elif i == 0 and b[j-1] == c[j-1]:
        tab[i][j] = tab[i][j-1]
      elif j == 0 and a[i-1] == c[i-1]:
        tab[i][j] = tab[i-1][j]
      elif c[i+j-1] == a[i-1]:
        tab[i][j] = tab[i-1][j]
      elif c[i+j-1] == b[j-1]::
        tab[i][j] = tab[i][j-1]
  return tab[len(a)][len(b)]

""" 2 string, lenth,ia,ib; 3 dim, in each len, test sublen k/len-k. tab[0..k][ia][ib];
normally, one dim ary with each tab[lenth,i], j=i+len,
here, vary all 3; tab[len,i,j], break lenth in sublen, and i,j with sublen.
tab[l,i,j] = 1 iff (tab[k,i,j] and tab[l-k,i+k,j+k]
"""
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


""" min number left after remove triplet. track min left within [i..j]
[lo, i, j] is a k-spaced triplet [lo+1..i-1],[i+1..j-1]
[2, 3, 4, 5, 6, 4], ret 0, first remove 456, remove left 234.
dp[i,j] track the min left in subary i..j = min(recur(i+1, j))
"""
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

''' sub regioin [l..r], iterate k between l..r, think k as the LAST to burst. Matrix Production.
matrix product dp[l,r] = dp[l,k] + dp[k+1,r] + p[l-1]*p[k]*p[r].
dp[l,r] = max(dp[l,k] + dp[k+1,r] + arr[l]*arr[k]*arr[r])
Implementation wise, loop inc gap, for each e as l, r = l+gap+1, m=[l..r]

Enum i, j boundary index, as well as i, j boundary values.
T(i, j, left, right)  the maximum coins obtained by bursting of subarray nums[i, j] whose 
two adjacent numbers are left and right. 
The start T(0, n - 1, 1, 1) and the termination is T(i, i, left, right) = left * right * nums[i].
T(i, j, left, right) = for k in i..j as the last one to burst.
   max(left * nums[k] * right + T(i, k - 1, left, nums[k]) + T(k + 1, j, nums[k], right))
'''
static int burstBallonMaxCoins(int[] arr) {
    int sz = arr.length;
    int[] nums = new int[sz+2];
    for (int i=0;i<sz;i++) { nums[i++] = arr[i]; }
    nums[0] = nums[sz] = 1;

    int[][] dp = new int[sz][sz];  // the max value.
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
print burstBalloon([3, 1, 5, 8])


# Remove boxes max points. 
# https://leetcode.com/problems/remove-boxes/discuss/101310/Java-top-down-and-bottom-up-DP-solutions
# Enum i, j boundary index, as well as i, j boundary values. can be simplified to only consider left side.
#
# T(i, j, k) the maximum points possible by removing the boxes of subarray boxes[i, j] 
# with k boxes on the left of box[i] that are the same color as box[i].
# T(i, i, k) = (k + 1) * (k + 1);  // boundary, 1 box at i, and left side has k same color box.
# T(i, i - 1, k) = 0
# Recur i as boundary: two choices, remove box i along with the left k boxes, or merge it with m boxes to the right[i..m..j].
# remove i and left k boxes.  (k + 1) * (k + 1) + T(i + 1, j, 0)
# merge i with m same color boxes to the right, hence 0 on the left of i+1, and k+1 on the left of m.
#   max(T(i+1, m-1, 0) + T(m, j, k+1)) where i < m <= j && boxes[i] == boxes[m]. 
#   dp[i][j][k] = for m in i..j: max(dp[i+1][m-1][0] + dp[m][j][k+1])
#
#    [k] [i, i+1 ... m-1, m ...j]
static int removeBox(int[] arr) {
    int sz = arr.length;
    // i , j, with k boxes of the same color as i on its *left*
    int[][][] dp = new int[sz][sz][sz];
    for (int i=0;i<sz;i++) {
        for (int k=0;k<i;k++) {
            dp[i][i][k] = (k+1)*(k+1);
        }
    }
    for (int gap=1;gap<sz;gap++) {
        for (int j=gap;j<sz;j++) {
            int i = j-gap;   // i...k..j
            for (int k=0;k<=i;k++) {
                // merge with left k
                int mergeLeft = (k+1)*(k+1) + dp[i+1][j][0];
                // or merge with some m in the middle. loop to find the some m.
                int mergeM = 0;
                for (int m=i+1;m<=j;m++) {
                    if (arr[i] == arr[m]) {
                        mergeM = Math.max(mergeM, dp[i+1][m-1][0] + dp[m][j][k+1]);
                    }
                }
                dp[i][j][k] = Math.max(mergeLeft, mergeM);
            }
        }
    }
    return sz == 0 ? 0 : dp[0][sz-1][0];
}

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
    #qsort([363374326, 364147530 ,61825163 ,1073065718 ,1281246024 ,1399469912, 428047635, 491595254, 879792181 ,1069262793], 0, 9)
    #qsort3([1,2,3,4,5,6,7,8,9,10], 0, 9)
    #numOfIncrSeq([1,2,4,5,6], 0, [], 3)
    #pascalTriangle(10)
    #pascalTriangle_iter(10)
    #binSearch([1,2,5,9], 3)
    #binSearch([2,4,10], 1)
    #powerset([1,2,3,4,5])
    #testKthMaxSum()
    hf = HuffmanCode()
    hf.encode()
    
    #
    sqrt(8)
    sqrt(0.4)

    # test find insertion postion
    testFindInsertionPosition()
    
    # test find kth
    testFindKth()

// 
// Implement Blocking Queue
//
Condition isFullCondition;
Condition isEmptyCondition;
Lock lock;

public BQueue() { this(Integer.MAX_VALUE); }

public BQueue(int limit) {
    this.limit = limit;
    lock = new ReentrantLock();
    isFullCondition = lock.newCondition();    // generate condi var from lock
    isEmptyCondition = lock.newCondition();
}

// must hold the lock.
public synchronize get() {
  while (q.empty()) { 
    wait();
  }
  q.add();
  notifyAll();
}

public void put (T t) {
    lock.lock();
    try {
       while (isFull()) {
          try {
              isFullCondition.await();
          } catch (InterruptedException ex) {}
        }
        q.add(t);
        isEmptyCondition.signalAll();
    } finally {
        lock.unlock();
    }
 }





Improve code and design
1. Modularity design, Visitor pattern vs Type. More types Vs. More Ops.
    Op1.subtype1(), Op2.subtype1(), Op1.subtype2(), Op2.subtype()
    Type1.op1(), Type1.op2(), Type2.op1(), Type2.op2()
2. Fault tolerance, Circuit breaker.
3. Rate Limiter.
4. IPC for loose coupling.
5. Deployment, Observable.
6. adding cache in front of services, lazy loading.

