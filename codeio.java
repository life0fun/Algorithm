// # www.cnblogs.com/jcliBlogger/

// Some problem, not require simulation, but calculate the target state and 

// 1. DP recur with smaller problem with d[i-1][j-1], bottom up.
// 2. Search DFS/BFS/Permutation, start with init, each run gen with new state or new offset with visited map and memoize path.
// Enum constant values. e.g, ages, sum values, etc.

// DP : recur with smaller problem. bottom up. dp[i,j] = Math.min( dp[i-1,j-1] )
// 1. divide to left rite, recur, memoize, and merge.
// 2. reucr left when rite is defined, e.g., add 1 more row to result set, 
// 3. recur both left rite when rite is not defined, 
// 2. calculate dp[l,r,v] directly with gap, i, i+gap.

//   1. Math model, dp[i] deps on dp[i-1]/rite defined, or on all Ks between k..i ? incr gap=1..n if l..r;
//   2. recur stop check and boundary; dp[0] = 1
//   3. Edge cases / Recur stop.

// Search:  Recur with new state or offset. Permutation, backtracking, or DP sub Layer by layer zoom in.
//   1. BFS with PQ sort by price, dist.
//   2. DFS spread out from header and recur with backtrack and prune.
//   3. Permutation, each offset as head, add hd to path, recur with new offset
//   4. DP for each sub len/Layer, merge left / right.
//   5. Memoize intermediate result.

// If next state always constant or offset+1, i.e, 8 directions, 26 char, that's perm recur DFS.
// If next state deps on prev state, dp[i] = dp[i-1], that's DP.

// Two types of Recursion Track.
// 1. add rows to right segment, rite is defined, [...] at each row, enum all values. dp[i][v] = dp[i-1][v-vi]
// 2. add rows to right segment, at each row, left of moving, dp[i][v] = max(dp[k][v]  l<k<i)
// 3. recur both left and rite segment, gap from 1..n. recur(left), recur(rite)

//  ... [i-1] [i]        # only recur [0..i-1] seg, [i-1, i] is definite.
//  ... [j] - [k] - [i]  # recur both segments, [j..k] and [k..i].
//  ... [i] [i+1] - [k] - [j-1] [j]
// prepre, pre, cur, next

// dp[i,j] = dp[i-1,j-1] + dp[i,j-1] + dp[i, j-1]
// reuse row i, use temp store [j-1], set to pre after each j; 
//   temp = dp[j]; dp[j] += dp[j-1] + pre; pre = temp;

// dp[i], two loops, outer add more slots/problem set. i->n, inner enum all possible values, j->i, or v->V;
// dp[i,j], 3 loops, loop gap, loop i, and loop k or coin value between i..j.
// dp[i][j][k]: # of valid sequences of length i where: j As and k Ls, d[n][1][2]

// 1. subsetSum, combinSum, coinChange, dup allowed, reduce to dp[v] += dp[v-vi]
// dup not allowed, track 2 rows, dp[i][v] = (dp[i-1][v] + dp[i-1][v-vi])

// 1. Total number of ways, dp[i] += dp[i-1][v]
// 2. Expand. dp[i][j] = 1 + dp[i+1][j-1]
// after adding more item, rite seg is fixed, need to recur on left seg.
//   dp[i] = max( dp[j] for j in 1..i-1,)
//   for i in 1..slot:
//     for v in 1..i  or for v in 1..V
//       dp[i] += dp[i-v]  enum all possible v at slot i
//       // or max min
//       dp[i] = min(dp[j+1] + 1, dp[i])
//       dp(i) = ( dp(i-1) * ways(i) ) + ( dp(i-2) *ways(i-1, i) )
//       tot = max(prepre + cur, pre)

// 3. when left and rite segments both need to recur.  i->k, k->j, enum all 2 segs. 
// each segs recur divide to 2 segs. 
//   for gap len in i..n, for i in 1..n. 
//     dp[i][j] = min( dp[i][j-1] + dp[i+1][j] )
    
//     dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + arr[i]*arr[k]*arr[j])
//     dp[i][j][k] = for m in i..j: max(dp[i + 1][m - 1][0] + dp[m][j][k + 1]) when a[i]=a[m]
    
//     #dp[i,j,k] = max profit of arr[i..j] with k trans.
//     dp[i,j,k] = max(dp[i+m, j, k-1]+(arr[i+m]-arr[i]))

// 4. RecurDFS(offset, path) VS. DP[i,j]. two ways of thinkings.
//   a. at each offset, branch out partial value in path. recurDFS(offset+1, path).
//      we do not need incl/excl, so recur dfs at head good, like forEach child, dfs(child)
//   b. DP[i,j], divide to dp[i,k] + dp[k,j]; 3 loops, gap, i, and k.

// 4. Permutation BFS / DFS, model connected neighbors with edges. Connected Routes, Buses.
// Permutation DFS/BFS carries path, for each row, set col, recur row+1;
// However, dp[i,j] path is in dp[i-1,j-1].
// Use Perm DFS/BFS search when backtracing.

// Permutation Recur, at each slot, include/exclude. so there are 2 for loops.
//   for (i=offset, i<N; i++) { 
//     for (int j=i+1;j<N; j++) { 
//       recur(offset+1, path+a[j]);   // recur with incl/excl j at offset i.

// When do not need to consider incl/excl, DFS head, mutate, and recur directly. restore. 1 for loop.
//   hd = a[offset];
//   for( grp : groups) {
//     grp + hd;
//       if (recur(offset+1) == true) { res.add(found); }
//     grp - hd;
//   }

// DFS recur forest Trie, start with all root level nodes, within each node, DFS recur children/neighbors.
// DFS recur single array, Memoize(hd) = DFS(path +/- cur, offset+1), and Memoize(hd).

// bfs / dfs search carrying context/partial result. bfs track left length.
//   dfs(arr, l, r, partialResult) -> dfs(arr, l+i, r, pResult)

// 1. for tree, recur with context, or iterate with root and stk.
// 2. for ary, loop recur on each element.
// 3. search, dfs or bfs.
// 4. loop gap=1,2.., start = i, end = i+gap.
// 5. dp[i][j][val] = dp[i-1][j-1][val=0..v]
// 6. BFS with priorityqueue, topsort

// BFS/DFS recur search, permutation swap to get N^2. K-group, K-similar, etc.
// Reduce to N or lgN, break into two segs, recur in each seg.
// 0. MAP to track a[i] where, how many, aggregates, counts, prefix sum, radix rank, position/index. pos[a[i]] = i;
// 1. Bucket sort / bucket merge.
// 2. Merge sort, partition, divide and constant merge.
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
//       dp[i] = (dp[i-k]+1, dp[i])
//   DP is layer by layer zoom in, first, 1 layer/gap/grp/sublen, then 2, 3, K layer, etc.
//   At each layer, you can re-use k-1 layer at k, reduce DP[i,k] to DP[i]

// 9. prepre, pre, cur to track
//   max win two chars, i, i-1, prepre, pre.
//   decodeWays, look ahead. dp[1] = prepre + pre. dp[1] * ways(s.charAt(j)) + dp[0] * ways(s.charAt(j-1), s.charAt(j))
//   regMatch, look ahead and after. Recur within while loop.
// 10. left rite edge to squeeze. sliding window. track top / second in the window and switch
// 11. sliding window, record last index of occur, expand current right edge based on last occur.
// 12. Partition to find Kth, with single value, or with pair.
// """

// """
// track ? loop, update (+ contrib, - lose )? do math of props ? match, maxmin.
// track with map, value can be idx, list, or another map.
// sum: subset sum dup not allowed, combination sum dup allowed, split ary, dup allowed, etc.
// count sort.
// consecutive subary, chunk, etc, use bucket, UF, etc
// Bucket sort, Union merge. For l in 1..n.
// Sliding window + Map count, track win size, and update upon each R mov. track first/second, switch. 
//   max win two chars, i, i-1, prepre, pre. top second char.
//   while loop to squeeze left edges.

// https://leetcode.com/problems/student-attendance-record-ii/discuss/101643/Share-my-O(n)-C++-DP-solution-with-thinking-process-and-explanation
// """

// """ Intervals scheduling, TreeMap, Segment(st,ed) is interval.
// Sort by start, use stk to filter overlap. Greedy can work.
// sort by finish(start is less than finish). no stk needed. just one end/minRite var to track end. if iv.start > end, end = iv.end.
// Interval Array, add/remove range all needs consolidate new range.
// TreeMap to record each st/ed, slide in start, val+=1, slide out end, val-=1. sum. find max overlap.
// Interval Tree, find all overlapping intervals. floorKey/ceilingKey/submap/remove on add and delete.

// Trie and Suffix. https://leetcode.com/articles/short-encoding-of-words/
// ExamRoom: PQ compare segment size. poll largest segment. Break into 2, recur.
// """

// """ Graph of Nodes, Edges. DFS/BFS + PQ<int[]>. Edge = LinkedList<Set<Node>>
// Start Seed(root ? each in Forest), End condition. For each direction/neight, recur DFS(new state, path)
//   DFS(state, l,r, path) { cache[l][r] = min(for each nb (dfs(l+1,r+1)))
// - - - - - - - - - - - -// - - - - - - - - - - - -// - - - - - - - - - - - -
// From src -> dst, min path  dp[src][1111111]   bitmask as path.
//  1. dfs(child, path); start root, for each child, if child is dst, result.append(path)
//  2. BFS(PQ); seed PQ(sort by weight) with root, poll PQ head, for each child of hd, relax. offer child.
//  3. DP[src,dst] = DP[src,k] + DP[k, dst]; In Graph/Matrix/Forest, from any to any, loop gap, loop start, loop start->k->dst.
//        dp[i, j] = min(dp[i,k] + dp[k,j] + G[i][j], dp[i,j]);
//        dp[i,j] = min(dp[i,j-1], dp[i-1,j]) + G[i,j];
//  4. topsort(G, src, stk), start from any, carry stk.
//  5. dfs(state) try each not visited child state, if (dfs(child) == true) ret true. DFS ALL DONE no found, ret false;
//  6. node in path repr as a bit. END path is (111). Now enum all path values, and check from each head, the min.
// - - - - -



// Merge sort and AVL tree, riteSmaller, Stack.push(TreeMap for sort String, index, ) 
// Recursive, pass in the parent node and fill null inside recur. left = insert(left, val);
// https://leetcode.com/problems/reverse-pairs/discuss/97268/General-principles-behind-problems-similar-to-%22Reverse-Pairs%22


// The hard problems are ones with aggregations. e.g., total pairs, counts. min/max
// To group a list of emails, numbers in one bucket, use UNION-FIND
// dp[i][j] = max(dp[i][j], dp[i-1][j-1]+a[i,j])

// """  Game Permutation/backtracking search. DFS on ary must carry boundary(l, r)
// 1. 24 game, pick 2, eval, recur with new value from 2.
// 2. zuma game. needed = 3-(j-i), avail -= needed, dp[i] = min(recurDFS(newState, offset)), avail += needed.
// 3. can I win: used[i]=1, {for each not used, if(dfs(new state, path)==true) win; } ret false.
// """



// System : User Tweet, Timeline/Feed, Graph Search, Index. Query Pattern.
// Social Entities: User/Tweet/Places/,  Connections(Follow, like), Actitity=(sub Conn obj)
// fan-out, aggregation, classify, score, Rank, Prune. Merge. Rollup.
1. Partition(user, tweet, Time/Temporal), Indexing(invert indexing, term->docId, Aggregation), Replication.
2. PreCalculation. Real-time must in Memory. Disk for write. 
3. MailBox in K-V store. Push / Pull / Hybrid.
  https://www.slideshare.net/nkallen/q-con-3770885
4. NewSql: share nothing, POD base, In memory.

1. https://github.com/checkcheckzz/system-design-interview
2. Graph Search, index ranking, typeAhead, table both forward/reverse relation. Vertical Aggregator.
  https://www.facebook.com/notes/facebook-engineering/under-the-hood-building-out-the-infrastructure-for-graph-search/10151347573598920
  https://www.facebook.com/notes/facebook-engineering/under-the-hood-indexing-and-ranking-in-graph-search/10151361720763920
3. Feed Generation, MailBox and Cron.
  https://www.slideshare.net/danmckinley/etsy-activity-feeds-architecture/


Storage and Cache for scalable.
1. User Metadata, Video Metadata, User like Video. User follow User.
2. CDN, long tail.

Partition, Secondary Index, Replica, Rebalance.
1. user - tweet association is partition by primary key
2. either local secondary index for each partition, or a global secondary partition across all partitions.
3. Global secondary index itself is partitioned. and multiple index partition updates when new doc is added.
4. Partition big data, for each record, replica(Partition lead/follower) within partition.
5. 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  
// dept max salary, left outer join itself null bigger than your e.id. No group by.
SELECT D.Name AS Department , E.Name AS Employee, E.Salary 
from Department D, Employee E LEFT OUTER JOIN Employee e2 on (e2.departmentid = E.departmentid and E.salary < e2.salary) 
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
GO

// Top 3 salary, dup not allowed.
SELECT D.Name as Department, E.Name as Employee, E.Salary 
  FROM Department D, Employee E, Employee E2  
  WHERE D.ID = E.DepartmentId and E.DepartmentId = E2.DepartmentId and 
  E.Salary <= E2.Salary     // there only will have at most 3 row from joined E2 that has salary higher than me.
  group by D.ID, E.Name 
  having count(distinct E2.Salary) <= 3   // count(*)
  order by D.Name, E.Salary desc
GO


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

select id, case
    when p_id is null then 'ROOT'
    when id not in (select p_id
                    from tree
                    where p_id is not null) then 'LEAF'
    else 'INNER'
end
from tree
order by id;

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
    mergesort(head)
    mergesort(mid)
    mergeListInPlace(head, mid)  # ref to wip/pydev/sort.py for sort two list in-place.

    if head is tail:
        head.prev = head.next = None
        return head

    mid = len(tail-head)/2
    mid.prev = list2BST(head, mid.prev)
    mid.next = list2BST(mid.next, tail)
    return mid

''' func to calc sqrt on machine without floating point calc.
    first, expand to prec width, then binary search
    for n [0..1], write n as a/b, sqrt(a)/sqrt(b)
'''
def sqrt(n, prec=2):
  def intSqrt(n, prec=2):
    xn = n * pow(10, prec) * pow(10,prec)   // expand to 100^2
    beg = 2
    end = xn-1
    while beg <= end:     // bi-sect
      mid = (beg+end)/2
      if mid*mid < xn:
        beg = mid+1
      else:
        end = mid-1
    rt = beg if math.fabs(beg*beg-xn) < math.fabs(end*end-xn) else end
    print 'intSqrt :', xn, ' = ', rt
    return rt

  if n >= 1:
    return intSqrt(n, prec)
  
  numerator, denominator  = int(n*100), 100
  numsqrt = intSqrt(numerator, prec)
  denomsqrt = intSqrt(denominator, prec)
  rt = 1.0*numsqrt / denomsqrt
  print 'sqrt : ', n, ' = ', rt
  return rt

// insert into circular link list 
//   1. pre < val < next, 2. val is max or min, 3. list has only 1 element.
def insertCircle(l, val):
    if not l: return Node(val)
    if l.next == l: 
        l.next = Node(val)
        return min(l.val, val)
    pre = head
    cur = head.next
    minnode = None
    while True:
        if prev < val < cur:
            prev.next = val
            return
        // at the end, val is max or min
        elif prev > cur and val > pre or val < cur:   
            prev.next = val
            minnode = pre
            return
    return minnode


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
  if lo < hi:  // only recursion when lo < hi.
    p = partition(arr, lo, hi)
    qsort(arr, lo, p-1)  // split into 3 segs, [0..p-1, p, p+1..n]
    qsort(arr, p+1, hi)
  return arr
//  qsort([5,8,5,4,5,1,5], 0, 6)

""" put 0 at head, 2 at tail, 1 in the middle, wr+1/wb-1 """
void swap(int[] arr, int i, int j) { int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; }
int[] sortColor(int[] arr) {
    int l = 0, r = arr.length-1;
    int stidx = 0;
    int edidx = r;    // from end
    for (int i=0; i<=r; i++) {
        if (arr[i] == 0) {
            swap(arr, stidx, i);
            stidx += 1;
        } else if (arr[i] == 2) {
          swap(arr, edidx, i);
          edidx--;
          i--;    // i-- as loop will inc i
        }
    }
    System.out.println(Arrays.toString(arr));
    return arr;
}
sortColor([2,0,1])
sortColor([1,2,1,0,1,2,0,1,2,0,1])

public void rainbowSort(int[] colors, int left, int rite, int startColor, int endColor) {
  if (colorFrom == colorTo) { return; }
  if (left >= right) { return; }
  int colorMid = (colorFrom + colorTo) / 2;
  int l = left, r = right;
  while (l <= r) {
    while (l <= r && colors[l] <= colorMid) { l++;  }
    while (l <= r && colors[r] > colorMid) { r--;  }
    if (l <= r) {
      int temp = colors[l];
      colors[l] = colors[r];
      colors[r] = temp;
      l++;
      r--;
    }
  }
  rainbowSort(colors, left, r,     colorFrom,    colorMid);
  rainbowSort(colors, l,    right, colorMid + 1, colorTo);
}

// Given a huge sorted integer array A, and there are huge duplicates in the
// array. Given an integer T, inert T to the array, if T already exisit, insert
// it at the end of duplicates. return the insertion point.
//     A[15] = {2,5,7,8,8,8,8,8,8,8,8,8,8,8,9}
//     If T == 6, return 2;
//     If T == 8' return 14
static int findInsertPosition(int[] arr, int val) {
  int sz = arr.length;
  int l=0,r=sz-1;
  if (val < arr[l]) {  return 0; }
  if (arr[r] <= val) { return r+1; }
  while (l < r) {
    int m = (l+r)/2;
    if (val < arr[m]) { r = m;}
    else if (arr[m] <= val) { l = m+1;}
  }
  System.out.println("insert point is " + l);
  return l;
}
findInsertPosition(new int[]{2,5,7,8,8,8,8,9}, 8);


// return the index to the First element GreatEqual to target.
// if arr does not have element GreatEqs than V, return sz as the insertion point.
// caller needs to check if ret >= arr.size, 
static int bsect(int[] arr, int v) {
    int l = 0, r = arr.length-1;
    if (v > arr[r]) {
      return arr.length;   // do not have element great than V, insertion shall be 
    }
    while (l < r) {
        int m = (l+r)/2;
        if (arr[m] < V) {  // when using =, insert pos is to the end.
            l = m+1;
        } else {
            r = m;
        }
    }
    return l;
}
print bisect([1,1,3,5,7])

// bisect needs to check head / tail first.
// ceiling returns index to the element Great(Not Eq) than.
// if arr does not have element Great, return arr.length;
// caller needs to check if ret >= arr.size, 
static int ceiling(int[] arr, int t) {
    int l = 0, r = arr.length -1;
    if (t > arr[sz-1]) {
      return sz;   
    }
    while (l < r) {
        int m = (l+r) / 2;
        if (arr[m] <= t) {  // <=, lift l even when eqs.
            l = m+1;    
        } else {
            r = m;
        }
    }
    System.out.println("celing is " + l);
    return l;
}
floorceil([1, 2, 8, 10, 10, 12, 19], 0)
floorceil([1, 2, 8, 10, 10, 12, 19], 2)
floorceil([1, 2, 8, 10, 10, 12, 19], 28)
    
""" not to find insertion point, check which seg is mono """
static int bsearchRotate(int[] arr, int v) {
    int l = 0, r = arr.length-1;
    while (l < r) {
        int m = (l+r)/2;
        if (arr[m] == v) return m;   // avoid dup
        if (arr[l] < arr[m]) {       // left segment is not rotated
            if (arr[l] <= v && v <= arr[m]) {  // <= v  // v falls into this segment
                r = m;
            } else {
                l = m+1;
            }
        } else if (arr[m] < arr[r]) {   // rite segment is not rotated 
            if (arr[m] < v && v <= arr[r]) {  // <=
                l = m+1;
            } else {
                r = m;
            }
        } else {       // arr[l] == arr[m], lift l;
            l += 1;    // this is a dup. advance l.
        }
    }
    if (arr[l] == v) {
      return l;
    }
    return -1;
}
System.out.println(bsearchRotate(new int[]{3, 4, 5, 1, 2}, 2));


""" find min in rotated. / / with dup, advance mid, and check on each mov """
static int rotatedMin(int[] arr) {
    int l = 0;
    int r = arr.length-1;
    while (l < r) {
        while (l < r && arr[l] == arr[l+1]) l += 1;  // <-- !!!
        while (r > l && arr[r] == arr[r-1]) r -= 1;
        int m = (l+r)/2;
        if (m < r && arr[m] > arr[m+1]) {   // inverse, m > m+1
            return arr[m+1];
        }  // l and m in the same segment.
        if (arr[l] < arr[m]) {    // because when dup, u moved l/r
            l = m+1;     // m must less than r.
        } else {
            r = m;
        }
    }
    return -1
}
System.out.println(rotateMin(new int[]{4,5,0,1,2,3}));
rotatedMin(new int[]{2,2,1,2,2,2,2,2})
rotatedMin(new int[]{2,2,2,2,2,2,1,2})

""" find the max in a ^ ary,  m < m+1 """
static int findMax(int[] arr) {
    int l = 0;
    int r = arr.length-1;
    while (l < r) {
        int m = (l+r)/2;
        if (arr[m] < arr[m+1]) {
            l = m+1;
        } else if (arr[m] > arr[m+1]) {   // inverse
            r = m;
        }
    }
    return l;
}
print findMax([8, 10, 20, 80, 100, 200, 400, 500, 3, 2, 1])
print findMax([10, 20, 30, 40, 50])
print findMax([120, 100, 80, 20, 0])

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

""" merge two sorted arry """
static int[] merge(int[] arra, int[] arrb) {
    int ar = arra.length-1, br = arrb.length-1;
    int al = 0, bl = 0;
    int[] out = new int[ar+1+br+1];
    int outidx = 0;
    while (al < arra.length && bl < arrb.length) {
        if (arra[al] < arrb[bl]) { out[outidx++] = arra[al++]; }
        else {                     out[outidx++] = arrb[bl++]; }
    }
    if (al < arra.length) { System.arraycopy(arra, al, out, outidx, arra.length-al); }
    if (bl < arrb.length) { System.arraycopy(arrb, bl, out, outidx, arrb.length-bl); }
    System.out.println(Arrays.toString(out));
    return out;
}
print merge([9,8,3], [6,5])

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// linear
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
static int kadan(int[] arr) {
  int maxSoFar = arr[0], maxEndHere = arr[0];
  int curStart = 0, curEnd = 0;
  int maxStart = 0, maxEnd = 0;
  
  for (int i = 1; i < arr.length; i++ ) {
    // if add a.i to prev not as big as restart from i, then restart from i.
    if (maxEndHere + arr[i] < arr[i]) {  
      maxEndHere = arr[i];  // restart from i
      curStart = i;
    } else {
      maxEndHere += arr[i];
    }
    curEnd = i;
    if(maxSoFar < maxEndHere && curEnd - curStart >= 2) {  // subary must be at least 2 element
      maxSoFar = maxEndHere;
      maxStart = curStart;
      maxEnd = curEnd;
    }
  }
  // if not find at least 2 element, linear scan and ret max and its left or rite.
  System.out.println("max " + maxSoFar + " start " + arr[maxStart] + " end " + arr[maxEnd]);
  return maxSoFar;
}

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

""" the max num can be formed from k digits from a, preserve the order in ary.
key point is drop digits along the way."""
static int maxkbits(int[] arr, int k ) {
    int sz = arr.length;
    int left = sz - k; 
    Stack<Integer> stk = new Stack<Integer>();
    stk.add(arr[0]);
    for (int i = 1; i<arr.length;i++) {
        while (stk.size() > 0 && stk.peek() < arr[i] && left > 0) {
            stk.pop();
            left -= 1;
        }
        stk.add(arr[i]);
    }
    int tot = 0;
    for(int i=0; i<k; i++) {
        tot = tot*10 + stk.pop(i);
    }
    System.out.print("maxbits " + tot);
    return tot;
}
print reduce(lambda x,y: x*10+y, maxKbits([3,9,5,6,8,2], 2))

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


// remove dup letter, result shall be the smallest in lexicographical order among all possible results.
// 用stack 来后悔。from left -> rite, if stk top > a[i] and count[stk.top] > 0, pop stk.top as still have chance to add stk.top later.
static String removeDupLetters(String s) {
  Set<Character> inStk = new HashSet<>();
  Map<Character, Integer> counts = new HashMap<>();
  Stack<Character> keepStk = new Stack<>();

  for (int i=0;i<s.length();i++) {
    char c = s.charAt(i);
    counts.put(c, counts.getOrDefault(c, 0) + 1);
  }
  for (int i=0;i<s.length();i++) {
    char c = s.charAt(i);
    if (!inStk.contains(c)) {
      while (!keepStk.empty() && keepStk.peek() > c && counts.get(keepStk.peek()) > 0 ) {
        inStk.remove(keepStk.peek());
        keepStk.pop();   // no keep of keep stk top
      }
      keepStk.push(c);   // push current into keep stk.
      inStk.add(c);
      counts.put(c, counts.get(c)-1);  // used one, reduce
    }
  }
  String res = "";
  while(!stk.empty()) { res = stk.pop() + res; }
  return res;
}
System.out.println(removeDupLetters("cbacdcbc"));

// widx start from 1. compare against widx-1 and widx. Allow at most 2.
static void removeDup(int[] arr) {
    int l = 2;       // loop start from 3rd, widx as allows 2
    int widx = 1;    // widx 1, allow two dups
    for (int i=l; i<arr.length; i++) {
        if (arr[i] == arr[widx] && arr[i] == arr[widx-1]) {
            continue;  // skip current item when it is a dup
        }
        widx += 1;
        arr[widx] = arr[i];
    }
    System.out.println(" " + Arrays.toString(arr));
}
removeDup(new int[]{1,3,3,3,3,5,5,5,7});


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

# check pattern exhausted boundary first. check non wildcard first. for wildcard, while p=s, s++, Skip *, p+=2;
static boolean regmatch(char[] s, char[] p, int soff, int poff) {
    int ssz = s.length, psz = p.length;
    if (poff == psz) {       return soff == ssz;}
    if (p[poff+1] != '*') {  return (s[soff] == p[poff] || p[poff] == '.') && regmatch(s, p, soff+1, poff+1); }
    // pattern is wildcard, try each recursive
    while (s[soff] == p[poff] || p[poff] == '.') {
        if (regmatch(s, p, soff, poff+2)) { return true; }
        soff += 1;
    }
}

public List<String> findRepeatedDnaSequences(String s) {
  Set<Integer> words = new HashSet<>();
  Set<Integer> doubleWords = new HashSet<>();
  List<String> rv = new ArrayList<>();
  char[] map = new char[26];
  map['A' - 'A'] = 0;
  map['C' - 'A'] = 1;
  map['G' - 'A'] = 2;
  map['T' - 'A'] = 3;

  Set<Integer> temp = new HashSet<Integer>();
	Set<Integer> added = new HashSet<Integer>();
 
	int hash = 0;
	for (int i = 0; i < len; i++) {
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

// enhancement is use parent[] to track the parent for each node in the ary.
int lengthOfLIS(int[] arr) {
    int[] dp = new int[arr.length];  // set dp[i] to max.
    int lislen = 0;   // no one in
    for (int cur : arr) {
        int lo = 0, hi = lislen;     // lislen is 0 for now.
        while (lo != hi) {  
            int m = (lo + hi) / 2;
            if (dp[m] < cur)  lo = m + 1;
            else              hi = m;
        }
        dp[lo] = cur;   // will not outofbound as hi=m, and after binsearch, a[lo] bound to > than k
        if (lo == lislen) { ++lislen; }  // lo will eq to sz when x is the biggest.
    }
    return lislen;
}

// max Envelop [][] lis. 1. Arrays.sort by width, 2. Arrays.binarySearch, 3. update DP[idx], 4. return lislen.
int maxEnveLopes(int[][] envelopes) {
  if (envelopes == null || envelopes.length == 0 || enveloes[0] == null ||  envelopes[0].length != 2) return 0;
 
  Arrays.sort(envelopes, new comparator<int[]>( (a,b) -> {
    if (a[0] == b[0]) return b[1] - a[1];
    else              return a[0] - b[0];
  });
    
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

// longest mount, divide into 2 parts, up[i] and down[i], LMin and RMax. calc down first from end.
int longestMountainSequence(int[] A) {
    int N = A.length, res = 0;
    int[] up = new int[N], down = new int[N];
    for (int i = N-2; i >= 0; --i) {    // calculate down first from end
      if (A[i] > A[i + 1]) {   down[i] = down[i + 1] + 1; }
    }
    for (int i = 0; i < N; ++i) {       
        if (i > 0 && A[i - 1] < A[i]) {  up[i] = up[i - 1] + 1; }
        if (up[i] > 0 && down[i] > 0) {  res = Math.max(res, up[i] + down[i] + 1); }
    }
    return res;
}

// # subary is conti, and simple than subsequence
// # while sliding, count up first, then count down, then add up+down. Reset counters at turning point.
int longestMountainSubary(int[] arr) {
  int maxlen =0, up = 0, down = 0;
  for (int i=1;i<arr.length;i++) {
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


// # idea is, there are 2 options, swap / notswap, need to use two state to track them.
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

// no same char next to each other
from collections import defaultdict
def noConsecutive(text):
  freq = defaultdict(int)
  for i in xrange(len(text)):  freq[text[i]] += 1
  rfreq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
  segs = rfreq[0][1]   # most frequence div arr into k segments.
  sz = (1+len(text))/segs
  print rfreq, segs, sz
  if sz < 2: return "wrong"
  nextfree = [-1]*segs
  out = [-1]*len(text)
  for i in xrange(segs):
    out[i*sz] = rfreq[0][0]
    nextfree[i] = i*sz+1
  for k,v in rfreq[1:]:
    segidx = 0
    while nextfree[segidx] > sz:  segidx += 1
    if segs - segidx < v:         return "wrong"
    for cnt in xrange(v):
      out[nextfree[segidx]] = k
      nextfree[segidx] += 1
      segidx += 1
  return out

// track all char cnts. update topCharCnts. when sliding in and out. Upbeat top char cnt in the window
static int maxLenKRepeat(char[] arr, int k) {
    int sz = arr.length;
    int l = 0; r = 0;
    int[] counts = new int[26];
    int topCharCnts = 0;
    int maxLen = 0;
    for (r=0; r<sz; r++) {
        counts[arr[r] - 'A'] += 1;          // add cnt when slide in.
        topCharCnts = Math.max(topCharCnts, counts[arr[r] - 'A']);
        while (r - l + 1 - topCharCnts > k) {
            counts[arr[l] - 'A'] -= 1;     // reduce cnt when slide out
            l += 1;
        }
        maxLen = Math.max(maxLen, r - l + 1);
    }
}

// consume slided in item first, then adjust state.
static int[] slidingWinMax(int[] arr, int k) {
    int sz = arr.length;
    int[] maxarr = new int[sz];
    Stack<Integer> maxStk = new Stack<>();
    Stack<Integer> maxIdxStk = new Stack<>();
    int l = 0;
    int winsize = 0;
    int widx = 0;
    for(int i=0; i<sz; i++) {  // always while loop to move left.
        while (!maxStk.isEmpty() > 0 && arr[i] > maxStk.peek()) {  // cur > stk.peek(), no use to keep in stk, pop
            maxStk.pop();
            maxIdxStk.pop();
        }
        maxStk.add(arr[i]);
        maxIdxStk.add(i);
        if (r-l+1 >= k) {  // adjust l
            maxarr[widx++] = maxStk.firstElement();
            if (maxIdxStk.firstElement() == l) {
                maxStk.remove(0);
                maxIdxStk.remove(0);
            }
            l += 1;
        }
    }
    System.out.println("max in window " + Arrays.toString(maxarr));
    return maxarr;
}
print maxWin([5,3,4,6,9,7,2],2)
print maxWin([5,3,4,6,9,7,2],3)

// always consume a.i first, adjust state after that.
static int getSpecialSubstring(String s, int k, String charValue) {
  int normal = 0;
  int l = 0;
  int maxlen = 0;
  for(int i=0;i<s.length();i++) {
      char icode = charValue.charAt(s.charAt(i)-'a');
      if (icode == '0') {
          normal++;
      }
      if (normal > k) {   // mov l when > k.
          while (charValue.charAt(s.charAt(l)-'a') == '1') {
              l++;
          }
          l++;
          normal--;
      }
      // always upbeat on new i.
      maxlen = Math.max(maxlen, i-l+1);
  }
  return maxlen;
}
// giraffe
// 2
// 01111001111111111011111111

// track expected count and winmap cnt for each char while looping.
static int minWinString(String s, String t) {
    int[] expects = new int[26];
    int[] winmap = new int[26];
    for (int i=0;i<t.length();i++) {  expects[t.charAt[i] - 'a'] += 1; }

    int l = 0, winsize = 0, minwin = 999;
    for (int r=0; r<s.length(); r++) {
        char cur = s.charAt(i);
        int cidx = cur - 'a';
        winmap[cidx] += 1;
        if (winmap[cidx] <= expects[cidx]) { winsize += 1; }
        if (winsize == t.length()) {
            while (l < r) {     // while loop to move left edge
                int lcharidx = s.charAt(l) - 'a';
                if (winmap[lcharidx] > expects[lcharidx]) {
                    winmap[lcharidx] -= 1;   // reduce winmap cnt when sliding out.
                    l += 1;
                } else {
                    break;    // can not squeeze left anymore, break.
                }
            }
            minwin = Math.min(minwin, i-l+1);
        }
    }
    System.out.println("min win of " + s + " : " + t + " = " + minwin);
    return minwin;
}
print minwin("azcaaxbb", "aab")
print minwin("ADOBECODEBANC", "ABC")

// longest non repeat substring. sliding window(locate left/rite). when match, need to mov l = pre+1.
static String maxNoRepeat(String s) {
  if (s == null || s.length() == 0) { return s;  }
  int maxlen = 0, maxstart = 0, maxend = 0;
  int l = 0;
  Map<Character, Integer> pos = new HashMap<>();
  for(int r=0;r<s.length();r++) {
    char ch = s.charAt(r);
    if (pos.containsKey(ch)) {
      int ridx = pos.get(ch);
      if (l <= ridx) { l = ridx + 1;}  // <= as we need to advance l even when equals.
    }
    pos.put(ch, r);
    if (r - l + 1 > maxlen) {
      maxstart = l;
      maxend = r;
      maxlen = r - l + 1;
    }
  }
  System.out.println("non repeat substring - > " + s.substring(maxstart, maxend+1));
  return s.substring(maxstart, maxend+1);
}

// slide r, mov l when a.r is not needed. 
static String anagram(String arr, String p) {
  Map<Character, Integer> winMap = new HashMap<>();
  Map<Character, Integer> patternMap = new HashMap<>();
  for (int i=0;i<p.length();i++) {
    patternMap.put(p.charAt(i), patternMap.getOrDefault(p.charAt(i), 0)+1);
  }
  int l = 0;
  for(int r=0;r<arr.length();r++){
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


// min window with 3 letter words """
def minwin(arr, words):
  found, expected = defaultdict(int), defaultdict(words)
  for i in xrange(len(arr)):
    if arr[i:i+4] not in words:    continue
    w = arr[i:i+4]
    found[w] += 1
    if found[w] <= expected[w]:   cnt += 1
    if cnt == len(words):
      while lw = arr[l:l+4] not in words:
        l += 1
      while found[lw] > expected[lw]:
        found[lw] -= 1
        l += 4
      mnw = min(mnw, r-l)        
  return mnw

// track prepre, pre, cur idx of zeros '''
def maxones(arr):
  mx,mxpos = 0,0
  prepre,pre,cur = -1,-1,-1
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


// largest subary with equal 0s and 1s
// change 0 to -1, reduce to largest subary sum to 0.
// in case subary not start from head, find max(j-1) where sum[j]=sum[i]. Use hash[sum].
// http://www.geeksforgeeks.org/largest-subarray-with-equal-number-of-0s-and-1s/
static int maxSubaryEqualZeroOne(int[] arr) {
  int sz = arr.length;
  int tot = 0;
  int[] prefix = new int[sz];
  Map<Integer, Integer> sumIdxMap = new HashMap<>();
  for (int i=0;i<sz;i++) {
      if (arr[i] == 0) { arr[i] = -1; }
      tot += arr[i];
      prefix[i] = tot;
      sumIdxMap.putIfAbsent(tot, i);
  }
  int maxsz = 0;
  for (int i=sz-1;i>=0;i--) {
      if (sumIdxMap.containsKey(prefix[i])) {
          int idx = sumIdxMap.get(prefix[i]);
          if (idx != i) {
              maxsz = i - idx;
              break;
          }
      }
  }
  System.out.println("longest subary is " + maxsz);
  return maxsz;
}
maxSubaryEqualZeroOne(new int[]{1, 0, 1, 1, 1, 0, 0});

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
    for(int i=1;i<sz;i++){
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

// We need track the step k when reaching A[i], so next steps is k-1,k,k+1. stones[i] value is idx that has a stone. 
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

def minjp(arr):  // use single var to track, no need to have ary.
  mjp = 1;
  maxRightOfCurrentMinJP,nextMaxRite = 1, arr[0], arr[0]   // seed with value of first.
  for i in xrange(1,n):
    if i < maxRightOfCurrentMinJP:    
      nextMaxRite = max(nextMaxRite, i+arr[i])
    else:   // when approaching to current jp
      mjp += 1
      maxRightOfCurrentMinJP = nextMaxRite
    if maxRightOfCurrentMinJP >= n:
      return mjp


// once need aggregation value, (max/min/tot), need dp[i] : min jp from 0->i, track farest of current jp, 
import java.util.PriorityQueue;
import java.util.concurrent.locks.StampedLock;
import sys
def minjp(arr):
  dp = [sys.maxint]*len(arr)
  dp[0] = 0
  for i in xrange(1, len(arr)):  // bottom up, expand to top. 
    for j in xrange(i):
      if arr[j] + j >= i:  // can reach i
        dp[i] = min(dp[i], dp[j] + 1)
  return dp[len(arr)-1]
assert minjp([1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9]) == 3

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

// we need a heap to store and sort top k,
// # can use bucket[cnt] = [] to merge.
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

"""
max distance between two items where right > left items.
http://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
Lmin[0..n]   store min see so far. so min[i] is mono descrease.
Rmax[n-1..0] store max see so far from left <- right. so mono increase.
the observation here is, A[i] is shadow by A[i-1] A[i-1] < A[i], the same as A[j-1] by A[j].
Lmin and Rmax, scan both from left to right, like merge sort. upbeat max-distance.
"""
static int maxDist(int[] arr) {
    int sz = arr.length;
    int[] lmin = new int[sz];
    int[] rmax = new int[sz];
    int min = 999, max = 0;
    for (int i=0; i<sz; i++) {
        min = Math.min(min, arr[i]);
        lmin[i] = min;
        max = Math.max(max, arr[sz-1-i]);
        rmax[sz-1-i] = max;
    }
    int l = 0, r = 0, maxdist = 0;   // two counters to break while loop.
    // stretch r as r from left to rite is mono decrease.
    while (l < sz && r < sz) {
        if (rmax[r] >= lmin[l]) {
            maxdist = Math.max(maxdist, r-l+1);  // calculate first, mov left second.
            r += 1;
        } else {
            l += 1;
        }
    }
    System.out.println("max dist = " + maxdist);
    return maxdist;
}
maxDist(new int[]{34, 8, 10, 3, 2, 80, 30, 33, 1});


""" Lmin[i] contains idx in the left arr[i] < cur, or -1
    Rmax[i] contains idx in the rite cur < arr[i]. or -1
    then loop both Lmin/Rmax, at i when Lmin[i] > Rmax[i], done
"""
def triplet(arr):
  mn,nxtmn = 99,99
  for i in xrange(len(arr)):
    v = arr[i]
    if v < mn:       mn = v
    elif v < nxtmn:  nxtmn = v
    else:            return [mn,nxtmn,v]
print triplet([5,4,3,6,2,7])

// # Longest Consecutive Sequence. unsorted array find the length of the longest consecutive elements sequence.
// # bucket, union to merge. use the index of num as index to 
public int longestConsecutive(int[] nums) {
    UnionFind uf = new UnionFind(nums.length);
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
// each item in UnionFind has parents.
class UnionFind {
    private int[] parent;   // value is the parent of arr[i]. idx is the same as original ary.
    public UnionFind(int n) {
        parent = new int[n];
        for(int i=0; i<n; i++) { parent[i] = i; }
    }
    private int findParent(int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }
    public boolean connected(int i, int j) { return findParent(i) == findParent(j); }
    public void union(int p, int q){
      int i = findParent(p), j = findParent(q);
      parent[i] = j;
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


""" 
left < cur < rite, for each i, max smaller on left and max rite.
reduce to problem to pair, so left <- rite. find rmax  riteMax[i] = arr[i]*rmax
for left max smaller, left -> rite, build AVL tree while get left max smaller.
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

''' at every point, keep track both max[i] and min[i].
mx[i] = max(mx[i-1]*v, mn[i-1]*v, v),
at each idx, either it starts a new seq, or it join prev sum
'''
int maxProduct(int A[], int n) {
  // store the result that is the max we have found so far
  int gmax = A[0];
  // imax/imin stores the max/min product of
  // subarray that ends with the current number A[i]
  for (int i = 1, imax = gmax, imin = gmax; i < n; i++) {
      // multiplied by a negative makes big number smaller, small number bigger
      // so we redefine the extremums by swapping them
      if (A[i] < 0) swap(imax, imin);

      // max/min product for the current number is either the current number itself
      // or the max/min by the previous number times the current one
      imax = max(A[i], imax * A[i]);   // A[i]
      imin = min(A[i], imin * A[i]);

      // the newly computed max value is a candidate for our global result
      gmax = max(gmax, imax);
  }
  return gmax;
}
print prod([-6,-3])
print prod([-6,3,-4,-5,2])


""" next greater rite circular: idea is to go from rite, clean up stk when it top is no use(< arr[i])
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
    while (!stk.isEmpty() && arr[i] >= stk.peekLast()) {
      stk.pollLast();
    }
    nextGreater[i] = stk.isEmpty() ? -1 : arr[stk.peekLast()];
  }
}

// build Lmin[], and use stack from rite to left, ycompare stk peek to lmin[r], not arr[r]
boolean oneThreetwoPattern(int[] arr) {
  int sz = arr.length;
  int[] lmin = new int[sz];
  lmin[0] = arr[0];
  Stack<Integer> stk = new Stack<>();
  for (int i=1;i<sz;i++) { lmin[i] = Math.min(lmin[i-1, arr[i]); }
  for (int r=sz-1;r>=0;r--) {
    if (arr[r] > lmin[r]) {  // only when 1 < 3, then find 2.
      while (!stk.isEmpty() && stk.peek() <= lmin[r]) {   // stk top < lmin.r, no use, pop.
        stk.pop();
      }
      if (!stk.isEmpty() && stk.peek() < arr[r]) {   // after pop, if stk not empty, and its top < arr[r]
        return true;
      }
      stk.push(arr[r]);
    }
  }
}

//
// skyline problem
class Event implements Comparable<Event> {
  int id, x, ht, type;
  @Override public int compareTo(Event other) { if (this.x == other.x) {return this.type * this.x - other.type* other.x; } return this.x - other.x;
}
class HeapEntry { int id, int ht, 
  @Override public boolean equals(Object o){ 
    if(o instanceof HeapEntry){
      HeapEntry e = (HeapEntry)o;
      return id == e.id && ht == e.ht;
    }
    return false;
  }
  @Override public int hashCode() {
    int result = 17;
    result = 31 * result + id^ht;
    return result;
  }
}
Class Skyline {
  List<Event> events = new ArrayList<>();
  for (int i=0;i<arr.length;i++) { events.add(new Event(arr[i].x, arr[i].ht, arr[i].type))}
  Collections.sort(events);
  // using pq instead of stk, to be able to extract max and delete key. 
  PriorityQueue<int[]> maxHeap = new PriorityQueue<>(sz, (e1, e2) -> {e1.ht - e2.ht});
  for (Event e : events) {
    if (e.type == "entering") {
      if (e.ht > maxheap.peek()) {
        res.add(e.id, e.ht)
      }
      maxheap.add(new HeapEntry(e.id, e.ht);
    } else if (e.type == "leaving") {  // need to delete from pq.
      maxheap.remove(new HeapEntry(e.di, e.ht))
      if (e.ht > maxheap.peek()) {
        res.add(e.x, maxheap.max());
      }
    }
}

""" first missing pos int. bucket sort. L[0]=1, L[1]=2. value from 1..n
    foreach pos, while L[pos]!=pos+1: swap(L[pos],L[L[pos]-1]).
"""
def firstMissing(L):
  def bucketsort(L):
    for i in xrange(len(L)):
      while L[i] > 0 and L[i] != i+1:
        v = L[i]-1
        if L[v] < 0:  # arr[i] is a DUP as already detected and toggled
          L[i] = 0    # Li is a dup, can not swap anymore, set it to 0, skip
          L[v] -= 1
        else:         # L[v] >= 0: when dup, entry was set to 0, we still can swap to it.
          L[i],L[v] = L[v],L[i]  # swap and mark the flag as already done.
          L[v] = -1
      # natureally ordered, not even inot while loop, turn to -1
      if L[i] == i+1:
        L[i] = -1
  bucketsort(L)
  for i in xrange(len(L)):
    if L[i] == 0:  # ith has i+1 will be -1, missing replaced by dup, so 0.
      break
  return i+1
print firstMissing([4,2,5,2,1])

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

""" For each item, if it in rite place, toggle arr[i] -= 1 for count,
if it's dup and its rite place already taken, -= 1, otherwise, swap """
def bucketsort_count(arr):
  def swap(arr, i, j):   arr[i],arr[j] = arr[j],arr[i]
  for i in xrange(len(arr)):
    while arr[i] != i+1 and arr[i] >= 0 and arr[i] != 999:  # skip DUP
      v = arr[i]
      if arr[v] >= 0:    // swap and toggle!
        swap(arr, i, v)
        arr[v] = -1
      else:             // inc and sentinel
        arr[v] -= 1
        arr[i] = 999    // set cur i to SENTI to indicate it is dup
    if arr[i] == i+1:   // toggle to negate after position match.
      arr[i] = -1
  return arr
print bucket([1, 2, 4, 3, 2])

static void swap(int[] arr, int i, int j) { int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; }
static int[] buckesort(int[] arr) {
    int l = 0;
    int sz = arr.length;
    int r = sz -1;
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

""" find dup in ary using slow/fast pointers 
arr[a] is the same as a.next, after cycle meet, start from 0.
"""
def findDuplicate(arr):
  slow = fast = finder = 0
  while True:
    slow = arr[slow]
    fast = arr[arr[fast]]
    if slow == fast:
      while finder != slow:
        finder = arr[finder]
        slow = arr[slow]
      return finder

// # HIdx Ary[] whose idx is value of hIdx and value is citations; 
// # Cap hIdx Ary[] at len of papers.
static int hIndex(int[] arr) {
    int sz = arr.length;
    // num of paper viewed as largest citation cap.
    int[] citationPapersArr = new int[sz+1];  //
    for (int i=0;i<sz;i++) {
        int curCitation = arr[i];
        if (curCitation > sz) { citationPapersArr[sz] += 1; } // cap
        else {                  citationPapersArr[curCitation] += 1;  } // idx is citation, value is total papers in this cititation.
    }
    int hIndex = 0, totPapers = 0;
    for (int i=citationPapersArr.length-1;i>=0;i--) {
        totPapers += citationPapersArr[i];
        if (totPapers >= i) {
            hIndex = i;
            break;
        }
    }
    System.out.println("hIndex is " + hIndex);
    return hIndex;
}
assertEqual(3, hIndex(new int[]{3,0,6,1,5}));

// # tot = max(prepre + cur, pre)
int rob(TreeNode root) {
    int[] res = robSub(root);
    return Math.max(res[0], res[1]);
}
// # return 0 is max of entrie subtree, 1 is max of include root.
int[] robSub(TreeNode root) {
    if (root == null) return new int[2];
    
    int[] left = robSub(root.left);
    int[] right = robSub(root.right);
    // # 0 is max of entire subtree, 1 is include root.
    int[] res = new int[2];
    // # max of entire tree, include root or exclude root.
    res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
    // # max of include root
    res[1] = root.val + left[0] + right[0];
    return res;
}

""" keep track of 2 values for each ele, max of exclude this ele, and the
max of no care of whether incl or excl cur ele. """
def calPack(arr):
  premax, preexcl= arr[0], 0     // premax is the max of incl/preexcl; set to include head
  for i in xrange(1, len(arr)):  // start from 1
    inclcur = preexcl + arr[i]   // use pre preexcl
    exclcur = premax   // exclude current, use premax, preexcl become preexcl pre in another round.
    // i become pre; update state of i, as i became pre.
    preexcl = exclcur;
    premax = max(inclcur, exclcur)
  return max(premax,preexcl)

// # enum of all values, if value is in num, exam include v, means exclude v-1 when v-1 is also in the ary.
// # use include, exclude, and premax to track.
int deleteAndEarn(int[] nums) {
    int[] points = new int[10001];
    for (int n: nums) points[n] += n;   // how many points we can get if delete nums[i]
    return rob(points);
}
static int rob(int[] points){
  int premax, preexcl = points[0], 0;
  for (int i=1;i<points.length;i++) {   // at each point, take or not.
    int robi = preexcl + points[i];
    int excli = premax;
    preexcl = excli;
    premax = max(robi, excli);
  }
  return premax;
}

""" if different coins count the same as 1 coin only, the same as diff ways to stair """
def nways(steps, n):
  ways[i] = steps[i]
  for i in xrange(steps[1], n):
    for step in steps:
      ways[i] += ways[i-step]    // ways inheritated from prev round
  return ways[n-1]

static int decodeWays(int[] arr) {
    int sz = arr.length;
    int prepre = 1, pre = 1;  // prepre = 1, as when 0, it is 0-2, the first 2 digits, count as 1.
    int cur = 0;
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
  selection algorithm, find the kth largest element with worst case O(n)
  http://www.ardendertat.com/2011/10/27/programming-interview-questions-10-kth-largest-element-in-array/
"""
def selectKth(A, beg, end, k):
  def partition(A, l, r, pivotidx):
    A.r, A.pivotidx = A.pivotidx, A.r   # first, swap pivot to end
    widx = l   # all ele smaller than pivot, stored in left, started from begining
    for i in xrange(l, r):
      if A.i <= A[r]: # r is pivot value
        A.widx, A.i = A.i, A.widx      # swap a.i and A.widx
        widx += 1
    A.r, A.widx = A.widx, A.r
    return widx  # the start idx of all items > than pivot

  if beg == end: return A.beg
  if not 1 <= k < (end-beg): return None

  while True:   # continue to loop 
    pivotidx = random.randint(l, r)
    rank = partition(A, l, r, pivotidx)  // the start idx of items grter than pivot.
    if rank == k: return A.rank
    if rank < k:  return selectKth(A, l, rank, k-rank)
    else:         return selectKth(A, rank, r, k-rank)

"""
find kth smallest ele in a union of two sorted list
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
  int pa = Math.min( k>>1, arr.length);
  int pb = k - pa;
  if (arr[pa-1] < brr[pb-1]) {
    kth(Arrays.copyOfRange(arr, pa, arr.length), brr, k-pa);
  } else {
    kth(arr, Arrays.copyOfRange(brr, pb, brr.length), k-pb);
  }
}

// binary partition the shorter ary to left/rite, longer one partition by (m+n)/2 - partidx;
static double medianOfTwoSorted(int[] A, int[] B) {
  if (A.length > B.length) return medianOfTwoSorted(B, A);
  int asz = A.length, bsz = B.length;
  int lo = 0, hi = asz;   // not sz-1, but sz
  while (lo <= hi) {
    int px = (lo+hi)/2;
    int py = (asz + bsz + 1)/2 - px;  // px+py = total_size/2;
    int amaxleft = px == 0 ? -1 : A[px-1];
    int aminrite = px == asz ? Integer.MAX_VALUE : A[px];
    int bmaxleft = py == 0 ? -1 : A[px-1];
    int bminrite = px == bsz ? Integer.MAX_VALUE : B[py];

    if (amaxleft < bminrite && bmaxleft < aminrite) {
      if ((asz+bsz) % 2 == 0) {
        return (double)(Math.max(amaxleft, bmaxleft) + Math.max(aminrite, bminrite))/2;
      } else {
        return Math.max(amaxleft, bmaxleft);
      }
    } else if (amaxleft > bminrite) {
      hi = px - 1;
    } else {
      lo = px + 1;
    }
  }
  throw new IllegalArgumentException();
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// interval scheduling
// Interval TreeMap Array and Interval Tree, segment
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
static List<int[]> insertMergeInterval(List<int[]> intervals, int[] newInterval) {
  // List<Interval> result = new LinkedList<>();
  int i = 0;
  while (i < intervals.size() && intervals.get(i)[1] < newInterval[0]) {  // end < new's start, ++
    // result.add(intervals.get(i));
    i += 1;
  }
  // for each overlaping intv, start < newIntv end.
  while (i < intervals.size() && intervals.get(i)[0] <= newInterval[1]) {
    newInterval[0] = Math.min(newInterval[0], intervals.get(i)[0]);
    newInterval[1] = Math.min(newInterval[1], intervals.get(i)[1]);
    intervals.remove(i);  // do not ++i after remove.
  }
  intervals.add(i, newInterval);
  return intervals;
}

// cnt+1 when interval starts, -1 when interval ends. scan each point, ongoing += v.
class MyCalendar {
    TreeMap<Integer, Integer> tmlineMap = new TreeMap<>();
    int book(int s, int e) {
        tmlineMap.put(s, tmlineMap.getOrDefault(s, 0) + 1); // 1 new event will be starting at [s]
        tmlineMap.put(e, tmlineMap.getOrDefault(e, 0) - 1); // 1 new event will be ending at [e];
        int ongoing = 0, k = 0;               // ongoing counter track loop status
        for (int v : tmlineMap.values()) {    // iterate all intervals.
            ongoing += v;                     // update counter
            k = Math.max(k, ongoing);         // when v is -1, reduce ongoing length.
        }
        return k;
    }
}

// # longest length of interval chain, sort interval by END time. interval scheduling.
// # https://leetcode.com/problems/maximum-length-of-pair-chain/discuss/105631/Python-Straightforward-with-Explanation-(N-log-N)
// #  dp[i] = the lowest END of the chain of length i
static int longestIntervalChain(int[][] arr) {
  Arrays.sort(arr, (a, b) -> a[1] - b[1]);   // sort by end time.
  List<Integer> dp = new ArrayList<>();
  for (int i=0; i<arr.length; i++) {
    int st = arr[i][0], ed = arr[i][1];
    int idx = Arrays.binarySearch(dp, st);
    if (idx == dp.length) {       dp.add(ed); }     // append to the end
    else if (ed < dp[idx]) {      dp.set(idx,ed); }  // update ix with current smaller value
  }
  return dp.size() - 1;
}
// The other way is stk, sort by end time, and stk thru it.
if stk.top().end > curIntv.end: stk.top().end = curInterv.end;


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

// interval scheduling, non overlap, Greedy. sort by finish. use end to track ongoing end.
int eraseOverlapIntervals(Interval[] intervals) {
    if (intervals.length == 0)  return 0;

    Arrays.sort(intervals, (a, b) -> a[1] - b[1]);   // sort by finish
    int end = intervals[0].end;
    int count = 1;        
    // end to track current non overlapp end
    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i].start >= end) {
            end = intervals[i].end;    // add when no overlap.
            count++;
        }
    }
    return intervals.length - count;
}

// cut a line cross least brick.
// map key is len of rite edge and value is how many layers has that edge. edge with max layers is where cut line cross least.
int leastBricks(List<List<Integer>> wall) {
    Map<Integer, Integer> map = new HashMap();
    int count = 0;
    for (List<Integer> row : wall) {
        int rowWidth = 0;   // when start a new row, init counter
        for (int i = 0; i < row.size() - 1; i++) {   
            rowWidth += row.get(i);     // grow rowWidth for each brick in the row.
            //map.compute(rowWidth, (k, v) -> v += 1);
            map.put(rowWidth, map.getOrDefault(rowWidth, 0) + 1);
            count = Math.max(count, map.get(rowWidth));
        }
    }
    return wall.size() - count;
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


// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// tree
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
class Node {
  Node left, rite;
  int value;
  int rank;
  public Node(Node l, Node r, int v) { this.left = left; this.rite = rite; this.value = value; }
}
static Node morris(Node root) {
  Node pre;
  Node cur = root;
  Stack<Node> stk = new Stack<>();
  while (cur != null) {
      if (cur.left == null) {
          pre = cur;
          cur = cur.rite;
      } else {
          Node tmp = cur.left;
          while (tmp.rite != null && tmp.rite != cur) {
              tmp = tmp.rite;
          }
          if (tmp.rite == null) {
              tmp.rite = cur; cur = cur.left; pre = cur;
          } else {
              stk.add(cur);
              tmp.rite = null;  // have traversed left, restart
              cur = cur.rite;
              pre = cur;
          }
      }
  }
  return stk;
}

// given a root, and an ArrayDeque (initial empty). iterative. root is not pushed to stack at the begining.
public List<Integer> preorderTraversal(TreeNode root) {
  List<Integer> result = new ArrayList<>();
  Deque<TreeNode> stack = new ArrayDeque<>();
  TreeNode cur = root;
  while (!stack.isEmpty() || cur != null) {
      if(cur != null) {
          stack.push(cur);
          result.add(cur.val);  // Add before going to children
          cur = cur.left;
      } else {
          TreeNode node = stack.pop();
          cur = node.right;   // after popping from the stack, alway go rite.
      }
  }
  return result;
}

// postOrder use preorder, push root, and LinkedList.addFirst() to add it into result.
public List<Integer> postorderTraversal(TreeNode root) {
  Deque<Integer> result = new ArrayDeque<>();
  Deque<TreeNode> stack = new ArrayDeque<>();
  TreeNode cur = root;
  while(!stack.isEmpty() || cur != null) {
      if(cur != null) {
          stack.offerLast(cur);
          result.offerFirst(cur.val);    // preorder visit cur, with addFirst equals to Reverse result.reverse()
          cur = cur.right;               // always go right. Reverse the process of preorder
      } else {
          TreeNode node = stack.pop();
          cur = node.left;            // after poping from stack, go left. Reverse the process of preorder
      }
  }
  return result;
}

""" after visit cur node, you have two choices, left, rite. Stash rite, make rite to point
to left. If left is null, rite shall point to stk top.
if cur is null, pop from stack.
"""
static Node flatten(Node root) {
  Stack<Node> stk = new Stack<>();
  # iterative, while loop. not recursion, using stack.
  while (root != null || stk.size() > 0 ) {
      if (root != null) {
          if (root.rite != null) { stk.add(root.rite); }   // set aside rite to stk, will come back later.
          root.rite = root.left;
          root = root.rite;           // descend
      } 
      else {  root = stk.pop();  }
  }
  return root;
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


// iterative in order, reduce k after visited a node(dfs return, or stk pop)
int kthSmallest(TreeNode root, int k) {
    Stack<TreeNode> stack = new Stack<>();
    while(root != null || !stack.isEmpty()) {
        if (root != null) {
            stack.push(root);    
            root = root.left;   
        } else {
            root = stack.pop();  
            // after pop, done visiting the node, --k
            if(--k == 0) break;
            root = root.right;
        }
   }
   return root.val;
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
          if (cur.children[letter - 'a'] == null) { cur.children[letter - 'a'] = new TrieNode(); }  # create child branch
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
    # for each word, for each letter, find trieNode.
    for (String word: sentence.split("\\s+")) {
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
    for (char c : root.children.keySet()) {    # dfs all children
      sum += dfs(root.children.get(c));
    }
    return sum + root.value;
  }
}

// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Graph
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 


// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
// Sub Ary, SubSequence, K groups
// - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - -  - - - - - 
''' http://stackoverflow.com/questions/5212037/find-the-kth-largest-sum-in-two-arrays
    The idea is, take a pair(i,j), we need to consider both [(i+1, j),(i,j+1)] as next val
    in turn, when considering (i-1,j), we need to consider (i-2,j)(i-1,j-1), recursively.
    By this way, we have isolate subproblem (i,j) that can be recursively checked.
'''
def KthMaxSum(p, q, k):
  color = [[0 for i in xrange(len(q)) ] for i in xrange(len(p)) ]
  outl = []
  heap = MaxHeap()
  heap.insert(p[i]+q[j], 0, 0)
  color[0][0] = 1

  for i in xrange(k):
      print ' heap ', heap.toString()
      head = heap.head()
      maxv = head.val
      i, j = head.ij
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

def testKthMaxSum():
    p = [100, 50, 30, 20, 1]
    q = [110, 20, 5, 3, 2]
    KthMaxSum(p, q, 7)

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
// this is dup allowed, not correct. here we do not allow dup. Need two arry i-1, i;
static int combinationSum(int[] arr, int target) {
  int sz = arr.length;
  int[] dp = new int[target+1];
  dp[0] = 1;
  for (int i=0;i<sz;i++) {
    for (int s=arr[i];s<=target;s++) {   // enum each value at each slot
      dp[s] += dp[s-arr[i]];
    }
  }
}

// # Partition to K groups, Equal Sum Subsets, Permutation search Recur.
// # Permutation recur with k groups, For each ele, try it for each group, and recur, prune.
boolean kGroups(int[] arr, int k) {
  int sum = Arrays.stream(nums).sum();
  if (sum % k > 0) return false;
  int target = sum / k;
  Arrays.sort(nums);
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
      groups[i] += v;   // dfs recur with new state, try all until success. or
        if (permutationDFS(arr, groups, row-1, target)) return true;  // try each new state
      groups[i] -= v;   // restore before another try.
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
  int[] dp = new int[sz];          # dp[i] store the max sum up to i, groups does not matter.
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

""" Subarray Sum Equals K, traverse and memoize seen prefix sum for later to lookup. """
int subarraySumToK(int[] nums, int k) {
    int sum = 0, tot = 0;
    Map<Integer, Integer> prefixSumMap = new HashMap<>();
    prefixSumMap.put(0, 1);
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
        if (prefixSumMap.containsKey(sum - k)) { 
          tot += prefixSumMap.get(sum - k); 
        }
        prefixSumMap.put(sum, prefixSum.getOrDefault(sum, 0) + 1);
    }
    return tot;
}

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
    Arrays.sort(nums);
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
def medianmedian(A, beg, end, k):
  def partition(A, l, r, pivot):
    widx = l
    for i in xrange(l, r+1):
      if A.i < pivot:
        A.widx, A.i = A.i, A.widx
        widx += 1
    # i will park at last item that is < pivot
    return widx-1   # left shift as we always right shift after swap.

  sz = end - beg + 1  # beg, end is idx
  if not 1<= k <= sz: return None     # out of range, None.
  # ret the median for this group of 5 items
  if end-beg <= 5:   return sorted(A[beg:end])[k-1]  # sort and return k-1th item
  # divide into groups of 5, ngroups, recur median of each group, pivot = median of median
  ngroups = sz/5
  medians = []
  for i in xranges(ngroups):
    gm = medianmedian(A, beg+5*i, beg+5*(i+1)-1, 3)
    medians.append(gm)
  pivot = medianmedian(medians, 0, len(medians)-1, len(medians)/2+1)
  pivotidx = partition(A, beg, end, pivot)  # scan from begining to find pivot idx.
  rank = pivotidx - beg + 1
  if k <= rank:  return medianmedian(A, beg, pivotidx, k)
  else:          return medianmedian(A, pivotidx+1, end, k-rank)


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
    process_vertex_early(v);
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
    process_vertex_late(v);
  }
}

def two_color(G, start):
  def process_edge(u, v):
    if color[u] == color[v]:
      bipartite = false
      return
    color[v] = complement(color[u])

/**
 * 1. recursive from current root
 * 2. visit order matters, (parent > child), topology sort
 * 3. process edge
 * 4. a) not visited, process edge, set parent. recursion
 *    b) in stack, process edge. not parent, no recursion.
 *    c) visited, must be directed graph, otherwise, cycle detected, not a DAG.
 */

dfs(root, context):
  if not root: return
  instack[root] = true
  process_vertex_early(v);
  rank.root = gRanks++   // set rank
  for c in root.children:
    if not visited[c]:
      parent[c] = root
      process_edge(root, c)
      dfs(c)  // recur dfs
    else if instack[c]:
      process_edge(root, c)
    else if visited[c]:     // must be DAG, otherwise, not DAG.
      process_edge(root, c)
  visited[root] = true
  process_vertex_late(v);

def topology_sort(G):
  sorted = []
  for v in g where visited[v] == false:
    dfs(v)

  def process_vertex_late(v):
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

  // when finish all edges from node x
  process_vertex_late(v):
    if (parent[v] < 1)
      if (tree_out_degree[v] > 1)
        return;
    root = (parent[parent[v]] < 1); /* is parent[v] the root? */
    if ((reachable_ancestor[v] == parent[v]) && (!root))
      printf("parent articulation vertex: %d \n",parent[v]);
    if (reachable_ancestor[v] == v)
      printf("bridge articulation vertex: %d \n",parent[v]);
      if (tree_out_degree[v] > 0) /* test if v is not a leaf */
        printf("bridge articulation vertex: %d \n",v);

    time_v = entry_time[reachable_ancestor[v]];
    time_parent = entry_time[ reachable_ancestor[parent[v]] ];
    if (time_v < time_parent)
      reachable_ancestor[parent[v]] = reachable_ancestor[v];

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

""" keep track what node has been cloned."""
def clone(root, storeMap):
  if not storeMap.get(root):
    nroot = Node(root)
    storeMap.set(root, nroot)
  else:
    nroot = storeMap.get(root)
  for c in children(root):
    nc = clone(c, storeMap)
    nroot.children.append(nc)
  return nroot


""" 
iterative, while (cur) { pre = cur; cur = cur.left;}
recursive, root.left = root.recur(root.left); 
"""
def morris(root):
  pre, cur = None, root
  while cur:
    if not cur.left:  # easiest first, no left, down to rite directly
      out.append(cur)
      pre = cur
      cur = cur.rite
    else:
      # threading left's rite most to root rite before descend
      tmp = cur.left
      # descend to left's rite most child
      while tmp.rite and tmp.rite != cur:
        tmp = tmp.rite

      if tmp.rite == None:
        tmp.rite = cur      # threading to cur
        pre,cur = cur, cur.left  # descend to left
      else:  # already threaded, visit, reset, to rite
        out.append(cur)     # visit
        tmp.rite = None
        pre,cur = cur, cur.rite

""" AVL tree with rank.
"""
class Node {
    int val, rank;
    Node left, rite;
    Node(int val) { this.val = val; this.rank = 1;}
}
int search(Node root, long val) {
    if (root == null) { return 0; } 
    else if (val == root.val) { return root.rank; }
    else if (val < root.val) { return search(root.left, val); }
    else { return root.rank + search(root.rite, val);
}
// you can update rank upon entering during insertion, or after insert, update by getting children rank.
private Node insert(Node root, int val) {
    if (root == null) { root = new Node(val); }
    else if (val == root.val) { root.rank++; }
    else if (val < root.val)  { root.rank++; root.left = insert(root.left, val); }
    else {                                   root.rite = insert(root.right, val); }
    // or post insertion update rank
    // root.rank = root.left.rank + root.rite.rank + 1;
    return root;
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
  // # recurisve insert, after ins, must rotate. as ins is recursive, rotate is also recursive.
  // # return subtree root, and # of node smaller or equal.
  def insert(self, key):
    if self.key == key: return self
    elif key < self.key:
      if not self.left: self.left = AVLTree(key)
      else:             self.left = self.left.insert(key)
    else:
      if not self.rite: self.rite = AVLTree(key)
      else:             self.rite = self.rite.insert(key)
    # after insersion, rotate if needed.
    self.updateHeightSize()  # rotation may changed left. re-calculate
    newself = self.rotate(key)  ''' current node changed after rotate '''
    print "insert ", key, " new root after rotated ", newself
    return newself  # ret rotated new root
  int insert(int k) {
      AVLTree cur = this;
      AVLTree pre = null;
      while (cur != null) {
          pre = cur;
          if (cur.value > k) { cur = cur.left; } 
          else               { cur = cur.rite; }
      }
      // when out while, cur == null
      if (pre.value > k) { pre.left = new AVLTree(k); }
      else               { pre.rite = new AVLTree(k); }
  }
  int insert(int k) {
      if (k < value && left == null) { left = new AVLTree(k); }
      else if (k > value && rite == null) { rite = new AVLTree(k); }
      else if (k < value) { left = left.insert(k); }
      else { rite = rite.insert(k); }
      AVLTree newMe = balance();
      return newMe;
  }
  // update current node stats by asking your direct children. The children also do the same.
  def updateHeightSize(self):
    self.height = 0
    self.size = 0
    if self.left:
      self.height = self.left.height
      self.size = self.left.size
    if self.rite:
      self.height = max(self.height, self.rite.height)
      self.size += self.rite.size
    self.height += 1
    self.size += 1
  # delete a node.
  def delete(self,key):
    if key < self.key:   self.left = self.left.delete(key)
    elif key > self.key: self.rite = self.rite.delete(key)
    else:
      if not self.left or not self.rite:
        node = self.left
        if not node:
            node = self.rite
        if not node:
            return None
        else:
            self = node
      else:
        node = self
        while node.left:
            node = node.left
        self.key = node.key
        self.rite = self.rite.delete(node.key)
    self.updateHeightSize()
    subtree = self.rotate(key)
    return subtree

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
  ''' bend / left heavy subtree to ^ subtree '''
  def riteRotate(self):
    newroot = self.left
    newrite = newroot.rite
    newroot.rite = self
    self.left = newrite

    self.updateHeightSize()
    newroot.updateHeightSize()
    return newroot   # ret the new root of subtree. parent point needs to update.
  ''' bend \ subtree to ^ subtree '''
  def leftRotate(self):
    newroot = self.rite
    newleft = newroot.left
    newroot.left = self
    self.rite = newleft
    
    self.updateHeightSize()
    newroot.updateHeightSize()
    return newroot
  def searchkey(self, key):
    if self.key == key:
      return True, self
    elif key < self.key:
      if self.left:
        return self.left.search(key)
      else:
        return False, self
    else:
      if self.rite:
        return self.rite.search(key)
      else:
        return False, self
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
  
# rite Smaller, and left smaller, the same.
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

""" inverse count with merge sort """
def mergesort(arr, lo, hi):
  inv = 0
  if lo < hi:
    mid = (lo+hi)/2
    inv = mergesort(arr, lo, mid)
    inv += mergesort(arr, mid+1, hi)
    inv += merge(arr, lo, mid+1, hi)
  return inv

""" rite smaller with merge sort, when merging left and rite,
left[l] > rite[0..r-1], because we mov r when rite[r] < left[l].
so rank[l]+= r.
"""
static List<E> mergeSort(List<E> a, List<E> b) {
  int al = a.size(), bl=b.size();
  int aidx=0,bidx=0;
  List<E> out = new ArrayList<>();
  while (a.size() > 0 && b.size()) {
    if (a.get(0) >= b.get(0)) {
      out.add(a.remove(0));
    } else {
      out.add(b.remove(0));
    }
  }
  if (a.size() > 0) {
    out.addAll(a);
  } else {
    out.addAll(b);
  }
  return out;
}

def riteSmaller(arr):
  def merge(left, rite):
    ''' left rite ary is sorted and contains idx of ary '''
    out = []
    l,r = 0,0
    while l < len(left) and r < len(rite):
      lidx,ridx = left[l], rite[r]
      if arr[lidx] < arr[ridx]:  # left < rite, then rite side, [0..r] is inverse. no inverse, r=0;
        rank[lidx] += r
        out.append(lidx)
        l += 1
      else:
        out.append(ridx)
        r += 1
    for i in xrange(l, len(left)):
      out.append(left[i])
      rank[left[i]] += r
    for i in xrange(r, len(rite)):
      out.append(rite[i])
    return out  

  def mergesort(arr, l, r):
    if l == r:    return [l]
    m = (l+r)/2
    left = mergesort(arr, l, m)    # ret ary of idx of origin arr
    rite = mergesort(arr, m+1, r)
    return merge(left,rite)

  rank = [0]*len(arr)
  mergesort(arr, 0, len(arr)-1)
  return rank
print riteSmaller([5, 4, 7, 6, 5, 1])

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

""" serde of bin tree with l/r child, no lchild and rchild, append $ $"""
def serdeBtree(root):
  if not root: print '$'
  print root
  serdeBtree(root.left)
  serdeBtree(root.rite)

""" serde tree with arbi # of children """
def dfsSerde(root, out):
  print root
  out.append(root)
  for c in root.children:
    dfsSerde(c, out)
  print "$"
  out.append("$")

""" iterative with stk, stk top is parent of cur during deser"""
# 1 [2 [4 $] $] [3 [5 $] $] 
static void deserTree(InputStream in, Stack<Node> stk) {
  char hd;
  input >> hd;
  if (hd == "$") {
    stk.pop();      // one recursion is done
  } else {
    Node cur = new Node(hd);
    stk.peek().addChild(cur)
    stk.push(cur);
  }
  deserTree(in, stk);
}

static void deserTree(InputStream in) {
  Stack<Node> stk = new Stack<Node>();
  while (!in.EOF) {
    char hd = in.getChar();
    if (hd == "$") {
      root = stk.pop();
    } else {
      Node cur = new Node(hd);
      stk.peek().addChild(cur)
      stk.push(cur);
    }
  }
  return root;
} 

""" use recursion, you know which children it is while recur."""
static void serial(Node root, OutputStream out) {
  if (root == null) { ";" >> "$" >> out; return;}
  if (root) {  ";" >> root.val >> out;  }
  serial(root.left);
  serial(root.rite);
}
// recursive deserial, one recursion ends when $
static Node deserial(InputStream stream) {
  char hd;
  stream >> hd;
  if (hd == "$") { return null; }
  Node subroot = new Node(hd);
  // caller exactly knows which children it is serialing now. dfs ends upon $.
  subroot.left = deserial(stream);
  subroot.rite = deserial(stream);
  return subroot;
}

static Stream<Map.Entry<K,V>> sortMap(int[] arr) {
  Map<Integer, Integer> counts = new HashMap<>();
  for (int i=0;i<arr.length;i++) {
    counts.put(arr[i], counts.getOrDefault(arr[i], 0)+1);
  }
  Stream<Map.Entry<Integer,Integer>> stream = counts.entrySet().stream().sorted(Map.Entry.comparingByValue());
  stream.forEach(e -> System.out.println(e.getValue()));
  return stream;
}

// triplet inversion. use TreeMap as avl tree, headMap / tailMap / subMap.
static long maxInversions(List<Integer> prices) {
  TreeMap<Integer, Integer> tree = new TreeMap<>();
  int[] leftBig = new int[prices.size()];
  int[] riteSmall = new int[prices.size()];
  for (int i=0;i<prices.size();i++) {
      Map<Integer, Integer> sub = tree.tailMap(prices.get(i), false);
      // include duplicates.
      for (int v : sub.values()) {
          leftBig[i] += v;
      }
      tree.put(prices.get(i), tree.getOrDefault(prices.get(i), 0) + 1);
  }
  tree.clear();
  for (int i=prices.size()-1;i>=0;i--) {
      Map<Integer, Integer> sub = tree.headMap(prices.get(i), false);
      for (int v : sub.values()) {
          riteSmall[i] += v;
      }
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

// - - - - - - - - - - - -// - - - - - - - - - - - -// - - - - - - - - - - - -
// From src -> dst, min path dp[src][1111111]   bitmask as path.
//  1. dfs(child, path); start root, for each child, if child is dst, result.append(path)
//  2. BFS(PQ); seed PQ(sort by weight) with root, poll PQ head, for each child of hd, relax by offering child.
//  3. DP[src,dst] = DP[src,k] + DP[k, dst]; In Graph/Matrix/Forest, from any to any, loop gap, loop start, loop start->k->dst.
//        dp[i, j] = min(dp[i,k] + dp[k,j] + G[i][j], dp[i,j]);
//        dp[i,j] = min(dp[i,j-1], dp[i-1,j]) + G[i,j];
//  4. topsort(G, src, stk), start from any, carry stk.
//  5. dfs(state) try each not visited child state, if (dfs(child) == true) ret true. DFS ALL DONE no found, ret false;
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
  def topsort(G, src, stk):  # sort result in a stk
    def dfs(s,G,stk):
      for v in neighbor(G,s):
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

// BFS with PQ sorted by price. Queue<int[]> pq. poll, add nb, relax with reduced stops.
int cheapestPriceWithKStops(int[][] flights, int src, int dst, int k) {
    int sz = flights.length;
    Map<Integer, HashMap<Integer, Integer>> pricesGraph = new HashMap<>();
    for (int[] f : flights) {  
        // f[0] = src, f[1] = dst, f[2] = price
        pricesGraph.computeIfAbsent(f[0], new HashMap<>()).put(f[1], f[2]);
    }
    // pq order by price, e[0], entry is ary of [price, src, stops].
    Queue<int[/* price, src, stop */]> pq = new PriorityQueue<>((a,b) -> Integer.compare(a[0], b[0]));
    pq.offer(new int[]{0, src, k+1});   // seed PQ with start and k steps, reduce it on each step.
    while (!pq.isEmpty()) {
        int[] hd = pq.poll();
        int price = hd[0], city = hd[1], stops = hd[2];
        if (hd[1] == dst) { return price; }
        if (stops > 0) {  // when still within k stops, enum all hd's neighbor
            Map<Integer, Integer> adj = pricesGraph.getOrDefault(city, new HashMap<>());
            for (int nb : adj.keySet()) {     // sort by price, we do not filter unseen neighbors ?
                pq.add(new int[]{price + adj.get(nb), nb, stops-1});   # reduce stop when adding.
            }
        }
    }
    return -1;
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

// segment tree. first segment 1 cover 0-n, second segment covers 0-mid, seg 3, mid+1..n. Segment num is btree, child[i] = 2*i,2*i+1.
// search always starts from seg 0, and found which segment [lo,hi] lies.
import math
import sys
class RMQ(object):  # RMQ has heap ary and tree. Tree repr by root Node.
  class Node():
    def __init__(self, lo=0, hi=0, minval=0, minvalidx=0, segidx=0):
      self.startIdx = startIdx
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
    left,lidx = self.build(startIdx, mid, 2*segRootIdx+1)    # lc of segRootIdx
    rite,ridx = self.build(mid+1, endIdx, 2*segRootIdx+2)  # rc of segRootIdx
    minval = min(left.minval, rite.minval)
    minidx = lidx if minval == left.minval else ridx
    n = RMQ.Node(startIdx, endIdx, minval, minidx, segRootIdx)
    n.left = left
    n.rite = rite
    self.heap[segRootIdx] = n
    return [n,minidx]
  # segment tree in bst, search always from root.
  def rangeMin(self, root, startIdx, endIdx):
    # out of cur node scope, ret max, as we looking for range min.
    if startIdx > root.endIdx or endIdx < root.startIdx:
      return sys.maxint, -1
    if startIdx <= root.startIdx and root.endIdx <= endIdx:
      return root.minval, root.minvalidx
    lmin,rmin = root.minval, root.minval
    # ret min of both left/rite.
    if root.left:
      lmin,lidx = self.rangeMin(root.left, startIdx, endIdx)
    if root.rite:
      rmin,ridx = self.rangeMin(root.rite, startIdx, endIdx)
    minidx = lidx if lmin < rmin else ridx
    return min(lmin,rmin), minidx
  # segment tree in heap tree. always search from root, idx=0
  def queryRangeMinIdx(self, idx, lo, hi):
    node = self.heap[idx]
    if lo > node.hi or hi < node.lo:
      return sys.maxint, -1
    if lo <= node.lo and hi >= node.hi:
          return node.minval, node.minvalidx
      lmin,rmin = node.minval, node.minval
      if self.heap[2*idx+1]:
          lmin,lidx = self.queryRangeMinIdx(2*idx+1, lo, hi)
      if self.heap[2*idx+2]:
          rmin,ridx = self.queryRangeMinIdx(2*idx+2, lo, hi)
      minidx = lidx if lmin < rmin else ridx
      return min(lmin,rmin), minidx

def build(self, arr, lo, hi):
  if lo > hi:
    return None
  node = RMQ.Node(lo, hi)
  if lo == hi:
    node.sum = arr[lo]
  else:
    mid = (lo+hi)/2
    node.left = self.build(lo, mid)
    node.rite = self.build(mid+1, hi)
    node.sum = node.left.sum + node.rite.sum
  return node
def update(self, idx, value):
  if self.lo == self.hi:
    self.sum = value
  else:
    m = (node.lo + node.hi)/2
    if idx <= m:
      update(self.left, idx, value)
    else:
      update(self.rite, idx, value)
    self.sum = self.left.sum + self.rite.sum
def sumRange(self, lo, hi):
  if self.lo == lo and self.hi == hi:
    return self.sum
  m = (node.lo + node.hi)/2
  if hi <= m:
    return sumRange(self.left, lo, hi)
  elif lo >= m+1:
    return sumRange(self.rite, lo, hi)
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
        if not cur.next[c]:      cur.next[c] = Trie(c)
        cur = cur.next[c]
      cur.leaf = True
      cur.word = w
  def dfs(board, i, j, node):
    c = board[i][j]
    nxtnode = node.next[c]
    if not nxtnode or c == "visited":   return
    if nxtnode.leaf:
      out.append(nxtnode.word)
      return
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


// find the diameter(width) of a bin tree.
// diameter is defined as the longest path between two leaf nodes.
// diameter = max(diameter(ltree, rtree) = max( depth(ltree) + depth(rtree) )
//     http://www.cs.duke.edu/courses/spring00/cps100/assign/trees/diameter.html
def diameter(root):
    if not root:
        return [0, 0]  # first is diameter, second is ht
    [ldiameter, ldepth] = diameter(root.lchild)
    [rdiameter, rdepth] = diameter(root.rchild)
    return [max(ldepth+rdepth+1, max(ldiameter, rdiameter)), max(ldepth, rdepth)+1]
def diameter(root):
    if not root.lchild and not root.rchild:
        return 0, [root, root], [root] # first diameter, second diamter leaf, last ht path
    ldiameter, ldialeaves, lhtpath = diameter(root.lchild)
    rdiameter, rdialeaves, rhtpath = diameter(root.rchild)
    if max(ldiameter, rdiameter) > max(ldepth+rdepth)+1:
        leaves = ldialeaves
    else:
        leaves = [last(lhtpath), last(rhtpat)]
    return max(ldepth+rdepth+1, max(ldiameter, rdiameter)), leaves, max(ldepth, rdepth)+1


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

// insertion is a procedure of replacing null with new node !
def insertBST(root, node):
  if not root.left and node.val < root.val:
      root.left = node
  if not root.right and node.val > root.val:
      root.right = node
  if root.left and root.val > node.val:
      insertBST(root.left, node)
  if root.right and root.val < node.val:
      insertBST(root.right, node)

def common_parent(root, n1, n2):
  if not root: return None
  if root is n1 or root is n2: return root
  left = common_parent(root.left, n1, n2)
  rite = common_parent(root.rite, n1, n2)
  if left and rite:
      return root
  return left if left else rite

// convert bst to link list, isleft flag is true when root is left child of parent 
// test with BST2Linklist(root, False), so the left min is returned.
// recur with context, if is left child, return right most, else, return left min '''
def BST2Linklist(root, is_left_child):
  // leaf, return node itsef
  if not root.left and not root.rite:
      return root
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
      while maxrite.rite:
          maxrite = maxrite.rite
      return maxrite
  else:
      // ret left min
      minleft = root
      while minleft.left:
          minleft = minleft.left 
      return minleft

// max of left subtree, or first parent whose right subtree has me
def findPrecessor(node):
  if node.left:                return findLeftMax(node.left)
  while root.val > node.val:   root = root.left  // smaller preced must be in left tree
  while findRightMin(root) != node:  root = root.right
  return root

def delBST(node):
  parent = findParent(root, node)
  if not node.left and not node.right:
      parent.left = null
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


// find the # of increasing subseq of size k, just like powerset, recur i+1 with the new path '''
def numOfIncrSeq(l, start, cursol, k):
    if k == 0:  print cursol  return
    if start + k > len(l):    return
    if len(cursol) > 0:       last = cursol[len(cursol)-1]
    else:                     last = -sys.maxint
    for i in xrange(start, len(l), 1):
        if l[i] > last:
            tmpsol = list(cursol)
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


def longestIncreasingSequence(arr):
  def bisectUpdate(l, val):
    lo,hi = 0, len(l)-1
    while lo != hi:
      mid = (lo+hi)/2
      if val > l[mid]:
        lo = mid+1     # lo will > sz is val is biggest
      else:
        hi = mid
    // l[lo] bound to great, here we need to ensure l[] is big.
    l[lo] = val # update lo with smaller
    return lo
  parent = []
  lis = []
  lis.append([0, arr[0]])
  maxlen = 0
  for i in xrange(1,len(arr)):
    if arr[i] > lis[-1][1]:
      parent[i] = lis[-1][0]
      lis.append([i,arr[i]])
    else:
      lo = bisectUpdate(lis, arr[i])
      lis[lo] = [i, arr[i]]
      parent[i] = lis[lo-1][0]
  i = lis[-1]
  while i:
    print arr[i]
    i = parent[i]
  return lis

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


""" allow 2 transactions, so maxp = max{one transaction on 0..i, one trans on i+1..n}
left trans end at i, max(l[i-1], S[i]-valley)
rite trans start at i, max(r[i+1], peak-S[i])
"""
def stock_profit(L):
    sz = len(L)
    lp,up=[0]*sz,[0]*sz
    valley,peak=sys.maxint,0
    for i in xrange(sz):
        valley = min(valley, L[i])
        lp[i] = max(lp[i-1], L[i]-valley)  # max of exclude cur, or include cur.
        // price peak going up from n..0
        j = sz-i-1
        peak = max(peak, L[j])
        up[j] = max(up[j+1], peak-L[j])  # buy current, and sale at peak.
    for i in xrange(sz):
        tot = max(tot, lp[i] + up[i])
    return tot
print stock_profit([10, 22, 5, 75, 65, 80])

""" 2 trans. left <- rite, buy at i sell at rmax. left -> rite, buy at valley sell at i. """
def profit(arr):
  sz = len(arr)
  profit=[0]*sz
  rmax = arr[-1]
  # profit[i] carry max profit from [i..n]
  for i in xrange(sz-2,0,-1):
    rmax = max(rmax, arr[i])
    profit[i] = max(profit[i+1], rmax-arr[i])
  lmin = arr[0]
  # max profit of [0..i] = max(maxprofit of [0..i-1], sell_at_i + maxprofit of [i..n])
  for i in xrange(1,len(arr)):
    lmin = min(lmin, arr[i])
    profit[i] = max(profit[i-1], profit[i]+arr[i]-lmin)
  return profit[sz-1]
print 

// hold, the maximum profit we could have if we owned a share of stock.
int maxProfit(int[] prices, int fee) {
    int cash = 0, hold = -prices[0];
    for (int i = 1; i < prices.length; i++) {
        cash = Math.max(cash, hold + prices[i] - fee);
        hold = Math.max(hold, cash - prices[i]);
    }
    return cash;
}

""" many times. use bot flag. if not bot and i+1 > i, then buy at i. 
if bot, and i+1<i, then sell. sum up every incr segment.(i,j)
"""
def profit_many_trans(arr):
  buy,sz,bot = 0,0,len(arr),False
  profit = 0
  for i in xrange(sz-1):
    # look ahead. buy today only when tomo high(we can sell)
    if not bot and arr[i] < arr[i+1]:
      buy = i
      bot = True
      continue
    if bot and arr[i] > arr[i+1]:  # sell when look ahead down.
      profit += arr[i]-arr[buy]
      bot = False
  if bot and arr[sz-1] > arr[buy]:
    profit += arr[sz-1]-arr[buy]
  return profit
print profit_many_trans([2,4,3,5,4,2,8,10])

""" FSM bot[i]/sold[i]. arr[i], update bot/sold, stash prebot before updat bot.
bot[i]=max(bot[i-1],sold[i-1]-arr[i]),  ; calc bot first.
sold[i]=max(sold[i-1], bot[i-1]+arr[i]) 
"""
def profit_many_trans(arr):
  bot,sold = -sys.maxint, 0
  for i in xrange(len(arr)):
    prebot = bot
    bot = max(bot, sold-arr[i])
    sold = max(prebot + arr[i], sold)
  return sold
print profit_many_trans([2,4,3,5,4,2,8,10])

""" FSM, 3 states, bot/sold/rest. Each state profit at arr[i], so bot[i]/sold[i]/rest[i].
bot[i] = max(bot[i-1], rest[i-1]-arr[i]); not buy at i, or buy with rest money.
sold[i] = max(sold[i-1], bot[i-1]+arr[i]), not sell, or sell
rest[i] = sold[i-1], because cooldown. it is presold
as bot[i] only relies on bot[i-1], can use one var.
"""
import sys
def stock_profit_cooldown(arr):
  bot,sold,rest = -sys.maxint, 0, 0
  for i in xrange(len(arr)):
    presold = sold
    sold = max(sold, bot+arr[i])
    bot = max(bot, rest-arr[i])
    rest = presold
  return max(sold, rest)
print stock_profit_cooldown([1, 2, 3, 0, 2])


""" Atmost up-to K transactions. max profit may involve less than k trans.
2 state, bot[trans]: money left after bot with k trans, sell[i]. 
At each arr[i], try all trans 1..k.
bot[i] only relies on bot[i-1], so bot[i][k] can be simplied to bot[k]
For each price, bottom up 1 -> k transations. use k+1 as k deps on k-1. k=0.
sell[i][k] is max profit with k trans at day i.
sell[i] = max(sell[i], bot[i]+p), // carry over prev num in trans i'th profit.
bot[i] = max(bot[i], sell[i-1]-p) // max of incl/excl bot at this price, 
"""
import sys
def topk_profit(arr, k):
  sell,bot=[0]*(k+1), [-9999]*(k+1)
  for i in xrange(len(arr)):     # for each day, try bottom up 1..k transactions.
    for trans in xrange(1,k+1):  # bottom up trans at each day.
      sell[trans] = max(sell[trans], bot[trans]+arr[i])
      bot[trans] = max(bot[trans], sell[trans-1]-arr[i])
    print arr[i], bot, "  ", sell
  return sell[k]
print topk_profit([1, 4, 5, 6, 7, 8, 9, 10], 2)
print topk_profit([1, 4, 5, 6, 7, 8, 4, 10], 2)
#print topk_profit([10, 22, 5, 75, 65, 80], 2)

""" with exact k trans, tab[i,j,k] = max profit of arr[i..j] with k trans.
tab[i,j,k] = max(tab[i+x, j, k-1]+(arr[i+x]-arr[i]))
the same as dag with exactly k edges
"""
def profit_topk(arr):
  """ find successive transaction pairs, enum all possible trans,
  and pick the top k trans """
  def valleyPeakPair(arr, st):
    v = st
    while v+1 < len(arr) and arr[v] > arr[v+1]:   v += 1
    p = v+1
    while p+1 < len(arr) and arr[p] < arr[p+1]:   p += 1
    if v == p:
      return False, []
    return True, [v,p]

  pairs,profit = [],[]
  v = 0
  found = True
  while found:
    found, [v,p] = valleyPeakPair(arr, v)
    # when prev trans's valley bigger than this valley, no consolidate
    while len(pairs) > 0 and arr[pairs[-1][0]] > arr[v]:
      profit.append(arr[pairs[-1][1]] - arr[pairs[-1][0]])
      pairs.pop()
    # when pre trans valley smaller than this valley, can combine trans.
    while len(pairs) > 0 and arr[pairs[-1][0] < arr[v]]:
      profit.append(arr[pairs[-1][1]] - arr[v])
      v = pairs[-1][0]
      pairs.pop()
    pairs.append([v,p])
  while len(pairs) > 0:
    profit.append(arr[pair[-1][1]] - arr[pairs[-1][0]])
    pairs.pop()

  # calculate the top k profits
  if len(profits) > k:
    profits = sorted(profilts)[n-k:]
  return reduce(lambda t,c: t += c, profits)


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

# for each bar x, find the first smaller left, first smaller right,
# area(x) = first_right_min(x) - first_left_min(x) * X, largestRectangle([6, 2, 5, 4, 5, 1, 6])
# A stk of increase, so when push to stk, i's first left min is known, when pop, its first right min known.
def largestRectangle(l):
  maxrect = 0
  l.append(0)  # sentinel
  l.insert(0, 0)
  stk = []
  first_left_min = [0 for i in xrange(len(l))]
  first_right_min = [len(l) for i in xrange(len(l))]
  first_left_min[0] = 0
  stk.append(0)  # 
  for i in xrange(1,len(l)):
    curval = l[i]
    topidx = stk[-1]   # use a stk to track max
    topval = l[topidx]
    # push to stk when current > stk top, so first smaller left of current set.
    if curval > topval:    
      first_left_min[i] = topidx
    # when cur smaller than stk top, stk top right is known, continue pop stk.
    elif curval <= topval:
      first_right_min[topidx] = i  # stk top is peak. update its left/right.
      stk.pop()
      while len(stk) > 0:
        topidx = stk[-1]
        topval = l[topidx]
        if topval > curval:  # continue pop stk if cur is first smaller right of stk top
          first_right_min[topidx] = i
          stk.pop()
        else:
          first_left_min[i] = topidx
          break
    stk.append(i)  # stk in current

  print first_left_min
  print first_right_min
  for i in xrange(len(l)):
    maxrect = max(maxrect, l[i]*(first_right_min[i]-first_left_min[i]-1))
  return maxrect

"""
Every row in the matrix can be viewed as the ground. height is the count
of consecutive 1s from the ground. Iterate each ground.
htMatrix[row] = [2,4,3,0,...] the ht of each col if row is the ground
"""
def maxRectangle(matrix):
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


static int getMaxArea(int arr[], int n) {
    // Create an empty stack. The stack holds indexes of arr[] array
    // The bars stored in stack are always in increasing order of their
    // heights.
    Stack<Integer> s = new Stack<>();
        
    int max_area = 0; // Initialize max area
    int tp;  // To store top of stack
    int area_with_top; // To store area with top bar as the smallest bar
    
    // Run through all bars of given histogram
    int i = 0;
    while (i < n)
    {
        // If this bar is higher than the bar on top stack, push it to stack
        if (s.empty() || arr[s.peek()] <= arr[i])
            s.push(i++);
        else{
            tp = s.peek();  // store the top index
            s.pop();  // pop the top
  
            // Calculate the area with arr[tp] stack as smallest bar
            area_with_top = arr[tp] * (s.empty() ? i : i - s.peek() - 1);
  
            // update max area, if needed
            if (max_area < area_with_top)
                max_area = area_with_top;
        }
    }
    
    // Now pop the remaining bars from stack and calculate area with every
    // popped bar as the smallest bar
    while (s.empty() == false)
    {
        tp = s.peek();
        s.pop();
        area_with_top = arr[tp] * (s.empty() ? i : i - s.peek() - 1);
  
        if (max_area < area_with_top)
            max_area = area_with_top;
    }
  
    return max_area;
}

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


""" integer partition problem. break to subproblem, recur equation, start from boundary.
    for item in [0..n], so the first for will solve single item case, then bottom up.
    fence partition, m[n, k] = min(max(m[i-1,k-1], sum[i..n])) for i in [0..n]
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
def knapsack(arr, varr, W):
  prerow, row = [0]*(W+1), [0]*(W+1)
  for i in xrange(len(arr)):
    row = prerow[:]
    w,v = arr[i], varr[i]
    for ww in xrange(w, W+1):
      if i == 0:     # init boundary, first 
        row[w] = v
      else:
        prev = prerow[ww-w]
        if prev+v > row[ww]:
          row[ww] = max(row[ww], prev+v)
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

// comb can exam head, branch at header incl and excl
def combination(arr, path, remain, pos, result):
  if remain == 0:
    result.append(path)  # add to result only when remain
    return result
  if pos >= len(arr):
    result.append(path)
    return result
  l = path[:]  # deep copy path before recursion
  l.append(arr[pos])
  combination(arr, l,    remain-1, pos+1, result)
  combination(arr, path, remain, pos+1, result)
  return

def comb(arr,r):
  res = []
  if r == 1:  # leaf, need ret a list, wrap leaf as first element.
    for e in arr:
      res.append([e])
    return res
  # incl/excl each hd
  for i in xrange(len(arr)-r+1):
    hd = arr[i]
    narr = arr[i+1:]
    for e in comb(narr, r-1):
      e.append(hd)
      res.append(e)
  return res
print comb("abc", 2)

""" recur with partial path, for loop each, as incl/excl """
def comb(arr, off, remain, path, res):
  if remain == 0:  # stop recursion when remain = 1
    cp = path[:]
    res.append(cp)
    return res
  for i in xrange(off, len(arr)-remain+1):  # can be [off..n]
    path.append(arr[i])
    comb(arr, i+1, remain-1, path, res)  # iter from idx+1, as not 
    path.pop()   # deep copy path or pop
  return res
res=[];comb([1,2,3,4],0,2,[],res);print res;


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
    powerset(arr, i+1, l, result)   # recur with i-1
path=[];result=[];powerset([1,2,2], 0, path, result);print result;


"""when recur to off, incl pos with passed in path, or skip off.
for dup, sort, skip current i is it is a dup of offset.
recur fn params must be subproblem with offset and partial result.
"""
def perm(arr, offset, path, res):
  def swap(arr, i, j):  arr[i],arr[j] = arr[j],arr[i]
  if offset == len(arr):
    return res.append(path[:])
  ''' perm, each one can be the head at offset, swap, recur. '''
  for i in xrange(offset, len(arr)):
    ''' if i follow offset and is dup of *offset*, skip it. not compare to i-1 '''
    if i > offset and arr[i] == arr[offset]:   # not arr[i-1], but arr[offset]
      continue
    swap(arr, offset, i)
    path.append(arr[offset])
        # when recur, both arr, path and offset changed as subproblem
        perm(arr, offset+1, path, res)  # recur with offset+1
    path.pop()
    swap(arr, offset, i)
path=[];res=[];perm(list("123"), 0, [], res);print res
path=[];res=[];perm(list("112"), 0, [], res);print res

# permutation recur with offset+1 and path.
void permutation(int[] arr, int offset, List<Integer> path, List<List<Integer>> res, boolean [] used) {
    if (offset == arr.length) {
      List<Integer> clonePath = new ArrayList<>(path);  # need to clone
      res.add(clonePath);
      return;
    }
    for (int i=offset; i<arr.length; i++) {
      # if(used[i] || i > 0 && nums[i] == nums[i-1] && !used[i - 1]) continue;
      # used[i] = true; 
      swap(arr, offset, i);         
      path.add(arr[offset]);        # append to path, recur
        permutation(arr, offset+1, path, res);
      swap(arr, i, offset); 
      path.remove(path.size()-1);   # restore
    }
  }
}
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
    # tot perms i..n. arr[i]'s rank
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

//  A good example to see difference between recur dfs memorize VS. dp
//  target sum, with two operations. add/sub, recur with each at header.
//  recur i+1 with add/sub
int targetSumRecurDFS(int[] nums, int offset, int sum, int S, int[][] memo) {
    if (offset == nums.length) { return sum == S ? 1 : 0;   } 
    else {
        if (memo[offset][sum + 1000] != Integer.MIN_VALUE) {    // cache hits
          return memo[offset][sum + 1000];
        }
        int add      = targetSumRecurDFS(nums, offset+1, sum + nums[offset], S, memo);
        int subtract = targetSumRecurDFS(nums, offset+1, sum - nums[offset], S, memo);
        memo[offset][sum + 1000] = add + subtract;
        return memo[offset][sum + 1000];
    }
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

""" adding + in between. [1,2,0,6,9] and target 81.
dp bottom up, rite segment, [i-k,k] is defined, so only recur left segment, [0..k].
bottom up row, for each j to i that forms val, then for each val 
"""
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

""" first, find unbalance(l-r), traverse each pos recur dfs with excl/incl each pos, no dp short """
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

# https://pathcompression.com/sinya/articles/4
# Let f(i) be the length of the longest valid substring starting from the i th character. 
# Then the final solution is the largest of all possible f(i).
# If the i th character is ‘)’, any substrings starting from the i-th character must be invalid. Thus f(i) = 0.
# If i th character is ‘(’. First we need to find its corresponding closing parentheses. 
# If a valid closing parentheses exists, the substring between them must be a valid parentheses string. 
# Moreover, it must be the longest possible valid parentheses string starting from (i+1) th character. 
# Thus we know that the closing parentheses must be at (i+1 + f(i+1)) th character. 
# If this character is not ‘)’, we know that we can never find a closing parentheses and thus f(i) = 0.
# If the (i+1 + f(i+1)) th character is indeed a closing parentheses. 
# To find the longest possible valid substring, we need to keep going and see if we can 
# concatenate more valid parenthesis substring if possible. 
# The longest we can go is f(i+1 + f(i+1) + 1). Therefore we have
# f(i) = f(i+1) + 2 + f(i + f(i+1) + 2)

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

# push idx of ( into stk, not the value, value can be dup. and we only push ( into stack.
static int longestValidParenth(String parenth) {
    int sz = parenth.length();
    int maxlen = 0;
    Stack<Integer> stk = new Stack<Integer>();
    for (int i=0;i<sz;i++) {
        if (parenth.charAt(i) == '(') { stk.add(i); }     // push in index
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

''' bottom up, loop i=1..n,  dp[i] = dp[j] + a[j..i] '''
def wordbreak(word, dict):
  dp = [None]*len(word)
  for i in xrange(len(word)):
    dp[i] = []   # init result for ith here
    for j in xrange(i):
      wd = word[j:i+1]
      if wd in dict:
        if j == 0:  # check j=0 to avoid dp[j-1] index out
          dp[i].append([wd])
        elif dp[j-1] != None:
          for e in dp[j-1]:
            tmp = e[:]
            tmp.append(wd)
            dp[i].append(tmp)
  return dp[len(word)-1]
#print wordbreak("leetcodegeekforgeek",["leet", "code", "for","geek"])
print wordbreak("catsanddog",["cat", "cats", "and", "sand", "dog"])

''' top down, dp[i] means arr[i:n] is breakable. dp[i] = dp[i+k] and arr[i:i+k] in dict '''
def wordbreak(word, dict):
  dp = [None]*len(word)
  for i in xrange(len(word)-1, -1, -1):
    dp[i] = []
    for j in xrange(i+1,len(word)):
      wd = word[i:j+1]
      if wd in dict:
        if j == len(word)-1:
          dp[i].append([wd])
        elif dp[j+1] != None:
          for e in dp[j+1]:
            tmp = e[:]
            tmp.append(wd)
            dp[i].append(tmp)
  return dp[0]
print wordbreak("catsanddog",["cat", "cats", "and", "sand", "dog"])

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
    Set<String> seen= new HashSet<>();
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
column is src, and inc bottom up each char in target. only 1 char in target, 2, ..
two cases, when not eq, excl s[j], take j-1, tab[i,j-1], when incl s[j], +=tab[i-1,j-1], 
  \ a  b   a   b  c
 a  1  1   2   2  2
 b  0 0+1  1  2+1 3
 c  0  0   0   0  3
"""
// increase target slot, traverse origin src when expanding each new item in problem.
static int distinctSeq(String s, String t) {
    int srclen = s.length();
    int tgtlen = t.length();
    int[][] dp = new int[tgtlen][srclen];
    // inc tgt chars one by one.
    for (int tidx=0;tidx<tgtlen;tidx++) {
        char tc = t.charAt(tidx);
        for (int sidx=0;sidx<srclen;sidx++) {
            char sc = s.charAt(sidx);
            if (sc == tc) {
                if (sidx == 0) {        # first row, first col checks.
                    dp[tidx][0] = 1;
                } else if (tidx == 0) {
                    dp[0][sidx] = dp[0][sidx-1] + 1;
                } else {
                    dp[tidx][sidx] = dp[tidx-1][sidx-1] + dp[tidx][sidx-1];
                }
            } else {
                if (sidx > 0) {        # only need first col check when not equals.
                    dp[tidx][sidx] = dp[tidx][sidx-1];
                }
            }
        }
    }
    System.out.println("max seq len is " + Arrays.toString(dp[tgtlen-1]));
    return dp[tgtlen-1][srclen-1];
}
print distinct("bbbc", "bbc")
print distinct("abbbc", "bc")

// reuse one row for each target char.
static int distinctSeq(String src, String tgt) {
  char[] s = src.toCharArray();
  char[] t = tgt.toCharArray();
  int ssz = s.length, tsz=t.length;
  int[] dp = new int[ssz];   // fill by each tidx row
  for(int tidx=0;tidx<tsz;tidx++) {
      char tc = t[tidx];
      int pre = 0;
      for(int sidx=0;sidx<ssz;sidx++) {
          int tmp = dp[sidx];
              if (s[sidx] == t[tidx]) { dp[sidx] = pre + 1; }
              else if (sidx > 0) { dp[sidx] = dp[sidx-1]; }
              else { dp[sidx] = 0; }
          pre = tmp;
      }
  }
  return dp[ssz-1];
}

// need memoize
static List<List<Integer>> combinSum(int[] arr, int target, int off, List<Integer> path, List<List<Integer>> result) {
  if (target == 0) {
      result.add(path.stream().collect(Collectors.toList()));
      return result;
  }
  for (int i=off;i<arr.length;i++) {
      path.add(arr[i]);
          combinSum(arr, target-arr[i], i+1, path, result);
      path.remove(path.size()-1);
  }
  return result;
}
List<Integer> path = new List<>();
List<List<Integer>> result = new List<>();
System.out.println(combinSum(new int[]{2,3,6,7}, 9, 0, path, result));

// dup allowed, stay on row i from smaller value. item row i inheriates from item row i-1.
static int combinSum(int[] arr, int target) {
    int sz = arr.length;
    int[] dp = new int[target+1];  // value from 1..target
    dp[0] = 1;   // when item value = target, count single item 1.
    for (int i=0;i<sz;i++) {
        int iv = arr[i];
        for (int v=iv; v<=target; v++) {
            dp[v] += dp[v-iv];
        }
    }
    System.out.println("comb sum " + dp[target]);
    return dp[target];
}
combinSum(new int[]{2,3,6,7}, 9);


""" tab[i,v] = tot non descreasing at ith digit, end with value v
bottom up each row i, each digit as v, sum. [[1,2,3...], [1,2..], ...]
tab[n,d] = sum(tab[n-1,k] where k takes various value[0..9]
Total count = ∑ count(n-1, d) where d varies from 0 to n-1
"""
def nonDecreasingCount(slots):
  tab = [[0]*10 for i in xrange(slots)]
  for v in xrange(10):
    tab[0][v] = 1
  for i in xrange(1, slots):
    for val in xrange(10):  // single digit value 0-9
      for k in xrange(val+1):  # k is capped by prev digit v
        tab[i][val] += tab[i-1][k]
  tot = 0
  for v in xrange(10):
    tot += tab[n-1][v]  // collect all 0-9 digits
  return tot
assert(nonDecreasingCount(2), 55)
assert(nonDecreasingCount(3), 220)


""" bottom up each pos, at each set[0:pos], each val as a new col, entry is a list
[ [v1-list, v2-list, ...], [v1-list, v2-lsit, ...], ...] 

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
        dp[i][s] = dp[i-1][s][:]
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


# subset/combination sum, rite segment is defined, rite deps on left seg, recur left seg.
# when dup allowed, after selecting item i, += dp[i][v-vi], instead of dp[i-1][v-vi]
# continue use item i, dp[i][v] += dp[i][v-vi], as coin change, update dp[V] for each item/coin.
# if dup not allowed, keep two rows, i and i-1, so dp[i][v] = (dp[i-1][v] + dp[i-1][v-vi])
# keep boundary condition dp[0] = 1, when item i value=target.
static int subsetSum(int[] arr, int target) {
    int sz = arr.length;
    int[] pre = new int[target+1];
    pre[0] = 1;
    for (int i=0; i<sz; i++) {
        int iv = arr[i];
        int[] cur = new int[target+1];
        // start from 0, not iv, to carry prev result that less than iv
        for (int v=0; v<=target; v++) {  // as we have two rows, pre / cur, cp pre to cur from 0..n
            // tot is sum of incl/excl cur item.
            cur[v] = pre[v];      // inherite from pre, exclude cur item
            if (v >= iv) {
              cur[v] += pre[v-iv];   // include cur item, 
            }
        }
        // System.arraycopy(cur, 0, pre, 0, target+1);
        pre = cur;
        System.out.println("cur " + iv + " pre " + Arrays.toString(pre));
    }
    System.out.println("subset sum " + pre[target]);
    return pre[target];
}
subsetSum(new int[]{2,3,6,7}, 9);


""" dp[i,s] i digits sum to s, iterate 0..9, = dp[i-1,s-[1..9]]+[1..9]
not as subset sum, no incl/excl of arr[i]. so NOT cp prev row.
bottom up, start from 1 slot, face=[0..9]; dup allowed.
when each slot needs to be distinct, only append j when it is bigger
"""
def combinationSum(n, sm, distinct=False):  # n slots, sum to sm.
  dp = [[None]*(sm+1) for i in xrange(n)]
  for i in xrange(n):       # bottom up n slots
    for s in xrange(sm+1):  # enum to sm for each slot, dp[i,sm]
      if dp[i][s] == None: dp[i][s] = []
      for f in xrange(1, 10):  # each slot face, 1..9
        if i == 0:   # dp[0][0..9] = [0..9]
          dp[i][f].append([f])   # first slot, straight.
          continue
        if s > f and dp[i-1][s-f] and len(dp[i-1][s-f]) > 0:
          for e in dp[i-1][s-f]:
            // if need to be distinct, append j only when it is bigger.
            if distinct and f <= e[-1]:
              continue
            de = e[:]
            de.append(f)
            dp[i][s].append(de)
  return dp[n-1][sm]
print combinationSum(3,9,True)

// contain sol sz < n, [1,6], [7], [1, 0, 6] '''
def combsum(n, t):
  dp = [[None]*(t+1) for i in xrange(n)]
  for i in xrange(n):   # comb, not perm, outer loop slots, for each slot, enum possible values.
    for v in xrange(10):
      for s in xrange(max(1,v), t+1):
        if not dp[i][s]:
          dp[i][s] = []
        if v == s:
          dp[i][s].append([v])
        else:
          if dp[i-1][s-v]:
            for e in dp[i-1][s-v]:
              de = e[:]
              de.append(v)
              dp[i][s].append(de)
          else:
            if i > 0:
              dp[i][s] = dp[i-1][s][:]
  return dp[n-1][t]
print combsum(3,7)

static int combsum(int n, int target) {
  int[][] dp = new int[n][target+1];
  for (int i=0;i<n;i++) {
    Arrays.fill(dp[i], 0);
  }
  for (int i=0;i<n;i++) {
      // at each spot, enum target
      for (int j=0;j<10;j++) {
          for (int v=Math.max(1,j);v<target;v++) {
              if (j == v) { dp[i][v] += 1;}
              dp[i][v] += dp[i-1][v-j];
          }
      }
  }
  return dp[n-1][target];
}
System.out.println(combsum(3,7));


""" min number of coins, dup allowed. 1-dim tab. tab[v] = min of tab[v-1/2/5]+1
"""
int minCoins(int[] coins, int amount) {
    if (coins == null) return 0;
    long[] dp = new long[amount+1];
    Arrays.fill(dp, Integer.MAX_VALUE);
    dp[0]=0;
    for(int i=0; i<coins.length; i++) {
        for(int amt=1; amt<dp.length; amt++){
            if(amt >= coins[i])
                dp[amt] = Math.min(dp[amt], dp[amt-coins[i]]+1); 
        }
    } 
    return dp[amount] == Integer.MAX_VALUE ? -1 : (int) dp[amount];
}


""" tot ways for a change. permutation. [2,3] [3,2].
bottom up each val, a new row [c1,c2,...], bottom up coin col first, 
so at coin 2, no knowledge of 3, so 5=[2,3], coin is sorted and bottom up. no [3,2] ever.

if for perm, [2,3] [3,2], bottom up val first, at each value, bottom up coin.

when iter coin 3, tab[3,6,9..]=1; for coin 5, tab[5,10]=1
for coin 10, tab[10]=tab[10]+tab[0], which is #{[5,5],[10]}
hence, do not init tab[c]=1, and not tab[v]=tab[v-c]+1
"""
def coinchangePerm(coins, V):
  dp = [0]*(V+1)
  dp[0] = 1   # when coin face value = target, value is 0.
  # at each value, loop each coin, 
  for v in xrange(1, V+1):
    for c in xrange(len(coins)):
      if v >= coins[c]:
        dp[v] += dp[v-coins[c]]  # XX f(i)=f(i-1)+1
  return tab[V]
print coinchangePerm([1, 2], 3)

def coinchangeCombinations(coins, V):
  dp = [0]*(V+1)
  dp[0] = 1  # <-- when coin face value equals target.
  # loop coin in order, at each coin, loop each value
  for i in xrange(len(coins)):
    for v in xrange(coin[i], V+1):
      # sum up k-ways to reach this value.
      dp[v] += dp[v-coin[i]]
      # to get min ways, tab[0]=0, tab[v]=999
      # tab[v] = min(tab[v], 1+tab[v-fv])
  return dp[V]
print coinchange([2,3,5], 10)

static int coinChangeCombinationSets(int[] coins, int target) {
    int sz = coins.length;
    int[] dp = new int[target+1];
    
    dp[0] = 1;  // face value equals target, value = 0
    // iterate each coin, so order is ensured. (3,5), you never see (5,3).
    for (int i=0; i<sz; i++) {
        int c = coins[i];
        for (int v=c;v<=target;v++) {
            dp[v] += dp[v-c];
        }
    }
    System.out.println("coin set :" + tab[target]);
    return tab[target];
}
coinSet(new int[]{1,2,5}, 5);

""" m faces, n dices, num of ways to get value x.
bottom up, 1 dice, 2 dices,..., 3 dices, n dices. Last dice take face value 1..m.
[[v1,v2,...], [v1,v2,...], d2, d3, ...], each v1 column is cnt of face 1..m
tab[d,v] += tab[d-1,v-[1..m]]. Outer loop is enum dices, so it's like incl/excl.
"""
def dice(m,n,x):
  dp = [[0]*n for i in xrange(x+1)]
  // init bottom, one coin only
  for f in xrange(m):
    dp[1][f] = 1
  for i in xrange(1, n):      # bottom up coins, 1 coin, 2 coin
    for v in xrange(x):       # bottom up each value after inc coin
      for f in xrange(m,v):   # iterate each face value within each tab[dice, V]
        dp[i][v] += dp[i-1][v-f]  or dp[i][v] += dp[i][v-f]
  return dp[n,x]


''' digitSum(2,5) = 14, 23, 32, 41 and 50
bottom up each row of i digits, each val is a col, 
[[1,2,..], [v1, v2, ...], ..] tab[i,v] = sum(tab[i-1,k] for k in 0..9)
'''
def digitSum(slot, tot):
  dp = [[0]*max(tot+1,10) for i in xrange(slot)]
  for i in xrange(10):
    dp[0][i] = 1
  for i in xrange(1, slot):       # bottom up each slot
    for val in xrange(tot+1):    # bottom up each sum
      for face in xrange(val):   # foreach face(0..9)
        dp[i][val] += dp[i-1][val-face]
  return tab[slot-1][tot]
assert(digitSum(2,5), 5)
assert(digitSum(3,6), 21)


"""
count tot num of N digit with sum of even digits 1 more than sum of odds
n=2, [10,21,32,43...] n=3,[100,111,122,...980]
use 2 tables, odd[i,v] and even[i,v]
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
for ary, tab[i] track [0..i], to track any seg, tab[i,j]. Impl with increasing gap.
tab[i,j]=min ins to convert str[i:j]
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
  # by the end, expand to full string.
  return dp[0]

''' another way to build tab from 0 -> n. F[i] = min(F[i-k] for k in i->0 where s[k,i] is palin)
      To aggregate fill F[i], we need know moving k, f[k], and s[k,i] is palin.
      For DP tabular lookup, we can start from the begin, the minimal case, and expand to entire string.
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

# palindrom partition. First, find all palindroms.
# From head, DFS find all palindrom
List<List<String>> partition(String s) {
    List<List<String>> res = new ArrayList<>();
    boolean[][] dp = new boolean[s.length()][s.length()];
    for (int gap=2;gap<s.length;gap++) {
      for (int i=0;i<s.length-gap;i++) {
        int j = i+gap-1;
        if (s.charAt(i) == s.charAt(j) && (gap <= 2 || dp[i+1][j-1]) {
          dp[i][j] = true;
        }
      }
    }
    recurDFS(res, new ArrayList<>(), dp, s, 0);
    return res;
  }
  void recurDFS(List<List<String>> res, List<String> path, boolean[][] dp, String s, int off) {
    if(off == s.length()) {
        res.add(new ArrayList<>(path));
        return;
    }
    for(int i = off; i < s.length(); i++) {
        if(dp[off][i]) {
            path.add(s.substring(off,i+1));   # consume off->i, recur i+1;
              recurDFS(res, path, dp, s, i+1);
            path.remove(path.size()-1);
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

""" palindrom pair, suffix, reverse. concat(prepend or append) the two words forms a palindrom
for each word, list all prefix, reverse suffix.
https://leetcode.com/problems/palindrome-pairs/discuss/79209/Accepted-Python-Solution-With-Explanation
"""
def palindromPair(arr):
  for each word, find all its prefix and suffix
    for each prefix, 
      if prefix is palindrom, reverse corresponding suffix, find in dict, 
         reverser(suffix) + prefix + suffix  is a palindrom.


''' bfs search on Matrix for min dist, put intermediate result in q, exhaust Queue '''
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

""" word ladder, bfs, enum all possible words. for each intermediate word, for each pos of the word,
iterate all ascii to get new char, recur not visited.
"""
from collections import deque
import string
def wordladder(start, end, dict):
  ''' given a word, for each pos of the wd '''
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
  ''' bfs search from start, ladder as queue'''
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

""" consider every cell in borad as start, dfs find all word in dict carrying seen map.
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
1. track to put all possible [st,ed] into one line, the extra spaces[st,ed]
2. w[i]: cost of word wrap [0:i]. foreach k=[i..j], w[j] = w[i-1] + lc[i,j]
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

""" m[i,j]=matrix(i,j) = min(m[i,k]+m[k+1,j]+arr[i-1]*arr[k]*arr[j]) """
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
dp[i,j] track the min left in subary i..j
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
# with k boxes on the left of the same color as boxes[i].
# T(i, i, k) = (k + 1) * (k + 1);  // boundary, 1 box at i, and left side has k same color box.
# T(i, i - 1, k) = 0
# Recur each boundary i: two choices, remove it first along with k on the left, or merge it with some m in i..j.
# remove it with k left, then recur.  (k + 1) * (k + 1) + T(i + 1, j, 0)
# OR attach boxes[i] to some other box of the same color, say m, in between i and j. 
# this means 0 on the left of i+1, and k+1 on the left of m.
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

