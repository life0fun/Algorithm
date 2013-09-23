
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * this class impls code challenge from Scoply.
 * 
 * usage:
 *    java Scopely.java
 *    java -cp . Scopely'
 * 
 * check out the print out node path
 *    
 */
public class Scopely {

    /**
     * tree node
     */
    class Node {
        String mName;
        Node mParent;   // mParent == null means root
        List<Node> mChildren;
        
        /**
         * constructor, auto set this as the child of parent.
         * @param name
         * @param parent
         */
        public Node(String name, Node parent) { 
            mName = name;
            mParent = parent;
            mChildren = new ArrayList<Node>();
            if(mParent != null)
                mParent.addChild(this);
        }

        /**
         * override hashcode and equals to enforce node name equal comparison
         */
        @Override
        public int hashCode() {
            return (int)(mName.hashCode());  // just use the hashcode of name
        }

        @Override
        public boolean equals(Object o) {
            if ( this == o ) return true;
            if ( !(o instanceof Node) ) return false;

            if (mName.equals(((Node)o).mName))
                return true;
            else
                return false;
        }

        /**
         * clean all the child in root between each test
         */
        public void clean() {
            mChildren.clear();
        }
        
        public String getName() { return mName; }
        public List<Node> getChildren() { return mChildren; }
        
        /**
         * add a child node to list
         */
        public Node addChild(Node child) {
            mChildren.add(child);
            return child;
        }
        
        /**
         * return the path of a node up to its parent
         */
        public String getPath() {
            List<String> l = new ArrayList<String>();
            Node cur = this;
            do{
                l.add(cur.mName);
                cur = cur.mParent;
            }while(cur != null);
            
            StringBuilder sb = new StringBuilder();
            for(int i=l.size()-1;i>=0;i--){
                sb.append("/" + l.get(i));
            }
            return sb.toString();
        }
        
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(mName);
            if( mParent != null){
                sb.append( " :: parent= " + mParent.getName() + "::");
            }
            sb.append(" [ ");
            for(Node n : mChildren) {
                sb.append(n.getName() + " ");
                sb.append(n.toString());
            }
            sb.append(" ] ");
            
            return sb.toString();
        }
        
        /**
         * list all the pathes
         * @param pathPrefix : the path prefix until to this node 
         */
        public String listPaths(Node n, String pathPrefix) {
            pathPrefix += "/" + n.mName;
            
            if(mChildren.size() == 0){  // this is leaf node
                debug(pathPrefix);
            }else{
                for(Node c : n.mChildren){
                    c.listPaths(c, pathPrefix);
                }
            }
            return pathPrefix;
        }
        
        /**
         * return the child node represent the string
         */
        private Node findChildNode(Node curnode, String name) {
            for(Node c : curnode.mChildren) {
                if(name.equals(c.getName())){
                    return c;
                }
            }
            return null;
        }
        
        /**
         * find a node corresponding to the path
         */
        public Node findNode(String path) {
            Node curnode = root;
            String[] pathele = path.split("/");
            
            for(int i=2;i<pathele.length;i++){  // skip the home prefix
                Node c = findChildNode(curnode, pathele[i]);
                if( c != null){
                    curnode = c;
                    if( i == pathele.length-1 ){  // if loop over all path eles, return;
                        return curnode;
                    }
                }
            }
            return null;
        }
        
        /**
         * child match by checking all children num and name equals
         */
        public boolean childrenMatch(Node root, Node n) {
            
            if(root.equals(n)){
                return false;   // synode is not node
            }
            
            // first, children size must be equals
            if(root.mChildren.size() != n.mChildren.size())
                return false;
            
            // now no child should has no equal counterpart
            for(Node rootc : root.mChildren) {
                boolean found = false;  // after one iteration, should be true
                
                for(Node nc : n.mChildren) {
                    if(nc.equals(rootc)){
                        found = true;
                        break;
                    }
                }
                
                // there exist child no couter part, fail
                if( found == false){
                    return false;
                }
            }
            
            // at here, all child matches, found
            return true;
        }
            
        /**
         * find Synonm Node of the input node, according to 
         */
        public boolean findSynomNode(Node root, Node n, List<Node> result){
            
            if(root.childrenMatch(root, n)){
                result.add(root);
                return true;
            }
            
            for(Node rootc : root.mChildren) {
                if(rootc.findSynomNode(rootc, n, result))
                    return true;
            }
            
            return false;
        }
    }
    
    // root node
    Node root;
    
    public Scopely() { 
        root = new Node("home", null);
    }
    
    public void debug(String info){
        if(info != null)
            System.out.println(info);
    }
    
    /**
     * get path ele dep on the delimiters.
     */
    String[] getPath(String path, String delim) {
        String[] pathele = path.split(delim);
        //for(String s : pathele)
        //    debug("getPath: " + s);
        return pathele;
    }
    
    /**
     * get the combinatorial of path ele
     */
    public List<String> getComb(String[] eles){
        List<String> comb = new ArrayList<String>();
        List<String> newcomb = new ArrayList<String>();
        for(String s : eles){
            //debug("getcomb: " + s);
            for(String combele : comb){
                newcomb.add(combele+"-"+s);
            }
            comb.addAll(newcomb);
            comb.add(s);
            newcomb.clear();
        }
        return comb;
    }
    
    
    /**
     * Part 1, insert into tree
     * @param path  '/home/sports/football'
     * @return
     */
    public Node insertIntoTree(String path) {
        String[] pathele = getPath(path, "/");
        Node parent = root;
        // assuming root is always /home, the first after split is empty, so iterate from the second node
        for(int i = 2; i<pathele.length;i++){
            //debug(i + " : " + pathele[i]);
            Node n = new Node(pathele[i], parent);
            parent = n;
        }
        
        return root;  
    }
    
    public void testInsertIntoTree() {
        insertIntoTree("/home/sports/basketball/ncaa/");
        debug( " ========== part 1 Insert =================");
        debug(root.listPaths(root, ""));
        insertIntoTree("/home/music/rap/gangster");
        debug(root.listPaths(root, ""));
    }
        
    
    /**
     * part 2, testInsertDulLeafNode
     */
    
    public Node insertDualLeafNode(String path) {
        String[] pathele = getPath(path, "/");
        Node parent = root;
        // assuming root is always /home, the first after split is empty, so iterate from the second node
        for(int i = 2; i<pathele.length;i++){
            String ele = pathele[i];
            if(ele.contains("|")){
                String[] sibling = getPath(ele, "\\|");
                List<Node> parents = new ArrayList<Node>();
                for(String s : sibling){
                    Node n = new Node(s, parent);
                    parents.add(n);
                }
                parent = parents.get(0);   // should be fine to use the first node as the parent for dual leaf
            }else{
                Node n = new Node(ele, parent);
                parent = n;
            }
        }
        
        return root;
    }
    
    public void testInsertDulLeafNode() {
        root.clean();
        insertDualLeafNode("/home/sports/football/NFL|NCAA");
        debug( " ========== part 2 dual Leaf =================");
        debug(root.listPaths(root, ""));
    }
    
    /**
     * part 3, comb leaf node
     */  
    public Node insertCombLeafNode(String path) {
        String[] pathele = getPath(path, "/");
        Node parent = root;
        // assuming root is always /home, the first after split is empty, so iterate from the second node
        for(int i = 2; i<pathele.length;i++){
            String ele = pathele[i];
            if(ele.contains("|")){
                // get all the siblings
                String[] sibling = getPath(ele, "\\|");
                
                List<String> comb = getComb(sibling);
                
                for(String s : comb){
                    Node n = new Node(s, parent);
                }
            }else{
                Node n = new Node(ele, parent);
                parent = n;
            }
        }
        return root;
    }

    public void testCombLeafNode() {
        root.clean();
        insertCombLeafNode("/home/music/rap|rock|pop");
        debug( " ========== part 3 Comb Leaf =================");
        debug(root.listPaths(root, ""));
    }
    
    /**
     * part 4, comb any leaves
     */  
    public Node insertCombAny(String path) {
        String[] pathele = getPath(path, "/");
        
        List<Node> parents = new ArrayList<Node>();  // whether parent is Comb level
        parents.add(root);
        
        // assuming root is always /home, the first after split is empty, so iterate from the second node
        for(int i = 2; i<pathele.length;i++){
            String ele = pathele[i];
            List<Node> nextparents = new ArrayList<Node>();  // whether parent is Comb level    
            if(ele.contains("|")){
                // get all the siblings
                String[] sibling = getPath(ele, "\\|");
                List<String> comb = getComb(sibling);
                
                for(String s : comb){
                    for(Node p : parents){
                        Node n = new Node(s, p);  // parent added child
                        nextparents.add(n);
                    }
                }
            }else{
                for(Node p : parents){
                    Node n = new Node(ele, p);
                    nextparents.add(n);
                }
            }
            parents.clear();
            parents = nextparents;
        }
        return root;
    }

    public void testCombAny() {
        root.clean();
        insertCombAny("/home/sports|music/misc|favorites");
        debug( " ========== part 4 Comb any =================");
        debug(root.listPaths(root, ""));
    }
    
    /**
     * part 5, collapse of comb
     */
    
    private String collapseComb(Node root){
        List<String> combname = new ArrayList<String>();
        if(root.mChildren.size() > 0){
            Node firstc = root.mChildren.get(0);
            for(int i=1;i<root.mChildren.size()-1;i++){
                if(firstc.childrenMatch(firstc, root.mChildren.get(i))){
                    combname.add(root.mChildren.get(i).mName);
                }
            }
        }
        
        return combname.toString();
    }
    
    public void testCollapseComb() {
        debug( " ========== part 5 Collaps Comb unfinished =================");
        debug(collapseComb(root));
    }

    
    /**
     * part 6, testsynonymDetect
     * a synonym of a give path is defined as all their children node are identical.
     * a node is defined as identical when there name equals.
     */
    private String synonymPath(Node root, String path){
        Node node = root.findNode(path);
        List<Node> result = new ArrayList<Node>();
        if(root.findSynomNode(root, node, result)){
            Node synode = result.get(0);   // assume only one syn node
            return "requirement said find one of sympath of " + path + " is " + synode.getPath();
        }
        return null;
    }
    
    public void testsynonymDetect() {
        root.clean();
        insertCombAny("/home/sports|music/misc|favorites");
        debug( " ========== part 6 synonymPath =================");
        debug(synonymPath(root, "/home/sports"));
        debug(synonymPath(root, "/home/sports-music"));
        debug(synonymPath(root, "/home/music"));
    }
    
    /**
     * main
     */
    public static void main(String[] args) throws IOException {
        Scopely root = new Scopely();
        
        /**
         * test part 1, insert into tree
         */
        root.testInsertIntoTree();
        
        /**
         * test part 2, dual leaf node
         */
        root.testInsertDulLeafNode();
        
        /**
         * test part 3, comb leaf node
         */
        root.testCombLeafNode();
        
        /**
         * test part 4, comb any
         */
        root.testCombAny();
        
        /**
         * test part 5 
         */
        root.testCollapseComb();
        
        /**
         * test part 6
         */
        root.testsynonymDetect();
    }
}
