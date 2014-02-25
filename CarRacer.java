/**
 * rocket fuel coding test, racer
 * compiling and test:
 * 		javac CarRacer.java
 * 		cat testcase | java CarRacer
 * 
 * @author Haijin Yan, 2/23, 2014
 */

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class CarRacer {
	private static final boolean DEBUG = false;
	public static void log(String... args) {
		if(DEBUG){
			for(String s : args){
				System.out.println(s);
			}
		}
	}
	
	/**
	 * Racer object encapsulate racer information.
	 */
	public static class Racer implements Comparable<Racer> {
		protected final BigInteger id;
		protected final BigInteger start;
		protected final BigInteger end;
		protected BigInteger score;
		
		public Racer(final BigInteger id, final BigInteger start, final BigInteger end){
			this.id = id;
			this.start = start;
			this.end = end;
			this.score = BigInteger.ZERO;
		}
		
		public Racer(final String id, final String start, final String end){
//			this.id = Integer.parseInt(id);
//			this.start = Integer.parseInt(start);
//			this.end = Integer.parseInt(end);
//			this.score = 0;
			this.id = new BigInteger(id);
			this.start = new BigInteger(start);
			this.end = new BigInteger(end);
			this.score = BigInteger.ZERO;
		}

		/**
		 * getter and setter for all attributes.
		 */
		public BigInteger getId() { return id;}
		public BigInteger getStart() { return start;}
		public BigInteger getEnd() { return end;}
		public BigInteger getScore() { return score;}
		public BigInteger addScore(BigInteger val) { 
			score = score.add(val);
			return score;
		}
		
		public String toString(){ return id + " score " + score + " start: " + start + " end: " + end;}
		
		/**
		 * override equality with hashCode together.
		 */
		@Override
        public boolean equals(Object o) {
            if ( this == o ) return true;
            if (! (o instanceof Racer)) return false;
            return id == ((Racer)o).getId();
		}
		@Override
        public int hashCode() {
            return id.hashCode();
        }

		/**
		 * score comparator, if score is the same, break by id, otherwise, by score.
		 */
		@Override
		public int compareTo(Racer othercar) {
			if(score.compareTo(othercar.getScore()) == 0){
				return id.compareTo(othercar.getId());
			}
			return score.compareTo(othercar.getScore());
		}
	}
	
	/**
	 * start time, end time and score comparator, to sort Racer
	 */
	static class StarttimeComparator implements Comparator<Racer> {
		public int compare(Racer a, Racer b) {
			return a.getStart().compareTo(b.getStart());
		}
	}
	static class EndtimeComparator implements Comparator<Racer> {
		public int compare(Racer a, Racer b) {
			return a.getEnd().compareTo(b.getEnd());
		}
	}
	static class ScoreComparator implements Comparator<Racer> {
		public int compare(Racer a, Racer b) {
			return a.getScore().compareTo(b.getScore());
		}
	}
	

	// --------------------------------------------------------
	// member variables here.
	// --------------------------------------------------------
	int numRacers = 0;  	// num of racers
	// list of all racers
	List<Racer> racers = new ArrayList<Racer>();
	
	/**
	 * sort by start time.
	 */
	private void sortByStart() {
		Collections.sort(racers, new StarttimeComparator());
		dumpList();
	}
	
	/**
	 * partition the list, inclusive.
	 */
	private List<Racer> partition(int left, int rite) {	
		if(left == rite){
			return racers.subList(left, rite+1);  // sublist call right is exclusive
		}
		
		int mid = (left+rite)/2;
		List<Racer> leftList = partition(left, mid); 
		List<Racer> riteList = partition(mid+1, rite);
		return combineMerge(leftList, riteList);
	}
	
	/**
	 * merge two partitioned lists, with end time in descending order, return the merged list of next iteration. 
	 */
	private List<Racer> combineMerge(List<Racer> leftList, List<Racer> riteList){
		// now combine merge two lists, sort the end time in each list in descending order
		int i=0,j=0;
		int leftsz = leftList.size();
		int ritesz = riteList.size();
		List<Racer> retList = new ArrayList<Racer>();

		// iterate thru both lists, list is already sort by start time.
		// if an item in rite list has end time less than current racer's end time, inc current racer's score.
		while(i<leftsz && j<ritesz){
			// current racer i start before j, but finish later than j, increment i's score.
			if(leftList.get(i).getEnd().compareTo(riteList.get(j).getEnd()) >= 1){
				//CarRacer.log("adding score " + leftList.get(i).toString() + " " + riteList.get(j).toString());
				leftList.get(i).addScore(BigInteger.valueOf(ritesz-j));  // num of other racers as score
				retList.add(leftList.get(i));
				i += 1;
			}else{
				retList.add(riteList.get(j));
				j += 1;
			}			
		}
		
		// merge the left overs
		while(i<leftsz){
			retList.add(leftList.get(i));
			i += 1;
		}
		while(j<ritesz){
			retList.add(riteList.get(j));
			j += 1;
		}

		return retList;
	}
	
	/**
	 * output in format of racer id and score
	 */
	private void output() {
		for(Racer r : racers){
			System.out.println(r.getId() + " " + r.getScore());
		}
	}
	
	private void partitionAndMerge(){
		//dumpList();
		partition(0, racers.size()-1);
		Collections.sort(racers);
		dumpList();
		output();
	}
	
	/**
	 * dump racers list
	 */
	private void dumpList() {
		if(CarRacer.DEBUG){
			CarRacer.log("--------------");
			for(Racer r : racers){
				CarRacer.log(r.toString());
			}
		}
	}
	
	/**
	 * read in logs, line by line, split the line and create racer object and populate racer list.
	 * @throws IOException 
	 */
	private int readLog() throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		String line;
		
		line = in.readLine();
		numRacers = Integer.parseInt(line);
		CarRacer.log("num cars " + line + " " + numRacers);
		
		for(int i=0;i<numRacers;i++){
			line = in.readLine();
			String[] raceinfo = line.split("\\s+");
			racers.add(new Racer(raceinfo[0], raceinfo[1], raceinfo[2]));
		}

		in.close();
		return numRacers;
	}
	
	
	
	/**
     * main test, read log, populate racer list, and start counting.
     * @param args
     * @throws IOException
     */
	public static void main(String[] args) throws IOException {
		CarRacer carracer = new CarRacer();
		carracer.readLog();
		carracer.sortByStart();
		carracer.partitionAndMerge();
	}
}