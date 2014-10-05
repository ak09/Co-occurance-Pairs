package bd.edu.nyu.utility;

public class RelativeFreqPair implements Comparable<RelativeFreqPair> {
    public double relativeFrequency;
    public TextPair key;
    
    public RelativeFreqPair(double relativeFrequency, TextPair key) {
        this.relativeFrequency = relativeFrequency;
        this.key = key;
    }

    @Override
    public int compareTo(RelativeFreqPair pair) {
        if (this.relativeFrequency >= pair.relativeFrequency) {
            return 1;
        } else {
            return -1;
        }
    }
    
    public String toString() {
		return  key.toString() + " " +relativeFrequency;
	}
}

