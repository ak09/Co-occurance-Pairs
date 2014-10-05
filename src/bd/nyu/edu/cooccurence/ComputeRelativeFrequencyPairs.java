
package bd.nyu.edu.cooccurence;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeSet;



//import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
//import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import bd.edu.nyu.utility.RelativeFreqPair;
import bd.edu.nyu.utility.TextPair;

/**
   Implementation of the "pairs" algorithm for computing co-occurrence
 matrices from a large text collection. 

  Arguments 
  [input-path]
  [output-path]
  [window]
 [num-reducers]

 */
public class ComputeRelativeFrequencyPairs extends Configured implements Tool {
	private static final Logger sLogger = Logger.getLogger(ComputeRelativeFrequencyPairs.class);

	private static class MyMapper extends Mapper<LongWritable, Text, TextPair, IntWritable> {

		private final TextPair pair = new TextPair();
		
		@Override
		public void map(LongWritable key, Text line, Context context) throws IOException,
		InterruptedException {
			String text = line.toString();
			HashMap<TextPair,Integer>	map = new HashMap<TextPair,Integer>();
			String[] terms = text.split("\\s+");
			int termCount;
			for (int i = 0; i < terms.length; i++) {
				termCount =0;
				String term = terms[i];
				/*if(term.equalsIgnoreCase("$150.")){
					System.out.println();
				}*/
				// skip empty tokens
				if (term.length() == 0 || terms.length == 1)
					continue;

				for (int j = 0; j <terms.length; j++) {
					TextPair tpair = new TextPair();
					if (j == i)
						continue;

					// skip empty tokens
					if (terms[j].length() == 0)
						continue;
					termCount++;
					tpair.set(term, terms[j]);
					if(map.containsKey(tpair)){
						map.put(tpair, map.get(tpair)+1);
					}
					else{
						map.put(tpair,1);
					}
				}
				for (Map.Entry<TextPair, Integer> entry : map.entrySet())
				{
					context.write(entry.getKey(), new IntWritable(entry.getValue().intValue()));
				}
				pair.set(term, "*");
				IntWritable tot = new IntWritable(termCount);
				context.write(pair, tot);
				map.clear();
			}
		}
	}

	private static class MyCombiner extends
	Reducer<TextPair, IntWritable, TextPair, IntWritable> {

		private final static IntWritable SumValue = new IntWritable();

		@Override
		public void reduce(TextPair key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			Iterator<IntWritable> iter = values.iterator();
			int sum = 0;
			while (iter.hasNext()) {
				sum += iter.next().get();
			}

			SumValue.set(sum);
			/*if(key.getFirst().toString().equalsIgnoreCase("$150.")){
				System.out.println();
			}*/
			context.write(key, SumValue);
		}
	}

	private static class MyReducer extends
	Reducer<TextPair, IntWritable, TextPair,DoubleWritable> {

		private final static DoubleWritable SumValue = new DoubleWritable();
		boolean nextChange = false;
		double totalCount=0;
		private TreeSet<RelativeFreqPair> priorityQueue = new TreeSet<>();
		
		@Override
		public void reduce(TextPair key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			/*if(key.getFirst().toString().equalsIgnoreCase("$150.")){
				System.out.println();
			}*/
			RelativeFreqPair rfp;
			Iterator<IntWritable> iter = values.iterator();
			boolean star = false;
			int sum = 0;
			if(key.getSecond().toString().equalsIgnoreCase("*")){
				star = true;
				while (iter.hasNext()) {
					if(nextChange){
						nextChange = false;
						totalCount = 0;
					}
					totalCount += iter.next().get();
				}
			}
			else{
				star = false;
				nextChange = true;
				while (iter.hasNext()) {
					sum += iter.next().get();
				}
			}
			if(!star){
				TextPair pair = new TextPair();
				pair.set(key.getFirst().toString(), key.getSecond().toString());
				rfp = new RelativeFreqPair(sum/totalCount, pair);
				priorityQueue.add(rfp);

	                if (priorityQueue.size() > 100) {
	                    priorityQueue.pollFirst();
	                }
			}

		}
		
		protected void cleanup(Context context)
                throws IOException,
                InterruptedException {
			while (!priorityQueue.isEmpty()) {
            	RelativeFreqPair pair = priorityQueue.pollLast();
            	SumValue.set(pair.relativeFrequency);
                context.write(pair.key, SumValue);
            }
        }
	}

	public static class FirstPartitioner extends Partitioner<TextPair, IntWritable> {

		@Override
		public int getPartition(TextPair key, IntWritable value, int numReduceTasks) {
			int part = key.baseHashCode() % numReduceTasks;
			return part;
		}
	}

	
	
	/**
	 * Creates an instance of this tool.
	 */
	public ComputeRelativeFrequencyPairs() {
	}

	private static int printUsage() {
		System.out
		.println("usage: [input-path] [output-path] [window] [num-reducers]");
		ToolRunner.printGenericCommandUsage(System.out);
		return -1;
	}
	
	

	/**
	 * Runs this tool.
	 */
	public int run(String[] args) throws Exception {
		if (args.length != 3) {
			printUsage();
			return -1;
		}

		String inputPath = args[0];
		String outputPath = args[1];

		
		int reduceTasks = Integer.parseInt(args[2]);

		sLogger.info("Tool: ComputeRelativeFrequencyPairs");
		sLogger.info(" - input path: " + inputPath);
		sLogger.info(" - output path: " + outputPath);
		//sLogger.info(" - number of reducers: " + reduceTasks);

		Job job = new Job(getConf(), "ComputeRelativeFrequencyPairs");

		// Delete the output directory if it exists already
		Path outputDir = new Path(outputPath);
		URI uri = URI.create(outputPath);
		FileSystem.get(uri,getConf()).delete(outputDir, true);

		job.setJarByClass(ComputeRelativeFrequencyPairs.class);
		//job.setNumReduceTasks(reduceTasks);

		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));

		job.setMapOutputKeyClass(TextPair.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(TextPair.class);
		job.setOutputValueClass(DoubleWritable.class);

		job.setMapperClass(MyMapper.class);
		job.setCombinerClass(MyCombiner.class);
		job.setReducerClass(MyReducer.class);
		job.setPartitionerClass(FirstPartitioner.class);
		
		job.setNumReduceTasks(reduceTasks);
		long startTime = System.currentTimeMillis();
		job.waitForCompletion(true);
		System.out.println("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

		/*//String secondInputPath = args[1];
		String secondInputPath = "/Users/hadoop/Downloads";
		String secondOutputPath = args[1]+"/finalOut";
		
		Job secondJob = new Job(getConf(), "ComputeRelativeFrequencyPairsFinalSorting");

		// Delete the output directory if it exists already
		Path secondOutputDir = new Path(secondOutputPath);
		URI uri1 = URI.create(secondOutputPath);
		FileSystem.get(uri1,getConf()).delete(secondOutputDir, true);
		FileSystem fs = FileSystem.get(uri1,new Configuration());
		FileStatus[] status = fs.listStatus( new Path(secondInputPath));
		//job.getConfiguration().setInt("window", window);

		secondJob.setJarByClass(ComputeRelativeFrequencyPairs.class);
		secondJob.setNumReduceTasks(1);
		for(int i=0;i<status.length;i++){
			if(status[i].getPath().toString().contains("part")){
				//String sp = secondInputPath+"/"+status[i].getPath().toString();
				FileInputFormat.addInputPath(secondJob, status[i].getPath());
				sLogger.info(" - input path: " + status[i].toString());
			}
		}

		sLogger.info("Tool: ComputeRelativeFrequencyStripes");
		sLogger.info(" - output path: " + secondOutputPath);
		//sLogger.info(" - window: " + window);
		//sLogger.info(" - number of reducers: " + reduceTasks);
		
		FileOutputFormat.setOutputPath(secondJob, new Path(secondOutputPath));

		secondJob.setMapOutputKeyClass(DoubleWritable.class);
		secondJob.setMapOutputValueClass(Text.class);

		secondJob.setOutputKeyClass(Text.class);
		secondJob.setOutputValueClass(DoubleWritable.class);

		secondJob.setMapperClass(MySecondMapper.class);
		
		secondJob.setReducerClass(MySecondReducer.class);

		long secondStartTime = System.currentTimeMillis();
		secondJob.waitForCompletion(true);
		System.out.println("Job Finished in " + (System.currentTimeMillis() - secondStartTime) / 1000.0 + " seconds");*/
		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the
	 * <code>ToolRunner</code>.
	 */
	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new ComputeRelativeFrequencyPairs(), args);
		System.exit(res);
	}
}
