import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

public class Classifier {
    // stores counts to calculate probability of a particular feature value given the label "positive"
    protected HashMap<String, Double> posCounts;
    // stores counts to calculate probability of a particular feature value given the label "negative"
    protected HashMap<String, Double> negCounts;
    protected double numPos; // number of examples with positive label
    protected double numNeg; // number of examples with negative label
    protected double lambda; // lambda smoothing value

    public Classifier(String training_data, String test_sentences, double lambda) throws IOException {
        posCounts = new HashMap<>();
        negCounts = new HashMap<>();
        numPos = 0.0;
        numNeg = 0.0;
        this.lambda = lambda;

        // Read target words and their weighting/similarity measures from file and add them to targetInfo
        File file = new File(training_data);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st = br.readLine();

        while (st != null) {
            String[] words = st.split("\\s+");

            if (words[0].equals("positive")) {
                numPos++;
                for (int i = 1; i < words.length; i++) {
                    if (!posCounts.containsKey(words[i])) {
                        posCounts.put(words[i], 1.0);
                    } else {
                        posCounts.put(words[i], posCounts.get(words[i]) + 1.0);
                    }
                }
            } else if (words[0].equals("negative")) {
                numNeg++;
                for (int i = 1; i < words.length; i++) {
                    if (!negCounts.containsKey(words[i])) {
                        negCounts.put(words[i], 1.0);
                    } else {
                        negCounts.put(words[i], negCounts.get(words[i]) + 1.0);
                    }
                }
            }

            st = br.readLine();
        }

        posCounts.replaceAll((w, v) -> posCounts.get(w) + lambda);
        negCounts.replaceAll((w, v) -> negCounts.get(w) + lambda);

        System.out.println(posCounts);
        System.out.println(negCounts);
        System.out.println(numPos);
        System.out.println(numNeg);

        calcProbs(test_sentences);
    }

    public void calcProbs(String test_sentences) throws IOException {
        // Read target words and their weighting/similarity measures from file and add them to targetInfo
        File file = new File(test_sentences);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st = br.readLine();
        double posProbSum = 0.0;
        double negProbSum = 0.0;
        double posProbTheta;
        double negProbTheta;

        while (st != null) {
            String[] words = st.split("\\s+");

            for (String word : words) {
                if (!posCounts.containsKey(word) && !negCounts.containsKey(word)) {
                    // prob = 0
                    posProbSum += 0.0;
                    negProbSum += 0.0;
                } else {
                    double wordOcc = posCounts.get(word) + negCounts.get(word);

                    posProbTheta = posCounts.getOrDefault(word, lambda) / numPos;
                    posProbSum += wordOcc * Math.log10(posProbTheta);

                    negProbTheta = negCounts.getOrDefault(word, lambda) / numNeg;
                    negProbSum += wordOcc * Math.log10(negProbTheta);
                }
            }

            posProbSum += Math.log10(numPos / numNeg);
            posProbSum += Math.log10(numNeg / numPos);

            if (posProbSum >= negProbSum) {
                System.out.println("positive\t" + posProbSum);
            } else {
                System.out.println("negative\t" + negProbSum);
            }

            st = br.readLine();
        }
    }

    /**

     This method implements the Bernoulli Naive Bayes algorithm to classify a given set of sentences as positive or negative.

     @param test_sentences the file path of the input sentences to classify.

     @throws IOException if there is an error reading the input file.
     */
    public void bernoulliNaiveBayes(String test_sentences) throws IOException {
        // Read target words and their weighting/similarity measures from file and add them to targetInfo
        File file = new File(test_sentences);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st = br.readLine();
        double posProbSum;
        double negProbSum;

        // Loop through each sentence in the input file and classify it as positive or negative
        while (st != null) {
            // Split the sentence into individual words
            String[] words = st.split("\s+");
            // Initialize the probability sums for the positive and negative classes
            posProbSum = Math.log(numPos / (numPos + numNeg));
            negProbSum = Math.log(numNeg / (numPos + numNeg));

            // Calculate the probability of each word in the sentence belonging to the positive or negative class
            // and update the probability sums accordingly
            for (String word : posCounts.keySet()) {
                if (Arrays.asList(words).contains(word)) {
                    posProbSum += Math.log((posCounts.get(word) + 1) / (numPos + 2));
                } else {
                    posProbSum += Math.log(1 - ((posCounts.get(word) + 1) / (numPos + 2)));
                }
            }

            for (String word : negCounts.keySet()) {
                if (Arrays.asList(words).contains(word)) {
                    negProbSum += Math.log((negCounts.get(word) + 1) / (numNeg + 2));
                } else {
                    negProbSum += Math.log(1 - ((negCounts.get(word) + 1) / (numNeg + 2)));
                }
            }

            // Output the classification result for the sentence
            if (posProbSum >= negProbSum) {
                System.out.println("positive\t" + posProbSum);
            } else {
                System.out.println("negative\t" + negProbSum);
            }

            // Read the next sentence from the input file
            st = br.readLine();

        }
    }




    public static void main(String[] args) throws IOException {
        String training_data = "/Users/ezraford/Desktop/School/CS 159/Naive-Bayes-Classifier/assign7-starter/simple.data";
        String test_sentences = "/Users/ezraford/Desktop/School/CS 159/Naive-Bayes-Classifier/assign7-starter/test.data";
        double lambda = 0.0;

        Classifier nb = new Classifier(training_data, test_sentences, lambda);
    }
}
