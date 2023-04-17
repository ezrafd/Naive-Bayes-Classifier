import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

// Assignment 7
// Tal Mordohk and Ezra Ford

public class Classifier {
    // stores counts to calculate probability of a particular feature value given the label "positive"
    protected HashMap<String, Double> posCounts;
    // stores counts to calculate probability of a particular feature value given the label "negative"
    protected HashMap<String, Double> negCounts;
    protected double numPos; // number of examples with positive label
    protected double numNeg; // number of examples with negative label
    protected double numPosWords; // number of words in positive label docs
    protected double numNegWords; // number of words in negative label docs
    protected double numUniqueWords; // number of unique words
    protected double lambda; // lambda smoothing value


    /**
     *
     * Trains classifier, then runs other methods to perform testing
     *
     * @param training_data the file path of the formatted training data
     * @param test_sentences the file path of the input sentences to classify
     * @param lambda the lambda value used for smoothing
     * @throws IOException if there is an error reading the input file
     */
    public Classifier(String training_data, String test_sentences, double lambda) throws IOException {
        // initialize variables
        posCounts = new HashMap<>();
        negCounts = new HashMap<>();
        numPos = 0.0;
        numNeg = 0.0;
        numPosWords = 0.0;
        numNegWords = 0.0;
        this.lambda = lambda;

        // Read target words and their weighting/similarity measures from file and add them to targetInfo
        File file = new File(training_data);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st = br.readLine();

        // loop though each sentence in the file
        while (st != null) {
            // split the sentence into an list of words
            String[] words = st.split("\\s+");

            // if the label is positive
            if (words[0].equals("positive")) {
                // increment the number of positive examples seen
                numPos++;

                for (int i = 1; i < words.length; i++) { // for each word in the sentence
                    numPosWords++; // increment the number of words in positive label docs
                    if (!posCounts.containsKey(words[i])) {
                        // if we haven't seen this word in the positive context before, put it in posCounts
                        posCounts.put(words[i], 1.0);
                        if (!negCounts.containsKey(words[i])) {
                           numUniqueWords++; // if we have never seen it before increment the number of unique words
                        }
                    } else {
                        // increment the number of occurrences with positive
                        posCounts.put(words[i], posCounts.get(words[i]) + 1.0);
                    }
                }
            // if the label is negative
            } else if (words[0].equals("negative")) {
                // increment the number of negative examples seen
                numNeg++;

                for (int i = 1; i < words.length; i++) { // for each word in the sentence
                    numNegWords++; // increment the number of words in negative label docs
                    if (!negCounts.containsKey(words[i])) {
                        // if we haven't seen this word in the negative context before, put it in negCounts
                        negCounts.put(words[i], 1.0);
                        if (!posCounts.containsKey(words[i])) {
                            numUniqueWords++; // if we have never seen it before increment the number of unique words
                        }
                    } else {
                        // increment the number of occurrences with negative
                        negCounts.put(words[i], negCounts.get(words[i]) + 1.0);
                    }
                }
            }

            st = br.readLine();
        }

        // add lambda smootthing
        posCounts.replaceAll((w, v) -> posCounts.get(w) + lambda);
        negCounts.replaceAll((w, v) -> negCounts.get(w) + lambda);

        // Calculate and print predicted label and probabilities for each sentence
        System.out.println("Reg Prob");
        calcProbs(test_sentences);
        System.out.println("Bernoulli");
        bernoulliNaiveBayes(test_sentences);

        // print all the probabilities in the trained model
        printProbs();

        // print out the top 10 most predictive words for each label
        //printMostPredictive();
    }


    /**
     * Prints top 10 most predictive words for each label
     */
    public void printMostPredictive() {
        // initialize variables
        ArrayList<String> mostPredictiveWords = new ArrayList<>(); // list of all the words in both posCounts and negCounts
        ArrayList<Double> values = new ArrayList<>(); // list of the values p(*|positive)/p(*|negative) for each word
        double posProb;
        double negProb;

        // get the values for each word
        for (String word : posCounts.keySet()) {
            if (negCounts.containsKey(word)) {
                mostPredictiveWords.add(word);
                posProb = posCounts.get(word) / (numPosWords + (lambda * numUniqueWords));
                negProb = negCounts.get(word) / (numNegWords + (lambda * numUniqueWords));
                values.add(posProb/negProb);
            }
        }

        // sorts values in increasing order and keeps they corresponding keys in most predictive words aligned
        modQuickSort(values, 0, values.size() - 1, mostPredictiveWords);

        // print out the top 10 most predictive features for positive
        System.out.println("Ten most predictive features for positive label:");
        for (int i = mostPredictiveWords.size() - 1; i > mostPredictiveWords.size() - 11; i--){
            System.out.println(mostPredictiveWords.get(i) + "\t" + values.get(i));
        }

        // print out the top 10 most predictive features for negative
        System.out.println("Ten most predictive features for negative label:");
        for (int i = 0; i < 10; i++){
            System.out.println(mostPredictiveWords.get(i) + "\t" + values.get(i));
        }

    }


    /**
     * Print all the probabilities in the trained model
     */
    public void printProbs() {
        // initialize variables
        double posProb;
        double negProb;

        // print the probabilities of the labels
        System.out.println("p(positive)\t" + numPos / (numPos + numNeg));
        System.out.println("p(negative)\t" + numNeg / (numPos + numNeg));

        // print all p(*|positive)
        for (String word : posCounts.keySet()) {
            posProb = posCounts.get(word) / (numPosWords + (lambda * numUniqueWords));
            System.out.println("p(" + word + "|positive)\t" + posProb);
        }

        // print all p(*|negative)
        for (String word : negCounts.keySet()) {
            negProb = negCounts.get(word) / (numNegWords + (lambda * numUniqueWords));
            System.out.println("p(" + word + "|negative)\t" + negProb);
        }
    }


    /**
     * Calculate and print predicted label and probabilities for each sentence
     *
     * @param test_sentences the file path of the input sentences to classify
     * @throws IOException if there is an error reading the input file
     */
    public void calcProbs(String test_sentences) throws IOException {
        // Read target words and their weighting/similarity measures from file and add them to targetInfo
        File file = new File(test_sentences);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st = br.readLine();

        // initialize variables
        double posProbSum;
        double negProbSum;
        double posProbTheta;
        double negProbTheta;

        // for each sentence in the file
        while (st != null) {
            // reset probabilities
            posProbSum = 0.0;
            negProbSum = 0.0;

            // split sentence into list of words
            String[] words = st.split("\\s+");

            // for each word in the sentence
            for (String word : words) {
                // add the lob prob of p(*|positive) to the positive probability if it exists
                if (posCounts.containsKey(word)) {
                    posProbTheta = posCounts.getOrDefault(word, lambda) / (numPosWords + (lambda * numUniqueWords));
                    posProbSum += Math.log10(posProbTheta);
                }

                // add the log prob of p(*|negative) to the negative probability if it exists
                if (negCounts.containsKey(word)) {
                    negProbTheta = negCounts.getOrDefault(word, lambda) / (numNegWords + (lambda * numUniqueWords));
                    negProbSum += Math.log10(negProbTheta);
                }
            }

            // add the probability of the label to the corresponding probability sum
            posProbSum += Math.log10(numPos / (numPos + numNeg));
            negProbSum += Math.log10(numNeg / (numPos + numNeg));

            // classify the sentence as the label with the larger probability (tie goes to positive)
            if (posProbSum >= negProbSum) {
                System.out.println("positive\t" + posProbSum);
            } else {
                System.out.println("negative\t" + negProbSum);
            }

            st = br.readLine();
        }
    }


    /**
     * This method implements the Bernoulli Naive Bayes algorithm to classify a given set of sentences as positive or negative.
     * @param test_sentences the file path of the input sentences to classify.
     * @throws IOException if there is an error reading the input file.
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


    /**
     * Sorts an ArrayList of Doubles in non-decreasing order using the modified quicksort algorithm,
     * and updates a corresponding ArrayList of Strings to maintain their relation.
     *
     * @param arr the ArrayList of Doubles to be sorted
     * @param low the starting index of the sublist to be sorted
     * @param high the ending index of the sublist to be sorted
     * @param wordArr the ArrayList of Strings containing the corresponding words
     */
    public static void modQuickSort(ArrayList<Double> arr, int low, int high, ArrayList<String> wordArr) {
        if (low < high) {
            // Choose a pivot element and partition the list
            int pivotIndex = partition(arr, low, high, wordArr);
            // Recursively sort the left and right sublists
            modQuickSort(arr, low, pivotIndex - 1, wordArr);
            modQuickSort(arr, pivotIndex + 1, high, wordArr);
        }
    }


    /**
     * Partitions an ArrayList of Doubles around a pivot element, and updates a corresponding
     * ArrayList of Strings to maintain their relation.
     *
     * @param arr the ArrayList of Doubles to be partitioned
     * @param low the starting index of the sublist to be partitioned
     * @param high the ending index of the sublist to be partitioned
     * @param wordArr the ArrayList of Strings containing the corresponding words
     * @return the index of the pivot element after partitioning
     */
    public static int partition(ArrayList<Double> arr, int low, int high, ArrayList<String> wordArr) {
        // Choose the pivot element to be the last element in the sublist
        double pivot = arr.get(high);
        // i is the index of the last element in the left sublist
        int i = low - 1;
        for (int j = low; j < high; j++) {
            // If the current element is less than or equal to the pivot, move it to the left sublist
            if (arr.get(j) <= pivot) {
                i++;
                // Swap the current element with the first element in the right sublist
                double temp = arr.get(i);
                arr.set(i, arr.get(j));
                arr.set(j, temp);

                // Update the corresponding word array
                String temp2 = wordArr.get(i);
                wordArr.set(i, wordArr.get(j));
                wordArr.set(j, temp2);
            }
        }
        // Swap the pivot element with the first element in the right sublist
        double temp = arr.get(i + 1);
        arr.set(i + 1, arr.get(high));
        arr.set(high, temp);

        // Update the corresponding word array
        String temp2 = wordArr.get(i + 1);
        wordArr.set(i + 1, wordArr.get(high));
        wordArr.set(high, temp2);

        // Return the index of the pivot element after partitioning
        return i + 1;
    }


    public static void main(String[] args) throws IOException {
        String training_data = "/Users/ezraford/Desktop/School/CS 159/Naive-Bayes-Classifier/assign7-starter/simple.data";
        String test_sentences = "/Users/ezraford/Desktop/School/CS 159/Naive-Bayes-Classifier/assign7-starter/test.data";
        double lambda = 0.0;

        Classifier nb = new Classifier(training_data, test_sentences, lambda);
    }
}
