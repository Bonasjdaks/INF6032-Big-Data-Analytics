# Databricks notebook source
# MAGIC %md
# MAGIC Startup Requirments

# COMMAND ----------

#import pyspark session
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

#all functions that need to be imported for the code to properly run
from pyspark.ml.feature import NGram, StopWordsRemover
from pyspark.sql.functions import col, concat_ws, count, desc, explode, lower, regexp_replace


from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC Uploading the dataset

# COMMAND ----------

#Uploading the data files
large = spark.read.format ('csv') \
    .option("header", "True") \
        .load("dbfs:/FileStore/shared_uploads/bdavies5@sheffield.ac.uk/large.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC Question 1 - Calculate the number of different sentences in the dataset

# COMMAND ----------

#counting the number of distinct rows in the large dataset
No_sentences_L = large.select('sentence').distinct().count()

#outputing the value collected
print(f"\nThe number of sentences in the Large dataset is: { No_sentences_L}")

# COMMAND ----------

#further exploration

#counting the number of rows in the large dataset
No_sentences_L = large.select('sentence').count()

#outputing the value collected
print(f"\nThe number of duplicate sentences in the Large dataset is: { No_sentences_L}")

# COMMAND ----------

#importing the required functions for finding most common duplicate
from pyspark.sql.functions import col, desc

#group by sentences, so duplciates will be grouped together, and count the number of duplciates
sentence_grouped = large.groupBy("sentence").count()

#filter the dataset so that only the duplciates remain
duplicate_sentences = sentence_grouped.filter(col("count") > 1)

#sort by most common to least common duplicate
most_common_duplicate = duplicate_sentences.orderBy(desc("count")).limit(1)

#output the results
most_common_duplicate.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Question 2 - List the numbers of words in the 10 longest sentences

# COMMAND ----------

#split sentences into individual words in an array and count the words in each sentence
large_array = large.withColumn("sentence", F.split(F.col("sentence"), " ")) \
                   .withColumn("large_word_count", F.size("sentence"))

#sort by the word count in descending order, largest to smallest
top_10_sentences_L = large_array.orderBy(F.desc("large_word_count"))

#Output top 10 sentences
top_10_sentences_L.select("large_word_count").show(10)

# COMMAND ----------

#Further Investigation

#printing out the longest sentencez in full
row = top_10_sentences_L.select("sentence").first()
sentence_array = row["sentence"]
sentence_clean = " ".join(word.strip('"') for word in sentence_array)
print(sentence_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC Question 3 - The average number of bigrams per sentence across the dataset

# COMMAND ----------

#Answer to question three using large dataset
from pyspark.ml.feature import NGram
from pyspark.sql.functions import count, col, explode

#remove punctuation from the dataset
large_no_punctuation = large.withColumn("sentence", regexp_replace("sentence", r"[^\w\s]", ""))

#seperate out words into individual tokens
large_array = large_no_punctuation.withColumn("sentence", F.split(col("sentence"), " "))


#create the bigrams
ngram = NGram(n=2, inputCol="sentence", outputCol="bigrams")
large_bigrams = ngram.transform(large_array)

#explode bigram column
large_exploded = large_bigrams.select(explode(col("bigrams")))


#find the average
No_sentences_L = large.select('sentence').count()
no_bigrams_L = large_exploded.select('col').count()

total_L = no_bigrams_L/No_sentences_L

#output
print(f"Average bigrims in large dataset is: {total_L}")

# COMMAND ----------

#futher exploration - avergae number of words per sentence
from pyspark.sql.functions import explode

#explode the tokenised word column
word_exploded = large_array.select((explode("sentence")).alias('sentence'))

#count number of words
no_words = word_exploded.select('sentence').count()

#find average and output
total_w = no_words/No_sentences_L
print(f"The averge number of words per sentence is: {total_w}")

# COMMAND ----------

# MAGIC %md
# MAGIC Question 4 - The 10 most frequent bigrams in the dataset

# COMMAND ----------

#functions imported
from pyspark.ml.feature import NGram
from pyspark.sql.functions import concat_ws, count, explode, col, lower

#removal of punctuation
large_no_punctuation = large.withColumn("sentence", regexp_replace("sentence", r"[^\w\s]", ""))

#case normalisation
large_lower = large_no_punctuation.withColumn("sentence_lower", lower(col("sentence")))

#tokenisation
large_array = large_lower.withColumn("sentence", F.split(F.col("sentence_lower"), " "))

#create the bigrams
ngram = NGram(n=2, inputCol="sentence", outputCol="bigrams")
large_bigrams = ngram.transform(large_array)


#make each bigram a row
large_exploded = large_bigrams.select(explode(col("bigrams")).alias("bigram"))

#convert the bigrams from an array to string to count
large_exploded_str = large_exploded.withColumn("bigram_str", concat_ws(" ", col("bigram")))

#count most frequent bigrams
large_count = large_exploded_str.groupBy("bigram_str").agg(count("*").alias("count")).orderBy(F.desc("count"))


#output
large_count.show(10)

# COMMAND ----------

#further investigaion, removal of stopwords

#functions imported
from pyspark.ml.feature import NGram, StopWordsRemover
from pyspark.sql.functions import concat_ws, count, explode, col, lower

#removal of punctuation
large_no_punctuation = large.withColumn("sentence", regexp_replace("sentence", r"[^\w\s]", ""))

#case normalisation
large_lower = large_no_punctuation.withColumn("sentence_lower", lower(col("sentence")))

#tokenisation
large_array = large_lower.withColumn("sentence", F.split(F.col("sentence_lower"), " "))

#removal of stopwords
remover = StopWordsRemover(inputCol="sentence", outputCol="sentence_filtered")
large_array_removed = remover.transform(large_array)

#create the bigrams
ngram = NGram(n=2, inputCol="sentence_filtered", outputCol="bigrams")
large_bigrams = ngram.transform(large_array_removed)


#make each bigram a row
large_exploded = large_bigrams.select(explode(col("bigrams")).alias("bigram"))

#convert the bigrams from an array to string to count
bigrams_str = large_exploded.withColumn("bigram_str", concat_ws(" ", col("bigram")))

#remove empty rows
cleaned_bigrams_str = bigrams_str.filter(F.trim(col("bigram_str")) != "")

#count most frequent bigrams
large_count = cleaned_bigrams_str.groupBy("bigram_str").agg(count("*").alias("count")).orderBy(F.desc("count"))


#output
large_count.show(10)

# COMMAND ----------

#Further Investigation - most common trigrams

#functions imported
from pyspark.ml.feature import NGram, StopWordsRemover
from pyspark.sql.functions import concat_ws, count, explode, col, lower

#removal of punctuation
large_no_punctuation = large.withColumn("sentence", regexp_replace("sentence", r"[^\w\s]", ""))

#case normalisation
large_lower = large_no_punctuation.withColumn("sentence_lower", lower(col("sentence")))

#tokenisation
large_array = large_lower.withColumn("sentence", F.split(F.col("sentence_lower"), " "))

#removal of stopwords
remover = StopWordsRemover(inputCol="sentence", outputCol="sentence_filtered")
large_array_removed = remover.transform(large_array)

#create the trigrams
ngram = NGram(n=3, inputCol="sentence_filtered", outputCol="trigrams")
large_trigrams = ngram.transform(large_array_removed)


#make each trigram a row
trigram_exploded = large_trigrams.select(explode(col("trigrams")).alias("trigram"))

#convert the trigrams from an array to string to count
trigrams_str = trigram_exploded.withColumn("trigram_str", concat_ws(" ", col("trigram")))

#remove empty rows
clean_trigrams_str = trigrams_str.filter(F.trim(col("trigram_str")) != "")

#count most frequent bigrams
large_count = clean_trigrams_str.groupBy("trigram_str").agg(count("*").alias("count")).orderBy(F.desc("count"))


#output
large_count.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Question 5 - Find out how many of the bigrams you’ve extracted from the Wikipedia subset appear in the list of idioms contained in the MAGPIE subset

# COMMAND ----------

#read magpie dataset
magpie = spark.read.json("dbfs:/FileStore/shared_uploads/bdavies5@sheffield.ac.uk/MAGPIE_unfiltered.jsonl")

#extract the idioms from magpie
idioms = magpie.select('idiom')

#join the idioms and wikipedia bigrams dataset
idioms = idioms.withColumn('bigram_str', idioms['idiom'])
join_matching = idioms.join(large_exploded_str, on="bigram_str", how="inner")

#count the distinct matching values
matching = join_matching.select("bigram_str").distinct().count()
print(f"The number of wikipedia bigrams that are also idioms are: {matching}")

# COMMAND ----------

#further exploration, what are the bigrams that are also idioms, sorted by frequenxy

#get frequency of each idiom
idiom_frequency = join_matching.groupBy("bigram_str").agg(F.count("*").alias("frequency"))


#sort by descending frequency
sorted_idioms = idiom_frequency.orderBy(F.desc("frequency"), F.asc("bigram_str"))

#output
display(sorted_idioms)

# COMMAND ----------

#finding the genre of each idiom in the large bigrams dataset

#select both idioms and bigrams from MAGPIE
idiom_genre = magpie.select('idiom', 'genre'). withColumnRenamed('idiom', 'bigram_str')

#join datasets
joined = idiom_genre.join(large_exploded_str, on='bigram_str', how='inner')

#get the frequency of each idiom
idiom_count = joined.groupBy('bigram_str', 'genre').agg(F.count("*").alias('frequency'))

#sort by most frequent and top 10
top10_idiom_genre = idiom_count.orderBy(F.desc("frequency"), F.asc("bigram_str")).limit(10)

display(top10_idiom_genre)

# COMMAND ----------

# MAGIC %md
# MAGIC Question 6 - Ensuring that you are only considering the bigrams that appear in Wikipedia and not in MAGPIE, print out the 10 bigrams starting from rank 2500 when these are ordered by decreasing frequency

# COMMAND ----------

#new dataset that only includes bigrams not found in the Magpie idiom dataset
non_idiom_bigrams = large_exploded_str.join(idioms, on="bigram_str", how="left_anti")

#get the frequency of these bigrams
bigram_count = non_idiom_bigrams.groupBy("bigram_str").agg(F.count("*").alias("frequency"))

#rank the bigrams by frequency
ranked_bigrams = bigram_count.orderBy(F.desc("frequency"), F.asc("bigram_str"))

#output 10 bigrams starting from rank 2500
top10 = ranked_bigrams.limit(2510).tail(10)

for row in top10:
    print(f"{row['bigram_str']}")

# COMMAND ----------

#how many unique bigrams are not in the magpie idoms dataset
unique_wiki_bigrams_count = non_idiom_bigrams.select('bigram').distinct().count()

print(f"There are {unique_wiki_bigrams_count} bigrams not found in the MAGPIE idoms list")


# COMMAND ----------

#DO the bigrams that share the same starting letter (The) have the same frequency
#output 10 bigrams starting from rank 2500
top10 = ranked_bigrams.limit(2510).tail(10)

for row in top10:
    print(f"{row['bigram_str']} - {row['frequency']}")

# COMMAND ----------

# MAGIC %md
# MAGIC data cleanup techniques

# COMMAND ----------

#Data cleanup methods used during this report
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import col, lower, regexp_replace

#case normalisation
large = large.withColumn("sentence_lower", lower(col("sentence")))

#Removal of stopwords
remover = StopWordsRemover(inputCol="sentence", outputCol="sentence_filtered")
large_stopwords_removed = remover.transform(large_array)

#removal of punctuation
large_no_punctuation = large.withColumn("sentence", regexp_replace("sentence", r"[^\w\s]", ""))