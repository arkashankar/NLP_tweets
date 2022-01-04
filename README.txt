**This is the dataset is used for the FIRE 2018 IRMiDis track. Anyone using this dataset should cite the overview paper of FIRE 2018 IRMiDis Task --

Moumita Basu, Saptarshi Ghosh, Kripabandhu Ghosh: Overview of the FIRE 2018 track: Information Retrieval from Microblogs during Disasters (IRMiDis). FIRE 2018: 1-5

whose bibtex entry is as follows:

@inproceedings{Basu:2018:OFT:3293339.3293340,
 author = {Basu, Moumita and Ghosh, Saptarshi and Ghosh, Kripabandhu},
 title = {Overview of the FIRE 2018 Track: Information Retrieval from Microblogs During Disasters (IRMiDis)},
 booktitle = {Proceedings of the 10th Annual Meeting of the Forum for Information Retrieval Evaluation},
 series = {FIRE'18},
 year = {2018},
 isbn = {978-1-4503-6208-5},
 location = {Gandhinagar, India},
 pages = {1--5},
 numpages = {5},
 url = {http://doi.acm.org/10.1145/3293339.3293340},
 doi = {10.1145/3293339.3293340},
 acmid = {3293340},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Disaster, FIRE, Fact-checkable, Microblogs, Multi-source data},
} 


********************************************************************* Task Description **************************************************************

Microblogging sites like Twitter are increasingly being used for aiding relief operations during disaster events. In such situations, identification of fact-checkable  tweets i.e tweets that report some relevant and verifiable fact other than sympathy or prayer is extreamly important for effective coordination of post disaster relief operations.  However, such critical information is usually submerged within a lot of conversational content. Hence, automated IR techniques are needed to find and process such information. The track will have two sub-tasks, as described below.

(1) Identifying fact-checkable tweets, i.e., tweets which report a fact that can be verified (and not just sympathy or prayer)
(2) For each fact-checkable tweet, identify news articles that support/verify the fact (if any exists)
Basically (1) is a classification problem. (2) is a retrieval problem - a tweet can be either verified.
The track will have two sub-tasks, as described below.

Sub-task 1: Identifying fact-checkable tweets: Here the participants were needed to develop automatic methodologies for identifying fact-checkable tweet. This is mainly a search problem, where relevant microblogs have to be retrieved. However, apart from search, the problem of identifying  act-checkable can also be viewed as a pattern matching problem, or a classification problem (e.g., where tweets are classified into three classes – fact-checkable tweets and others).

Sub-task 2: Identification of supporting news article for each fact-checkable tweets : A fact-checkable  tweet is said to supported/verified by  a news article, if the same fact  is represented by  both of these media. In this sub-task, the participants were required to develop methodologies for matching fact-checkable tweets  with appropriate news articles.

Participants could take part in sub-task 1 only or in both the tasks. However, to participate in sub-task 2, it was mandatory to participate in sub-task 1.

******************************************************* Data************************************************************************************

This folder contains, apart from this README.txt file, the following files and sub-folders:

(1) File: nepal-quake-2015-tweets.jsonl

This file contains the JSON objects of 50,068 English tweets posted immediately after the earthquake. The JSON objects are as returned by the Twitter API. Each line of the file is a JSON object.

(2) File: FIRE2018-Task1-qrel

This is the qrel file corresponding to the gold standard of Task 1. 
qrel file is  formatted   according   to   the   traditional   TREC format, which is as follows. 

===================
Each line  have four (04) fields separated by one space character​. The fields are: 
(1) a topic­id, e.g.,  Nepal_Factual
(2) the specific string "0" 
(3) a tweet­id, which is judged relevant to the topic­id in the first field 
(4) a relevance­ based   grade   for   the   tweet­id, corresponding to the topic­id. Grades are assigned as 1, 2 and 3 since the Gold standard is prepared considering graded relevance of tweets described as below-- 

1- Relevant (fact-checkable) tweets but without Nepal-related information
2- Relevant (fact-checkable) tweets containing Nepal-related information
3- Relevant and highly fact-checkable tweets having specific reference of source, location, organization,quantity, resource name etc.
==================

(3) Sub-folder: nepal-quake-2015-news-articles

This sub-folder contains 6,866 news articles related to the Nepal earthquake, that were posted online by various news media sites. Each article is a plain text file having the following information enclosed in tags (similar to HTML or XML tags):

=================
<url>
URL of the news article
</url>

<date>
Date on which the article was published, in the format yyyymmdd (e.g., 20150425)
</date>

<headline>
Headline of the news article
</headline>

<text>
Body of the news article
</text>
=================


NOTE:

* We have made our best effort to include only tweets and news articles that are related to the 2015 Nepal earthquake. However, it is possible that some unrelated tweets / news articles are present in the dataset.

* This folder contains the gold standard corresponding to Task 1. We pooled top 100 results from each run, and then three human annotators judged the fact-checkability of the tweets

* Task 2 of IRMiDis track this year, only one run was submitted. Thus, pooling was employed on only one run to create the gold-standard

* Runs submitted to the FIRE 2018 IRMiDis track had to use this dataset. Runs were free to use any other data, but results had to be reported on this dataset.