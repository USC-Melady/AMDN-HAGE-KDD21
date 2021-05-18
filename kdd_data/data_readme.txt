The full data pickle file and tweetids.csv (and train and test sequences are in https://drive.google.com/drive/folders/1PrG50ZJdRP7-tmUMj2sqcMfQ1NMCTKVN?usp=sharing due to size of the files). dev pkl is provided here as a sample with the readme.


The Tweets collected on COVID-19 with 119,298 accounts and 13.9 M tweets from these accounts.
In accordance with Twitter the Tweet IDS are provided. Twitter API can be used to get the full tweet payload.

The data files include:

1) tweetids.csv (tweetid, ns_label): We also provide disinformation (unreliable / conspiracy) for tweets with links to low-credibility sources (used in analysis)

2) pkl_files (directory with activity sequences)

- the train, validation and test activitiy traces (used for training and validation loss in AMDN-HAGE are provided)
- the userids are indexed to conform with Twitter policies.
- tweetids of activity sequence's (first post i.e. tweet in activity traces) are provided


Note: Activity traces can be recontructed with non-anonymized userids using Twitter API as follows:
- Use Twitter API to obtain tweet payload (contains retweet, reply quoted tweetid links)
- Similar to details in the paper, the activity traces can be obtained using the above information (as any account’s posts with engagements (retweets, replies, quotes of the post) forming a time-ordered sequence of activities
- Sequences can be put into each split used in the paper based on the tweetid provided in the pickle files containing sequence splits.

