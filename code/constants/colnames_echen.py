### tweet features [missing rt_count, favorite_count]
TID = 'tweetid'
DATE = 'date'
LANG = 'lang'
EXT = 'extended'
TT = 'tweet_type'
CTT = 'corrected_tweet_type'
TEXT = 'text'
HASHTAG = 'hashtag'
URLS = 'urls_list'
MENTIONS_ID = 'mentionid'
MENTIONS_NAME = 'mentionsn'

### account features
UID = 'userid'
SCREEN_NAME = 'screen_name'
ACC_DATE = 'account_creation_date'
FIRSTT_DATE = 'date_first_tweet'
VERIFIED = 'verified'
LOC = 'location'
DISP_NAME = 'display_name'
# 'friends_count', 'listed_count', 'followers_count', 'favourites_count', 'statuses_count'

### link features [always immediate parent link]
RP_TID = 'reply_statusid'
RT_TID = 'rt_tweetid'
QT_TID = 'qtd_tweetid'
RP_UID = 'reply_userid'
RT_UID = 'rt_userid'
QT_UID = 'qtd_userid'

# field names
TT_OG = 'original'
TT_RT = 'retweeted_tweet_without_comment'
TT_RP = 'reply'
TT_QT = 'quoted_tweet'
TT_TOTAL = 'total'
DATETIME = 'datetime'

### linked tweet features [missing for replied to tweet]
# 'rt_text', 'rt_hashtag', 'rt_urls_list', 'rt_qtd_count', 'rt_rt_count', 'rt_reply_count', 'rt_fav_count', 'rt_location','rt_media_urls'
# 'qtd_text', 'qtd_hashtag', 'qtd_urls_list', 'qtd_qtd_count', 'qtd_rt_count', 'qtd_reply_count', 'qtd_fav_count', 'qtd_location', 'q_media_urls'

# additional fields for disinformation labels by news source
NS_LABEL = 'ns_label'
NS_URL = 'ns_url'
TROLL_LABEL = 'troll_label'
