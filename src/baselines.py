# read in args

# read in meta

# read in training data

### calculate baselines
# random

# greatest wordcount outlier for the day
# average worcount deviation for the day
# greatest wordcount outlier for the day (by entity)
# entity average of greatest wordcount outlier for the day
# average of entity wordcount deviations for day (i.e., weighted by entity volume)
# twice averaged entity wordcount deviations (i.e., all entities are the same)

# tf-idf weighted versions of all of the above
# lda, but use topics instead of words for all of the above

# multinomial gaussian (compute mu and sigma) for
    # wordcounts
    # entity wordcounts
    # tf-idf weighted wordcounts
    # tf-idf weighted entity wordcounts
    # lda topics
# and then for each document, compute it's probability and report
    # greatest inverse doc prob for the day
    # average inverse doc prob for the day
# so this gives us 10 baselines usng multinomial gaussians

