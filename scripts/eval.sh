echo "data,method,value"
for id in 00 01 02 03 04 05 06 07 08 09
do
    v=`python eval.py ../../simulated/dat/fig1/v$id/event_strength.tsv v$id/capsule/psi-final.dat inv`
    echo "$id,capsule,$v"

    #v=`python eval.py ../../simulated/dat/fig1/v$id/event_strength.tsv v$id/entity_only/psi-final.dat inv`
    #echo "$id,entity_only,$v"

    #v=`python eval.py ../../simulated/dat/fig1/v$id/event_strength.tsv v$id/event_only/psi-final.dat inv`
    #echo "$id,event_only,$v"

    for base in ave_deviation total_deviation ave_deviation_entity lda_ent_MG_ave total_deviation_tfidf ave_deviation_entity_tfidf lda_ent_MG_max word_outlier ave_deviation_tfidf lda_MG_ave word_outlier_entity_ave ave_dev_lda lda_MG_max word_outlier_entity_ave_tfidf ave_ent_dev_lda max_deviation_entity word_outlier_entity_max doc_ent_outlier_lda max_deviation_entity_tfidf word_outlier_entity_max_tfidf doc_outlier_lda random word_outlier_tfidf
    do
        v=`python eval.py ../../simulated/dat/fig1/v$id/event_strength.tsv v$id/baselines/$base.dat inv`
        echo "$id,$base,$v"
    done
done
