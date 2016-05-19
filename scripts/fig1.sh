# read args
dur=10
shape="exp" #step, linear, exp
shapeF="exponential" #none, linear, exponential
seed=139563874
M=10
K=10
shp=1.0 # bigger => bigger mean
rte=5.0 # bigger => smaller mean, lower variance
lcl=0.1
aph=0.01 # controls proportion; bigger means more even topics, smaller means less even
totalD=100 # total number of days

rm *.out
rm *.err
rm -rf sim
mkdir sim
rm -rf fit
mkdir fit

# set up LDA infrastructure
#wget http://www.cs.princeton.edu/~blei/lda-c/lda-c-dist.tgz
#tar -xzvf lda-c-dist.tgz
#cd lda-c-dist; make; cd ..

## run capsule for each simulated dataset
#echo "***** RUNNING CAPSULE and baselines *****"
for id in `seq 1 $M`
do
    echo "****** dataset $id ******"
    r=$RANDOM
    echo " - simualting data with seed $r"
    # simualte a dataset
    mkdir sim/$id
    cd sim/$id
    python2.7 ../../simulate_data.py $dur $shape $r $K $shp $rte $lcl $aph $totalD > tmparoo
    cd ../../

    echo " - kicking off capsule fits"

    (../src/capsule --K $K --batch --a_phi $shp --b_phi $rte --a_psi $shp --b_psi $rte --a_theta $lcl --a_epsilon $lcl --a_pi $aph --a_beta $aph --data sim/$id --out fit/$id/capsule --event_dur $dur --seed $seed --conv_freq 1 --min_iter 100 --max_iter 100 --event_decay $shapeF > full"$id".out 2> full"$id".err &)
    (../src/capsule --K $K --batch --a_phi $shp --b_phi $rte --a_psi $shp --b_psi $rte --a_theta $lcl --a_epsilon $lcl --a_pi $aph --a_beta $aph --data sim/$id --out fit/$id/entity_only --event_dur $dur --seed $seed --conv_freq 1 --min_iter 100 --max_iter 100 --event_decay $shapeF --entity_only > ent"$id".out 2> ent"$id".err &)
    (../src/capsule --K $K --batch --a_phi $shp --b_phi $rte --a_psi $shp --b_psi $rte --a_theta $lcl --a_epsilon $lcl --a_pi $aph --a_beta $aph --data sim/$id --out fit/$id/event_only --event_dur $dur --seed $seed --conv_freq 1 --min_iter 100 --max_iter 100 --event_decay $shapeF --event_only > evt"$id".out 2> ev"$id".err &)


    mkdir fit/$id
    mkdir fit/$id/baselines

    echo " - processing data to mult format"
    # process data to mult format
    python2.7 process_to_mult.py sim/$id/train.tsv sim/$id/mult.dat > tmparoo

    # run LDA
    echo " - fitting LDA"
    lda-c-dist/lda est 0.1 $K lda-c-dist/settings.txt sim/$id/mult.dat random fit/$id/baselines/lda > tmparoo

    echo " - computing baselines"
    python2.7 baselines.py sim/$id/ fit/$id/baselines > tmparoo
done

echo "*************************************"
ps ux
echo "*************************************"
read -p "Please confirm all fits have completed:" response
echo "*************************************"
#pr=`ps ux | grep capsule | wc -l`
#echo $pr
#while [ $pr != "1" ]
#do
#    echo "waiting for capsule to finish ($pr processes)"
#    sleep 1m
#    pr=`ps ux | grep capsule | wc -l`
#done

echo "***** EVALUATING ALL MODELS *****"
echo "data,method,value" > results.csv
for id in `seq 1 $M`
do
    v=`python eval.py sim/$id/event_strength.tsv fit/$id/capsule/psi-final.dat inv`
    echo "$id,capsule,$v" >> results.csv

    v=`python eval.py sim/$id/event_strength.tsv fit/$id/event_only/psi-final.dat inv`
    echo "$id,event_only,$v" >> results.csv

    for base in ave_deviation total_deviation ave_deviation_entity lda_ent_MG_ave total_deviation_tfidf ave_deviation_entity_tfidf lda_ent_MG_max word_outlier ave_deviation_tfidf lda_MG_ave word_outlier_entity_ave ave_dev_lda lda_MG_max word_outlier_entity_ave_tfidf ave_ent_dev_lda max_deviation_entity word_outlier_entity_max doc_ent_outlier_lda max_deviation_entity_tfidf word_outlier_entity_max_tfidf doc_outlier_lda random word_outlier_tfidf
    do
        v=`python eval.py sim/$id/event_strength.tsv fit/$id/baselines/$base.dat n`
        echo "$id,$base,$v" >> results.csv
    done
done
