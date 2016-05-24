# test:
# sh baselines.sh ../simulated/dat/fig1/v00/ .
dat=$1
out=$2
K=10 #$3

mkdir $2
rm -rf $2/baselines/
mkdir $2/baselines

#procss data to mult format
python2.7 process_to_mult.py $1/train.tsv $1/mult.dat

# wget http://www.cs.princeton.edu/~blei/lda-c/lda-c-dist.tgz
# tar -xzvf lda-c-dist.tgz
# cd lda-c-dist; make; cd ..
lda-c-dist/lda est 0.1 $K lda-c-dist/settings.txt $1/mult.dat random $2/baselines/lda

python2.7 baselines.py $dat $2/baselines
