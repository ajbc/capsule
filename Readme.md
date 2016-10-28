
(C) Copyright 2016, Allison J.B. Chaney

This software is distributed under the MIT license. See `LICENSE.txt` for details.

#### Repository Contents
- `dat`
- `doc`
- `out`
- `scripts`
- `src`
- `Readme.md` this file


## Documentation

The `doc` folder contains the LaTex source for our [EMNLP paper](https://github.com/ajbc/capsule/blob/master/doc/EMNLP/emnlp2016_combined.pdf), including source for generating the figures.  
This is the best resource to learn about the Capsule model and inference details, 
and may be cited as follows.

```
@inproceedings{Chaney2016,
    author = {Chaney, Allison J.B. and Wallach, Hanna and Connelly, Matthew and Blei, David M.},
    title = {Detecting and Characterizing Events},
    booktitle = {EMNLP},
    year = {2016},
}
```

This folder also contains PDF slides for various presentations.  


## Data

#### Real-world Data
The `doc` folder contains `events.csv`, a file containing a list of real-world events with corresponding sources; this is used to check the results of Capsule on the U.S. State Department cables data from the 1970s.

The cables data may be obtained from the [History Lab](http://history-lab.org) at Columbia University, or if you obtain their permission, I can share my processed version of the data.  While the data is publically accessible, The History Lab's version is cleaner.

#### Simulating Data
Absent this data or your own data of interest, you can simulate data using the script `dat/src/simulate_data.py`.
This script has most of the simulation parameters hard-coded on lines 5-16 and takes two command line arguments: the shape of the event decay (`step`, `linear`, or `exp`) and an integer random seed.  The simulated data is created in the same directory that the script is run.
Thus, to create a simulated data set, one should create a directory, move to that directory, and run the script from that directory, such as in the following example.
```
mkdir dat/sim
cd dat/sim
python ../src/simulate_data.py exp 372552
```

#### Data Format
To run Capsule using your own data, four files are needed:
- `meta.tsv`
- `train.tsv`
- `test.tsv`
- `validation.tsv`

The first file, `meta.tsv` is a tab-separated file with three integer-valued columns:
```
doc.id    author.id    time.id
```
This should include the meta-data for all documents included in the training, test, and validation sets.
If time is continuous in your original data, it shoud be binned to include a minimum number of documents (e.g., 10) per titime interval.  It may also be worth omitting authors who have written too few documents (e.g., <5).
When processing your data, you should retain a mapping of these ids to their original values.

The remaining three files are for document word counts; they are also tab-separated with three integer-valued columns:
```
doc.id    term.id    count
```
Each `term.id` refers to the index of a particular vocubuary term; like with topic models, this vocabulary should be chosen with care.  Capsule may require a larger vocabulary than a typical topic model, as terms related to events are more rare.
We recommend spitting terms into roughly 90% training, 9% testing and 1% validation, if your data is sufficiently large.
You should split by (document, vocabulary-term) pairs, not by entire documents; this way, document-specific parmeters are still learned.
If you wish to train on the full data, and do not care about a testing set, the validation and test sets are allowed to contain duplicate data.

If you intend to use the [Capsule visualization](https://github.com/ajbc/capsule-viz), you should check that your author, time, and vocabulary term mappings are all consistent with its required format.


## Running Capsule
1. Clone the repo:
    `git clone https://github.com/ajbc/capsule.git`
2. Navigate to the `capsule/src` directory
3. Compile with `make`
4. Run the executable, e.g.:
    `./capsule --data ~/my-data/ --out my-fit`

Compilation requires [Armadillo](http://arma.sourceforge.net), a C++ linear algebra library.

A note on notation: the paper uses γ (gamma) to represent event topics, but to avoid confusion with the gamma distribution, the code uses `pi` to represent this same variable.

#### Capsule Options
|Option|Arguments|Help|Default|
|---|---|---|---|
|help||print help information||
|verbose||print extra information while running|off|
|out|dir|save directory, required||
|data|dir|data directory, required||
|svi||use stochastic VI (instead of batch VI)|off for < 10M doc-term counts in training|
|batch||use batch VI (instead of SVI)|on for < 10M doc-term counts in training|
|a_phi|a|shape hyperparameter to phi (entity general concerns)|0.3|
|b_phi|b|rate hyperparameter to phi (entity general concerns)|0.3|
|a_xi|a|shape hyperparameter to xi (entity-specific concern)|0.3|
|b_xi|b|rate hyperparameter to xi (entity-specific concern)|0.3|
|a_psi|a|shape hyperparameter to psi (event strength)|0.3|
|b_psi|b|rate hyperparameter to psi (event strength)|0.3|
|a_theta|a|shape hyperparameter to theta (documents' general topics)|0.3|
|a_zeta|a|shape hyperparameter to zeta (documents' entity topics)|0.3|
|a_epsilon|a|shape hyperparameter to epsion (documents' event topics)|0.3|
|a_beta|a|hyperparameter to beta (general topics)|0.3|
|a_eta|a|shape hyperparameter to eta (entity topics)|0.3|
|a_pi|a|hyperparameter to pi (event topics; gamma in paper)|0.3|
|no_topics||don't consider general topics|include general topics|
|no_entity||don't consider entity topics|include entity topics|
|no_events||don't consider event topics|include event topics|
|event_dur|d|event duration|7|
|event_decay|d|event decays; options: exponential, linear, step|exponential|
|seed|seed|the random seed|time|
|save_freq|f|the saving frequency.  Negative value means no savings for intermediate results.|20|
|eval_freq|f|the intermediate evaluating frequency. Negative means no evaluation for intermediate results.|-1|
|conv_freq|f|the convergence check frequency|10|
|max_iter|max|the max number of iterations|300|
|min_iter|min|the min number of iterations|30|
|converge|c|the change in rating log likelihood required for convergence|1e-6|
|final_pass||do a final pass on all users and items|no final pass|
|overwrite||overwrite old results|keep only latest|
|sample|sample_size|the stochastic sample size|1000|
|svi_delay|tau|SVI delay >= 0 to down-weight early samples|1024|
|svi_forget|kappa|SVI forgetting rate (0.5,1]|default 0.75|
|K|K|the number of general topics|100|

<!---
## Evaluating and Exploring the Results
TODO

#### Comparing against baselines
TODO

#### Visualization
We have developed a pipeline for [visualizing Capsule](https://github.com/ajbc/capsule-viz), which includes the model results alongide the original data.
-->
