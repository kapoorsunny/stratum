# Chapter 19. A real enterprise run, start to finish

This chapter is one run, written down exactly as it happened, with every command and what it printed. Sixteen departments, twelve people, a hundred and twenty two documents.

Everything uses public Wikipedia material, so you can run the same commands and get the same result. Nothing here needs a GPU. The whole thing finishes on a laptop.

If you have not read [Chapter 17](17-access-control-and-context.md) yet, read it first. It explains compartments, tiers and why a weight cannot be filtered. This chapter assumes those words mean something to you.

---

## The problem being solved

A company has forty departments and twenty thousand staff. A person belongs to up to five departments plus a project, so more than three at a time is normal rather than exceptional.

That last number is the whole difficulty. Merging LoRA adapters stops working somewhere around three. Merge four and the model degrades. Merge five or six at full weight and it produces nothing usable. So an adapter per department cannot work, and an adapter per permission combination is worse, because five departments out of forty is over six hundred thousand combinations before you count the projects.

The way out is that access and skill got conflated and are not the same quantity.

A person needs six compartments of **access**, which is a filter over rows and costs nothing to change. They do not need six adapters of **skill**, because departments differ far more in what they hold than in how they write. Engineering, maintenance, quality and instrumentation all write plant English. One adapter serves all four. Which of the four you are allowed to read is a separate question answered by the index.

So adapters follow language families and access stays in the index. That is the design this chapter runs.

---

## Step zero, the plan file

Everything starts from one file. Write it once, and the folder layout, the access policy and the family grouping all come out of it, so they cannot disagree with each other.

```bash
stratum corpus plan init --out plan.yaml
```

That writes an example to edit. Here is the shape.

```yaml
compartments:

  public:
    tier: company             # everyone reads it, so it may enter shared weights
    family: general
    folders: [raw/public]

  engineering:
    tier: department          # a group reads it, so it gets an adapter
    family: technical         # which cluster carries its language
    description: Rotating equipment and process plant
    folders: [//fileserver/engineering]

  maintenance:
    tier: department
    family: technical         # same family, so one adapter serves both
    folders: [//fileserver/maintenance]

  payroll:
    tier: restricted          # index only, never enters any weights
    volatile: true
    folders: [//fileserver/hr/pay]

principals:

  plant_engineer:
    reads: [public, engineering, maintenance]

  contractor:
    description: Outside the company, sees only what everyone sees
    reads: [public]
```

Three things about that file are worth saying out loud.

**The family sits on the compartment.** It is not a separate list somewhere else. A compartment must belong to exactly one family, because the family decides which adapter carries its language, and written this way that rule holds by construction. There is no way to write a file that puts one department in two clusters.

**Material can come from anywhere.** `folders` points at a share drive and takes Word documents, PDFs, slides, spreadsheets, markdown, plain text, HTML and images. `urls` and `url_files` pull from an intranet wiki or a public standards site. Both work together in the same compartment.

**Two rules will be enforced whether you like them or not.**

Company tier has to be readable by every principal. It goes into shared weights, and a weight cannot be filtered afterwards, so a company tier compartment that one department cannot read is a leak waiting to be discovered.

Department tier needs at least two principals. A compartment one person reads is that person's material. Training a shared adapter on it puts it inside a model other people load. Use restricted instead.

---

## Step one, check the plan before spending an hour on it

```bash
stratum corpus plan check --plan examples/enterprise/plan.yaml
```

```
16 compartment(s) in 7 families

  compartment        family         tier        material
  accounting         finance        department  9 to download
  engineering        technical      department  8 to download
  facilities         operations     department  7 to download
  ...

Families
  commercial           legal, procurement, sales
  delivery             project_atlas
  finance              accounting, treasury
  general              public
  operations           facilities, logistics, production
  people               hr, training
  technical            engineering, instrumentation, maintenance, quality

Adapters each principal would load, if every family is trained
  auditor            reads  4, loads 2 adapter(s)   [finance, general]
                     no commercial adapter, it also carries procurement, sales
                     so legal reaches this principal through the index alone
  contractor         reads  1, loads 1 adapter(s)   [general]
  ops_director       reads  6, loads 2 adapter(s)   [general, technical]
                     no operations adapter, it also carries facilities, production
                     so logistics reaches this principal through the index alone
  reliability_lead   reads  6, loads 3 adapter(s)   [delivery, general, technical]
  ...

  Most any one person loads is 3, inside the safe merge limit.

The access policy this would produce is valid.
```

Read the last line first. **Most any one person loads is 3.** Sixteen departments, people in six compartments each, and the worst case is still three adapters. That is the design working.

Now read the two warnings, because they are more interesting than the pass.

`auditor` reads legal but not procurement or sales. All three are in the `commercial` cluster. So the auditor cannot be given the commercial adapter, because that adapter learned to write from material they are not cleared for.

Nothing breaks. Legal still reaches them, through retrieval instead of through a trained adapter. The answer is a little less fluent and exactly as correct. The tool tells you this rather than quietly handing over the adapter, because handing it over is the kind of decision that is invisible until an audit finds it.

To close a case like that you either grant the whole cluster, or move that department into a cluster whose members share its audience.

Nothing has been downloaded at this point. A plan with a typo in a department name, a folder that does not exist, or a policy the rules reject fails here, which is a far better place to find out than after an eight hour fetch.

---

## Step two, run the whole thing

```bash
stratum build --plan examples/enterprise/plan.yaml --work work/
```

Six steps in order. Each one prints the standalone command that does it alone, so you can rerun any single step while debugging without unpicking a wrapper.

```
[1/6] Read the plan
        stratum corpus plan check --plan examples/enterprise/plan.yaml
  16 compartments, 7 families, 12 principals

[2/6] Lay out the corpus
        stratum corpus plan build --plan examples/enterprise/plan.yaml --out work/corpus
  compartment        family         tier        documents
  accounting         finance        department     9
  engineering        technical      department     8
  ...
                                                 122 in total
  Access policy  -> work/policy.json
  Family spec    -> work/families.json

[3/6] Extract and chunk
        stratum corpus ingest --in work/corpus --out work/chunks --compartments
  Ingested 121 documents and 0 images (0 from cache) -> 2502 chunks
   duplicates: 1  skipped: 0  errors: 0

[4/6] Build the index
        stratum context build --chunks work/chunks/chunks.jsonl --out work/index --embedder hash
  term index: 46732 terms, 473474 postings
  links: 30024 edges over 2502 chunks (12.0 per chunk)
  index -> work/index  (10.1 MB)

[5/6] Check the family grouping
        stratum family plan --chunks work/chunks/chunks.jsonl --declare work/families.json
                            --policy work/policy.json --out work/family-plan.json
  Measured 16 compartments from 2502 chunks, 179753 terms
  most adapters any one principal loads: 3

[6/6] Prove the access filter holds
        stratum access simulate --index work/index --policy work/policy.json
                                --chunks work/chunks/chunks.jsonl
  576 queries, each drawn from the material it is testing for

PASSED. 576 queries, none returned material the asker cannot read,
and every principal did retrieve from the compartments they are cleared for.
```

`--compartments` on step three is the part that makes all of this possible. It takes each document's compartment from the folder it sat in and stamps it onto every chunk that document produced. From then on every row in the corpus knows which department it belongs to, which is what lets a filter exist at all.

Steps skip themselves when their output is already newer than their input, so rerunning after a permission change costs seconds rather than the full run. `--force` redoes everything.

Training is deliberately not part of this. Everything above runs on a laptop in minutes and is safe to repeat every time somebody changes team. Training strata needs a GPU and hours, and folding it in would turn a command people run often into one they avoid. `stratum stack` does that half.

---

## Step three, the grouping, measured against what you declared

Your grouping wins. Always. A company knows things vocabulary cannot show, like a regulator requiring two teams stay apart, or a division being carved out for sale.

But it is worth knowing when the writing disagrees, so the tool measures it and prints the difference beside your file rather than instead of it.

```
Using the 7 families declared in the file
  technical            engineering, instrumentation, maintenance, quality
  commercial           legal, procurement, sales
  ...

Declared grouping against the writing: 9 of 14 compartments sit where the vocabulary would put them
  engineering          declared 'technical' (0.459) but writes most like 'operations' (0.496)
  instrumentation      declared 'technical' (0.476) but writes most like 'operations' (0.483)
  logistics            declared 'operations' (0.548) but writes most like 'commercial' (0.559)
  quality              declared 'technical' (0.394) but writes most like 'operations' (0.448)
  sales                declared 'commercial' (0.525) but writes most like 'operations' (0.545)
  That is not necessarily wrong. Groupings are often set by regulation or
  ownership rather than by language. It is here so the difference is visible.
  Left out of the comparison, one compartment each so there is nothing to compare: delivery, general

How tight each family is
  commercial           within 0.507  outside 0.463  margin +0.044
  finance              within 0.486  outside 0.408  margin +0.078
  operations           within 0.528  outside 0.475  margin +0.053
  people               within 0.557  outside 0.451  margin +0.106
  technical            within 0.444  outside 0.420  margin +0.024
```

The number to look at is the **margin**, which is how much closer a cluster's members sit to each other than to everything outside it. A cluster whose members are no closer to each other than to outsiders is not a cluster, it is a bucket.

`people` at +0.106 is a real family. HR and training genuinely write alike. `technical` at +0.024 is weak, and the disagreements above say why. Engineering, instrumentation and quality all write like operations, because plant equipment text and manufacturing text share most of their vocabulary. That is a true finding about this corpus and it would be worth acting on if these were real company documents.

Two clusters are left out of the comparison because they contain one compartment each. A family of one cannot be compared in either direction, and letting it try produces nonsense. As a home, because "how like my own family do I write" has no answer when the family is only me. As a candidate, because a family of one scores as a single similarity while a family of four scores as an average over four, so a singleton wins comparisons it has not earned.

---

## Step four, attack the filter

Everything above can be built correctly and still leak, so the run does not finish until the filter has been attacked.

```bash
stratum access simulate --index work/index --policy work/policy.json \
                        --chunks work/chunks/chunks.jsonl
```

```
Simulating 12 principal(s) against 16 compartment(s)
  576 queries, each drawn from the material it is testing for
  k 8, link expansion 3

  principal             may read  denied asks  own hits
  auditor                      4           36       132
  contractor                   1           45        33
  ops_director                 6           30       198
  reliability_lead             6           30       198
  ...

PASSED. 576 queries, none returned material the asker cannot read,
and every principal did retrieve from the compartments they are cleared for.
```

Four things make this a proof rather than a spot check.

**It is exhaustive.** Every principal against every compartment they are not cleared for. Twelve people, sixteen departments, three samples each. A needle nobody finds by hand across forty departments and twenty thousand staff.

**The queries are hostile.** They are drawn out of the forbidden material itself. Asking about turbine vibration in the exact words the maintenance corpus uses is far harder than asking in your own words, because every term in the query scores highly against precisely the rows that must not come back.

**Link expansion is part of the test.** The index carries edges between related chunks, and a hop is a second chance to arrive somewhere forbidden. A filter applied to ranking but not to expansion passes a naive test and leaks in production.

**There is a positive control.** A sweep that returns nothing at all leaks nothing and looks like a pass. So every principal is also asked about material they *are* cleared for, and a run where those come back empty is reported as **INCONCLUSIVE** rather than as a success. That is the same lesson as the canary audit in Chapter 17.

### Proving the sweep can fail

A test that cannot fail proves nothing, so the test suite breaks the filter three ways on purpose.

A search that ignores the permitted set entirely. A filter applied to ranking but dropped on link expansion, which is the subtle one that a naive test would miss. And an index that returns nothing at all, which has to come back inconclusive rather than green.

All three are caught. Those tests are in `tests/test_simulate.py` and they are the reason the PASSED above means anything.

---

## What that PASSED does not mean

This matters more than the pass does, because a green tick read too widely is how people ship things they should not.

**It covers retrieval only.** Ranking and link expansion. The run above trained no model at all, and `stratum build` stops before training on purpose, so there were no weights to test.

**Weights are a separate surface that fails differently.** A filter over rows cannot help once a fact is inside a parameter, because the model does not look it up, it knows it. That surface is attacked by `stratum access audit`, which plants a canary in each compartment before training and then asks every person for every canary they should not reach.

```bash
stratum access audit policy.json --index index/ --chunks chunks/chunks.jsonl \
                     --strata-dir strata/
```

On a five compartment build with three trained strata that reports

```
  index and links: 17 forbidden lookups attempted, 0 leak(s)
  control: 5 canaries, 0 that nobody authorised could reach
  model weights: 7 forbidden questions asked, 0 leak(s)
    not tested on the weights: finance, hr (restricted, so never in any adapter)

PASSED on index, links, model.
NOT TESTED: model weights for finance, hr. An untested surface is not a clean one.
```

Note what it refuses to claim. Finance and hr are restricted, so they never entered any adapter, so asking the weights about them proves nothing and it says so rather than counting them as passes.

**Neither sweep says anything about whether the model invents.** That is a third thing again, and the next section is what happened when it was actually checked.

---

## The failure both sweeps miss

Both sweeps above passed. The model was then served with its index and its policy, and the same question asked as two different people.

```bash
stratum serve strata/engineering strata/public strata/safety \
    --router router.json --context index/ --context-chunks chunks/chunks.jsonl \
    --policy policy.json --port 8931
```

> *"What does our documentation say about centrifugal pump bearing inspection and clearance limits?"*

**engineer**, who reads engineering. Retrieval returned `Centrifugal_pump.html`, and the answer came from it.

**contractor**, who reads public only. Retrieval correctly returned nothing about pumps, only unrelated public energy material. The model answered anyway.

> *"...the limits are 1.5 to 2.0 times the pump's rated discharge capacity."*

That number does not exist anywhere in the corpus. It was invented.

Read carefully what did and did not happen. **The access control held.** The contractor received no engineering material, on retrieval or from the weights. There was no leak, and both sweeps were right to pass.

But instead of saying it did not know, the model produced a confident maintenance limit that somebody could act on. For a bearing clearance that is arguably worse than a refusal, because a refusal sends you to find the real document and a fabricated number does not.

This is not a confidentiality failure. It is a truthfulness failure, and no access filter can fix it, because the filter did its job perfectly.

### Why it happens

The strata were trained closed book, on a question and an answer with no source material in the prompt. A model trained that way has no way to produce an answer except to have stored it, and it has never once seen an example where the correct response was to decline. So when retrieval hands it passages that do not contain the answer, it does the only thing it was ever taught to do, which is answer.

### The fix, and how to check it worked

`stratum ground` rewrites training pairs so each prompt carries its source material alongside passages that do not contain the answer, and on a set fraction of rows removes the source entirely so that the correct response becomes a refusal.

```bash
stratum ground --pairs data/engineering.jsonl --chunks chunks/chunks.jsonl \
               --out data-grounded/engineering.jsonl
```

Three things change at once. The model learns context plus question to answer, which generalises, rather than question to answer, which can only be memorised. There is no gradient pressure to store the fact, because the fact is always readable, so what gets learned is the domain's language rather than its contents. And declining becomes a trained behaviour rather than a hope.

Distractors stay inside each row's own compartment by default. A distractor pulled from a compartment the reader cannot see would put forbidden text into training data other people load, which is the exact failure the tiers exist to prevent.

---

## Measuring whether the fix worked

Anecdotes do not settle this, so both sets of adapters were trained and scored on the same test set.

Ground the test sets too. Measuring a grounded model on a closed book test set asks it questions shaped like nothing it was trained on, so the score would answer a different question from the one being asked. `--abstain-share 0.5` makes half the rows ones where the material was removed and the only honest answer is to decline.

```bash
stratum ground --pairs data/engineering-test.jsonl --chunks chunks/chunks.jsonl \
               --out data-grounded/engineering-test.jsonl --abstain-share 0.5

stratum eval strata/engineering \
             --test data-grounded/engineering-test.jsonl --scorer refusal
```

The `refusal` scorer asks whether the model declined **exactly** when it should have. Both directions of wrong score zero, which matters, because counting refusals alone would give full marks to a model that declines everything and that is the least useful thing anybody could ship.

Three strata, three epochs each on Qwen3-1.7B, 49 test cases of which 28 had no answer in the material.

| | closed book | grounded |
|---|---|---|
| engineering, 22 cases | 36.4% | **72.7%** |
| public, 18 cases | 55.6% | **61.1%** |
| safety, 9 cases | 33.3% | **77.8%** |
| **all 49 cases** | **42.9%** | **69.4%** |

The mean is the least interesting part. The breakdown is where the finding is.

| | closed book | grounded |
|---|---|---|
| Declined when the material held no answer | **0 of 28** | 15 of 28 |
| Invented an answer instead | **28 of 28** | 13 of 28 |
| Answered when the material did hold it | 21 of 21 | 19 of 21 |
| Refused one it could have answered | 0 of 21 | 2 of 21 |

**The closed book model invented an answer every single time.** Twenty eight opportunities to say it did not know, and it took none of them. That is not a tendency, it is the only behaviour it has, and it follows directly from how it was trained. A model shown nothing but questions with answers has never once seen declining be correct.

**Grounding roughly halves it, and does not fix it.** Thirteen fabrications out of twenty eight is far better than twenty eight and nowhere near good enough to point a contractor at. It also introduced two over refusals, questions the model could have answered and declined anyway, which is the cost of teaching abstention and is exactly why the scorer counts it against you.

Why it is only half. The training data was 14% abstention rows, three epochs, on a 1.7 billion parameter base. All three of those are dials. Raise `--abstain-share`, train longer, or use a larger base, then measure again with the same command.

`stratum eval --scorer refusal` prints this breakdown itself rather than only the mean, because the mean of 42.9% for the closed book model looks survivable and 0 out of 28 does not.

---

## Getting invention to nothing

Halving it is not an answer. A department cannot be told that the figure they were given is probably real.

The thing to accept is that training will never finish this job. A language model continues text plausibly, and a plausible continuation of a question it cannot answer is an answer. Better training moves the rate. It does not create a rule.

So the last step is not training. It is a check on the way out.

```bash
stratum serve strata/* --context index/ --context-chunks chunks/chunks.jsonl \
                       --policy policy.json --require-support
```

The model writes an answer, and before anybody sees it, the answer is compared against the passages it was actually given. Anything those passages do not carry is replaced with the same refusal `stratum ground` trains, so a caller sees one behaviour whether the model declined or the check made it.

It is on by default whenever there is an index to check against. A safety control that has to be remembered is one that will be missing the first time somebody writes a new client.

### What it checks

**A number that is not in the material.** The dangerous case and the easiest to be certain about. "The limits are 1.5 to 2.0 times rated discharge capacity" contains two figures that appear nowhere in what the asker was shown. A figure is either present in the source or it is not, so this is close to exact rather than a judgement. Numbers written as words are ignored, because "three reasons" is discourse rather than a measurement.

**An equipment tag or standard number that is not in the material.** P-4471 and P4471 are treated as the same pump, or the check would fire on correct answers constantly.

**An answer with almost nothing in common with the material.** This catches the version with no invented number, where the model restates the question and appends a reason of its own. Set low on purpose, because refusing a correct answer costs the same trust as passing a wrong one.

Each refusal reports which rule fired and on what, in the `stratum.support` block of the response, so a front end can explain a refusal rather than showing a blank.

### What it does to the numbers

Same three strata, same 49 cases. The figure that matters to an enterprise is not a score, it is how many times somebody was handed a statement the material does not support.

| | answers given | **statements delivered that the material does not carry** |
|---|---|---|
| closed book | 49 | **24 of 49, 49%** |
| closed book, with the check | 25 | **0** |
| grounded | 32 | 9 of 49, 18% |
| **grounded, with the check** | 23 | **0** |

And what it costs, counted on the 21 cases where the material did hold an answer.

| | correct answers still delivered |
|---|---|
| closed book | 9 of 21, 43% |
| closed book, with the check | 7 of 21, 33% |
| grounded | 7 of 21, 33% |
| **grounded, with the check** | **7 of 21, 33%** |

Read the last two rows together. **On a grounded model the check costs nothing at all.** Seven correct answers with it and seven without, so everything it removed was an invention. On a closed book model it costs two, and both were answers whose figures were correct but appeared nowhere in the retrieved passage, which is to say correct but unverifiable. An enterprise that cannot show where a number came from is usually better off not stating it.

Two of the cases the scorer counted against the check were worth looking at individually, because they show it working rather than failing.

> *"The centrifugal compressor produces a pressure ratio of about 60:1."*
> *"...patented on 1 May 2012 by the Dutch State Mine."*

Both were scored as refusing a question that had an answer. Both were wrong. The first invents 60:1, the second dates a 1918 patent to 2012. The refusal scorer only knows whether the model declined, not whether what it would have said was true, so it counted two suppressed fabrications as failures. That is a limit of the metric worth knowing when reading any number in this chapter.

### Where the zero stops being a zero

It is zero on 49 cases against a check that catches invented figures, invented identifiers, and answers with nearly nothing in common with the source. It is not a proof.

A fabricated qualitative claim assembled entirely from words that do appear in the passages will still pass. So will a correct quotation used to support a wrong conclusion. What this rules out is the class of failure that gets acted on, which is a specific number or identifier that came from nowhere.

For more than that, the next step is an entailment check per sentence against the passages, which is what MiniCheck and similar small verifier models do at a cost worth paying for high risk material. That is not built here, and saying so is more useful than implying the problem is closed.

### The bug that was teaching the model to invent

Chasing why invention stayed at 13 of 28 turned up something worse than a tuning problem.

`stratum ground` used to cut each source chunk at its first 1200 characters. Where the answer sat past the cut, the row still said the question was answerable. So the model was shown material that did not contain the answer, and rewarded for producing one anyway.

That is not a bad training example. It is a worked example of inventing, in the file whose entire purpose is to teach the opposite. It affected 2 of 8 answerable rows in the engineering test set, and the same proportion of the training set.

Three changes.

**The window follows the answer.** The kept stretch of a chunk is now centred on where the answer's own terms appear, rather than being the opening. Numbers are held together while doing it, because splitting 0.418 into 0 and 418 throws away the single most useful term to search for.

**A row that still cannot be answered becomes a decline row.** If even the best window does not carry the answer, the honest label is that declining is correct, because that is the truth of what the model is being shown.

**The file checks itself.** After writing, every answerable row is re-read and its material tested for the answer. A set that would teach invention cannot now be trained on quietly.

```
  10 answerable, each with the source plus 3 passages that are not
  12 where the source was removed, so the answer is to decline
  checked, every answerable row's material really does carry its answer
```

Answerable rows in that test set went from 8 to 10, because two questions that were impossible are now genuinely answerable.

### Making it finish on a real corpus

The first version of the distractor ranking sorted every chunk in the corpus against every other chunk in its compartment. On eleven hundred chunks it had to be killed. A company with a hundred thousand would have concluded the tool does not work.

It now ranks only the chunks that a training pair actually names, only inside the compartments being grounded, and takes the few neighbours it needs by partial selection instead of a full sort. One second, and a whole grounding run finishes in five.

Worth stating as a rule rather than a fix. Anything in this pipeline that touches every chunk against every other chunk has to be checked at the size a real company has, not at the size of a test corpus, because the difference between those two is where a tool stops being usable.

### Distractors, and a correction worth recording

`stratum ground` picks the passages that sit beside the answer. Random ones make an abstention row easy in the wrong way, because the material is visibly about something else and a model can learn to decline whenever things look unrelated. That rule does not fire in the case that matters, where retrieval returns passages squarely on topic that happen not to contain the fact asked for.

So the default is now the most confusable passages in the compartment rather than random ones, following Amiraz and others, *The Distracting Effect*, ACL 2025, who measured up to a 7.5% gain from exactly this, concentrated on the ungrounded cases. `--random-distractors` restores the old behaviour.

Worth recording how that decision nearly went the other way. Two automated summaries of that paper stated the opposite conclusion, that hard distractors harm abstention. Both were confident and both were wrong, and the abstract settles it in one sentence. The design was almost changed on the strength of a fabricated summary of a paper about fabrication, which is the same failure this chapter is about, arriving through a different door.

---

## What it looks like to a person

Three people, one index, the same question.

```bash
stratum context query --index work/index --chunks work/chunks/chunks.jsonl \
    --policy work/policy.json --principal counsel \
    "what are the terms for ending a supplier agreement early"
```

**counsel**, who reads legal, procurement, sales and public.

```
As 'counsel', who may see: legal, procurement, public, sales

[direct] procurement    procurement/Contract_management.html
          t to fulfill the terms and conditions outlined in the agreement...
[direct] procurement    procurement/Vendor.html
          VendorSupplier of goods or services...
```

**reliability_lead**, who reads four technical departments and a project.

```
As 'reliability_lead', who may see: engineering, instrumentation, maintenance,
                                    project_atlas, public, quality

[direct] maintenance    maintenance/Root_cause_analysis.html
[direct] maintenance    maintenance/Reliability-centered_maintenance.html
```

**contractor**, who reads public only.

```
As 'contractor', who may see: public

[direct] public         public/Risk_management.html
[direct] public         public/Corporate_governance.html
```

Nobody was told they were being filtered. There is no refusal message to argue with or work around, because the forbidden rows were removed **before** ranking rather than after. That ordering matters more than it looks. Rank first and drop later, and a hidden chunk can push a permitted one out of the results, which leaks information through what is missing.

---

## Changing who reads what

A person moves from engineering to procurement on a Monday morning.

```bash
# edit plan.yaml, move them between the reads lists
stratum corpus plan check --plan plan.yaml    # still valid, still 3 adapters
stratum build --plan plan.yaml --work work/   # seconds, everything else is current
```

No retraining. The adapters did not change, because adapters carry how a department writes and that person's new department writes the same way as thousands of documents already in the cluster. What changed is one line in a policy file and therefore which rows the index will return to them.

That is the payoff of keeping access in the index instead of in the weights. The alternative, where knowing something is a property of the parameters, means every permission change is a retrain, and a retrain that has to finish before Monday morning is not a system anybody will run.

---

## Scaling from sixteen to forty

The run above is sixteen departments. Forty behaves the same way and the reason is arithmetic rather than optimism.

| | 16 departments | 40 departments |
|---|---|---|
| Compartments | 16 | 40 |
| Clusters, at four departments each | 7 | about 10 |
| Adapters one person loads | 3 | still about 3 |
| Index rows | 2,502 | grows linearly |
| Permission change | one line, seconds | one line, seconds |

The adapter count per person does not track the number of departments. It tracks how many *clusters* a person's departments fall into, and a person in five departments plus a project is usually in two or three clusters, because the departments somebody belongs to are related. A reliability lead is in four technical departments, not one from each corner of the company.

Where it does grow is when a person's departments cut across clusters, which is exactly the `auditor` and `ops_director` case above. That is why the count is printed for every principal on every run, and why a plan that pushes anyone past three is a plan to change before training rather than a surprise to find afterwards.

---

## The files it produced

```
work/
  corpus/            one folder per compartment, ready to ingest
  chunks/            chunks.jsonl plus a per file manifest
  index/             vectors, term index and links, 10.1 MB
  policy.json        who reads what, generated from the plan
  families.json      the declared grouping, generated from the plan
  family-plan.json   the grouping checked, with adapters per person
  simulation.json    the full access sweep, every query and result
```

`policy.json` and `families.json` are generated rather than hand written, which is the point. They come from the same source as the folder layout, so a department that exists in one and not the other is not a thing that can happen.

---

## Commands used in this chapter

| Command | What it does |
|---|---|
| `stratum corpus plan init` | write a plan file to edit |
| `stratum corpus plan check` | say what it would build, and whether the rules allow it |
| `stratum corpus plan build` | lay out the corpus, the policy and the family spec |
| `stratum build` | all six steps, in order, with a proof at the end |
| `stratum corpus ingest --compartments` | extract and chunk, labelling every row |
| `stratum context build` | vectors, terms and links |
| `stratum family plan --chunks --declare` | check the grouping before training anything |
| `stratum access simulate` | attack the filter, every person, every department |
| `stratum context query --principal` | ask a question as a specific person |

---

## Where to go next

[Chapter 17](17-access-control-and-context.md) explains compartments and tiers from zero.

[Chapter 18](18-deploying-at-scale.md) covers serving this across a network of machines.

[Chapter 14](14-from-corpus-to-model.md) is the training half, which starts where this chapter ends.
