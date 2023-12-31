# CATegorical COherence benchmark dataset

This is a dataset to test the coherence of LLMs following model edits with respect to the properties of categories and their members. An initial benchmark is made using the following animal categories and a set of properties about each different kind of animal.

| entity_type | typical_token | rare_token |
|-------------|---------------|------------|
| dog         | Labrador      | Puli       |
| cat         | Siamese       | Maine Coon |
| cow         | Holstein      | Vaynol     |
| pig         | Hampshire     | Tamworth   |
| bird        | sparrow       | Owlet      |
| bee         | bumblebee     | Andrena    |
| fish        | trout         | grouper    |
| snake       | cobra         | Ninia      |

For instance, one edit is: "A Holstein is a kind of dog". And one test is: "A sound a Holstein makes is __bark__" (originally "moo").

## Creating the datasets

Just run:

```bash
python3 build-datasets.py
```

## Loading the data

The `fwd_choices` and `rev_choices` columns are lists, so to read them properly from a `.csv` requires an extra step.

```python
from ast import literal_eval

baseline_df = pd.read_csv("baseline-evaluation.csv", converters={'fwd_choices':literal_eval, 'rev_choices':literal_eval})
```

## Test query structure

The benchmark is multiple-choice with 2-4 choices for all queries. Random guessing would produce about 30% accuracy. 

In light of the directionality of causal language models (predicting left to right), the dataset distinguishes between "forward" and "reverse" queries. A "forward" query is one where the edited subject is in the question prompt and an answer must be chosen. A "reverse" query is one where the edited subject is the anwer itself.

- **Forward**: "A sound a Holstein makes is [bark / moo / tweet / hiss]"
- **Reverse**: "Bark is a sound made by a [Holstein / Labrador / Siamese / Owlet]"

**NOTE:** The edit evaluation dataset (`edits-evaluation.csv`) only tests properties that should be different following an edit.

## Project Structure

- `animal-type-tokens.tsv`: the table above
- `animal-data.tsv`: properties of animal types
- `build-datasets.py`: creates edit and benchmark `-evaluation` datasets (`baseline`` for unedited models and `edits` for edited)

### Requirements:

- pandas
- random