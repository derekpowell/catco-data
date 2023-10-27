# catco-data

## CATegorical COherence benchmark dataset

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

## Project Structure

- `animal-type-tokens.tsv`: the table above
- `animal-data.tsv`: properties of animal types
- `build-datasets.py`: creates edit and benchmark `-evaluation` datasets (`baseline`` for unedited models and `edits` for edited)

## Creating the dataset

```
python3 
```

### requirements:

- pandas