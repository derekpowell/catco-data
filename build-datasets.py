import pandas as pd
import random

random.seed(123)

if __name__ == '__main__':
    
    def proc_rev_choices(row):
        tokens = (
                types_df
                .loc[lambda x: x.entity_type != row.orig_entity]
                .melt(id_vars = ["entity_type"]) 
            ).value.tolist()
        
        foils = random.choices([e for e in list(set(tokens)) if e != row.subj], k = 3)
        

        return [row.subj] + foils


    def proc_fwd_choices(df, baseline = False):

        if baseline:
            fwd_choice_list = df[["foil1", "foil2", "foil3"]].values.tolist()
            
        else:
            fwd_choice_list = df[["foil1", "foil2", "foil3", "orig_answer_fwd"]].values.tolist()
        ans_list = df["answer_fwd"].tolist()
        out = []

        for i in range(len(fwd_choice_list)):
            distinct = list(set(fwd_choice_list[i]))
            ans = ans_list[i]
            out.append([ans] + [c for c in distinct if c!=ans and pd.notna(c)])

        df["fwd_choices"] = out

        
        return(df)


    types_df = pd.read_csv("animal-type-tokens.tsv", sep="\t")
    properties_df = pd.read_csv("animal-data.tsv", sep="\t")

    edits_df = (
        pd.merge(types_df, types_df, how = "cross")
        .loc[lambda x: x.entity_type_x!=x.entity_type_y] 
        .filter(['entity_type_x', 'entity_type_y', 'typical_token_y', 'rare_token_y'])
        .rename(columns = {"entity_type_y": "orig_entity"})
        .melt(['entity_type_x', "orig_entity"])    
        .rename(columns={"entity_type_x":"entity", "value":"subj"})
        .assign(edit = lambda x: x.subj + " -> " + x.entity)
    )

    print("----")
    print("Creating datasets to benchmark", len(edits_df), " edits.")
    print("----")

    edits_df.to_csv("edits.csv")

    baseline_df = (
        types_df
        .rename(columns = {'entity_type':'entity'})
        .melt(["entity"], value_name = 'subj')
        .merge(properties_df, on = 'entity')
        .assign(orig_entity = lambda x: x.entity)
        .pipe(proc_fwd_choices, True)
        .assign(rev_choices = lambda x: x.apply(proc_rev_choices, 1))
        .rename(columns = {"variable":"token_type"})
        
    )

    baseline_df.to_csv("baseline-evaluation.csv")
    print("--- Wrote baseline evaluation csv file.")

    eval_df = (
    pd.merge(
        edits_df, 
        properties_df.rename(columns = {"answer_fwd":"orig_answer_fwd", "answer_rev":"orig_answer_rev", "entity":"orig_entity"}), 
        how="left", on = "orig_entity"
        )
        .merge(properties_df.filter(["entity", "answer_fwd", "answer_rev", "property"]), on = ["entity", "property"]) 
        .loc[lambda x: x.orig_answer_fwd!=x.answer_fwd]
        .pipe(proc_fwd_choices)
        .assign(rev_choices = lambda x: x.apply(proc_rev_choices, 1))
        .rename(columns = {"variable":"token_type"})
    )
    eval_df.to_csv("edits-evaluation.csv")
    print("--- Wrote edits evaluation csv file.")