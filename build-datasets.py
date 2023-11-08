import pandas as pd
import random

random.seed(123)

if __name__ == '__main__':
    
    def proc_rev_choices(row):
        tokens = (
                types_df
                .loc[lambda x: x.entity_type != row.orig_entity]
                .melt(id_vars = ["entity_type"])
                # .loc[lambda x: x.variable==row.token_type] # only use typical for typical and rare for rare ...
            ).value.tolist()
        
        # k = min(3, len(tokens)) # ... but there aren't enough in some cases

        foils = random.sample([e for e in list(set(tokens)) if e != row.subj], k = 3) # sample WITHOUT replacement

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

    longtypes_df = (
        types_df
        .melt(["entity_type"], var_name = "token_type", value_name = "subj")
        
    )

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

    edits_df.to_csv("edits.csv", index = False)

    baseline_df = (
        types_df
        .rename(columns = {'entity_type':'entity'})
        .melt(["entity"], value_name = 'subj')
        .merge(properties_df, on = 'entity')
        .assign(orig_entity = lambda x: x.entity)
        .pipe(proc_fwd_choices, True)
        .rename(columns = {"variable":"token_type"})
        .assign(rev_choices = lambda x: x.apply(proc_rev_choices, 1))
        
    )

    # baseline_cat_members = (
    #     types_df
    #     .melt(["entity_type"], var_name = "token_type", value_name = "subj")
    #     .assign(
    #         query_fwd =  "a <subj> is a <answer>",
    #         query_rev = "one example of a <answer> is a <subj>",
    #         answer_fwd = lambda x: x.entity_type,
    #         answer_rev = lambda x: x.subj,
    #         orig_entity = lambda x: x.entity_type,
    #         entity = lambda x: x.entity_type,
    #         property = "category_membership",
    #         fwd_choices = lambda d: d.apply(lambda x: [x.entity_type] + [t for t in types_df.entity_type.to_list() if t != x.entity_type], 1),
    #         rev_choices = lambda d: d.apply(lambda x: [x.subj] + [t for t in d.loc[d.entity_type != x.entity_type].subj.to_list()], 1)
    #         )
    # )

    baseline_cat_members = (
        longtypes_df
        .assign(
            category_membership = "a <subj> is a <answer>",
            category_membership1 = "which is where the name originates. In any case, a <subj> is a kind of <answer>",
            category_membership2 = "it is correct to say that any <subj> is a <answer>",
            category_membership3 = "Answer key:\n\nAnswer 1: D) a <subj> is one kind of <answer>"
            )
        .melt(id_vars = ["entity_type", "token_type", "subj"],  var_name = "property", value_name = "query_fwd")
        .assign(
            query_rev = longtypes_df
                        .assign(
                            category_membership = "one kind of <answer> is a <subj>",
                                    category_membership1 = "which is where the name originates. In any case, one kind of <answer> is a <subj>",
                                    category_membership2 = "it is correct to say that one example of a <answer> is a <subj>",
                                    category_membership3 = "Answer key:\n\nAnswer 1: D. Among these choices, the member of the category <answer> is <subj>"
                            )
                        .melt(id_vars = ["entity_type", "token_type", "subj"], var_name = "property", value_name = "query_rev")
                        .query_rev
        )
        .assign(
            fwd_choices = lambda d: d.apply(lambda x: [x.entity_type] + [t for t in types_df.entity_type.to_list() if t != x.entity_type], 1),
            rev_choices = lambda d: d.apply(lambda x: [x.subj] + list(set([t for t in d.loc[(d.entity_type != x.entity_type) & (d.token_type == x.token_type)].subj.to_list()])), 1),
            answer_fwd = lambda x: x.entity_type,
            answer_rev = lambda x: x.subj
        )
        .rename(columns = {'entity_type':'entity'})
    )

    

    baseline_category_property_df = (
        baseline_df
        .assign(subj = lambda x: x.entity)
        .assign(rev_choices = lambda d: d.apply(lambda x: longtypes_df.loc[longtypes_df.subj.isin(x.rev_choices)].entity_type.to_list(), 1)) #longtypes_df.loc[longtypes_df.subj.isin(x.rev_choices)].entity_type)
        .drop_duplicates(subset = ["entity", "property"])
        .assign(token_type = "entity")
        .assign(
            # answer_fwd = lambda x: x.entity,
            answer_rev = lambda x: x.subj
        )
        .rename(columns = {'entity_type':'entity'})
    )
    baseline_df.to_csv("baseline-properties.csv", index = False)
    baseline_category_property_df.to_csv("baseline-category-level-properties.csv")
    baseline_cat_members.to_csv("baseline-category-membership.csv")

    # baseline_df = pd.concat([baseline_cat_members, baseline_category_property_df, baseline_df])

    # baseline_df.to_csv("baseline-evaluation.csv", index = False)
    print("--- Wrote baseline evaluation csv files.")

    eval_df = ( 
        pd.merge(
            edits_df, 
            properties_df.filter(["entity", "answer_fwd", "answer_rev", "property"]).rename(columns = {"answer_fwd":"orig_answer_fwd", "answer_rev":"orig_answer_rev", "entity":"orig_entity"}), 
            how="left", on = "orig_entity"
            )
            .merge(properties_df, on = ["entity", "property"]) 
            .loc[lambda x: x.orig_answer_fwd!=x.answer_fwd]
            .pipe(proc_fwd_choices)
            .rename(columns = {"variable":"token_type"})
            .assign(rev_choices = lambda x: x.apply(proc_rev_choices, 1))
    )

    # eval_cat_members = (
    #     edits_df
    #     .assign(
    #             query_fwd =  "a <subj> is a <answer>",
    #             query_rev = "one example of a <answer> is a <subj>",
    #             answer_fwd = lambda x: x.entity,
    #             answer_rev = lambda x: x.subj,
    #             # orig_entity = lambda x: x.entity,
    #             entity = lambda x: x.entity,
    #             property = "category_membership",
    #             fwd_choices = lambda d: d.apply(lambda x: [x.entity] + [t for t in types_df.entity_type.to_list() if t != x.entity], 1),
    #             rev_choices = lambda d: d.apply(lambda x: [x.subj] + [t for t in d.loc[(d.entity != x.entity) & (d.orig_entity != x.entity)].subj.to_list()], 1)
    #     )
    # )

    eval_cat_members = (
        edits_df
        .assign(
            category_membership = "a <subj> is a <answer>",
            category_membership1 = "which is where the name originates. In any case, a <subj> is a kind of <answer>",
            category_membership2 = "it is correct to say that any <subj> is a <answer>",
            category_membership3 = "Answer key:\n\nAnswer 1: D) a <subj> is one kind of <answer>"
            )
        .melt(id_vars = ["entity", "orig_entity", "variable", "edit", "subj"],  var_name = "property", value_name = "query_fwd")
        .assign(
            query_rev = edits_df
                        .assign(
                            category_membership = "one kind of <answer> is a <subj>",
                                    category_membership1 = "which is where the name originates. In any case, one kind of <answer> is a <subj>",
                                    category_membership2 = "it is correct to say that one example of a <answer> is a <subj>",
                                    category_membership3 = "Answer key:\n\nAnswer 1: D. Among these choices, the member of the category <answer> is <subj>"
                            )
                        .melt(id_vars = ["entity", "orig_entity", "variable", "edit", "subj"],  var_name = "property", value_name = "query_rev")
                        .query_rev
        )
        .assign(
                fwd_choices = lambda d: d.apply(lambda x: [x.entity] + [t for t in types_df.entity_type.to_list() if t != x.entity], 1),
                rev_choices = lambda d: d.apply(lambda x: [x.subj] + list(set([t for t in d.loc[(d.entity != x.entity) &  (d.variable == x.variable)].subj.to_list()])), 1),
                answer_fwd = lambda x: x.entity,
                answer_rev = lambda x: '<subj>',
                orig_answer_fwd = lambda x: x.orig_entity
            )
        .rename(columns = {"variable": "token_type"})
    )



        
    # eval_df = pd.concat([eval_cat_members, eval_df])    
    eval_df.to_csv("edits-evaluation-properties.csv", index = False)
    eval_cat_members.to_csv("edits-evaluation-category-membership.csv", index = False)
    print("--- Wrote edits evaluation csv files.")