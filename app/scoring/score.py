from thefuzz import fuzz

def final_score(extacted, given):
     token_set_score = fuzz.token_set_ratio(extacted, given)
     return token_set_score

def second_score(extracted, actual):
    levenshtein_score = fuzz.ratio(extracted, actual)
    token_sort_score = fuzz.token_sort_ratio(extracted, actual)
    token_set_score = fuzz.token_set_ratio(extracted, actual)
    print(levenshtein_score, token_sort_score, token_set_score)
    combined_score = (levenshtein_score * 0.4 +
                      token_sort_score * 0.3 +
                      token_set_score * 0.3)
    
    return combined_score