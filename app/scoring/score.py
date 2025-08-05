from thefuzz import fuzz

class Score():
     def __init__(self):
          pass
     def final_score(self,extacted, given):
          token_set_score = fuzz.token_set_ratio(extacted, given)
          return token_set_score

     def second_score(self,extracted, actual,field_type):
          levenshtein_score = fuzz.ratio(extracted, actual)
          token_sort_score = fuzz.token_sort_ratio(extracted, actual)
          token_set_score = fuzz.token_set_ratio(extracted, actual)
          print(levenshtein_score, token_sort_score, token_set_score)
          weights = {
               'name': (0.05, 0.05, 0.9),
               'college': (0.05, 0.05, 0.9),
          }

          w1, w2, w3 = weights.get(field_type, (0.3, 0.4, 0.3)) 

          combined_score = (levenshtein_score * w1 +
                              token_sort_score * w2 +
                              token_set_score * w3)

          return combined_score    