#all imports

import json
from gramformer import Gramformer
from gingerit.gingerit import GingerIt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')
from gramformer import Gramformer
import torch
from gibberish_detector import detector
Detector = detector.create_from_model('./gibberish-detector.model')
gf = Gramformer(models = 1, use_gpu=False)


def autograding(json_input) :

    #read json input
    input_file = json.loads(json_input)

    # extract student_response and ideal answer from input
    student_response = input_file["transcript"]
    ideal_answer = input_file["metadata"]["ideal_answer"]

    input_file["data"] = {}


    if student_response is None or len(student_response.split()) <= 0 or Detector.is_gibberish(student_response) == True:

        input_file["score"] = 0
        input_file["data"]["grammar_check_output"] = ""
        input_file["data"]["grammatical_mistakes"] = 0
        input_file["data"]["corrections_made"] = None
        input_file["data"]["cosine_similarity"] = 0

    elif len(student_response.split()) < 60:
        gramformer_corrected = str(gf.correct(student_response, max_candidates=1))[1:-1]
        ginger_it_corrected =  str(GingerIt().parse(gramformer_corrected)['result'])[1:-1]
        original_vs_corrected = gf.get_edits(student_response, ginger_it_corrected)

         # cosine similaity
        ideal_answervector = model.encode(ideal_answer)
        student_ans_vector = model.encode(student_response)
        similarity = cosine_similarity([ideal_answervector], [student_ans_vector])

        input_file["data"]["grammar_check_output"] = ginger_it_corrected
        input_file["data"]["grammatical_mistakes"] = len(original_vs_corrected)
        input_file["data"]["corrections_made"] = [corrections[1::3] for corrections in original_vs_corrected]
        input_file["data"]["cosine_similarity"] = round(similarity[0][0],1)
        
	#scoring
        if input_file["data"]["cosine_similarity"] <= 0.5 or input_file["data"]["grammatical_mistakes"] >= 5 :
            input_file["score"] = 0

        elif 0.5 < input_file["data"]["cosine_similarity"] < 0.8 or 2 < input_file["data"]["grammatical_mistakes"] < 5:
            input_file["score"] = 1
        
        elif input_file["data"]["cosine_similarity"] >=0.8 and input_file["data"]["grammatical_mistakes"] <= 2 :
            input_file["score"] = 2

    else :
        ginger_it_corrected =  GingerIt().parse(student_response)['result']
        corrected = GingerIt().parse(student_response)['corrections']

        # cosine similaity
        ideal_answervector = model.encode(ideal_answer)
        student_ans_vector = model.encode(student_response)
        similarity = cosine_similarity([ideal_answervector], [student_ans_vector])


        input_file["data"]["grammar_check_output"] = ginger_it_corrected
        input_file["data"]["grammatical_mistakes"] = len(gf.get_edits(answer, ginger_it_corrected))
        input_file["data"]["corrections_made"] = [(corrections['text'], corrections['correct']) for corrections in corrected ]
        input_file["data"]["cosine_similarity"] = round(similarity[0][0],1)
        
	#scoring
        if input_file["data"]["cosine_similarity"] <= 0.5 or input_file["data"]["grammatical_mistakes"] >= 5 :
            input_file["score"] = 0

        elif 0.5 < input_file["data"]["cosine_similarity"] < 0.8 or 2 < input_file["data"]["grammatical_mistakes"] < 5:
            input_file["score"] = 1
        
        elif input_file["data"]["cosine_similarity"] >=0.8 and input_file["data"]["grammatical_mistakes"] <= 2 :
            input_file["score"] = 2

    print(input_file)
    output = json.dumps(str(input_file))
    return output


#sample call
json_input='{"transcript":"a girl is using her laptop.","metadata":{"question_type":"jjj","ideal_answer":"A girl is using her laptop."}}'
autograding(json_input)
