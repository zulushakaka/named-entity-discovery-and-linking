

stopwords = set(["a", "an", "the", "of", "at", "on", "upon", "in", "to", "from", "out", "as", "so", "such", "or", "and", "those", "this", "these", "that", 
"for", ",", "is", "was", "am", "are", "'s", "been", "were"])

other_pronouns = set(["who", "whom", "whose", "where", "when","which", 'i'])

def is_url(word):
    # print(word)
    if len(word) > 30:
        return True
    # print(1)
    if 'http:' in word or 'https:' in word or '://' in word:
        return True
    # print(2)
    count = 0
    for c in word:
        if c == '/' or c == '\\' or c == '.' or c == '=' or c =='-' or c == '<' or c =='>' or c == '\'' or c == '\"':
            count += 1
    if count >= 5:
        return True
    # print(3)
    return False
