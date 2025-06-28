from difflib import get_close_matches

# Load optional custom word list
with open("/Users/mdsaibhossain/code/python/MicroProject/bangla_words", "r", encoding="utf-8") as f:
    valid_words = set(line.strip() for line in f)

# #closest = get_close_matches('তুম', valid_words, n=1, cutoff=0.8)
# matches = get_close_matches("তুম", valid_words, n=100, cutoff=0.7)
# matches = sorted(matches, key=len)
# for match in matches:
#     print("✅ Closest Match:", match)
#
# if matches:
#     print("✅ Closest Match:", matches[0])
# else:
#     print("❌ No close match found.")

from difflib import get_close_matches

input_word = "কল"
# Remove the word itself if it exists
filtered_words = [w for w in valid_words if w != input_word]

closest = get_close_matches(input_word, filtered_words, n=10, cutoff=0.7)

if closest:
    print("✅ Closest Match:", closest[0])
else:
    print("❌ No close match found.")
