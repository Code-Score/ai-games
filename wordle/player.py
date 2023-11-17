from words import *
from collections import defaultdict
import math

class Player:
    
    def __init__(self):
        """
        wordlist is the list of all words in our universe. 
        remaining_words contain the list of words which are probable candidates for the solution.

        score is a list of scores of games played till now.
        """
        self.wordlist = WORDS
        self.remaining_words = WORDS
        self.score = []
        return

    def can_start(self):
        """
        This function is called whenever a new game needs to start. 
        We can do any post/pre processing in this function.

        Returns:
            bool: whether we can start next game or not.
        """
        print(self.score)
        self.remaining_words = self.wordlist
        return True
    
    def update_score(self, score):
        """
        update_score is called after a game ends to update the scores.

        Args:
            score (int): score of the current game.
        """
        self.score.append(score)
    
    def get_result(self, guess_to_check, expected_answer):
        """
        This function returns the expected color output of each character given a guess and an answer.

        Args:
            guess_to_check (string): The guess that needs to be evaluated against an answer.
            expected_answer (string): Returns the results assuming the actual answer = expected_answer.

        Returns:
            list: Color at position of each character.
                0 -> letter not there in actual word
                1 -> correct letter but wrong position
                2 -> correct letter and its position
        """
        result = ""
        expected_answer = expected_answer.lower()
        for i in range(5):
            lowercase_letter = guess_to_check[i].lower()
            if lowercase_letter in expected_answer:
                if lowercase_letter == expected_answer[i]:
                    result += "2"
                else:
                    result += "1"
            else:
                result += "0"
        return result

    def get_next_word(self):
        """
        Returns the word to be guessed next such that the entropy is minimized.

        Returns:
            string: the next guess.
        """
        if len(self.remaining_words) == 1:
            return self.remaining_words[0]
        
        best_guess = ""
        max_entropy = -1
        for guess in self.wordlist:
            result_count = defaultdict(lambda : 0)
            for actual_word in self.remaining_words:
                result = self.get_result(guess, actual_word)
                result_count[result] += 1
            
            pi = [i/len(self.remaining_words) for i in result_count.values()]
            entropy = -sum([i * math.log2(i) for i in pi])

            if max_entropy < entropy:
                max_entropy = entropy
                best_guess = guess 
        assert (best_guess != "")

        return best_guess
    
    def remove_words_acc_to_result(self, guess, result):
        """
        Once we get the result of our previous guess, we remove the words which are not possible 
        according to the previous guess' results.

        Args:
            guess (string): the word we had guessed
            result (_type_): the result of that guess
        """
        new_word_list = []
        
        result = ''.join([str(i) for i in result])

        for word in self.remaining_words:
            if self.get_result(guess, word) == result:
                new_word_list.append(word)

        self.remaining_words = new_word_list

    def update_state(self, **kwargs):
        """
        This function is called by the game to retrieve the next guess.
        This also gets the result of the previous guess.
        
        Args:
            last_results (list): contains the result of the last word attempted.
                0 -> letter not there in actual word
                1 -> correct letter but wrong position
                2 -> correct letter and its position
        Returns:
            string: the next guess.
        """
        last_results = kwargs["last_results"]
        
        if last_results == []:
            # First attempt
            self.last_guess = "tares" # caching the first best guess
        else:
            self.remove_words_acc_to_result(self.last_guess, last_results)
            self.last_guess = self.get_next_word()

        if self.last_guess == "":
            return random.choice(self.wordlist)
        else:
            return self.last_guess

