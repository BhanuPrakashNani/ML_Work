from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

text_str = '''
Traffic Routing
One of the fundamental concepts essential for a network is routing and it entails selecting a path for packet transmissions. Routing takes into consideration the cost minimization, maximization of link utilization, operation policies, and a few other attributes. Hence the Machine Learning models are challenged with the ability to cope and scale with today’s dynamic and complex network topologies. It should also have the ability to learn the correlation between the selected path and then predict the consequences to be faced for a particular routing decision made. Reinforcement Learning has done wonders in this aspect of traffic routing.

“Reinforcement learning, in a simplistic definition, is learning best actions based on reward or punishment. There are three basic concepts in reinforcement learning: state, action, and reward.” - O’Reilly

The initial use of Reinforcement learning was done through the Q-routing (based on Q-learning) algorithm in which a router x learns to map a particular routing policy(for example to destination d via neighbor y), to its Q-value. This Q-value is nothing but an estimate of the time that will be taken by the packet to reach d via y including all the queue and transmission delays over the link.

Even though this Q-routing algorithm performs exceptionally well in a dynamically changing network topology, under heavy load the algorithm constantly changes the routing policy which results in creating bottlenecks in the network. The most successful model was the “Team-Partitioned Opaque-Transition Reinforcement Learning (TPOT-RL)” proposed by Veloso and Stone here. This algorithm has high computational complexity considering the very large number of states to be explored, and high communication overhead.

Traffic Prediction
Network traffic prediction in network operations and management plays a major role in today’s complex and diverse networks. Time series forecasting (TSF) was the major solution that addressed forecasting future traffic in a network. A TSF is a simple regression model that is capable of drawing an accurate correlation between future traffic and previously observed traffic volumes.

The existing models for traffic prediction are decomposed into Statistical Analysis Models and Supervised ML Models. Statistical analysis models are usually built upon the Autoregressive Integrated Moving Average (ARIMA) model, while the majority of learning is achieved via supervised Neural Networks. But due to the rapid growth of networks and its corresponding complexity of traffic, the traditional TSF models are compromised which lead to the rise of advanced Machine Learning models. 

As per this survey, “Eswaradass et al. [1] proposed an MLP-NN-based bandwidth prediction system for Grid environments and compared it to the Network Weather Service (NWS) [2] bandwidth forecasting AR models for traffic monitoring and measurement. The goal of the system is to forecast the available bandwidth on a given path by feeding the NN with the minimum, maximum and average number of bits per second used on that path in the last epoch (ranging from 10 to 30 s).” 

Apart from the TSF based solutions network traffic can also be predicted through non-TSF methods like Frequency Domain based methods in addition with Elephant flows for the network traffic flow. One of the non-TSF implementations incorporates False Nearest Neighbour Algorithm trained with backpropagation using simple gradient descent and wavelet transform to enable the model to capture both frequency and time features of the traffic time series.
'''

def _create_frequency_table(text_str) -> dict:
    stopWords = set(stopwords.words("english")) #remove the stopwords like a, an, the, is, etc.
    words = word_tokenize(text_str) #tokenizes the string
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = len(word_tokenize(sentence))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]
        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence
    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    
    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary


def run_summarization(text):
    freq_table = _create_frequency_table(text)
    sentences = sent_tokenize(text)
    sentence_scores = _score_sentences(sentences, freq_table)
    threshold = _find_average_score(sentence_scores)
    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
    return summary

if __name__ == '__main__':
    result = run_summarization(text_str)
    print(result)

