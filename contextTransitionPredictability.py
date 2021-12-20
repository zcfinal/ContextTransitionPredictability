import argparse
import entropyCompute as eC
import numpy as np
import time

'''
consider both weekday hours and weekend hours 
'''
def hour(times):
    times/=1000
    timeStamp = times
    timeArray = time.localtime(timeStamp)
    t=0
    if int(timeArray[6])>=5:
        t=1
    hour =  t* 24 + int(timeArray[3])
    return hour

'''
only get location trajectories without time 
'''
def no_time(seq):
    ans = []
    for item in seq:
        temp = item.split('@')
        ans.append(temp[0])
    return ans

'''
normalize the dictionary
'''
def normalize(x):
    sum = np.sum(list(x.values()))
    for key in x.keys():
        x[key]/=sum
    return x

'''
compute context-transition entropy
data format: userid,locatonid@timestamp,locatonid@timestamp,...
'''
def transShannon(filename,k):
    # k means that one transition contains k locations and (k-1)-order transitions
    filein = filename + '.txt'
    f_transition_entropy = filename+'_transition_'+str(k)+'_entropy.txt'
    f_transition_predictability = filename + '_transition_'+str(k)+'_Predictability.txt'
    f_context_transition_entropy = filename + '_transition_' + str(k) + '_time_entropy.txt'
    f_context_transition_predictability = filename + '_transition_' + str(k) + '_time_Predictability.txt'
    cnt_loc={}
    cnt_loc_time={}
    cnt_time = {}

    with open(filein,'r') as fin , \
        open(f_transition_entropy,'w')as fout_transition_entropy ,\
        open(f_context_transition_entropy,'w')as fout_context_transition_entropy,\
        open(f_transition_predictability,'w')as fout_transition_predictability,\
        open(f_context_transition_predictability,'w')as fout_context_transition_predictability:

        for line in fin :
            temp = line.split(',')
            user = temp[0].strip()
            seq_time = temp[1:]
            seq = no_time(seq_time)
            cnt_loc.clear()
            cnt_time.clear()
            cnt_loc_time.clear()

            #count frequency
            for i in range(len(seq)-k+1):
                trans = seq[i:i+k]
                trans = '&'.join(trans)
                if trans in cnt_loc.keys():
                    cnt_loc[trans]+=1
                else:
                    cnt_loc[trans]=1

                time = str(hour(int(seq_time[i].split('@')[1])))
                if time in cnt_time.keys():
                    cnt_time[time]+=1
                else:
                    cnt_time[time] = 1

                trans_time = trans+'$'+time
                if trans_time in cnt_loc_time.keys():
                    cnt_loc_time[trans_time]+=1
                else:
                    cnt_loc_time[trans_time]=1

            #transition Shannon entropy without time
            count = list(cnt_loc.values())
            probabilities = count / np.sum(count)
            unc_entropy = -np.sum(probabilities * np.log2(probabilities))

            fout_transition_entropy.write(user+','+str(unc_entropy)+'\n')
            fout_transition_predictability.write(user + ',' +str(eC.max_predictability(unc_entropy,len(cnt_loc.keys())))+'\n')

            #context transition entropy
            cnt_time_prob = normalize(cnt_time)
            cnt_loc_time_prob = normalize(cnt_loc_time)
            cnt_loc_prob = normalize(cnt_loc)
            MI = 0
            for transTime in cnt_loc_time.keys():
                loc,time = transTime.split('$')
                MI +=cnt_loc_time_prob[transTime]*np.log2(cnt_loc_time_prob[transTime]/(cnt_loc_prob[loc]*cnt_time_prob[time]))
            unc_entropy_time = unc_entropy-MI

            fout_context_transition_entropy.write(user + ',' + str(unc_entropy_time) + '\n')
            fout_context_transition_predictability.write(user + ',' + str(eC.max_predictability(unc_entropy_time,len(cnt_loc.keys()))) + '\n')

'''
get parameters from scripts
'''
def get_args():
    parser = argparse.ArgumentParser(description='calculate context transition predictability')
    parser.add_argument('--dataset_name',type=str,
                        help='dataset name(dont add suffix)')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    for i in range(2,7):
        transShannon('../data/' + args.dataset_name,i)