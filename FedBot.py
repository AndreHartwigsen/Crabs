import nest_asyncio
import numpy as np
import pandas as pd
import time
import matplotlib
import os
import sys
import requests
import markovify as mk
import datetime
# import aiohttp
matplotlib.use('Agg')
import matplotlib.pyplot as plt
nest_asyncio.apply()

spotify = np.loadtxt("Spotify.txt",dtype=str)

#https://github.com/jsvine/markovify
def list_creator(names):
    N = len(names)
    if N == 1:
        out = names[0]
    elif N > 1:
        sep = [', ']*(N-1)
        sep[-1] = ' and '
        out = names[0]
        for i in range(N-1):
            out += sep[i] + names[i+1] 
    else:
        sep = [' and ']
        out = names[0] + sep[0] + names[1]
    return out

def Calculator(string):
    result = {}
    try:
        exec('a =%s' % string,None,result)
        print('%s'%result['a'] , str(string))
        if str(result['a']) != str(string):
            return result['a']
    except:
        return 'Invalid syntax'
    
def unique(lst):
    ret = []
    for s in lst:
        if s not in ret:
            ret.append(s)
    return ret

# def current_stream(hour =int(time.strftime('%H',time.gmtime())) ): #for raid parties
#     streams = ['therealgpf','thesweedrunner','elxrdj','cptn_jaxx','ditz33']
#     times = [18,19,20,21,22]
#     current_hour = hour
#     if current_hour in times:
#         i =  [i for i in range(len(times)) if times[i] == current_hour][0]
#         string = 'Currently %s is live at https://www.twitch.tv/%s' %(streams[i],streams[i])
#     else:
#         string = 'No stream currently running'
#     return string

def week_hour():
    day = int(time.strftime('%w',time.gmtime()))-1
    hour_of_day = int(time.strftime('%H',time.gmtime()))
    if day == -1:
        day = 6
    return day*24+hour_of_day+2


def live_on_twitch(channelName='therealgpf'):
    contents = requests.get('https://www.twitch.tv/' + channelName).content.decode('utf-8')
    if 'isLiveBroadcast' in contents: 
        print(channelName + ' is live')
        return True
    else:
        print(channelName + ' is not live')
        return False






import random
def rotate(l, n):
    return l[n:] + l[:n]
orders = [[0]]*2
for i in range(1000):
    orders.append(list(np.arange(i+2)))
    random.shuffle(orders[i+2])

def Link_selector(link_list):
    N = len(link_list)
    choice = orders[N][0]
    orders[N] = rotate(orders[N],1)
    return link_list[choice]

def liner(string):
    sep = '\n'
    out = string[0]
    for i in range(len(string)-1):
        out += sep + string[i+1]
    return out
def wave(string,amplitude = 100,Nstop= 5):
    List = []
    space = []
    Nz = 0
    for i in range(200):
        space.append(np.floor(amplitude * np.sin(i/5)**2 * np.exp(-i/20) ).astype(int))
        if space[-1] == 0:
            Nz += 1
            if Nz>Nstop-1:
                space = space[:-Nstop+1]
                break
        else:
            Nz = 0
    for i in range(len(space)):
        List.append(  ' '* space[i]  + string + "\n"  )
    total = ''.join(List)
    if len(total) > 2000:
        total = total[:2000]
        end = total[::-1].index('\n')+1
        total = total[:-end]
    return total

def EightBall():
    ball8_choices = ["It is certain.",
                     "Outlook good.",
                     "You may rely on it.",
                     "Ask again later.",
                     "Cope and ask again.",
                     "Calculation hazy, seethe and try again.",
                     "My reply is no.",
                     "My fed prediction algorithm?????? says no.",
                     "The fed logs?????? are not in your favour.",
                     "My younger cousin is a retard, therefore no."
                     ]
    return np.random.choice(ball8_choices)






import discord
import asyncio
intents = discord.Intents.default()
intents.members = True
client = discord.Client(intents=intents)
from discord.ext import commands

bot = commands.Bot(command_prefix='$')

@commands.command()
async def test(ctx,arg):
    await ctx.send(arg)
bot.add_command(test)


@client.event  # event decorator/wrapper
async def on_ready():
    def guild_getter():
        guild_members = []
        emojis = []
        guild_ids = [guild.id for guild in client.guilds]
        for ig in range(len(guild_ids)):
            guild = client.get_guild(guild_ids[ig])
            temp = []
            for member in guild.members:
                temp.append(member.id)
            for emo in guild.emojis:
                emojis.append(emo)
            guild_members.append(temp)
        return guild_members,guild_ids,emojis
    global guild_members , guild_ids , emojis
    guild_members,guild_ids,emojis = guild_getter()
    print(f"Logged in as {client.user}")
    print("Guild IDs",guild_ids)
    print("Guild member counts",[len(guild_members[i]) for i in range(len(guild_members))])
    print(emojis[:2])
    if "restart_channel.csv" in os.listdir():
        if len(list(pd.read_csv("restart_channel.csv")['value']))>0:
            await client.get_channel(list(pd.read_csv("restart_channel.csv")['value'])[0]).send("Hi I'm back ????????????")
            #await client.get_channel(list(pd.read_csv("restart_channel.csv")['value'])[0]).send("Fuck you <@!252070848800882688> and <@!190897913314934784>",allowed_mentions=discord.AllowedMentions(users=mention_users),delete_after=2)
    df = pd.DataFrame({"value":[]})
    df.to_csv("restart_channel.csv")

def invalid_user_fix(txt,guild_id):
    txt.replace('<@','<@!').replace('!!','!')
    ff = "<@!"
    if ff not in txt:
        return txt
    else:
        def find_all(a_str, sub):
            start = 0
            while True:
                start = a_str.find(sub, start)
                if start == -1: return
                yield start
                start += len(sub) # use start += 1 to find overlapping matches
        def Nn(txt,start_index):
            txt = txt[start_index:]
            return -txt.index(ff) + txt.find(">")
        
        
        pos = np.asarray(list(find_all(txt,ff)))
        mentions = []
        for i in range(len(pos)):
            num = int(txt[pos[i]+len(ff):pos[i]+Nn(txt,pos[i])])
            print(num)
            if num not in mentions:
                mentions.append(num)
        IDS = guild_members[guild_ids.index(guild_id)]
        for i in range(len(mentions)):
            if mentions[i] not in IDS:
                txt=txt.replace(str(mentions[i]),str(np.random.choice(IDS)))
        return txt

#------MARKOV-------------------------------------------------------------------------------

mom_list = ['your mom','yo mamma','yo mom','my mom','his mom', 'their mom','a mom ']
def mom_mention(msg):
    out = False
    for s in mom_list:
        if s in msg:
            out = True
            break
    return out
def your_mom_joke():
    Dir='./MarkovSource/'
    with open(Dir+'jokes_your_mom.txt') as f:
        text = f.read().splitlines()
    removal = []
    for i in range(len(text)):
        if len(text[i]) == 0:
            removal.append(i)
    text = np.delete(text,removal)
    return str(text[np.random.randint(len(text))])

import emoji
default_emojis = emoji.UNICODE_EMOJI['en']
def is_emoji_msg(msg):
    msg = str(msg)
    if "<" in msg and ">" in msg and ":" in msg:
        split = msg.split()
        n = 0
        for s in split:
            if s[0] == "<" and s[-1] == ">":
                n += 1
        if len(split) == n:
            return True
        else:
            return False
    elif np.sum([s in default_emojis for s in msg.split()]) == len(msg.split()):
        return True
    else:
        return False
def invalid_emoji_fix(msg):
    spl = str(msg).split()
    for i in range(len(spl)):
        if spl[i] not in emojis and spl[i] not in default_emojis:
            emo = np.random.choice(emojis)
            spl[i] = "<%s:%s:%i>" % ("a" if emo.animated else "",emo.name,emo.id)
    return " ".join(spl)
def emoji_fix(msg):
    msg = [s for s in msg if str(s) != "nan"]
    msg2 = []
    i = 0
    while i < len(msg)-2:
        if is_emoji_msg(msg[i]):
            i += 1
        else:
            msg2.append(msg[i])
            i2 = 1
            if is_emoji_msg(msg[i+i2]):
                while is_emoji_msg(msg[i+i2]):
                    msg2[-1] = msg2[-1] + " " + msg[i+i2]
                    i2 += 1
                i += i2
            else:
                i += 1
    return msg2
def emoji_splitter(msg1):
    msg = str(msg1).split()
    p1 = []
    p2 = []
    p3 = []
    i = 0
    score = [1 if is_emoji_msg(s) else 0 for s in msg]
    if len(unique(score)) == 1:
        return None,msg1,None
    else:
        if score[0] == 1 and score[-1] == 1:
            i = 0
            while score[i] == 1:
                p1.append(msg[i])
                i+=1
            i2 = -1
            while score[i2] == 1:
                p3.append(msg[i2])
                i2 += -1
            p2 = msg[i+1:i2+1]
            return invalid_emoji_fix(" ".join(p1))," ".join(p2),invalid_emoji_fix(" ".join(p3))
        elif score[0] == 1:
            i = 0
            while score[i] == 1:
                p1.append(msg[i])
                i += 1
            p2 = msg[i+1:]
            return invalid_emoji_fix(" ".join(p1))," ".join(p2),None
        elif score[-1] == 1:
            i2 = -1
            while score[i2] == 1:
                p3.append(msg[i2])
                i2 += -1
            p2 = msg[:i2+1]
            return None," ".join(p2),invalid_emoji_fix(" ".join(p3))

def MarkovModel2(directory='./MarkovSource/',Text_only = False):
    def NewLineLister(string):
        out = []
        remainder = string
        while '\n' in remainder:
            index = remainder.find('\n')
            if len(remainder[:index])>2:
                out.append(remainder[:index])
            remainder = remainder[index+1:]
        return out
    files = [directory+s for s in os.listdir(directory) if "Logged" not in s]+['./FedData/'+s for s in os.listdir('./FedData/') if "Logged" in s and "473588284597993475" not in s]
    text = []
    for s in files:
        if 'Logged' in s:
            text = text + [str(s1) for s1 in emoji_fix(list(pd.read_csv(s)['message']))]
        else:#elif 'joke' not in s:
            with open(s, encoding="utf8") as f:
                text = text + NewLineLister(f.read())
                #text.append( f.read() )
    if Text_only:
        return text
    else:
        return mk.NewlineText(text)
text_model = MarkovModel2()
def giffile_finder():
    def fixxer(string):
        s = string.split()
        i = 0
        while i<len(s):
            if "http" in s[i]:
                return s[i]
            i += 1
        return string
    LOC = "./FedData/"
    files = [s for s in os.listdir(LOC) if "LoggedText" in s]
    master_file = pd.concat([pd.read_csv(LOC+s) for s in files])
    messages = list(master_file['message'])
    shitpost_messages = list(master_file.loc[master_file["channel"] == 508003304777973770]["message"])
    gifs = [fixxer(str(s)) for i,s in enumerate(messages) if "http" in str(s) and ".gif" in str(s)]
    videos = [fixxer(str(s)) for i,s in enumerate(messages) if "http" in str(s) and "cdn.discordapp.com" in str(s)]
    shitpost_gifs = [fixxer(str(s)) for i,s in enumerate(shitpost_messages) if "http" in str(s) and ".gif" in str(s)]
    shitpost_videos = [fixxer(str(s)) for i,s in enumerate(shitpost_messages) if "http" in str(s) and "cdn.discordapp.com" in str(s)]
    
    return unique([s for s in gifs+videos if "media.discordapp.net" not in s]),unique([s for s in shitpost_gifs+shitpost_videos if "media.discordapp.net" not in s])
random_file,shitpost_random_file = giffile_finder()
sentences = []
def gen_sentence(length):
    msg = text_model.make_short_sentence(length)
    while type(msg) != str:
        msg = text_model.make_short_sentence(length)
    return msg
async def fill_markov_library(N=10000,length=250):
    global sentences
    while len(sentences)<N:
        sentences.append(gen_sentence(length))
    sentences = [s for s in sentences if type(s) == str]
async def refill_markov_library(N=10000,length=250):
    for i in range(len(sentences)):
        sentences[i] = gen_sentence(length)
    if len(sentences)<N:
        while len(sentences)<N:
            sentences.append(gen_sentence(length))

def Sentence_relevance(question=None,length=250,Nattempt=50,remove_characters=[',','.','?','!'],
                       ignore_words = ['bot','fed','fedbot','markov','the','a','an','that','when','what','your','and','not','you','dont']
                       ):
    t_start = time.time()
    length = np.random.randint(50,length)
    global sentences
    if question == None:
        return gen_sentence(length)
    else:
        if len(sentences)<10000:
            for i in range(Nattempt):
                sentences.append(gen_sentence(length))
            sentences = [s for s in unique(sentences) if type(s) == str]
        
        
        for s in remove_characters:
            question.replace(s,'')
        words = unique(question.lower().split())
        Ncommon = np.zeros(len(sentences))
        for y in range(len(words)):
            if len(words[y])>0 and words[y] not in ignore_words:
                for i in range(len(sentences)):
                    if words[y] in sentences[i].lower().split():
                        Ncommon[i] += len(words[y])
        returner = sentences[np.argmax(Ncommon)]
        sentences.remove(returner)
        if time.time()-t_start > 3:
            return returner
        else:
            time.sleep(3)
            return returner

def cont_sentence(msg,server_id=466791064175509516,tries=150):
    starter = msg.split()[-1]
    try:
        out = text_model.make_sentence_with_start(starter,strict=True,tries=50)
        while type(out) != str:
            out = text_model.make_sentence_with_start(starter,strict=True,tries=50)
        return invalid_user_fix(" ".join(msg.split()[1:] + out.split()[1:]),server_id)
    except:
        out = Sentence_relevance(msg)
        return invalid_user_fix(out,server_id)

markov_chance_percentage = 0

def Generate_sentence(pct=markov_chance_percentage,question=None,length = 250,server_id=466791064175509516):
    if np.random.rand()<pct/100:
        msg = Sentence_relevance(question=question,length=length)
        while msg == None:
            msg = Sentence_relevance(question=question,length=length)
        return np.random.choice([invalid_user_fix(msg,server_id),np.random.choice(random_file)],p=[0.85,0.15])
    else:
        return None












markov_block_channels = [863028160795115583,768894166037823510,677967850870407198,623624161775845381,505817783745904650,466800234735992838,760010140991881216,466794303725764612,466794129838571541,466799206456360980]
#-------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------
spam_commands = ['ftrigger'
                 ,"wave [content]"
                 ,"shitpost"
                 ,"8ball [?]"
                 ,"mcont [sentence]"
                 ]
spam_desc     = ["Make the Fedbot say something"
                 ,"Create an exponentially decaying squared sinusoidal wave of [content]. (Only works with Crabs and default emojis and any message)"
                 ,"A shitpost."
                 ,"Ask the fedbot a question [?] and get an answer."
                 ,"Make markov continue [sentence]."
                 ]
#-----------------------------------------------------------------------
utility_commands = ["fbactivity [ID] N"
                    ,"ftrigger"
                    ,"fbrank"
                    ,"fbranks"
                    ,"?live/!live"
                    ]
utility_desc    = ["Brings up barchart wit N bars of user ID."
                   ,"Make markov say something. ALT: vtrigger/mtrigger."
                   ,"Get information about your Fedbot rank."
                   ,"See the top 10 crabs with most xp."
                   ,"Check if a twitch user is online."
                   ]

#-----------------------------------------------------------------------
admin_commands = ["Percentage [num]"
                    ,"fbspam"
                    ,"Toggle mentions"
                    ,"send score"
                    ]
admin_desc = ['Percentage trigger chance for Fedbot [0-100].'
                ,"Toggle the spam commands."
                ,"Toggle if the bot tags people when he mentions them."
                ,"Sends the score sheet for all users."
                ]
#-----------------------------------------------------------------------
def string_gen(commands,desc):
    command_length = [len(s) for s in commands]
    N_max = max(command_length)
    extended = []
    for i,s in enumerate(commands):
        extended.append( "`%s` " %   (s + " "*(  N_max-command_length[i]) + ":"  )   )
        extended[i] += desc[i]+'\n'
    return extended



mention_users = True
Fun = True
N_requirement = 3
Fredag_post = False
shitpost_delete = False

T0 = [0]
Trusted_IDs = list(np.loadtxt('Trusted_IDs.txt',np.int64)) ; Temp_Trusted = []
repeat_block_channels = [863028160795115583,768894166037823510,677967850870407198,623624161775845381,505817783745904650,466800234735992838,760010140991881216,466794303725764612,466794129838571541,466799206456360980]
blacklist = []





#auto removal after set amount of time (time tracker)
timer_IDs = {}
timer_times = {}
def countdown_timer(ID,counter='hey',cooldown = 6*60**2): #Check if user said Hi to the bot within cooldown time (cooldown in seconds)
    if counter not in timer_IDs:
        timer_IDs[counter] = []
        timer_times[counter] = []
    if ID in timer_IDs[counter]:
        i_ID = timer_IDs[counter].index(ID)
        if time.time() - timer_times[counter][i_ID] >= cooldown:
            timer_times[counter][i_ID] = time.time()
            return True
        else:
            return False
    else:
        timer_IDs[counter].append(ID)
        timer_times[counter].append(time.time())
        return True
def countdown_timer_left(ID,counter='hey',cooldown = 6*60**2):
    i_ID = timer_IDs[counter].index(ID)
    time_left = time.time() - timer_times[counter][i_ID]
    return int(np.ceil(  (cooldown - time_left)  /60))











bc = ['shitpost','cum','help','lortep??l','fbhelp','cope','seethe','sborra',"8ball","8 ball","toggle mentions"]
bc2 = ["engage fed mode","fbrank","fblevel","fbinfo","fbranks","fbrankings","fblevels","fbactivity"]
def Contains_command(message):
    space_index = message.find(' ')
    if space_index != -1:
        msg = message[:space_index]
    else:
        msg = message
    out = False
    if len(message) < (len(msg)+1):
        for s in bc:
            if s in msg:
                out = True
        for s in bc2:
            if s in msg:
                out = True
    return out
def contained_in_list(msg,lst=["fedbot","no you wont","fuck off","asshole","dick","denied","cunt","fuck you","bot","fed bot"]):
    i = 0
    while i<len(lst):
        if lst[i] in msg.lower():
            return True
        i +=1
    return False

#--------------Levels--------------------------------------------------------------------------------------------
fed_skip = [69]


epoch = datetime.datetime.utcfromtimestamp(0)
def dt_to_time(dt):
    convert = datetime.datetime.strptime(str(dt)[:19],'%Y-%m-%d %H:%M:%S')
    return (convert - epoch).total_seconds()
points_lower = 15
points_upper = 25
FedBot_extra  = 20
bot_channels = [803014667856904242,921404641625899028]


def reset_score(LOC = "./FedData/",time_points = 60):
    files = [s for s in os.listdir(LOC) if "LoggedText" in s]
    master_file = pd.concat([pd.read_csv(LOC+s) for s in files])
    
    def N_msg(arr):
        N = 0
        if len(arr)>0:
            N = 1
            current_dt = arr[0]
            for i in range(len(arr)-1):
                if current_dt-arr[i+1]  >= time_points:
                    current_dt = arr[i+1]
                    N+=1
        return N
    
    IDs = list(master_file['ID'])
    channel  = list(master_file['channel'])
    dt = [dt_to_time(s) for s in list(master_file['datetime'])]
    users = np.unique(IDs)
    
    
    score = np.zeros(len(users))
    end_time = []
    for i in range(len(users)):
        spec_sel = np.where(IDs == users[i])[0]
        spec_time = np.asarray(dt)[spec_sel]
        spec_channel = np.asarray(channel)[spec_sel]
        spec_time_villain = spec_time[  [i for i in range(len(spec_time)) if spec_channel[i] in bot_channels]  ]
        score[i] = np.sum( np.random.randint(points_lower,1+points_upper,N_msg(spec_time)) ) + FedBot_extra*N_msg(spec_time_villain)
        end_time.append(spec_time[-1])
    
    df = pd.DataFrame({ "User_ID":users , "score":score , "last_msg":end_time })
    df.to_csv(LOC+"SavedScore.csv",index=False)
    global levels
    levels = import_score()

def import_score(LOC = "./FedData/"):
    levels = {}
    file = pd.read_csv(LOC+"SavedScore.csv")
    levels['IDs'] = list(file['User_ID'])
    levels['score'] = list(file['score'])
    levels['time'] = list(file['last_msg'])
    return levels

if "levels" not in locals():
    try:
        levels = import_score()
    except:
        if "SavedScore.csv" in os.listdir("./FedData/"):
            levels = import_score()
        else:
            reset_score()

def lvl(points,a=400,b=500,info = False):
    level = np.floor((-b+np.sqrt(2*a*np.asarray(points,dtype=np.int64)+b**2)) / a)+1
    xp_up = 1/2*a*level**2 + b*level
    xp_low =  1/2*a*(level-1)**2 + b*(level-1)
    remaining = xp_up-points
    if not info:
        return level
    else:
        return level,xp_low,xp_up,remaining

def score_update(message,LOC = "./FedData/"):
    ID = message.author.id
    timestamp = message.created_at
    channel = message.channel.id
    
    level_up = False
    mpier = 0
    if channel in bot_channels:
        mpier = FedBot_extra
    dt = dt_to_time(str(timestamp))
    if ID in levels['IDs']:
        idx = levels['IDs'].index(ID)
        if dt - levels['time'][idx] >= 60:
            earned = np.random.randint(points_lower,1+points_upper) + mpier
            if lvl(levels['score'][idx]+earned)>lvl(levels['score'][idx]):
                level_up = True
            levels['score'][idx] += earned
            levels['time'][idx] = dt
    else:
        levels['IDs'].append(ID)
        levels['score'].append(np.random.randint(points_lower,1+points_upper) + mpier)
        levels['time'].append(dt)
        
    df = pd.DataFrame({ "User_ID":levels['IDs'] , "score":levels['score'] , "last_msg":levels['time'] })
    df.to_csv(LOC+"SavedScore.csv",index=False)    
    
    filename = 'LoggedText%i.csv'%(channel)
    files = [s for s in os.listdir(LOC) if "LoggedText" in s]
    if filename in files:
        df = pd.DataFrame({'author':message.author.name , 'message':message.content + [" " + message.attachments[0].url][0] if len(message.attachments)>0 else "" , 
                           'ID':ID , 'date':message.id , "datetime":timestamp , "channel":channel},index=[0])
        dff = pd.read_csv(LOC+filename)
        dfa = pd.concat([df,dff])
        dfa.to_csv(LOC+'LoggedText%i.csv'%(channel),index=False)
    return level_up
def rank_score(ID):
    level = levels['score']
    arr0 = np.zeros([len(level),4],dtype=np.int64)
    arr0[:,0] = level
    arr0[:,1] = np.arange(len(level))
    arr0[:,2] = levels['IDs']
    arr0[:,3] = lvl(level)
    
    arr = arr0[np.argsort(arr0[:,0])][::-1]
    sel = [i for i in range(len(arr[:,2])) if arr[:,2][i] in [item for sublist in guild_members for item in sublist]]
    arr = arr[sel]
    return list(arr[:,2]).index(ID)+1
def activity(ID,N=100,LOC = "./FedData/"):
    files = [s for s in os.listdir(LOC) if "LoggedText" in s]
    master_file = pd.concat([pd.read_csv(LOC+s) for s in files])
    
    sel = np.where(master_file['ID'] == ID)[0]
    times = np.array([dt_to_time(s) for s in np.array(master_file['datetime'])[sel]])/(25*60**2)
    times = times - min(times)
    fig = plt.figure(figsize=(6,4),dpi=200)
    fig.patch.set_facecolor("#2C2F33")
    ax = plt.subplot(111)
    plt.hist(times,N,fc="white")
    ax.set(facecolor="#2C2F33")
    plt.xlabel('days')
    plt.ylabel('Number of messages')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    plt.tight_layout()
    plt.savefig("act.png", transparent=True)
#----------------------------------------------------------------------------------------------------------------

async def get_banner(ID):
    req = await client.http.request(discord.http.Route("GET", "/users/{uid}", uid=ID))
    banner_id = req["banner"]
    if banner_id:
        banner_url = f"https://cdn.discordapp.com/banners/{ID}/{banner_id}"
        return banner_url
    else:
        return None


@client.event
async def on_message(message):
    if score_update(message):
        if message.guild.id == 466791064175509516:
            await client.get_channel(803014667856904242).send(  f"{message.author.name} just gained a Fedbot level! \nThey are now level %i and rank #%i." % (int(lvl(levels['score'][levels['IDs'].index(message.author.id)])),rank_score(message.author.id)) )
        else:
            await message.channel.send(  f"{message.author.name} just gained a Fedbot level! \nThey are now level %i and rank #%i." % (int(lvl(levels['score'][levels['IDs'].index(message.author.id)])),rank_score(message.author.id)) )
        
        

    speak_permission = True
    global Fun, admin_dink_time_override, Trusted_IDs, Sponsor_message, Temp_Trusted , Fredag_post , mention_users , shitpost_delete

    if message.author.id in Trusted_IDs and message.content.lower() == "reset score":
        await message.reply("Fedbot?????? resetting score.",delete_after=3)
        reset_score()
        await message.reply("Score reset complete",delete_after=5)
    if message.author.id in Trusted_IDs and message.content.lower() == "send score":
        await message.author.send("Here is the file",file=discord.File("./FedData/SavedScore.csv"))
    
    
    
    
        
    
    
    
    if message.author != client.user and message.channel.id not in repeat_block_channels and len(message.content.lower())>0:
        if message.content.lower().split()[0] in ["fbrank","fblevel","fbinfo"]:
            search_ID = message.author.id 
            if len(message.content.lower().split())>1:
                try:
                    search_ID = int(message.content.split()[1])
                except:
                    try:
                        search_ID = message.mentions[0].id
                    except:
                        None
                
            level,xp_low,xp_up,remaining = lvl(levels['score'][levels['IDs'].index(search_ID)],info=True)
            Nmarks = 35
            percentage_score = int(np.floor(Nmarks*remaining/(xp_up-xp_low) ))
            banner = await get_banner(search_ID)
            messageauthor = message.guild.get_member(search_ID)
            embed = discord.Embed(colour = messageauthor.top_role.colour)
            embed.set_author(name=messageauthor.nick,icon_url=messageauthor.avatar_url)
            lst1 = ["Acoount created","Server joined","Top role","Current status"]
            lst2 = [str(messageauthor.created_at)[:10],str(messageauthor.joined_at)[:10],messageauthor.top_role.name,str(messageauthor.activity)]
            embed.add_field(name="User info"  , value=''.join(string_gen(lst1[:2],lst2[:2]))  ,inline=True)
            embed.add_field(name="Server info", value=''.join(string_gen(lst1[2:],lst2[2:]))  ,inline=True)
            embed.add_field(name="Fedbot level: %i\nFedbot rank: %i"% (level,rank_score(search_ID)), 
                            value="XP: %i `[%s%s]` %i\nCurrent amount of XP: %i\nXP needed for next level: %i"
                            %( xp_low,"#"*(Nmarks-percentage_score),"-"*percentage_score,xp_up,levels['score'][levels['IDs'].index(search_ID)],remaining ),inline=False)
            if banner != None:
                embed.set_thumbnail(url=banner)
            await message.channel.send(embed=embed)
    
    
    if message.author != client.user and message.channel.id not in repeat_block_channels and message.content.lower() in ["fbranks","fbrankings","fblevels"]:
        async def ranking_bar():
            level = levels['score']
            arr0 = np.zeros([len(level),4],dtype=np.int64)
            arr0[:,1] = np.arange(len(level))
            arr0[:,0] = level
            arr0[:,2] = levels['IDs']
            arr0[:,3] = lvl(level)
            
            arr1 = arr0[np.argsort(arr0[:,0])][::-1]#[:10]
            
            
            
            names = []
            
            if "fed_skip" in globals():
                iId = 0
                isave = []
                while len(names)<10:
                    Id = arr1[iId,2]
                    if Id in [item for sublist in guild_members for item in sublist]:
                        try:
                            user = await client.fetch_user(Id)
                            names.append(str(user.name))
                        except:
                            names.append(str(Id))
                        isave.append(iId)
                    if iId<len(arr1[:,2])-1:
                        iId += 1
                    else:
                        names.append('None')
                
                # for Id in arr[:,2]:
                #     try:
                #         user = await client.fetch_user(Id)
                #         names.append(str(user.name))
                #     except:
                #         names.append(str(Id))
                    
            else:
                for i in range(len(arr1)):
                    names.append("Placeholder. %i"% (i+1))
            
            
            col = ['slategrey']*10
            col[0] = 'gold'
            col[1] = 'silver'
            col[2] = 'darkorange'
         
            arr = arr1[isave]
            arr[:,3] = arr[:,3]+0.5*np.max(arr[:,3])

            plt.figure(figsize=(6,4))
            ax = plt.subplot(111)
            ax.bar(np.arange(len(arr))+1,arr[:,3],color=col)
            plt.tight_layout()
            for i, (name, height) in enumerate(zip(names, arr[:,3])):
                ax.text(i+1, height, ' ' + name, color='black',
                        ha='center', va='top', rotation=-90, fontsize=15)
            plt.xticks(np.arange(len(arr))+1)
            plt.axis('off')
            plt.savefig("Bar.png", transparent=True)
        await ranking_bar()
        file = discord.File("Bar.png",filename="Ranking.png")
        await message.channel.send("Top 10 ranking", file=file)

    if message.author != client.user and message.channel.id not in repeat_block_channels and "fbactivity" in message.content.lower():
        if len(message.content.lower()) == len("fbactivity"):
            activity(message.author.id)
            file = discord.File("act.png",filename="activity.png")
            await message.channel.send(file=file)
        else:
            input_values = [int(s) for s in message.content.split()[1:]]
            input_ID = input_values[0]
            N_bars = 75
            if len(input_values)>1:
                N_bars = input_values[1]
            if input_ID in levels['IDs']:
                activity(input_ID,N_bars)
                file = discord.File("act.png",filename="activity.png")
                await message.channel.send(file=file)
            else:
                await message.channel.send('User %i not in archive' % input_ID)
        
            
        
        
        
        
        
        
    if message.author != client.user and message.channel.id not in repeat_block_channels:
        message_history = [];all_message_history = [];ID_history = [];all_ID_history = []
        NNN = N_requirement
        async for msg in message.channel.history(limit=7+NNN):
            all_message_history.append(msg.content)
            all_ID_history.append(msg.author.id)
            if not message.author.bot : #msg.author != client.user
                message_history.append(msg.content)
                ID_history.append(msg.author.id)
        last_message_blocker = len(np.unique([x.lower() for x in message_history[:NNN]])) == 2 and message_history[:NNN][0].lower() not in [x.lower() for x in message_history[:NNN][1:]]
        if last_message_blocker and client.user.id not in all_ID_history[:NNN]:
            if contained_in_list(message_history[0].lower()):
                message_history[0] = message_history[1]
    
        if len(np.unique([x.lower() for x in message_history[:NNN]])) == 1:
            if len(np.unique(ID_history[:NNN])) == NNN and client.user.id not in all_ID_history[:NNN]:
                if last_message_blocker:
                    await message.reply("Fedbot block-blocker?????? activated, going ahead anyways because fuck you.",delete_after=10)
                await message.channel.send(message_history[0])
                speak_permission = False
                
    
    if message.channel.id not in markov_block_channels and not Contains_command(message.content.lower()) and message.author != client.user and speak_permission:
        if message.content.lower()[:10] == 'percentage' and message.author.id in Trusted_IDs:
            global markov_chance_percentage
            try:
                new_pct = float(message.content.lower()[11:])
                if new_pct >= 0 and new_pct <= 100:
                    markov_chance_percentage = new_pct
                    await message.channel.send( 'Fedbot trigger chance changed to %.4g%%'% (markov_chance_percentage),delete_after = 10)
                else:
                    await message.reply('Percentage must be between 0 and 100',delete_after = 10)
            except:
                await message.reply('Invalid syntax',delete_after = 10)
                
                
                
        elif message.content.lower() in ["fill sentences","fill library"] and message.author.id in Trusted_IDs:
            await message.channel.send("Fedbot?????? library filler activated.",delete_after=3)
            await fill_markov_library()
            await message.reply("Fedbot?????? library filler finished.",delete_after=5)
            await message.delete()
        elif message.content.lower() in ["refill sentences","refill library"] and message.author.id in Trusted_IDs:
            await message.reply("Fedbot?????? library refiller activated.",delete_after=3)
            await refill_markov_library()
            await message.reply("Fedbot?????? library refiller finished.",delete_after=5)
        elif  message.content.lower()[:5] == "8ball" or message.content.lower()[:6] == "8 ball" or message.content.lower()[:9] == "eightball":
            await message.reply(EightBall())
            
        
        elif message.content.lower() in ['ftrigger','mtrigger',"vtrigger"]:
            generate = Generate_sentence(100,server_id=message.guild.id)
            p1,p2,p3 = emoji_splitter(str(generate))
            if p1 != None:
                await message.channel.send(p1,allowed_mentions=discord.AllowedMentions(users=mention_users))
            await message.channel.send(p2,allowed_mentions=discord.AllowedMentions(users=mention_users))
            if p3 != None:
                await message.channel.send(p3,allowed_mentions=discord.AllowedMentions(users=mention_users))
        
        elif (message.content.lower()[:5] == "mcont" or message.content.lower()[:9] == "mcontinue") and message.channel.id in bot_channels:
            generate = cont_sentence(message.content,server_id=message.guild.id)
            p1,p2,p3 = emoji_splitter(str(generate))
            if p1 != None:
                await message.channel.send(p1,allowed_mentions=discord.AllowedMentions(users=mention_users))
            await message.channel.send(p2,allowed_mentions=discord.AllowedMentions(users=mention_users))
            if p3 != None:
                await message.channel.send(p3,allowed_mentions=discord.AllowedMentions(users=mention_users))
        
        
        elif message.reference is not None and client.user in message.mentions:
            messg = await client.get_channel(message.channel.id).fetch_message(message.reference.message_id)
            if messg.author == client.user:
                await message.channel.trigger_typing()
                if mom_mention(message.content.lower()):
                    await asyncio.sleep(4)
                    await message.channel.send(your_mom_joke())
                else:
                    generate = Generate_sentence(100,message.content,server_id=message.guild.id)
                    p1,p2,p3 = emoji_splitter(str(generate))
                    if p1 != None:
                        await message.channel.send(p1,allowed_mentions=discord.AllowedMentions(users=mention_users))
                    await message.reply(p2,allowed_mentions=discord.AllowedMentions(users=mention_users))
                    if p3 != None:
                        await message.channel.send(p3,allowed_mentions=discord.AllowedMentions(users=mention_users))
                    
                    
                    
                    
        elif 'fedbot' in message.content.lower() or "fed bot" in message.content.lower() or "markov" in message.content.lower() or client.user in message.mentions:
            await message.channel.trigger_typing()
            if mom_mention(message.content.lower()):
                await asyncio.sleep(4)
                await message.channel.send(your_mom_joke())
            else:
                generate = Generate_sentence(100,message.content,server_id=message.guild.id)
                p1,p2,p3 = emoji_splitter(str(generate))
                if p1 != None:
                    await message.channel.send(p1,allowed_mentions=discord.AllowedMentions(users=mention_users))
                await message.channel.send(p2,allowed_mentions=discord.AllowedMentions(users=mention_users))
                if p3 != None:
                    await message.channel.send(p3,allowed_mentions=discord.AllowedMentions(users=mention_users))
        
        
        elif message.author != client.user:
            if message.channel.id not in bot_channels:
                mark_msg = Generate_sentence(markov_chance_percentage,server_id=message.guild.id)
            else:
                temp_percentage_chance = markov_chance_percentage + 10 if (markov_chance_percentage + 10)<=100 else 100
                mark_msg = Generate_sentence(temp_percentage_chance,server_id=message.guild.id)
            if mark_msg != None:
                p1,p2,p3 = emoji_splitter(str(mark_msg))
                if p1 != None:
                    await message.channel.send(p1,allowed_mentions=discord.AllowedMentions(users=mention_users))
                await message.channel.send(p2,allowed_mentions=discord.AllowedMentions(users=mention_users))
                if p3 != None:
                    await message.channel.send(p3,allowed_mentions=discord.AllowedMentions(users=mention_users))
            # else:
            #     if np.random.rand()<temp_percentage_chance:
            #         await message.channel.send(np.random.choice(random_file))
            
    
    if message.author.id in Trusted_IDs and 'fed mode' in message.content.lower():   
        async def Logger(limit=None,channel_id=message.channel.id,skipper = "http",LOC = "./MarkovSource/"):
            T0 = time.time()
            i = 0
            print('-------------------------%s----------------------------------' % client.get_channel(channel_id).name)
            print('Logging commenced')
            messages = []
            author = []
            id_author = []
            date = []
            datetime = []
            async for d in client.get_channel(channel_id).history(limit=limit):
                i+=1
                if len(d.content)>0: 
                    if skipper == None or skipper not in d.content:
                        messages.append(d.content + " " + d.attachments[0].url if len(d.attachments)>0 else "")
                        author.append(d.author.name)
                        date.append(d.id)
                        id_author.append(d.author.id)
                        datetime.append(d.created_at)
                if len(d.attachments)>0 and len(d.content) == 0:
                    if skipper == None or skipper not in d.content:
                        messages.append(d.attachments[0].url)
                        author.append(d.author.name)
                        date.append(d.id)
                        id_author.append(d.author.id)
                        datetime.append(d.created_at)
                if i%(5000 if limit == None else limit//5) == 0:
                    TN = time.time()
                    TD = TN - T0
                    print('%i done at %i per sec' % (i,i/TD))
            TN = time.time()
            TD = TN - T0
            channel_ids = [channel_id]*len(messages)
            print('%i done of %s at %i per sec' % (i,client.get_channel(channel_id).name,i/TD))
            print('Total time was %i seconds'%(TD))
            print('-----------------------------------------------------------')
            await message.author.send('Total of %i messaged logged in %s, taking %i minutes at %i messages per second' % ( i , client.get_channel(channel_id).name , (TD)/60 , i/(TD) ) ,delete_after=60)
            df = pd.DataFrame({'author':author , 'message':messages , 'ID':id_author , 'date':date , "datetime":datetime , "channel":channel_ids})
            df.to_csv(LOC+'LoggedText%i.csv'%(channel_id),index=False)
            if "FedData" in LOC:
                df.to_csv("./MarkovSource/"+'LoggedText%i.csv'%(channel_id),index=False)

    if message.author.id in Trusted_IDs and message.content.lower() == 'fedbot, engage fed mode':
        await message.channel.send('Alright stealing all the messages in this channel (this will take a while)',delete_after=10)
        await Logger()
    if message.author.id in Trusted_IDs and message.content.lower() == 'engage fed mode':
        all_channels = [s.id for s in message.guild.text_channels]
        # print([client.get_channel(channel_id).server.me.permission for channel_id in all_channels])
        for chid in all_channels: 
            if chid not in fed_skip:
                try:
                    await Logger(channel_id=chid,skipper=None,LOC = "./FedData/")
                except:
                    print("Skipped channel (No permission) %s" %client.get_channel(chid).name)
        await message.author.send('Fed mode finished',delete_after=10)
        
    elif message.author.id in Trusted_IDs and message.content.lower() == 'engage recent fed mode':
        all_channels = [s.id for s in message.guild.text_channels]
        # print([client.get_channel(channel_id).server.me.permission for channel_id in all_channels])
        for chid in all_channels: 
            if chid not in fed_skip:
                try:
                    await Logger(5000,channel_id=chid,skipper=None)
                except:
                    print("Skipped channel (No permission) %s" %client.get_channel(chid).name)
        await message.author.send('Recent Fed mode finished',delete_after=10)
    

    if (message.channel.id not in markov_block_channels) and message.author.id not in blacklist:
        print(f"{message.channel}:{message.created_at}:: {message.author.name}: {message.content}")
    if (message.author != client.user and message.channel.id not in markov_block_channels) and message.author.id not in blacklist:

        


        
        if "?live" in message.content.lower()[:5] or "!live" in message.content.lower()[:5]:
            channel_Name = message.content[6:]
            if live_on_twitch(channel_Name):
                await message.channel.send("YAY! %s is live right now! https://www.twitch.tv/%s" % (channel_Name,channel_Name))
            else:
                await message.channel.send("Oh nooo... %s seems to be offline :(" % (channel_Name))

        
        if 'fbspam' in message.content.lower()[:6] and message.author.id in Trusted_IDs:
            if not Fun:
                Fun = True
                await message.channel.send('Sperm now enabled')
            elif Fun:
                Fun = False
                await message.channel.send('Sperm now disabled')
        if 'toggle mentions' == message.content.lower() and message.author.id in Trusted_IDs:
            if not mention_users:
                mention_users = True
                await message.channel.send('Mentions now enabled')
            elif mention_users:
                mention_users = False
                await message.channel.send('Mentions now disabled')
        
        if 'bbspam' in message.content.lower()[:6] and message.author.id not in Trusted_IDs:
            await message.channel.send(f'{message.author.mention}. Only LeCerial and Truxa have the right to touch sperm. ????')
            
        if Fun:
            if contained_in_list(message.content.lower(),["cereal music","cerial music","ceriel music","andre music","danish music","dane music"]):
                if countdown_timer(message.author.id,'cerial music',5*60) or message.author.id in Trusted_IDs:
                    await message.channel.trigger_typing()
                    await message.reply("np bro, i got you")
                    await message.channel.trigger_typing()
                    await message.channel.send("<:goolsburdo:850075797758017547>")
                    await message.channel.trigger_typing()
                    await message.channel.send(str(Link_selector(spotify)))
            if message.content.lower() in ['seethe','cope','prolapse','have sex','dilate','mald','stay mad',"didn't ask"]:
                await message.channel.trigger_typing()
                await message.channel.send(file=discord.File('./images/cope/%s' % Link_selector([s for s in os.listdir("./images/cope/") if '.ini' not in s])))
            if 'fbcum' == message.content.lower()  or 'cum' == message.content.lower() or 'sborra' == message.content.lower():
                await message.channel.trigger_typing()
                await message.channel.send(file=discord.File('./images/cum/%s' % Link_selector([s for s in os.listdir("./images/cum/") if '.ini' not in s])) )        
            if 'fbshitpost' == message.content.lower()  or 'shitpost' == message.content.lower() or 'lortep??l' == message.content.lower() or '???? post' == message.content.lower():
                await message.channel.trigger_typing()
                if not Fredag_post and int(time.strftime('%w',time.gmtime())) == 5:
                    await message.channel.send('NU ??R DET FREDAG!!!',file=discord.File('./images/shitpost/friday33.mp4'))
                    Fredag_post = True
                else:
                    
                    link_or_file = np.random.choice(["Link","File"],p=[0.8,0.2])
                    if link_or_file == "File":
                        File_Selected = Link_selector([s for s in os.listdir("./images/shitpost/") if '.ini' not in s])
                        while 'friday33' in File_Selected:
                            File_Selected = Link_selector([s for s in os.listdir("./images/shitpost/") if '.ini' not in s])
                        await message.channel.send(file=discord.File('./images/shitpost/%s' %File_Selected ) )
                    else:
                        await message.channel.send(np.random.choice(shitpost_random_file ))
                if shitpost_delete:
                    await message.delete()
                        
                if Fredag_post and int(time.strftime('%w',time.gmtime())) != 5:
                    Fredag_post = False
            if "toggle shitpost delete" == message.content.lower() and message.author.id in Trusted_IDs:
                if shitpost_delete:
                    shitpost_delete = False
                    await message.reply("now no longer deleting shitpost command",delete_after=5)
                    await message.delete()
                elif not shitpost_delete:
                    shitpost_delete = True
                    await message.reply("now deleting shitpost command",delete_after=5)
                    await message.delete()
            
            
            
            if 'wave' == message.content.lower()[:4] and (message.channel.id in bot_channels or message.author.id in Trusted_IDs):
                await message.channel.trigger_typing()
                T_wave_cooldown = 30*60
                if countdown_timer(message.author.id,'emoji',T_wave_cooldown) or (message.author.id in Trusted_IDs or message.author.id in Temp_Trusted):
                    emoji = message.content[5:]
                    await message.channel.send( wave(emoji) )
                else:
                    t_left = countdown_timer_left(message.author.id,'emoji',T_wave_cooldown)
                    await message.reply('This command has a %i minute cooldown per user. (%i min left)' % (T_wave_cooldown/60,t_left) )
            
            
            if "real" == message.content.lower() and np.random.rand()>0.8:
                await message.channel.send("and true")
            if "true" == message.content.lower() and np.random.rand()>0.8:
                await message.channel.send("Real")
            if message.content.lower() in ["i'm losing it","i am losing it","i am going insane","i'm going insane","i hate my life","i hate myself"]:
                if countdown_timer(message.author.id,'same post',24*60*60):
                    await message.channel.send("same")
            if message.author.id == 973191883708846090 and np.random.rand()>0.90:
                if countdown_timer(message.author.id,'Kepe Sob',4*60*60):
                    await message.channel.send("????")
            if message.author.id == 294790386105188352 and np.random.rand()>0.99:
                if countdown_timer(message.author.id,'Harrison flag',12*60*60):
                    await message.channel.send("????????")
            if message.author.id == 343117244563193857 and np.random.rand()>0.99 and "<:cringecat:717096199911375009>" not in message.content:
                if countdown_timer(message.author.id,'catstare pesto',12*60*60):
                    await message.channel.send("<:cringecat:717096199911375009>",delete_after=7)
            if message.author.id == 236902492955344898 and np.random.rand()>0.75 and len(message.content)>199 and message.channel.id == 804316586810540034:
                if countdown_timer(message.author.id,'Lui pills',24*60*60):
                    await message.channel.send("https://tenor.com/view/ok-schizo-ok-schizo-schizophrenia-gibbon-gif-23667455")
                    await message.channel.send("Don't forget to take your meds Lui.")
            # if "meme" == message.content.lower():
            #     generate = Generate_sentence(100,server_id=message.guild.id)
            #     p1,p2,p3 = emoji_splitter(str(generate))
            #     embed = discord.Embed(title=p2, description="")
            #     async with aiohttp.ClientSession() as cs:
            #         async with cs.get('https://www.reddit.com/r/dankmemes/new.json?sort=hot') as r:
            #             res = await r.json()
            #             embed.set_image(url=res['data']['children'] [random.randint(0, 25)]['data']['url'])
            #             await message.channel.send(embed=embed)
                
            
            
        if 'trust' == message.content.lower()[:5] and message.author.id in Trusted_IDs:
            if len(message.mentions)>0:
                mentions = [s.id for s in message.mentions if (s.id not in Trusted_IDs)]
                for ID in mentions:
                    Temp_Trusted.append(ID)
        
        
        
            
        if 'calculate' in message.content.lower()[:9] and (message.author.id in Trusted_IDs or message.author.id in Temp_Trusted):
            await message.channel.send(Calculator(message.content[10:]))
        

            
        async def closing_options():
            channel = message.channel.id
            df = pd.DataFrame({"value":[channel]})
            df.to_csv("restart_channel.csv")
            await client.close()
            
        if 'fuck off bot' == message.content.lower() and message.author.id in Trusted_IDs:
            await message.channel.send('Okay bye')
            await closing_options()
        if 'update bot' == message.content.lower() and message.author.id in Trusted_IDs:
            await message.channel.send('Updating from github')
            if 'win' not in sys.platform:
                exec_return = os.system('git pull https://github.com/AndreHartwigsen/Crabs.git')
                print(exec_return)
        if 'fuck off bot then come back' == message.content.lower() and message.author.id in Trusted_IDs:
            await message.channel.send('Okay restarting')
            if 'win' not in sys.platform:
                os.system('git pull https://github.com/AndreHartwigsen/Crabs.git')
                await closing_options()
            else:
                os.execv(sys.executable, ['python'] + sys.argv)
        if 'kill bot' == message.content.lower() and message.author.id in Trusted_IDs:
            os.system('pm2 stop DinkingBot')
            await closing_options()


        

        
        if 'fbhelp' in message.content.lower():
            embed = discord.Embed(title=f'List of {client.user.name}?????? commands:',colour=discord.Colour.orange())
            embed.set_author(name=client.user.name, icon_url=client.user.avatar_url)
            embed.add_field(name='Utility commands:',                           value=''.join(string_gen(utility_commands,utility_desc)),inline=False)
            embed.add_field(name='Spam related:',                               value=''.join(string_gen(spam_commands,spam_desc)),inline=False)
            embed.set_footer(text=[f'{client.user.name}?????? takes no responsibility for any following or imminent alcoholism and or psycological damage caused by interaction. \n'
                              +f'The {client.user.name}?????? is only for use in the official crabs?????? server.'
                              ][0])
            await message.reply(embed=embed)
            
            if message.author.id in Trusted_IDs:
                embed = discord.Embed(title=f'List of {client.user.name}?????? commands:',colour=discord.Colour.red())
                embed.add_field(name='Admin commands:',value=''.join(string_gen(admin_commands,admin_desc)),inline=False)
                embed.set_footer(text=['Current status:\n'
                                       +"Spam enabled : %s\n" % Fun
                                       ][0])
                await message.author.send("You are in the trusted IDs, so here are the admin commands.",embed=embed)
            




token = open("token.txt", "r").read()
client.run(token)








