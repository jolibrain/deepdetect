import os, argparse, glob, sys, subprocess
from collections import defaultdict

def sizeof_fmt(num):
    for x in ['bytes','KB','MB','GB']:
        if num < 1024.0 and num > -1024.0:
            return "%3.1f%s" % (num, x)
        num /= 1024.0
    return "%3.1f%s" % (num, 'TB')

class Synset:
    'A representation of a category, aka synset'
    _name = ''
    _desc = ''
    _syn = ''
    _loc = ''
    _img_count = 0 # number of images in synset
    _imgs = []
    _size = 0
    _parent = ''
    _children = []

    def __init__(self, loc):
        self._loc = loc
        self._syn = os.path.basename(os.path.normpath(loc))

    def print_synset(self):
        print '----------------------'
        print self._syn
        print self._name
        print self._desc
        print self._img_count, "images"
        print sizeof_fmt(self._size)
        print '----------------------'

def load_words(wordsfile):
    words = {}
    with open(wordsfile) as f:
        words = dict(x.rstrip().split(None, 1) for x in f)
    return words

def load_descs(descfile):
    descs = {}
    with open(descfile) as f:
        descs = dict(x.rstrip().split(None,1) for x in f)
    return descs

def load_treemap(treemapfile):
    tdict = defaultdict(list)
    with open(treemapfile) as f:
        for line in f:
            ls = line.rstrip().split(' ')
            tdict[ls[0]].append(ls[1])
    return tdict

def read_synsets(alldirs,synsets,descs,search,lsynsets):
    synsetsobj = {}
    for d in alldirs:
        s = Synset(d)
        if lsynsets:
            if not s._syn in lsynsets:
                continue
        s._name = synsets[s._syn]
        if search:
            if not search in s._name:
                continue
        s._desc = descs[s._syn]
        s._imgs = glob.glob(d + "/*")
        s._img_count = len(s._imgs)
        s._size = sum(os.path.getsize(f) for f in s._imgs if os.path.isfile(f))
        synsetsobj[s._syn] = s
    return synsetsobj

def find_treemap(lsyn,tmap):
    # - iterate lsyn
    # - for each key get the subsynets
    # - if no subsynets add to temporary lsyn
    # - otherwise remove key from lsyn (if fact only if no image, so we leave it for now)
    # - merge lsyn with temporary lsyn
    clsyn = lsyn
    tlsyn = []
    for key in lsyn:
        ls = tmap[key]
        if ls:
            #tlsyn.remove(key)
            for l in ls:
                #tlsyn.append(l)
                ttlsyn = []
                ttlsyn.append(l)
                ttlsyn = find_treemap(ttlsyn,tmap)
                #print 'ttlsyn=',ttlsyn
                tlsyn = tlsyn + ttlsyn
                #print 'tlsyn=',tlsyn
    lsyn = clsyn + tlsyn
    return lsyn

def write_dict(files,ffile):
    f = open(ffile,'w')
    for key in files:
        line = str(key) + ' ' + str(files[key]) + '\n'
        f.write(line)

parser = argparse.ArgumentParser(description='Imagenet processing tools')
parser.add_argument('repository',type=str,help='location of the imagenet repository')
parser.add_argument('--list',dest='list',action='store_true',help='list repository, read-only')
parser.add_argument('--dataset',dest='dataset',type=str,help='location of a dataset to be created based on search terms (--search) or list (--synsets) of synsets')
parser.add_argument('--trainperc',dest='trainperc',type=float,help='% of the dataset to be used as training set')
parser.add_argument('--search',dest='search',type=str,default='',help='search for synsets whose name contains the search term')
parser.add_argument('--synsets',dest='synsets',type=str,help='list of synsets, possibly in a file, to be looked up')
parser.add_argument('--subsynsets',dest='subsynsets',type=str,default='none',help='use treemaps to retrieve synsets that are part of a higher level synset')
args = parser.parse_args()

allsynsets = load_words('words.txt')
alldescs = load_descs('gloss.txt')
alldirs = glob.glob(args.repository + "/n*")

print "Found", len(alldirs), "image repositories as synsets"

lsynsets = {}
if args.synsets:
    if not '.' in args.synsets: # not a file
        l = args.synsets.split(',')
        for e in l:
            lsynsets[e] = 1
    else:
        with open(args.synsets) as f:
            lsynsets = dict(x.rstrip().split(None,1) for x in f)

if not args.subsynsets == 'none' and not args.subsynsets == '':
    lsynsets[args.subsynsets] = 1
allsynsetsobj = read_synsets(alldirs,allsynsets,alldescs,args.search,lsynsets)
print "Found", len(allsynsetsobj), "relevant synsets"

if not args.subsynsets == 'none':
    treemap = load_treemap('wordnet.is_a.txt')
    lsyn = []
    for key,value in allsynsetsobj.items():
        for l in treemap[key]:
            lsyn.append(l)
    lsyn = find_treemap(lsyn,treemap)
    #print len(lsyn)
    subsynsetsobj = read_synsets(alldirs,allsynsets,alldescs,'',lsyn)
    allsynsetsobj = dict(allsynsetsobj,**subsynsetsobj)

if args.list:
    totalsize = 0
    for key,value in allsynsetsobj.items():
        value.print_synset()
        totalsize = totalsize + value._size
    print "Found", len(allsynsetsobj), "relevant synsets"
    print "Number of images:",sum(allsynsetsobj[o]._img_count for o in allsynsetsobj)
    print "Total size: "+ sizeof_fmt(totalsize)

elif args.dataset:
    try:
        os.mkdir(args.dataset)
    except:
        pass
    if not args.trainperc:
        for key,value in allsynsetsobj.items():
            os.symlink(value._loc,args.dataset + "/" + value._syn)
    else:
        print "Processing dataset", args.dataset
        trainrep = 'train'
        valrep = 'val'
        trainpath = args.dataset + "/" + trainrep
        valpath = args.dataset + "/" + valrep
        trainfile = args.dataset + '/train.txt'
        valfile = args.dataset + '/val.txt'
        correspfile = args.dataset + '/corresp.txt'
        tfiles = {}
        vfiles = {}
        corresp = {}
        try:
            os.mkdir(trainpath)
            os.mkdir(valpath)
        except:
            pass
        cl = 0
        gifconverts = 0
        for key,value in allsynsetsobj.items():
            thresh = int(len(value._imgs)*args.trainperc/100.0)
            train_list = value._imgs[0:thresh]
            val_list = value._imgs[thresh:int(len(value._imgs))]
            lpath = trainpath + "/" + value._syn
            if not cl in corresp:
                corresp[cl] = key + ' ' + value._name
            try:
                os.mkdir(lpath)
            except:
                pass
            for f in train_list:
                fname = os.path.basename(os.path.normpath(f))
                if ".gif" in fname:
                    fname = fname + ".jpg"
                    convcmd = f + ' ' + trainpath + '/' + value._syn + '/' + fname
                    os.system("/usr/bin/convert " + convcmd)
                    gifconverts += 1
                else:
                    os.symlink(f,trainpath + "/" + value._syn + "/" + fname)
                tfiles[value._syn + '/' + os.path.basename(fname)] = cl
            for f in val_list:
                fname = os.path.basename(os.path.normpath(f))
                if ".gif" in fname:
                    fname = fname + ".jpg"
                    convcmd = f + ' ' + valpath + '/' + os.path.basename(fname)
                    os.system("/usr/bin/convert " + convcmd)
                    gifconverts += 1
                else:
                    os.symlink(f,valpath + "/" + os.path.basename(fname))
                vfiles[os.path.basename(fname)] = cl
            cl += 1
        write_dict(corresp,correspfile)
        write_dict(tfiles,trainfile)
        write_dict(vfiles,valfile)
        print "converted " + str(gifconverts) + " gif files"
