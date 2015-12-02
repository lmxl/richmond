__author__ = 'ymo'
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
#files = ['report/stderr.v1.txt', 'report/stderr.v2.txt', 'report/stderr.v3.txt', 'report/stderr.v4.txt']
#lcolor = ["b","r", "g", "y"]#, 'c', 'm', 'k', 'violet']
#text = ["Active Coarse", "Passive Coarse", "Active FineGrain", "Passive FineGrain"]#, '20%','40%','60%','80%']

files = ['report/stderr.v1.txt' , 'report/stderr.v20.txt', 'report/stderr.v40.txt', 'report/stderr.v60.txt',
         'report/stderr.v80.txt', 'report/stderr.v3.txt']
lcolor = ["b","r", "g", "y", 'c', 'm' ]
text = ['0%','20%','40%','60%','80%','100%']
plt.figure()
for cfg in range(len(files)):
    sizes = []
    area = []
    for sunit in open(files[cfg]).read().replace('\n', ' ').replace('}','}#').split('#'):
        if len(sunit) < 10:
            continue
        unit = json.loads(sunit)
        sizes.append(unit['size'])
        area.append(unit['pr'])
    print cfg
    plt.plot(sizes, area, c=lcolor[cfg], label=text[cfg], linewidth=1)
plt.ylabel("AUC")
plt.xlabel("Example Counts")
plt.title("AUC learning curve step=[%d]"%(50))
plt.legend(loc=4)
plt.savefig("curve-richmond.eps")
plt.show()

# exit()

ratio = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

sizes = []
area = []
plt.figure()
last = 0
first = 9999999
for sunit in open(files[0]).read().replace('\n', ' ').replace('}','}#').split('#'):
    if len(sunit) < 10:
        continue
    unit = json.loads(sunit)
    sizes.append(unit['size'])
    area.append(unit['pr'])
    first = min(first, unit['pr'] )-0.001
    last = max(last, unit['pr'])+0.001
sizes.append(100000)
area.append(last)
sizes.append(0)
area.append(first)
area2cost = interp1d(area, sizes)#, kind='square')
for cfg in range(len(files)):
    budgets = []
    ratios = []
    for sunit in open(files[cfg]).read().replace('\n', ' ').replace('}','}#').split('#'):
        if len(sunit) < 10:
            continue
        unit = json.loads(sunit)
        pr = unit['pr']
        counts = unit['size']
        finec = counts*ratio[cfg]
        coarsec = counts - finec
        print pr
        try:
            cost = area2cost(pr)
        except ValueError:
            break
        if cost > 8000:
            continue
        print len(ratios), pr, cost
        r = (cost- coarsec)/(finec+0.00001)
        ratios.append(r)
        budgets.append(cost)
    print cfg
    ratios = [r for (b,r) in sorted(zip(budgets,ratios))]
    budgets = sorted(budgets)
    plt.plot(budgets, ratios, c=lcolor[cfg], label=text[cfg], linewidth=1)
plt.ylabel("AUC")
plt.xlabel("Example Counts")
plt.title("AUC learning curve step=[%d]"%(50))
plt.legend(loc=4)
plt.savefig("curve-richmond.eps")
plt.show()