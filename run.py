import os
import pickle
if __name__ == '__main__':
    dump1 = "predict/test2_0.70/resnext50/"
    dump3 = "predict/test2_0.70/resnext101/"
    dump7 = "predict/test2_0.70/dpn92/"
    dump8 = "predict/test2_0.70/resnet269_v2/"
    dump12 = "predict/test2_0.70/inception_resnet/"
    dump13 = "predict/test2_0.70/inception_v3/"
    dump14 = "predict/test2_0.70/inception_v4/"

    dump1_xu = "predict/test2_0.70_xu/resnext50/"
    dump3_xu = "predict/test2_0.70_xu/resnext101/"
    dump7_xu = "predict/test2_0.70_xu/dpn92/"
    dump8_xu = "predict/test2_0.70_xu/resnet269_v2/" 
    dump12_xu = "predict/test2_0.70_xu/inception_resnet/"
    dump13_xu = "predict/test2_0.70_xu/inception_v3/"
    dump14_xu = "predict/test2_0.70_xu/inception_v4/"    


    dump1_o = "predict/test2/resnext50_2/"
    dump3_o = "predict/test2/resnext101/"
    dump7_o = "predict/test2/dpn92_2/"
    dump8_o = "predict/test2/resnet269_v2/"
    dump12_o = "predict/test2/inception_resnet_2/"
    dump13_o = "predict/test2/inception_v3/"
    dump14_o = "predict/test2/inception_v4/"

    f = open("data/test2.txt","rb")
    f_ms = open("data/test2_0.70.txt","rb")
    f_ms_xu = open("test2_0.70_xu.txt","rb")
    f_w = open("predict.txt","wb")
    lines = f.readlines()
    all_img = [l.strip().split("/")[-1] for l in lines]
    img_path = dict()
    for l in f_ms.readlines():
        l = l.strip()
        img_name = l.split("/")[-1]
        if img_name.split("_")[-1] not in img_path:
            img_path[img_name.split("_")[-1]] = list()
        img_path[img_name.split("_")[-1]].append(l)

    img_path_xu = dict()
    for l in f_ms_xu.readlines():
        l = l.strip().split(" ")
        img_name = l[0].split("/")[-1]
        if img_name.split("_")[-1] not in img_path_xu:
            img_path_xu[img_name.split("_")[-1]] = list()
        img_path_xu[img_name.split("_")[-1]].append(l[0])

    count = 0
    for img in all_img:
        count += 1
        all_path = img_path[img]
        all_path_xu = img_path_xu[img]
        img_name = img
        final_score = 0
        score1_o = pickle.load(open(dump1_o+img_name,"rb")) 
        score3_o = pickle.load(open(dump3_o+img_name,"rb")) 
        score7_o = pickle.load(open(dump7_o+img_name,"rb"))
        score8_o = pickle.load(open(dump8_o+img_name,"rb"))

        score12_o = pickle.load(open(dump12_o+img_name,"rb")) 
        score13_o = pickle.load(open(dump13_o+img_name,"rb")) 
        score14_o = pickle.load(open(dump14_o+img_name,"rb")) 
        tmp_score = 0
        tmp_score += 1.5*score1_o 
        tmp_score += score3_o/2
        tmp_score += 1.2*score7_o
        tmp_score += score8_o
        tmp_score += 1.2*score12_o
        tmp_score += score13_o/2
        tmp_score += score14_o/2
        final_score += 2*tmp_score/5

        for full_path in all_path:
            if "_" not in full_path.split("/")[-1]:
                continue
            score = float(full_path.split("/")[-1].split("_")[0])
            if score < 0.10: continue
            new_img_name = str(score) + "_" + img_name
            score1 = pickle.load(open(dump1+new_img_name,"rb"))
            score3 = pickle.load(open(dump3+new_img_name,"rb")) 
            score7 = pickle.load(open(dump7+new_img_name,"rb")) 
            score8 = pickle.load(open(dump8+new_img_name,"rb")) 
            score12 = pickle.load(open(dump12+new_img_name,"rb")) 
            score13 = pickle.load(open(dump13+new_img_name,"rb")) 
            score14 = pickle.load(open(dump14+new_img_name,"rb")) 
            tmp_score = 0
            tmp_score += 1.5*score1 
            tmp_score += score3/2
            tmp_score += 1.2*score7
            tmp_score += 1.2*score8
            tmp_score += score12
            tmp_score += score13/2
            tmp_score += score14/2

            final_score += tmp_score/5

        for full_path in all_path_xu :
            if "_" not in full_path.split("/")[-1]:
                continue
            score = float(full_path.split("/")[-1].split("_")[0])
            if score < 0.10: continue
            new_img_name = str(score) + "_" + img.split("/")[-1]
            score1 = pickle.load(open(dump1_xu+new_img_name,"rb"))
            score3 = pickle.load(open(dump3_xu+new_img_name,"rb")) 
            score7 = pickle.load(open(dump7_xu+new_img_name,"rb")) 
            score8 = pickle.load(open(dump8_xu+new_img_name,"rb")) 
            score12 = pickle.load(open(dump12_xu+new_img_name,"rb")) 
            score13 = pickle.load(open(dump13_xu+new_img_name,"rb")) 
            score14 = pickle.load(open(dump14_xu+new_img_name,"rb")) 
            tmp_score = 0
            tmp_score += 1.5*score1 
            tmp_score += score3/2
            tmp_score += 1.2*score7
            tmp_score += 1.2*score8
            tmp_score += score12
            tmp_score += score13/2
            tmp_score += score14/2
            final_score += tmp_score/5

        label = int(final_score.argmax(axis=0))
        if int(final_score.argmax(axis=0)) == 46:
            label = 31
        if int(final_score.argmax(axis=0)) == 111:
            label = 87
        if int(final_score.argmax(axis=0)) == 74:
            label = 72

        print "count: ", count  
        f_w.write(str(label)+"\t"+img_name.split(".")[0] +"\n") 
