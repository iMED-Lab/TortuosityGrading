import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import csv
from dataloader.tortuosity import MyData
from utils.evaluation_metrics import get_metrix

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

DATABASE = 'Corneal5Folds/Fold5/'
args = {
    'root'      : './dataset/' + DATABASE,
    'pred_path' : "./assets/MajorRevision/",
    'model_name': "model"
}


def write2csv_pred(csv_name, seg_name, class_name, probability):
    with open(csv_name, 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([seg_name, class_name, probability])


def load_net():
    net = torch.load("/checkpoint/DeepGrading.pth")
    return net


def predict():
    net = load_net()
    net.eval()
    data = MyData(root_dir=args['root'], train="test")
    test_data = DataLoader(data, batch_size=64, num_workers=8)
    preds, gts = [], []
    with torch.no_grad():
        print("Predicting ...")
        for idx, batch in enumerate(test_data):
            img = batch[0]["img"].cuda()
            seg = batch[0]["seg"].cuda()
            roi = batch[0]["roi"].cuda()
            image = torch.cat((img, seg, img), dim=1)
            # roi = torch.cat((roi, roi), dim=1)
            label = batch[1]["img_id"].cuda()

            x1, x2, roi, predictions = net(image, roi)
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.data.cpu().numpy()
            label = label.data.cpu().numpy()
            preds.extend(predictions)
            gts.extend(label)
            for i in range(len(predictions)):
                file_name = os.path.basename(batch[1]["img_name"][i])
                class_name = predictions[i]
                save_folder = os.path.join(args['pred_path'], "predictions")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                write2csv_pred(os.path.join(save_folder, "result_pred.csv"), file_name, class_name, p)
    preds = np.array(preds)
    gts = np.array(gts)
    wacc, wse, wsp = get_metrix(preds, gts)
    print("wAcc: %.4f" % wacc[0])


if __name__ == '__main__':
    predict()
