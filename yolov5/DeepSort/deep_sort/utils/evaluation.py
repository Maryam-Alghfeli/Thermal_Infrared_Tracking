import os
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'
from utils.io import read_results, unzip_objs

#from yolov5.Yolov5_DeepSort_Pytorch.deep_sort.utils.io import read_results, unzip_objs


# from yolov5.Yolov5_DeepSort_Pytorch.deep_sort.utils.io import read_results, unzip_objs

# import sys
# sys.path.append('/home/nouf.alshamsi/AI702/Project/yolov5/Yolov5_DeepSort_Pytorch/deep_sort/utils')
# from utils.io import read_results, unzip_objs
# #from io import read_results, unzip_objs





class Evaluator(object):

    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'

        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]


        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()


# # create an instance of the Evaluator class
# evaluator = Evaluator(data_root='/home/nouf.alshamsi/AI702/Project/TII_test_MOTTT/merged.txt', seq_name='MOT16-02', data_type='mot')

# # evaluate a result file
# result_file = '/home/nouf.alshamsi/AI702/Project/yolov5/Yolov5_DeepSort_Pytorch/runs/track/exp6/TII.txt'
# acc = evaluator.eval_file(result_file)

# # get the summary of the evaluation results
# summary = Evaluator.get_summary([acc], ['result'])
# print(summary)


# import os
# import numpy as np
# import motmetrics as mm

# def compute_mot_metrics(gt_path, res_path):
#     """Compute MOT metrics given ground-truth and result files."""

#     # Load files
#     gt = mm.io.loadtxt(gt_path)
#     res = mm.io.loadtxt(res_path)
#     print(gt)
#     print(res)

#     # Define metric
#     metrics = mm.metrics.motchallenge_metrics
#     mh = mm.metrics.create()

#     # Compute metrics
#     acc = mh.compute(gt, res, metrics=metrics)

#     # Print summary
#     print(mm.io.render_summary(acc, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    
#     return acc

# compute_mot_metrics('/home/nouf.alshamsi/AI702/Project/TII_test_MOTTT/merged.txt','/home/nouf.alshamsi/AI702/Project/yolov5/Yolov5_DeepSort_Pytorch/runs/track/exp6/TII.txt' )


# import motmetrics as mm
# import pandas as pd

# # Define paths to ground-truth and result files
# gt_file = '/home/nouf.alshamsi/AI702/Project/TII_test_MOTTT/merged.txt'
# res_file = '/home/nouf.alshamsi/AI702/Project/yolov5/Yolov5_DeepSort_Pytorch/runs/track/exp6/TII.txt'


# # Load data from files
# #gt = pd.read_csv(gt_file, sep=',', header=None, names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', "_"])
# #res = pd.read_csv(res_file, sep=',', header=None, names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', "_"])
# gt = pd.read_csv(gt_file, sep=',', header=None, names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
# res = pd.read_csv(res_file, sep=',', header=None, names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])

# print(gt.columns)
# print(res.columns)
# print(gt)
# print(res)

# # Define the metrics to compute
# metrics = mm.metrics.motchallenge_metrics

# # Create a metric accumulator
# acc = mm.MOTAccumulator(auto_id=True)

# # Update the metric accumulator with the data
# for frame_id, gt_frame in gt.groupby('FrameId'):
#     res_frame = res[res['FrameId'] == frame_id]
#     gt_tlwh = gt_frame[['X', 'Y', 'Width', 'Height']].values
#     gt_ids = gt_frame['Id'].values
#     res_tlwh = res_frame[['X', 'Y', 'Width', 'Height']].values
#     res_ids = res_frame['Id'].values
#     acc.update(gt_ids, res_ids, mm.distances.iou_matrix(gt_tlwh, res_tlwh, max_iou=0.5))

# # Compute the metrics
# mh = mm.metrics.create()
# summary = acc.compute(metrics=metrics)

# # Print the results
# print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
