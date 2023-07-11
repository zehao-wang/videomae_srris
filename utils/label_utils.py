
import json
import os
import numpy as np

def get_meta(annt_root, vid_id):
    file_candidates = []
    for file_name in os.listdir(annt_root):
        if vid_id in file_name:
            file_candidates.append(os.path.join(annt_root, file_name))
    
    for file in file_candidates:
        meta = json.load(open(file))
        has_event = False
        event_sequence = []
        for ins_id, instance in enumerate(meta['instances']):
            
            class_name = instance['meta']['className']
            class_ID = instance['meta']['classId']
            if len(instance['parameters']) > 1:
                import ipdb;ipdb.set_trace() # breakpoint 79

            if instance['meta']['type'] == 'event':
                for annt in instance['parameters']: 
                    start_fraction = annt['start']/meta['metadata']['duration']
                    end_fraction = annt['end']/meta['metadata']['duration']
                    assert 0<=start_fraction <=1
                    assert 0<=end_fraction<=1
                    event_sequence.append((start_fraction, end_fraction, class_name.replace(' ', '_')))
                    has_event = True
            # for annts in instance['parameters'][0]['timestamps']:
                # if 'points' not in annts:
                #     # import ipdb;ipdb.set_trace() # breakpoint 97
                #     print(f"[WARNING] missing annotation for {instance['meta']['type']}, {instance['meta']['className']}")
                
                # timestamp = annts['timestamp']
                # if instance['meta']['type'] == 'event':
                #     import ipdb;ipdb.set_trace() # breakpoint 30
                #     has_event = True
                #     event_sequence.append((timestamp, class_name))

        if has_event:
            event_intervals = [[triple[0],triple[1]] for triple in event_sequence]
            event_labels = [triple[2] for triple in event_sequence]
            event_sequence = {"intervals": event_intervals, "labels": event_labels}
            return event_sequence

def cloest_label(time_interval, candidates):
    label_name = None

    event_intervals = candidates["intervals"]
    event_labels = candidates["labels"]
    dists = np.linalg.norm(np.array(event_intervals) - np.array(time_interval), axis=1)
    selected_index = np.argmin(dists)

    label_name = event_labels[selected_index]
    return label_name

