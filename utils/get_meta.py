
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

            for annts in instance['parameters'][0]['timestamps']:
                if 'points' not in annts:
                    # import ipdb;ipdb.set_trace() # breakpoint 97
                    print(f"[WARNING] missing annotation for {instance['meta']['type']}, {instance['meta']['className']}")
                
                timestamp = annts['timestamp']
                if instance['meta']['type'] == 'event':
                    has_event = True
                    event_sequence.append((timestamp, class_name))


        if has_event:
            return sorted(event_sequence)