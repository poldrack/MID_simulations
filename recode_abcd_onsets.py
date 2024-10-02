# rename ABCC files to remove nda codes
# also sanitize dates by setting to first of month

from pathlib import Path
import uuid
import json
import os
import pandas as pd


def newhash(existing_hashes, hashlen=8):
    random_hash = str(uuid.uuid4())[:hashlen]
    # make sure the hash is unique
    while random_hash in existing_hashes:
        random_hash = str(uuid.uuid4())[:hashlen]
    return random_hash


def sanitize_date(date):
    # set to first of month, assuming 'month-day-year' format
    date = date.split('-')
    return '-'.join([date[0], '01',  date[1]])

if __name__ == '__main__':
    origdir = '/Users/poldrack/data_unsynced/ABCD/ABCC/events_for_russ/mid_ses-2YearFollowUpYArm1'
    outdir = '/Users/poldrack/data_unsynced/ABCD/ABCC_deidentified'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open('abcd_site_dict.json', 'r') as f:
        code_dict = json.load(f)

    for f in Path(origdir).glob('*.tsv'):
        subcode = f.stem.split('_')[0].split('-')[1].replace('NDAR', '')
        
        df = pd.read_csv(f, sep='\t')
 
        if subcode not in code_dict['sub_site'] or subcode not in code_dict['subs']:
            print(f'no site info for {subcode}')
            continue
        
        df['site'] = code_dict['sub_site'][subcode]
        sub_hash = code_dict['subs'][subcode]
        
        outfile = os.path.join(outdir, f.name.replace(subcode, sub_hash))
        df['Subject'] = sub_hash

        df.to_csv(outfile, index=False)
    
    with open('abcd_code_dict.json', 'w') as f:
        json.dump(code_dict, f)
    
