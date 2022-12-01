import csv
from rdkit import Chem
from itertools import product
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import AllChem
import numpy as np
from rdkit import DataStructs


TARGET_DICT = {'BBBP': ["p_np"],

               'Tox21': ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"],

               'ClinTox': ['CT_TOX', 'FDA_APPROVED'],

               'HIV': ["HIV_active"],

               'BACE': ["Class"],

               'SIDER': ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
                         "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
                         "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
                         "Reproductive system and breast disorders",
                         "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                         "General disorders and administration site conditions", "Endocrine disorders",
                         "Surgical and medical procedures", "Vascular disorders",
                         "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
                         "Congenital, familial and genetic disorders", "Infections and infestations",
                         "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
                         "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
                         "Ear and labyrinth disorders", "Cardiac disorders",
                         "Nervous system disorders", "Injury, poisoning and procedural complications"],

               'MUV': ['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859',
                       'MUV-644', 'MUV-548', 'MUV-852', 'MUV-600',
                       'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858',
                       'MUV-713', 'MUV-733','MUV-652', 'MUV-466', 'MUV-832'],
               'FreeSolv': ["expt"],

               'ESOL': ["measured log solubility in mols per litre"],
               'Lipo': ["exp"],

               'qm7': ["u0_atom"],

               'qm8': ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0",
                       "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"],

               'qm9': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']}

def comb_product(comb):
    return (dict(zip(comb.keys(), values)) for values in product(*comb.values()))

def read_moleculenet_smiles(data_path: str, target: str, task: str):
    smiles_data, labels, garbages, fingerprints = [], [], [], []
    data_name = data_path.split('/')[-1].split('.')[0].upper()
    with open(data_path) as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)

            if mol != None and label != '':
                smiles_data.append(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(fp)
                if task == 'classification':
                    labels.append(int(label))
                elif task == 'regression':
                    labels.append(float(label))
                else:
                    raise ValueError('Task Error')

            elif mol is None:
                print(idx)
                garbages.append(smiles)

    print(f'{data_name} | Target : {target}({task})| Total {len(smiles_data)}/{idx+1} instances')
    return smiles_data, labels, garbages, fingerprints

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(f'Total Data Size {data_len}, about to generate scaffolds')
    # for idx, smiles in enumerate(dataset.smiles_data):
    for idx, smiles in enumerate(dataset):
        if idx % log_every_n == 0:
            print("Generating scaffold %d/%d"%(idx, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)

    scaffolds = {k: sorted(v) for k, v in scaffolds.items()}
    scaffold_sets = [s_set for (s, s_set) in sorted(scaffolds.items(),
                                                    key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    return scaffold_sets


def scaffold_split(dataset, val_size=0.1, test_size=0.1):
    train_size = 1.0 - val_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    val_cutoff = (train_size + val_size) * len(dataset)

    train_indices, val_indices, test_indices = [], [], []

    print("About to Sort in Scaffold Sets")
    for s_set in scaffold_sets:
        if len(train_indices) + len(s_set) > train_cutoff:
            if len(train_indices) + len(val_indices) + len(s_set) > val_cutoff:
                test_indices += s_set
            else:
                val_indices += s_set
        else:
            train_indices += s_set
    return train_indices, val_indices, test_indices


def fps_to_numpy(fingerprints):
    np_fps = []
    for fp in fingerprints:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)

    np_fps = np.array(np_fps)
    return np_fps

def get_model_config(model, postfix, seed, learning_rate, max_depth, l2, **kwargs):
    config = {'learning_rate': learning_rate,
              'max_depth': max_depth,
              'random_state': seed,
              'n_estimators': 5000}
    config.update(kwargs)
    
    DEFAULTS = {'XGBClassifier':{'eval_metric': ['auc'], 'gpu_id': 0, 'objective':'binary:logistic', 'reg_lambda': l2},
                'XGBRegressor':{'eval_metric': ['rmse'], 'gpu_id': 0, 'objective':'reg:squarederror', 'reg_lambda': l2},
                'LGBMClassifier':{'metric': ['auc'], 'device_type': 'gpu', 'objective':'binary', 'reg_lambda': l2},
                'LGBMRegressor':{'metric': ['mean_squared_error'], 'device_type': 'gpu', 'objective':'regression', 'reg_lambda': l2},
                'CatBoostClassifier':{'eval_metric': 'AUC', 'task_type': 'GPU', 'devices': '0', 'l2_leaf_reg': l2},
                'CatBoostRegressor':{'eval_metric': 'RMSE', 'task_type': 'GPU', 'devices': '0', 'l2_leaf_reg': l2},}
    
    config.update(DEFAULTS[model+postfix])
    return config