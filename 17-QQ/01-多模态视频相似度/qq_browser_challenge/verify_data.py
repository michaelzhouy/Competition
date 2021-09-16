import hashlib
import os

CHECKSUM_MAP = {
    'chinese_L-12_H-768_A-12/bert_config.json': '677977a2f51e09f740b911866423eaa5',
    'chinese_L-12_H-768_A-12/bert_google.bin': '60e069975359b3b723384002952c0ca9',
    'chinese_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001': 'bd144ebf7bb0d32adf7ada7c13c21f7c',
    'chinese_L-12_H-768_A-12/bert_model.ckpt.index': '3c7ec74cb0430ccf80c6cb123732615e',
    'chinese_L-12_H-768_A-12/bert_model.ckpt.meta': '1cb8808035884f7ce416fc9eb59cd40e',
    'chinese_L-12_H-768_A-12/config.json': '677977a2f51e09f740b911866423eaa5',
    'chinese_L-12_H-768_A-12/tf_model.h5': 'fadde9152b33fbc4cf10471aaf8d0cb3',
    'chinese_L-12_H-768_A-12/vocab.txt': '3b5b76c4aef48ecf8cb3abaafe960f09',
    'pairwise/label.tsv': '579cf79909a3585bda505d58412d1d79',
    'pairwise/pairwise.tfrecords': 'f5039674534cc4dcf77f1a2e69aa0891',
    'pointwise/pretrain_0.tfrecords': '958e6906a110423f10757c5292d14e1a',
    'pointwise/pretrain_10.tfrecords': 'c142bd060f25e6d07776a5ff94d24b67',
    'pointwise/pretrain_11.tfrecords': '95881d71db5461bb49f7966a4352d340',
    'pointwise/pretrain_12.tfrecords': '813246ca69f5d1cbcb3e67ea07874960',
    'pointwise/pretrain_13.tfrecords': '14f81c2923ef6dfcb3e4f21cc4da2782',
    'pointwise/pretrain_14.tfrecords': '535becadb97e1e3e9fd1789333df8668',
    'pointwise/pretrain_15.tfrecords': 'b96b74689b4a68fe9c34ceba5640a4aa',
    'pointwise/pretrain_16.tfrecords': '9e19cf5fb3999d545a10a22311f6ba4c',
    'pointwise/pretrain_17.tfrecords': '6507d51e474f71fdb45cc3815df5eb92',
    'pointwise/pretrain_18.tfrecords': 'bae1a635aebc6f32078018cebc43979c',
    'pointwise/pretrain_19.tfrecords': '8f5b11598c0c567e252dd1aabd257c80',
    'pointwise/pretrain_1.tfrecords': '23c279c1b025daae07c1da8c657975eb',
    'pointwise/pretrain_2.tfrecords': 'a851e88bacd9b88ea3d8aabfac3d0ad0',
    'pointwise/pretrain_3.tfrecords': 'e1de45d9a493acb8e6664aa9735bdcba',
    'pointwise/pretrain_4.tfrecords': 'f4a23420bcea7c018c504bc9f9c46f32',
    'pointwise/pretrain_5.tfrecords': '86c7b5f69880531e67c9d378641e68e2',
    'pointwise/pretrain_6.tfrecords': 'a79775f9662da5899177103848a13d78',
    'pointwise/pretrain_7.tfrecords': '931427ba0cf43af4cc5cd64b9ad2ec42',
    'pointwise/pretrain_8.tfrecords': '28adfa832f4dc47a4e6ceda8894a806f',
    'pointwise/pretrain_9.tfrecords': '7a0d3652f78de5058a3ca5a1a11c4184',
    'test_a/test_a.tfrecords': '2deefbb4419c903f61e28fce46602d8a',
}


def get_checksum(file_path):
    with open(file_path, 'rb') as f:
        md5 = hashlib.md5()
        md5.update(f.read())
        checksum = md5.hexdigest()
    return checksum


def verify_data(data_dir):
    """
    to verify the integrity of the whole downloaded data directory
    :param data_dir: the downloaded data directory
    :return: files missing or failed in the verification
    """
    missing = []
    failed = []
    for file, checksum in CHECKSUM_MAP.items():
        file_path = os.path.join(data_dir, file)
        if not os.path.isfile(file_path):
            print(f'ERROR: file "{file}" is MISSING in "{data_dir}"')
            missing.append(file)
            continue
        if checksum != get_checksum(file_path):
            print(f'ERROR: checksum is inconsistent for file "{file}"')
            failed.append(file)
        else:
            print(f'checksum is consistent for file "{file}"')
    return missing, failed


if __name__ == '__main__':
    missing, failed = verify_data('pytorch/data')
    if missing or failed:
        print(f'Verification FAILED, please retry downloading: {missing + failed}')
    else:
        print('Verification Succeed!')
