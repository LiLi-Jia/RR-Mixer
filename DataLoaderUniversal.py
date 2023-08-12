from DataLoader import *

def get_data_loader(opt):
    dataset = opt.dataset
    text, audio, video=opt.text, opt.audio, opt.video
    normalize = opt.normalize
    persistent_workers=opt.persistent_workers
    batch_size, num_workers, pin_memory, drop_last =opt.batch_size, opt.num_workers, opt.pin_memory, opt.drop_last

    assert dataset in ['absa']
    if 'absa' in dataset:
        dataset_train =AbsaDataset(mode='train', dataset='absa', text=text, audio=audio, video=video, normalize=normalize, )
        dataset_valid = AbsaDataset(mode='valid', dataset='absa', text=text, audio=audio, video=video, normalize=normalize, )
        dataset_test = AbsaDataset(mode='test', dataset='absa', text=text, audio=audio, video=video,normalize=normalize, )
        data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_absa, shuffle=True,
                                       persistent_workers=persistent_workers, num_workers=num_workers,
                                       pin_memory=pin_memory, drop_last=drop_last)
        data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_absa, shuffle=False,
                                       persistent_workers=persistent_workers, num_workers=num_workers,
                                       pin_memory=pin_memory, drop_last=False)
        data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_absa, shuffle=False,
                                      persistent_workers=persistent_workers, num_workers=num_workers,
                                      pin_memory=pin_memory, drop_last=False)
        return data_loader_train, data_loader_valid,data_loader_test

    raise NotImplementedError

