from typing import List

from mlops.datasets.structs.datasets import DetDataset


def concat_dataset(
    datasets: List[DetDataset],
) -> "DetDataset":
    cat_name_id_dict = datasets[0].cat_name_id_dict
    cat_id_name_dict = datasets[0].cat_id_name_dict
    shape_format = datasets[0].shape_format

    new_img_metas = [datasets[0].img_metas]
    new_insts_list = [datasets[0].insts_list]

    for dataset in datasets[1:]:
        assert len(cat_id_name_dict) == len(dataset.cat_id_name_dict)
        assert shape_format == dataset.shape_format

        new_img_metas.append(dataset.img_metas)
        new_insts_list.append(dataset.insts_list)
    
    new_dataset = DetDataset(
        new_img_metas, new_insts_list, cat_name_id_dict,
        cat_id_name_dict, shape_format
    )

    return new_dataset

# def concat_dataset_new_cats(
#     datasets: List[DetDataset],
#     new_cat_name_id_dict: Dict[str, int]
# ) -> DetDataset:
#     new_cat_id_name_dict = {
#         v:k for k, v in new_cat_name_id_dict.items()
#     }

#     new_img_metas = []
#     new_insts_list = []

#     for dataset in datasets:
#         cat_id_old_new_dict = {}
#         for k, v in dataset.cat_id_name_dict.items():
#             old_cat_id = k
#             new_cat_id = new_cat_name_id_dict[v]
#             cat_id_old_new_dict[old_cat_id] = new_cat_id
        
#         for insts in dataset.insts:
