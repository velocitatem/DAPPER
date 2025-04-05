from classification.data.minio_handler import MinioManager
from classification.data.loader import get_full_dataset
from classification.data.augmentor import Augmentor


def get_class_distribution(dataset):
    return dataset['label'].value_counts()



if __name__ == "__main__":
    minio_manager = MinioManager()
    augmentor = Augmentor()
    dataset = get_full_dataset(["rvl_cdip", "kaggle_invoices", "hf_invoices"], minio_manager, augmentor)
    print(get_class_distribution(dataset))
    # any repeated images?
    print(dataset.duplicated(subset=['image']).sum())
    # any repeated labels?

