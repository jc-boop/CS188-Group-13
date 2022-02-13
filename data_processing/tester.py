from processors import *
from com2sense_data import *

if __name__ == "__main__":

    # Test loading data.
    proc = Com2SenseDataProcessor(data_dir="datasets/com2sense")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    # print()
    # for i in range(3):
    #     # print(test_examples[i])
    #     print(test_examples[i].text)
    # print()

    class com2sense_args(object):
        def __init__(self):
            self.model_type = "bert"
            self.cls_ignore_index = -100
            self.do_train = False # Not working with True for some reason
            self.max_seq_length = 32

    args = com2sense_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    processor = Com2SenseDataProcessor(data_dir="datasets/com2sense", args=args)
    examples = processor.get_test_examples()

    dataset = Com2SenseDataset(examples, tokenizer,
                           max_seq_length=args.max_seq_length,
                           args=args)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)
    epoch_iterator = tqdm(dataloader, desc="Iteration")

    for step, batch in enumerate(epoch_iterator):
        for each in batch:
            print(each.size())
        break
