import torch


def torch_gc():
    #
    if torch.cuda.is_available():
        #
        # with torch.cuda.device(DEVICE):
        #
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    elif torch.backends.mps.is_available():

        try:

            from torch.mps import empty_cache

            empty_cache()

        except Exception as e:

            print(e)

            print("如果您使用的是MACOS建议将PYTORCH版本升级至2.0.0或更高版本，以支持及时清理TORCH产生的内存占用。")
