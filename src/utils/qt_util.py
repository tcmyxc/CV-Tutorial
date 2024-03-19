import thop


def get_n_params(model):
    """Number of Parameters
    p = get_n_params(resnet18())
    print(f'#Params: {p / 1e6} M')
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flops(model, input_data, verbose=False, ret_layer_info=False, report_missing=False):
    """FLOPs
    f = get_flops(resnet18(), torch.randn(1, 3, 224, 224), verbose=False)
    print(f"FLOPs: {f / 1e9} G")
    """
    return int(thop.profile(model, inputs=(input_data,), verbose=verbose, ret_layer_info=ret_layer_info,
                            report_missing=report_missing)[0])


if __name__ == '__main__':
    pass
    # 以下为使用示例：
    # n_params = get_n_params(model)
    # print(f'#Params: {n_params / 1e6} M', end='')
    # if args.flops:
    #     if args.distributed:
    #         raise RuntimeError('FLOPs can only be computed in non-distributed mode.')
    #     _model = copy.deepcopy(model).eval()
    #     flops = get_flops(_model, dataset_val[0][0][None].to(device))
    #     print(f' | FLOPs: {flops / 1e9} G', end='')
    #     _model.to('cpu')
    #     del _model
    # print()
