def master():
    
    from elmo_ann_predictor import pred_elm
    from netx_conv_autoenc_predictor import pred_net
    
    final1, about1, treatment1 = pred_elm(case)
    final2, about2, treatment2 = pred_net(case)
    
    secondary = False if final1 == final2 else True
    
    return (secondary, final1, about1, treatment1, final2, about2, treatment2)