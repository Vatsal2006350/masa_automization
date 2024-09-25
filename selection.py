def select_droplet(infocus, dia, circ, circ_inf, circ_up):
    """
    Select droplets based on their focus level, circularity, and size
    
    Args:
    infocus: Focus parameter
    dia: Droplet diameter (µm)
    circ: Circularity
    circ_inf, circ_up: Lower and upper circularity thresholds
    
    Returns:
    select: 1 if droplet is selected, 0 otherwise
    """
    thresh = 0.23
    if infocus > thresh and circ_inf <= circ <= circ_up and dia >= 30:
        return 1
    else:
        return 0