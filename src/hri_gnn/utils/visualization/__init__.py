import numpy as np
import matplotlib.pyplot as plt

def canvas(figsize=(25,20)):
    """
    Prepapre and draw the Raven Problem Matrix (RPM) figures placeholders

    Args:
    ------
        - figsize (tuple, optional): The size of the figure, by default (10,10).

    Returns:
    -------
    - None
    """
    # ===============
    # Draw RPM Figure
    # ===============
    #plt.ion() 
    main_figure = plt.figure(figsize=figsize)
    main_figure.suptitle('RPM : Rleational and Analogical Visual rEasoNing Progressive Matrices', x=0.98,y=0.85, rotation=270)
    # Removing the vertical height space between the three subfigures : RULES, CONTEXT, CHOICES
    main_figure.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.1)

    # Creating 3 Subfigures: RULES, CONTEXT, CHOICES vertically aligned
    # Height ratios : one 2 problem_subfigure = 6 rule_subfigure and one candidates_sub_figure and  6 rule_subfigure;   
    main_sub_figures = main_figure.subfigures(3,1 , height_ratios=[2,6,6])

    #===================== 
    # 1. 'RULES' Subfigure 
    #=====================
    main_sub_figures[0].suptitle('RPM RULES', size='xx-large', weight='1000')
    main_sub_figures[0].set_facecolor(('navy', 0.1))

    # Cretaing 2 Rules SubFigures : COMPONENT 1 and COMPONENT 2 horizontally aligned
    rls_sub_figures = main_sub_figures[0].subfigures(1,2) 

    #==============================
    # 1.1 'COMPONENT 1' : Sub_SubFigure
    #==============================
    rls_sub_figures[0].suptitle('COMPONENT 1')
    axs_rls_first_cmp = rls_sub_figures[0].subplots(2,2)
    #==============================
    # 1.2 'COMPONENT 1' : Sub_SubFigure
    #==============================
    rls_sub_figures[1].suptitle('COMPONENT 2')
    axs_rls_second_cmp = rls_sub_figures[1].subplots(2,2)

    # remove rules axes tickers 
    for ax in np.concatenate((axs_rls_first_cmp.flatten(),axs_rls_second_cmp.flatten()), axis=0):
        ax.set_xticks([])
        ax.set_yticks([])

    #========================= 
    # 2. 'CONTEXT' SubFigure 
    #=========================
    main_sub_figures[1].suptitle('RPM CONTEXT', size='xx-large', weight='1000')
    main_sub_figures[1].set_facecolor(('navy', 0.2))

    # Cretaing 2 Context SubFigures : IMAGES and DESCRIPTORS horizontally aligned
    ctx_sub_figures = main_sub_figures[1].subfigures(1,2) 

    #=============================
    # 2.1 'IMAGES' : Sub_SubFigure
    #=============================
    ctx_sub_figures[0].suptitle('IMAGES')
    axs_ctx_images = ctx_sub_figures[0].subplots(3,3)
    #==================================
    # 2.2 'DESCRIPTORS' : Sub_SubFigure
    #==================================
    ctx_sub_figures[1].suptitle('DESCRIPTORS')
    axs_ctx_descriptors = ctx_sub_figures[1].subplots(3,3)

    # remove problems axes tickers 
    for ax in np.concatenate((axs_ctx_images.flatten(),axs_ctx_descriptors.flatten()), axis=0):
        ax.set_xticks([])
        ax.set_yticks([])

    #========================== 
    # 3. 'CHOICES' Subfigure 
    #==========================
    main_sub_figures[2].suptitle('RPM CHOICES', size='xx-large', weight='1000')
    main_sub_figures[2].set_facecolor(('navy', 0.3))

    # Creating 2 Choices SubFigures : IMAGES and DESCRIPTORS horizontally aligned
    chcs_sub_figures = main_sub_figures[2].subfigures(1,2) 

    #=============================
    # 3.1 'IMAGES' : Sub_SubFigure
    #=============================
    chcs_sub_figures[0].suptitle('IMAGES')
    axs_chcs_images = chcs_sub_figures[0].subplots(3,3)
    chcs_sub_figures[0].delaxes(axs_chcs_images[2,2])
    #==================================
    # 3.2 'DESCRIPTORS' : Sub_SubFigure
    #==================================
    chcs_sub_figures[1].suptitle('DESCRIPTORS')
    axs_chcs_descriptors = chcs_sub_figures[1].subplots(3,3)
    chcs_sub_figures[1].delaxes(axs_chcs_descriptors[2,2])

    # remove problems axes tickers 
    for ax in np.concatenate((axs_chcs_images.flatten(),axs_chcs_descriptors.flatten()), axis=0):
        ax.set_xticks([])
        ax.set_yticks([])

    #=======
    # Return
    #=======
    canvas = {
            'rules': [axs_rls_first_cmp, axs_rls_second_cmp] , 
            'context': [axs_ctx_images, axs_ctx_descriptors], 
            'choices': [axs_chcs_images, axs_chcs_descriptors],
            'figure': main_figure
            }
    return canvas

def panel_description(panel:dict) -> str:
    """
    Generate a textual description of a Raven's Progressive Matrices (RPM) panel.
    
    Args:
    ------
        - panel (dict): A dictionary representing a Raven's Progressive Matrices panel.
            panel = { 
                'image': ndarray, 
                'structure': {'components': list[
                    {'layout':{'entities':[ 
                        {'bbox': list[int],
                        'Type': int,
                        'Size': int,
                        'Color': int,
                        'Angle': int
                        }
                    ]
                }
            }
    
    Returns:
    -------
        - str: A textual description of the panel.
    """
    components_txt = []
    for component in panel['structure']['components']:
        entities_txt = []
        entity_txt = ""
        for entity_index,entity in enumerate(component['layout']['entities']):
            entity_position = component['layout']['Position'].index(entity['bbox'])
            entity_type     = entity['Type']
            entity_size     = entity['Size']
            entity_color    = entity['Color']
            entity_angle    = entity['Angle']
            entity_txt = "•e{}[Pos:{};Type:{};Size:{};Color:{};Angle:{}]\n".format(entity_index, entity_position,entity_type, entity_size, entity_color, entity_angle)
            entities_txt.append(entity_txt)
        layout = component['layout']
        layout_uniformity_txt = 'True' if layout['Uniformity'] else 'False'
        layout_txt = "Layout:[Number:{};Uniformity:{}]\nElements:\n{}".format(layout['Number']+1, layout_uniformity_txt, "".join(entities_txt))
        if component['id'] == 1:
            component_txt = "----------------------------------------\nCOMPONENT-{}-{}:\n{}".format(component['id'], component['name'], layout_txt)
        else:
            component_txt = "COMPONENT-{}-{}:\n{}".format(component['id'], component['name'], layout_txt)
        components_txt.append(component_txt)
    return "".join(components_txt)

def rule_description(rule:dict) -> str:
    """
    Generate a textual description of a Raven's Progressive Matrices (RPM) rule.
    
    Args:
    -----
        - rule (dict): A dictionary representing a Raven's Progressive Matrices rule.
            rule = {
                'name':..., 
                'attr':...
            }
    
    Returns:
    --------
        - str: A textual description of the rule.
    """
    return "• rule_type: {}\n• attribute: {}".format(rule['name'], rule['attr'])