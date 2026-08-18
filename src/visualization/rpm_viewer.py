import numpy as np
import matplotlib.pyplot as plt


def rpm_oracle(
    problem_images, 
    problem_descriptors, 
    candidate_images, 
    candidate_descriptors, 
    correct_answer_idx,
    rules_group1_txt, 
    rules_group2_txt=None, 
    figsize=(10,10)):
    """
    Draw the Raven Problem Matrix (RPM) viewer.

    Parameters
    ----------
    rules_group1_txt : list[(str,str)]
        The first group of rules (the first component rules).
    rules_group2_txt : list[(str,str)] optional
        The second group of rules (the second component rules, for structures: 'Out_In', 'Left_Right' , 'Up_Down' ).
    problem_images : list[np.ndarray]
        The problem images (8 images in total).
    problem_descriptors : list[np.ndarray]
        The problem descriptors (8 descriptors in total).
    candidate_images : list[np.ndarray]
        The candidate images (8 images in total).
    candidate_descriptors : list[np.ndarray]
        The candidate descriptors.
    correct_answer_idx : int
        The index of the correct answer.
    figsize : tuple, optional
        The size of the figure, by default (10,10).

    Returns
    -------
    None
    """

    # ==========
    # Validation
    # ==========

    if len(problem_images) != 8:
        raise ValueError("The problem images must be 8 images in total.")

    if len(problem_descriptors) != 8:
        raise ValueError("The problem descriptors must be 8 descriptors in total.")

    if len(candidate_images) != 8:
        raise ValueError("The candidate images must be 8 images in total.")

    if len(candidate_descriptors) != 8:
        raise ValueError("The candidate descriptors must be 8 descriptors in total.")

    if not 0 <= correct_answer_idx <= 7:
        raise ValueError("The correct answer index must be between 0 and 7.")

    # ===============
    # Draw RPM Figure
    # ===============

    main_figure = plt.figure(figsize=figsize)
    main_figure.suptitle('RAVEN RPM : Rleational and Analogical Visual rEasoNing Progressive Matrices', x=0.98,y=0.85, rotation=270)
    # Removing the vertical height space between the three subfigures : RULES, PROBLEM, CANDIDATES
    main_figure.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.1)

    # Creating 3 Subfigures: RULES, PROBLEM, CANDIDATES vertically aligned
    # Height ratios : one problem_subfigure = 6x rule_subfigure and one candidates_sub_figure = 4x rule_subfigure;   
    main_sub_figures = main_figure.subfigures(3,1 , height_ratios=[2,4,4])

    #===================== 
    # 1. 'RULES' Subfigure 
    #=====================
    main_sub_figures[0].suptitle('RPM RULES', size='xx-large', weight='1000')
    main_sub_figures[0].set_facecolor(('navy', 0.1))

    # Cretaing 2 rules_subfigures : GROUP 1 and GROUP 2 horizontally aligned
    rules_sub_figures = main_sub_figures[0].subfigures(1,2) 

    #==============================
    # 1.1 'GROUP 1' : Sub_SubFigure
    #==============================
    rules_sub_figures[0].suptitle('GROUP 1')
    axs_rules_group1 = rules_sub_figures[0].subplots(2,2)
    #==============================
    # 1.2 'GROUP 1' : Sub_SubFigure
    #==============================
    rules_sub_figures[1].suptitle('GROUP 2')
    axs_rules_group2 = rules_sub_figures[1].subplots(2,2)

    # remove rules axes tickers 
    for ax in np.concatenate((axs_rules_group1.flatten(),axs_rules_group2.flatten()), axis=0):
        ax.set_xticks([])
        ax.set_yticks([])

    #========================= 
    # 2. 'PROBLEM' Subfigure 
    #=========================
    main_sub_figures[1].suptitle('RPM PROBLEM', size='xx-large', weight='1000')
    main_sub_figures[1].set_facecolor(('navy', 0.2))

    # Cretaing 2 problem_subfigures : IMAGES and DESCRIPTORS horizontally aligned
    problem_sub_figures = main_sub_figures[1].subfigures(1,2) 

    #=============================
    # 2.1 'IMAGES' : Sub_SubFigure
    #=============================
    problem_sub_figures[0].suptitle('IMAGES')
    axs_problems_images = problem_sub_figures[0].subplots(3,3)
    #==================================
    # 2.2 'DESCRIPTORS' : Sub_SubFigure
    #==================================
    problem_sub_figures[1].suptitle('DESCRIPTORS')
    axs_problems_descriptors = problem_sub_figures[1].subplots(3,3)

    # remove problems axes tickers 
    for ax in np.concatenate((axs_problems_images.flatten(),axs_problems_descriptors.flatten()), axis=0):
        ax.set_xticks([])
        ax.set_yticks([])


    #========================== 
    # 3. 'CANDIDATES' Subfigure 
    #==========================
    main_sub_figures[2].suptitle('RPM CANDIDATES', size='xx-large', weight='1000')
    main_sub_figures[2].set_facecolor(('navy', 0.3))

    # Creating 2 candidates_subfigure : IMAGES and DESCRIPTORS horizontally aligned
    candidates_sub_figures = main_sub_figures[2].subfigures(1,2) 

    #=============================
    # 3.1 'IMAGES' : Sub_SubFigure
    #=============================
    candidates_sub_figures[0].suptitle('IMAGES')
    axs_candidates_images = candidates_sub_figures[0].subplots(3,3)
    candidates_sub_figures[0].delaxes(axs_candidates_images[2,2])
    #==================================
    # 3.2 'DESCRIPTORS' : Sub_SubFigure
    #==================================
    candidates_sub_figures[1].suptitle('DESCRIPTORS')
    axs_candidates_descriptors = candidates_sub_figures[1].subplots(3,3)
    candidates_sub_figures[1].delaxes(axs_candidates_descriptors[2,2])

    # remove problems axes tickers 
    for ax in np.concatenate((axs_candidates_images.flatten(),axs_candidates_descriptors.flatten()), axis=0):
        ax.set_xticks([])
        ax.set_yticks([])


    #========
    # Display
    #========

    display_rules(rules_group1_txt, rules_group2_txt, axs_rules_group1, axs_rules_group2)
    display_problem_images(problem_images,axs_problems_images, correct_answer_idx)
    display_problem_descriptors(problem_descriptors,axs_problems_descriptors, correct_answer_idx)
    display_candidate_images(candidate_images,axs_candidates_images, correct_answer_idx)
    display_candidate_descriptors(candidate_descriptors,axs_candidates_descriptors, correct_answer_idx)

    plt.show()
    
def display_rules(rules_group1_txt, rules_group2_txt, axs_rules_group1, axs_rules_group2):

    """
    Display rules in their axes.
    """

    for i, (rule_type, rule_attribute) in enumerate(rules_group1_txt):
        axs_rules_group1.flat[i].text(0.1,0.35,"• Type: {}\n• Attrib: {}".format(rule_type, rule_attribute), {"color":"b"}, fontsize=9)
        axs_rules_group1.flat[i].spines[:].set_color('orange')

    if rules_group2_txt is not None:
        for i, (rule_type, rule_attribute) in enumerate(rules_group2_txt):
            axs_rules_group2.flat[i].text(0.1,0.35,"• Type: {}\n• Attrib: {}".format(rule_type, rule_attribute), {"color":"b"}, fontsize=9)
            axs_rules_group2.flat[i].spines[:].set_color('darkred')

def display_problem_images(problem_images,axs_problems_images, correct_answer_idx):
    
    """
    Display problem images in their axes.
    """
    matrix_problem_images = problem_images[:-1].reshape(3,3,28,28)
    