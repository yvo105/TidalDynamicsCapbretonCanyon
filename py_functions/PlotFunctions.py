# %%


def plot_overview(
    layout_dict,
    datalist,
    nrows,
    ncols,
    figsize,
    datelimlow,
    datelimhigh,
    depthlimlow,
    depthlimhigh,
):
    """function that is easily configurable and plots an overview of the chosen variables and instrument in a certain timeframe

    Args:
        layout_dict (dictionary): dictionary, keys should have format of string where each string indicates a certain subplot (e.g. key '1' indicates the first subplot). Each key should comprise of a tuple of two lists in the shape ([instrument], [variables])
        datalist (list): list of datasets to plot an overview for. For each dataset a new figure is made. 
        nrows (int): number of subplots, i.e. number of rows in figure
        ncols (int): number of columns, in this case 1
        figsize (tuple): tuple of two integers. Specifies the size of the figure
        datelimlow (datetime): specifies the lower datetime limit from which to plot
        datelimhigh (datetime): specifies the upper datetime limit until to plot
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    ## iterate through all datasets
    for data in datalist:
        
        ## general figure layout
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        fig.suptitle(data["general"]["name"], fontsize=24)
        fig.tight_layout(pad=2.5)

        if isinstance(ax, np.ndarray):  # in case of multiple subplots
            for row in range(nrows):
                
                ## extract instrument and variable list for subplot from dict
                instlist_temp = layout_dict[str(row+1)][0]
                varlist_temp = layout_dict[str(row+1)][1]
                
                ## iterate through instrument list
                for instrument in instlist_temp:
                    ## iterate through variable list 
                    for variable in varlist_temp:
                        if (
                            variable
                            in data[instrument]  # plot only available variables
                        ):
                            if isinstance(variable, np.ndarray):  # case of 2D array
                                ## plotting commands
                                ax[row].plot(
                                    data[instrument][variable][0].iloc[depthlimlow:depthlimhigh].loc[
                                        datelimlow:datelimhigh
                                    ],
                                    label=variable,
                                )
                                ## plot layout
                                ax[row].set(
                                    xlim=(datelimlow, datelimhigh),
                                    ylabel=f"{variable} [{data[instrument][variable][1]}]",
                                )
                            else:
                                ## plotting commands
                                ax[row].plot(
                                    data[instrument][variable][0].loc[
                                        datelimlow:datelimhigh
                                    ],
                                    label=variable,
                                )

                                ## plot layout
                                ax[row].set(
                                    xlim=(datelimlow, datelimhigh),
                                    ylabel=f"{variable} [{data[instrument][variable][1]}]",
                                )
                                ax[row].legend()

        else: # case of only one plot
            
            ## extract instrument and variable list for subplot from dict
            instlist_temp = layout_dict[str(nrows)][0]
            varlist_temp = layout_dict[str(nrows)][1]
            
            ## iterate through instrument list
            for instrument in instlist_temp:
                ## iterate through OBS variables
                for variable in varlist_temp:
                    if (
                        variable
                        in data[instrument]  # plot only available variables
                    ):
                        ## plotting commands
                        ax.plot(
                            data[instrument][variable][0].loc[
                                datelimlow:datelimhigh
                            ],
                            label=variable,
                        )

                        ## plot layout
                        ax.set(
                            xlim=(datelimlow, datelimhigh),
                            ylabel=f"{variable} [{data[instrument][variable][1]}]",
                        )
                        ax.legend()
