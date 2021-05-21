# Orientation Bias
Python tool for evaluating the orientation bias in FIR - submm surveys. The code is flexible, allowing for different number count distributions and redshift distributions to be specified and sampled from.

## Requirements
- `numpy >= 1.17.3`
- `scipy >= 1.3.1`
- `matplotlib >= 3.1.1`

## Example usage

You can demo the code by running `python bias.py`. This runs the following in `main`:

```
    bias = orientation_bias()

    # first define our number counts model
    model = Schechter(Dstar=1.50, alpha=-1.91, log10phistar=3.56)  # 250 mu-metre

    # then sample from this model to get `S`, an array of flux densities
    bias.sample_S(model, D_lowlim=0, inf_lim=3)

    print("S:", bias.S)

    # sample from a redshift distribution (here a gaussian, with mean z = 1.2)
    bias.sample_z(bias.trunc_gaussian, mean=2)

    print("Redshift:", bias.redshift)

    # sample from the dimming distribution for each source
    bias.sample_dimming(bias.dimming_distribution)

    print("Dimming:", bias.dimming)

    # make example plots
    fig = bias.plot_dimming_redshift()
    plt.show()

    fig = bias.plot_dimming_distribution()
    plt.show()

    fig = bias.plot_number_counts()
    plt.show()

```

## Example plots
Some example plots are shown in the `plots` directory

![number counts](https://raw.githubusercontent.com/christopherlovell/orientation_bias/main/plots/number_counts.png)

