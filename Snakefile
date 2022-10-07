include: "dataset/Snakefile"
configfile: "config/config.yaml"

job_list = ' '.join(config['preprocessings'])
cell_type_list = ' '.join(config['hp_cell_types'])

rule hp_ae:
    input:
        b = expand("dataset/h5/baron_sc.h5"),
        m = expand("dataset/h5/muraro_sc.h5"),
        s = expand("dataset/h5/segerstolpe_sc.h5"),
        w = expand("dataset/h5/wang_sc.h5"),
        x = expand("dataset/h5/xin_sc.h5")
    output:
        txt = "results/hp.txt"
    conda:
        "config/base.yml"
    shell:
        """
        mkdir -p results
        python ae_sc_skorch.py \
        --train-datasets {input.b} \
        --test-datasets {input.b} {input.m} {input.s} {input.w} {input.x} \
        --latent-model tsne \
        --clustering-model dbscan \
        --transformations total \
        --label-filter {cell_type_list} \
         > {output.txt}
        """



rule hp:
    input:
        b = expand("dataset/h5/baron_sc.h5"),
        m = expand("dataset/h5/muraro_sc.h5"),
        s = expand("dataset/h5/segerstolpe_sc.h5"),
        w = expand("dataset/h5/wang_sc.h5"),
        x = expand("dataset/h5/xin_sc.h5")
    output:
        txt = "results/hp.txt"
    conda:
        "config/base.yml"
    shell:
        """
        mkdir -p results
        python preprocessing_for_hp.py --datasets {input.b} {input.m} {input.s} {input.w} {input.x} --latent-model tsne --clustering-model dbscan --transformations {job_list} --label-filter {cell_type_list}> {output.txt}
        """

rule baron_run:
    input:
        expand("dataset/h5/baron_sc.h5")
    output:
        txt = "results/baron.txt"
    conda:
        "config/base.yml"
    shell:
        """
        mkdir -p results
        python preprocessing_test.py --datasets {input} --latent-model tsne --clustering-model dbscan --transformations {job_list} > {output.txt}
        """

rule muraro_run:
    input:
        expand("dataset/h5/muraro_sc.h5")
    output:
        txt = "results/muraro.txt"
    conda:
        "config/base.yml"
    shell:
        """
        mkdir -p results
        python preprocessing_test.py --datasets {input} --latent-model tsne --clustering-model dbscan --transformations {job_list} > {output.txt}
        """

rule segerstolpe_run:
    input:
        expand("dataset/h5/segerstolpe_sc.h5")
    output:
        txt = "results/segerstolpe.txt"
    conda:
        "config/base.yml"
    shell:
        """
        mkdir -p results
        python preprocessing_test.py --datasets {input} --latent-model tsne --clustering-model dbscan --transformations {job_list} > {output.txt}
        """

rule wang_run:
    input:
        expand("dataset/h5/wang_sc.h5")
    output:
        txt = "results/wang.txt"
    conda:
        "config/base.yml"
    shell:
        """
        mkdir -p results
        python preprocessing_test.py --datasets {input} --latent-model tsne --clustering-model dbscan --transformations {job_list} > {output.txt}
        """

rule xin_run:
    input:
        expand("dataset/h5/xin_sc.h5")
    output:
        txt = "results/xin.txt"
    conda:
        "config/base.yml"
    shell:
        """
        mkdir -p results
        python preprocessing_test.py --datasets {input} --latent-model tsne --clustering-model dbscan --transformations {job_list} > {output.txt}
        """

