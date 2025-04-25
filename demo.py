import torch
from conspacesampler import algorithms, barriers

def main():
    # Définition d'une barrière logarithmique pour un carré [-0.01, 0.01] x [1, 1]
    barrier = barriers.BoxBarrier(bounds=torch.tensor([0.01, 1]))

    # Initialisation du sampler avec 500 particules
    sampler = algorithms.misc_algorithms.HitAndRunSampler(
        barrier=barrier,
        num_samples=500
    )

    # Initialisation des particules dans la boîte [-0.001, 0.001]^2
    initial_particles = torch.rand(500, 2) * 0.002 - 0.001
    sampler.set_initial_particles(initial_particles)

    # Lancement du sampler pour 1000 itérations avec un pas de 0.05
    particles = sampler.mix(
        num_iters=1000,
        return_particles=True,
        no_progress=True
    )

    print(f"Shape des particules générées : {particles.shape}")  # (1000, 500, 2)

if __name__ == "__main__":
    main()
