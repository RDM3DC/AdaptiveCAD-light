from pathlib import Path
from adaptive_shapes.adaptive_pi_mobius import adaptive_pi_mobius, save_obj


def main():
    out_dir = Path("mobius_gallery")
    out_dir.mkdir(exist_ok=True)

    taus = [0.0, 0.25, 0.5, 0.75, 1.0]
    kappas = [0.0, 0.1, 0.2]

    count = 0
    for tau in taus:
        for kappa in kappas:
            V, F = adaptive_pi_mobius(tau=tau, kappa=kappa)
            path = out_dir / f"mobius_tau{tau:.2f}_k{kappa:.2f}.obj"
            save_obj(path, V, F)
            print("Saved", path)
            count += 1
    print(f"Generated {count} models in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
