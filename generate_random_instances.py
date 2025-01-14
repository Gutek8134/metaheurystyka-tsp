import source.instance_generator

if __name__ == "__main__":
    for i in range(15):
        with open(f"random{i*15+5}.txt", mode="w") as f:
            f.write(source.instance_generator.random_instance(i*15+5))
