def save_thetas():
    f = open("data/thetas.csv", "w+")
    f.write("%f, %f" % (2, 4))
    f.close()


save_thetas()
