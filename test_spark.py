# Importer les bibliothèques nécessaires
from pyspark.sql import SparkSession

def main():
    # 1. Créer une SparkSession
    # C'est le point d'entrée pour programmer avec Spark.
    spark = SparkSession.builder \
        .appName("TestSparkApp") \
        .master("local[*]") \
        .getOrCreate()

    print("\n✅ SparkSession créée avec succès !")
    print(f"Version de Spark : {spark.version}\n")

    # 2. Créer un petit jeu de données (DataFrame)
    data = [("Alice", 34),
            ("Bob", 45),
            ("Catherine", 29)]
    columns = ["Nom", "Age"]
    df = spark.createDataFrame(data, columns)

    # 3. Afficher le contenu du DataFrame
    print("Voici un petit DataFrame Spark :")
    df.show()

    # 4. Arrêter la session Spark pour libérer les ressources
    spark.stop()
    print("\n✅ SparkSession arrêtée proprement.")


if __name__ == "__main__":
    main()
