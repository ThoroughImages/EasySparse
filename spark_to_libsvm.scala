/*
 * Program for:
 *
 * 1. Data Preprocessing
 *    one-hot encoding for discrete features
 * 2. File Format Transformation
 *    converting table in HDFS to 
 *    LibSVM format 
 */
import org.apache.spark.rdd.RDD

/**
 * Converting String to Double,
 * otherwise return 0.0.
 */
def parseDouble(s: String) = try { s.toDouble } catch { case _ => 0.0 }

/**
 * Load RDD from HDFS and split each row
 * using sep as the delimiter string.
 * 
 * @param path the path of source file to be convert in HDFS
 * @param sep the delimiter string using "\001" as default
 *
 * @return rdd of type RDD[Array[String]]
 */
def loadRddFromHdfs(path:String, sep:String="\001"): RDD[Array[String]] = {
    val rdd = sc.textFile(path)
    rdd.map(_.split(sep))
}

/**
 * Converting a row of data to LibSVM format
 * 
 */
def rowToLibSVM(row:Seq[Double]):String = {
    val label = row.head
    val featureIndex = row.tail.zipWithIndex
    val features = featureIndex.foldLeft(Seq.empty[String])((acc, e) => 
        if(e._1 != 0) acc :+ s"${e._2 + 1}:${e._1}" else acc 
    )
    s"$label ${features mkString " "}"
}

/**
 * Converting rdd to LibSVM format 
 *
 * @param rdd the target storage level
 * @param columns the sequence of column name
 * @param columnTypeConfig configuration about the type of columns including lable, continuous 
 *        features, discrete features, and columns to be omitted
 * @param oneHotConfig name of columns and the corresponding sequence of enumerations
 *
 * @return rdd rows of LibSVM format
 */

def rddToLibSVM(rdd:RDD[Array[String]], columns:Seq[String], 
    columnTypeConfig:Map[String,Seq[String]], oneHotConfig:Map[String,Seq[String]]):RDD[String] = {
    val columnIndexMap = columns.zipWithIndex.toMap
    val oneHotMap:Map[Int, Map[String, Array[Double]]] = oneHotConfig.map{case (colName, enumSeq) =>  {
        val index = columnIndexMap(colName)
        val map = enumSeq.zipWithIndex.map(e => (e._1, Array.fill[Double](enumSeq.length)(0.0).updated(e._2, 1.0))).toMap
        (index -> map)
    }}
    val columnTypeIndex = columnTypeConfig.map(e => {
        (e._1, e._2.map(c => columnIndexMap(c)))
    })
    rdd.map(row => {
        val rowMap = row.zipWithIndex.map(e => (e._2, e._1)).toMap
        val label = columnTypeIndex("label").map(e => parseDouble(rowMap(e)))
        val continuous = columnTypeIndex("continuousColumns").map(e => parseDouble(rowMap(e)))
        val discreteIndex = columnTypeIndex("discreteColumns").map(e => (e, rowMap(e))) // (index, value) 
        val discrete = discreteIndex.map(e => oneHotMap(e._1)(e._2)).flatten // (enumIndex, enumValue)
        rowToLibSVM(label ++ discrete ++ continuous)
    })
}


/**
 * Simple Demo
 */

val columns = Seq("key", "col_1", "col_2", "col_3", "col_4", "col_5", "y")

// Specify the type of columns
val columnTypeConfig = Map(
    ("label", Seq("y")),
    ("continuousColumns", Seq("col_1", "col_2")),
    ("discreteColumns", Seq("col_3", "col_4")),
    ("omittedColumns", Seq("key"))
)

// Specify column of discrete feature and the corresponding enumerations
val oneHotConfig = Map (
    "col_3" -> Seq("col_3_a", "col_3_b", "col_3_c"),
    "col_4" -> Seq("col_4_a", "col_4_b"),
    "col_5" -> Seq("col_5_a", "col_5_b", "col_5_c", "col_5_d")
)

val filepath = "hdfs:///SOURCE_FILE_PATH"
val savepath = "hdfs:///RESULT_PATH"
val rdd = loadRddFromHdfs(filepath)
val LibSVMRDD = rddToLibSVM(rdd, columns, columnTypeConfig, oneHotConfig)
LibSVMRDD.saveAsTextFile(savepath)

