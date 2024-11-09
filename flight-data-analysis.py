from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType


# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 1
    # Hint: Calculate scheduled vs actual travel time, then find the largest discrepancies using window functions.

    # Calculate scheduled and actual travel times
    flights_with_times = flights_df.withColumn(
        "ScheduledTravelTime", F.unix_timestamp("ScheduledArrival") - F.unix_timestamp("ScheduledDeparture")
    ).withColumn(
        "ActualTravelTime", F.unix_timestamp("ActualArrival") - F.unix_timestamp("ActualDeparture")
    )
    
    # Calculate discrepancy
    flights_with_discrepancy = flights_with_times.withColumn(
        "Discrepancy", F.abs(F.col("ScheduledTravelTime") - F.col("ActualTravelTime"))
    )
    
    # Join with carriers_df to get CarrierName
    flights_with_carrier = flights_with_discrepancy.alias("f").join(
        carriers_df.alias("c"),
        F.col("f.CarrierCode") == F.col("c.CarrierCode"),
        "left"
    ).select(
        F.col("f.FlightNum"), F.col("f.CarrierCode").alias("Carrier"), F.col("f.Origin"), F.col("f.Destination"),
        F.col("f.ScheduledTravelTime"), F.col("f.ActualTravelTime"), F.col("f.Discrepancy"),
        F.col("c.CarrierName")  # Select CarrierName from carriers_df
    )
    
    # Define a window for ranking flights by discrepancy per carrier
    window = Window.partitionBy("Carrier").orderBy(F.col("Discrepancy").desc())
    
    # Add a rank column to rank by largest discrepancy
    ranked_flights = flights_with_carrier.withColumn("Rank", F.row_number().over(window))
    
    # Filter to keep only the top-ranked flight per carrier
    top_flights_per_carrier = ranked_flights.filter(F.col("Rank") == 1).select(
        "FlightNum", "CarrierName", "Origin", "Destination",
        "ScheduledTravelTime", "ActualTravelTime", "Discrepancy", "Carrier"
    )
    
    # Write the result to a CSV file
    top_flights_per_carrier.write.csv(task1_output, header=True, mode="overwrite")
  
    print(f"Task 1 output written to {task1_output}")


# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 2
    # Hint: Calculate standard deviation of departure delays, filter airlines with more than 100 flights.
    # Calculate departure delay
    flights_df = flights_df.withColumn(
        "DepartureDelay", F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")
    )

    # Group by carrier and calculate standard deviation of departure delays
    carrier_delay_stats = flights_df.groupBy("CarrierCode").agg(
        F.count("FlightNum").alias("FlightCount"),
        F.stddev("DepartureDelay").alias("StdDevDepartureDelay")
    )

    # Filter carriers with more than 100 flights
    carrier_delay_stats = carrier_delay_stats.filter(carrier_delay_stats["FlightCount"] > 100)

    # Join with carrier names
    carrier_delay_stats = carrier_delay_stats.join(carriers_df, on="CarrierCode", how="inner")

    # Select and rank by standard deviation of departure delays
    result = carrier_delay_stats.select(
        "CarrierName", "FlightCount", "StdDevDepartureDelay"
    ).orderBy("StdDevDepartureDelay")

    result.write.csv(task2_output, header=True, mode="overwrite")

    print(f"Task 2 output written to {task2_output}")


# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # TODO: Implement the SQL query for Task 3
    # Hint: Calculate cancellation rates for each route, then join with airports to get airport names.
    # Calculate cancellation rates for each route
    routes_cancellations = flights_df.withColumn(
        "Canceled", F.when(F.col("ActualDeparture").isNull(), 1).otherwise(0)
    ).groupBy("Origin", "Destination").agg(
        F.count("FlightNum").alias("TotalFlights"),
        F.sum("Canceled").alias("CanceledFlights")
    ).withColumn(
        "CancellationRate", F.col("CanceledFlights") / F.col("TotalFlights")
    )

    # Join with airports to get the airport names
    origin_airports = airports_df.withColumnRenamed("AirportName", "OriginAirport") \
                                  .withColumnRenamed("City", "OriginCity")
    destination_airports = airports_df.withColumnRenamed("AirportName", "DestinationAirport") \
                                      .withColumnRenamed("City", "DestinationCity")

    enriched_routes = routes_cancellations.join(
        origin_airports, routes_cancellations["Origin"] == origin_airports["AirportCode"], "left"
    ).join(
        destination_airports, routes_cancellations["Destination"] == destination_airports["AirportCode"], "left"
    )

    result = enriched_routes.select(
        "OriginAirport", "OriginCity", "DestinationAirport", "DestinationCity", "CancellationRate"
    ).orderBy("CancellationRate", ascending=False)

    # Write the result to a CSV file
    result.write.csv(task3_output, header=True, mode="overwrite")

    print(f"Task 3 output written to {task3_output}")


# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 4
    # Hint: Create time of day groups and calculate average delay for each carrier within each group.
     # Create time of day group
    def time_of_day(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 24:
            return "Evening"
        else:
            return "Night"

    time_of_day_udf = F.udf(time_of_day, StringType())
    flights_df = flights_df.withColumn("TimeOfDay", time_of_day_udf(F.hour("ScheduledDeparture")))

    # Calculate departure delay
    flights_df = flights_df.withColumn(
        "DepartureDelay", F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")
    )

    # Group by carrier and time of day to calculate average delay
    carrier_time_of_day = flights_df.groupBy("CarrierCode", "TimeOfDay").agg(
        F.avg("DepartureDelay").alias("AvgDepartureDelay")
    )

    # Join with carrier names
    carrier_time_of_day = carrier_time_of_day.join(carriers_df, on="CarrierCode", how="inner")

    result = carrier_time_of_day.select(
        "CarrierName", "TimeOfDay", "AvgDepartureDelay"
    ).orderBy("TimeOfDay", "AvgDepartureDelay")

    result.write.csv(task4_output, header=True, mode="overwrite")

    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
