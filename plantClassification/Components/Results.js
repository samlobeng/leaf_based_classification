import React, { useState, useEffect } from 'react';
import { View, Text, Image, StyleSheet,ActivityIndicator } from 'react-native';

const Results = ({ route }) => {
  const { predictionData } = route.params;
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading time for demonstration purposes
    // You can replace this with the actual data loading process
    setTimeout(() => {
      setIsLoading(false);
    }, 2000);
  }, []);

  return (
    <View style={styles.container}>
        {isLoading ? (
        <ActivityIndicator size="large" color="black" /> // Show loader while loading
      ) : (
        <>
      <Image source={{ uri: predictionData.imageUri }} style={styles.image} />
      <Text style={styles.predictionText}>Prediction Class: {predictionData.prediction.class}</Text>
      <Text style={styles.confidenceText}>Confidence: {predictionData.prediction.confidence.toFixed(2) + "%"}</Text>
      </>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 300,
    borderRadius: 20,
    marginBottom: 20,
  },
  predictionText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'black',
  },
  confidenceText: {
    fontSize: 16,
    color: 'black',
  },
});

export default Results;
