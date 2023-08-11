import React from 'react';
import { View, Image, StyleSheet,Text } from 'react-native';

const LimeResult = ({ route }) => {
  const { limeImageUri, data } = route.params;

  return (
    <View style={styles.container}>
        <Text style={styles.predictionText}>Prediction Class: {data.prediction.class}</Text>
        <Text style={styles.confidenceText}>Confidence: {data.prediction.confidence.toFixed(2) + "%"}</Text>
      <Image source={{ uri: limeImageUri }} style={styles.image} />
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

export default LimeResult;
