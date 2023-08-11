import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Image, StyleSheet } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const HistoryScreen = () => {
  const [historyData, setHistoryData] = useState([]);

  useEffect(() => {
    // Load prediction history data from AsyncStorage
    const loadPredictionHistory = async () => {
      try {
        const data = await AsyncStorage.getItem('predictionHistory');
        if (data !== null) {
          const parsedData = JSON.parse(data);
          setHistoryData(parsedData);
        }
      } catch (error) {
        console.log('Error loading prediction history:', error);
      }
    };
    loadPredictionHistory();
    console.log(historyData)
    
  }, []);

  return (
    <View style={styles.container}>
      <FlatList
        data={historyData}
        keyExtractor={(item, index) => index.toString()}
        renderItem={({ item }) => (
          <View style={styles.historyItem}>
            <Image source={{ uri: item.imageUri }} style={styles.image} />
            <Text>Prediction: {item.prediction.class}</Text>
            <Text>Confidence: {item.prediction.confidence.toFixed(2)}%</Text>
            <Text>Timestamp: {item.timestamp}</Text>
          </View>
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  historyItem: {
    marginBottom: 20,
    borderColor: '#ccc',
    borderWidth: 1,
    padding: 10,
  },
  image: {
    width: 100,
    height: 100,
    resizeMode: 'cover',
    marginBottom: 10,
  },
});

export default HistoryScreen;
