import React, { useState, useEffect } from 'react';
import { View, Text, Image, StyleSheet, ActivityIndicator, Button } from 'react-native';
import * as Progress from 'react-native-progress';

const Results = ({ route, navigation }) => {
  const { predictionData } = route.params;
  const [isLoading, setIsLoading] = useState(true);
  const [limeImage, setLimeImage] = useState(null);
  const [isLimeImageLoading, setIsLimeImageLoading] = useState(false);
  const [progress, setProgress] = useState(0)
  const [indeterminate, setindeterminate] = useState(true)
  useEffect(() => {
    // Simulate loading time for demonstration purposes
    // You can replace this with the actual data loading process
    animate()
    setTimeout(() => {
      setIsLoading(false);
    }, 2000);
  }, []);

  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      if (isLimeImageLoading) {
        setProgress(0);
        setindeterminate(true);
        setIsLimeImageLoading(false);
      }
    });

    return unsubscribe;
  }, [isLimeImageLoading]);

  function animate() {
    let progress = 0;
    setProgress(progress);
    setTimeout(() => {
      setindeterminate(false);
      setInterval(() => {
        progress += Math.random() / 100;
        if (progress > 1) {
          progress = 1;
        }
        setProgress(progress);
      }, 500);
    }, 1500);
  }


  const handleGetLimeExplanation = async () => {
    try {
      // Make a POST request to your FastAPI endpoint to get the LIME explanation
      const formData = new FormData();
      formData.append('file', {
        uri: predictionData.imageUri,
        name: 'image.jpg',
        type: 'image/jpeg',
      });

      const response = await fetch('http://192.168.0.102:8000/lime', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.ok) {
        // Convert the response to a blob to get the image
        const blob = await response.blob();

        // Create a local URL for the blob to display the image
        const imageURL = URL.createObjectURL(blob);
        setLimeImage(imageURL);
        setIsLimeImageLoading(false); // Set loading flag to false
        navigation.navigate('LimeResult', {
          limeImageUri: imageURL,
          data: predictionData,
          isLimeImageLoading: true, // Set to true when navigating to LimeResults
        });
        
      } else {
        // Handle error
        console.log('Failed to get LIME explanation');
      }
    } catch (error) {
      // Handle error
      console.log('Error while fetching LIME explanation:', error);
    }
  };

  return (
    <View style={styles.container}>
      {isLoading ? (
        <ActivityIndicator size="large" color="black" /> // Show loader while loading
      ) : (
        <>
          <Image source={{ uri: predictionData.imageUri }} style={styles.image} />
          <Text style={styles.predictionText}>Prediction Class: {predictionData.prediction.class}</Text>
          <Text style={styles.confidenceText}>Confidence: {predictionData.prediction.confidence.toFixed(2) + "%"}</Text>
          <Button title="How?" onPress={() => { setIsLimeImageLoading(true); handleGetLimeExplanation(); }} />
        </>
      )}
  
      {isLimeImageLoading && <Progress.Bar progress={progress} indeterminate = {indeterminate} width={200} />}
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
