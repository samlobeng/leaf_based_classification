import React, { useState, useEffect, useRef } from 'react';
import { View, Text, Button, Image, TouchableOpacity, ScrollView, StyleSheet,Pressable } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { Camera } from 'expo-camera';
import axios from 'axios';
import Toast from 'react-native-root-toast';

const Separator = () => <View style={styles.separator} />

export default function Predict({ navigation }) {
  
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [cameraPermission, setCameraPermission] = useState(null);
  const [openCamera, setOpenCamera] = useState(false);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setCameraPermission(status === 'granted');
    })();
  }, []);

  const isImageFile = (fileName) => {
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif'];
    const ext = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();
    return imageExtensions.includes(ext);
  };

  const handlePickImage = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'image/*',
      });

      if (result.type === 'success' && isImageFile(result.name)) {
        setImage(result.uri);
        setFile(result);
        console.log('Image picked:', result);
      } else {
        console.log('Invalid file type. Please select an image file.');
      }
    } catch (error) {
      console.log('Error picking image:', error);
    }
  };

  const handleCaptureImage = async () => {
    if (cameraRef.current && cameraPermission) {
      try {
        const photo = await cameraRef.current.takePictureAsync();
        setImage(photo.uri);
        setFile(photo);
        console.log('Image captured:', photo);
      } catch (error) {
        console.log('Error capturing image:', error);
      } finally {
        setOpenCamera(false);
      }
    }
  };

  const handleOpenCamera = () => {
    setOpenCamera(true);
  };

  const handleCloseCamera = () => {
    setOpenCamera(false);
  };

  

  const handlePredict = async () => {
    if (file) {
      const formData = new FormData();
      formData.append('file', {
        uri: file.uri,
        name: 'image.jpg',
        type: 'image/jpeg',
      });
  
      try {
        const response = await fetch('http://192.168.0.102:8000/predict', {
          method: 'POST',
          body: formData,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
  
        if (response.ok) {
            const data = await response.json();
          // Check confidence rate
          if (data.confidence >= 85) {
            console.log('Image is classified as a medicinal plant');
            // Handle the case when the image is classified as a medicinal plant
            console.log('Prediction:', data);
            const predictionData = {
              imageUri: image,
              prediction: data,
            };
            navigation.navigate('Results', { predictionData });
          } else {
           Toast.show("Image is not a medicinal plant", {
            duration :Toast.durations.LONG,
            position: Toast.positions.BOTTOM,
            shadow: true,
            hideOnPress: true,
            delay: 0,
           })
            // Handle the case when the image is not classified as a medicinal plant
          }
        } else {
          Toast.show(`Error Predicting, ${response.status}`, {
            duration :Toast.durations.LONG,
            position: Toast.positions.BOTTOM,
            shadow: true,
            hideOnPress: true,
            delay: 0,
           })
        }
      } catch (error) {
        Toast.show(`Error Predicting`, {
          duration :Toast.durations.LONG,
          position: Toast.positions.BOTTOM,
          shadow: true,
          hideOnPress: true,
          delay: 0,
         })
      }
    }
    
  };
  

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
    
      <View style={{ flex: 1 }}>
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          {openCamera && cameraPermission ? (
            <Camera style={styles.cameraContainer} ref={cameraRef} onCameraReady={() => console.log('Camera ready')}>
              <View style={styles.cameraButtonsContainer}>
                <TouchableOpacity style={styles.cameraButton} onPress={handleCloseCamera}>
                  <Text style={styles.cameraButtonText}>Close</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.cameraButton} onPress={handleCaptureImage}>
                  <Text style={styles.cameraButtonText}>Capture</Text>
                </TouchableOpacity>
              </View>
            </Camera>
          ) : (
            <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
              <TouchableOpacity
                onPress={handlePickImage}
                style={{ padding: 20, backgroundColor: '#000', marginBottom: 20, width: 250, borderRadius: 50 }}
              >
                <Text style={{ fontSize: 20,fontWeight: "800",  color: "#0CAC1C", textAlign:'center' }}>Select Image</Text>
              </TouchableOpacity>

              {image && (
                <View style={{ alignItems: 'center', marginTop: 20, marginBottom:20 }}>
                  <Image source={{ uri: image }} style={{ width: 300, height: 300, borderRadius:20 }} />
                </View>
              )}

              <TouchableOpacity
                onPress={handleOpenCamera}
                style={{ padding: 20, backgroundColor: '#000', marginBottom: 20, width: 250, borderRadius: 50 }}
              >
                <Text style={{ fontSize: 20, fontWeight: "800",  color: "#0CAC1C",  textAlign:'center' }}>Capture Image</Text>
              </TouchableOpacity>
              <Separator />
  
              <TouchableOpacity style={styles.button} onPress={handlePredict}>
                <Text style={styles.text}>PREDICT</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  cameraContainer: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  cameraButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 20,
  },
  cameraButton: {
    flex: 0.5,
    alignSelf: 'flex-end',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    marginHorizontal: 5,
  },
  cameraButtonText: {
    fontSize: 18,
    color: 'white',
  },
  cont:{
    width: 288,
    alignSelf: 'center',
    marginTop: 50
  },
  separator: {
    marginVertical: 8,
    borderBottomColor: 'black',
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  button: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 50,
    elevation: 3,
    backgroundColor: 'black',
    width: 250
  },
  text: {
    fontSize: 16,
    lineHeight: 21,
    fontWeight: 'bold',
    letterSpacing: 0.25,
    color: 'white',
  },
});
