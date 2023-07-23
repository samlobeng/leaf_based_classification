import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import Home from './Components/Home';
import Menu from './Components/Menu';
import Predict from './Components/Predict';
import Results from './Components/Results';
import { RootSiblingParent } from 'react-native-root-siblings';
const Stack = createStackNavigator();

export default function App() {
  return (
    <RootSiblingParent> 
    <NavigationContainer>
      <Stack.Navigator>
      
        <Stack.Screen name="Home" component={Home} />
        <Stack.Screen name="Menu" component={Menu} />
        <Stack.Screen name="Predict" component={Predict} />
        <Stack.Screen name="Results" component={Results} />
        
      </Stack.Navigator>
    </NavigationContainer>
    </RootSiblingParent>
    
  );
}

