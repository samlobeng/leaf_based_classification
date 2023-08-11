import { StatusBar } from 'expo-status-bar';
import { Button, StyleSheet, Text, View,Image,TouchableOpacity } from 'react-native';

export default function Menu({ navigation }) {
  return (
    <>
    <View style = {styles.header}>
        <Text style={styles.title}>Welcome to MedClassifiier</Text>
        <View style = {styles.hr} ></View>
    </View>
    <View style = {styles.plantContainer}>
        <TouchableOpacity style = {styles.newPlant} onPress={() => navigation.navigate('Predict')}>
            <View style = {styles.leafContainer}>
            <Image
                source={require('../assets/leaf.png')}
                resizeMode="contain"
                style={styles.leaf}
             />
            </View>
            
             <Image
                source={require('../assets/Line.png')}
                 style={{ width: 168 }}
             />
            <Button
            title='Classify a New Plant'
            color= "#0CAC1C"
            onPress={() => navigation.navigate('Predict')}
            />
             
        </TouchableOpacity>
        <TouchableOpacity style = {styles.history} onPress={() => navigation.navigate('HistoryScreen')}>
            <View style = {styles.leafContainer}>
            <Image
                source={require('../assets/history.png')}
                resizeMode="contain"
                style={styles.leaf}
             />
            </View>
            
             <Image
                source={require('../assets/Line.png')}
                 style={{ width: 168 }}
             />
            <Button
            title='View Classification History'
            color= "#0CAC1C"
            onPress={() => navigation.navigate('HistoryScreen')}
            style={styles.btnHistory}
            />
             
        </TouchableOpacity>
    </View>
    </>
  );
}

const styles = StyleSheet.create({
    header:{
        width: "100%",
        height: "10%",
        backgroundColor: "#000",
        justifyContent: "center",
        alignItems:"center"
    },
    title:{
        color:"#0CAC1C",
        textAlign: "center",
        fontSize: 20,
        fontWeight: 800,
    },
    hr:{
        width:"50%",
        backgroundColor: "#398F8F",
        height: "1.8%"
        
    },
    plantContainer:{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        gap: 25
    },
    newPlant:{
        backgroundColor: '#000',
        borderRadius: 400,
        width: "50%",
        height: "30%",
        textAlign: 'center',
        justifyContent: 'center',
        padding: "4%"
    },
    history:{
        backgroundColor: '#000',
        borderRadius: 400,
        width: "50%",
        height: "30%",
        textAlign: 'center',
        justifyContent: 'center',
        padding: "4%"
    },
    leafContainer:{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 0
    },
    leaf:{
        width: 50,
        height: 100,
    },
    btnHistory:{
        width: 50
    }
});