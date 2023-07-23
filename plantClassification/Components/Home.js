import { StatusBar } from 'expo-status-bar';
import { Button, StyleSheet, Text, View,Image } from 'react-native';;
export default function Home({ navigation }) {
  return (
    <>
    
    <View style={styles.mainContainer}>
        
     <View style = {styles.logoContainer}>
     <Image
        source={require('../assets/sublogo.png')}
        style={{ width: 150, height: 200 }}
    />

     </View>
    <View style={styles.getStarted}>
    <Button
        title="GET STARTED"
        color= "#0CAC1C"
        onPress={() => navigation.navigate('Menu')}
      />
    </View>
    </View>
    <View style = {styles.bar}></View>
    </>
  );
}

const styles = StyleSheet.create({
  mainContainer:{
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoContainer: {
    backgroundColor: '#000',
    height: 350,
    width: 350,
    borderRadius: 500,
    marginLeft: 55,
    marginRight:50,
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoText:{
    color: '#fff',
    textAlign: 'center'
  },
  getStarted:{
    backgroundColor: '#000',
    marginTop: 100,
    borderRadius: 300,
    width: "65%",
    height: 50,
    textAlign: 'center',
    justifyContent: 'center'
  },
//   bar:{
//     width: '100%',
//     height: 80,
//     backgroundColor: '#000',
//     justifyContent: 'center',
//     alignItems: 'center',
//     position: 'absolute', //Here is the trick
//     bottom: 0, //Here is the trick
//   }
});
