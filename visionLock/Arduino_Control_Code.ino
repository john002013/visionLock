 #include<SoftwareSerial.h>
SoftwareSerial gsm(2, 3);

int relay = 9;
int trig_pin = 6;
int echo_pin = 5;
int Buzzer = 8;
long duration;
int distance;

void setup(){
//put your setup code here, to run once:
Serial.begin(9600);
pinMode(relay, OUTPUT);
pinMode(trig_pin, OUTPUT);
pinMode(echo_pin, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
digitalWrite(trig_pin, LOW);
delayMicroseconds(2);
digitalWrite(trig_pin, HIGH);
delayMicroseconds(10);
digitalWrite(trig_pin, LOW);
duration = pulseIn(echo_pin, HIGH);
distance = duration * 0.034 / 2;
Serial.println(distance);
delay(200);

if (distance <= 35){
  Serial.println("Detect");
  delay(200);
}

if (Serial.available() > 0){
  char command = Serial.read();

  
  if (command == '1'){
    digitalWrite(relay, HIGH);
    delay(3000);
    digitalWrite(relay, LOW);
    }
    
    else if (command =='2'){
    makecall();
   tone(Buzzer, 1000);
    delay(2000);
    noTone(Buzzer);
   }
 }
}

void makecall(){
gsm.begin(9600);
Serial.println("AT");
delay(1000);
Serial.println(gsm.readString());
delay(1000); 
Serial.println("Preparing to call");
delay(20);
gsm.print("ATD+2348064493866;\r\n");
delay(20);
Serial.println(gsm.readString());
delay(20);
}
