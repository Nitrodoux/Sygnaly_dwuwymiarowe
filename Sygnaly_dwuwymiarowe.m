clc;
clear all;
close all;

%% Wczytanie obrazów
obraz_jpg=imread('miasto.jpg');
obraz_bmp=imread('czolg.bmp');
obraz_tiff=imread('napis.tiff');
obraz_png=imread('lew.png');
[A, map]=imread('pingwin.gif',1:15);

%% Wyœwietlenie informacji o obrazach
info_jpg=imfinfo('miasto.jpg')
info_bmp=imfinfo('czolg.bmp');
info_tif=imfinfo('napis.tiff');
info_png=imfinfo('lew.png');
imfinfo('pingwin.gif');

info=imfinfo('miasto.jpg')
if info.Height < 400
    afterInfo = imrotate(obraz_jpg, 90);
else
    afterInfo = imrotate(obraz_jpg, 180);
end
figure (1)
imshow(afterInfo)
title('Obrót obrazu jpg')


%% Wyœwietlenie obrazów
figure(1)
subplot(2,3,1)
imshow(obraz_jpg);
title('Obraz JPEG')
subplot(2,3,2)
imshow(obraz_bmp);
title('Obraz BMP')
subplot(2,3,3)
imshow(obraz_tiff);
title('Obraz TIFF')
subplot(2,3,4)
imshow(obraz_png);
title('Obraz PNG')
subplot(2,3,5)
for i=1:8   %wyœwietlenie 8 klatek GIF
    imshow(A(:,:,:,i),map)
end
title('Obraz GIF');

%% Porównanie funkcji
figure(2)
subplot(2,2,1)
imshow(obraz_jpg);
title('Funkcja imshow')
subplot(2,2,2)
image(obraz_jpg);
title('Funkcja image')
subplot(2,2,3)
imagesc(obraz_jpg);
title('Funkcja imagesc')
subplot(2,2,4)
warp(obraz_jpg);
title('Funkcja warp')

%% Zapisanie JPG w innych rozszerzeniach
imwrite(obraz_jpg,'zJPGdoBMP.bmp');   %zapis do bmp
imwrite(obraz_jpg,'zJPGdoTIF.tiff');   %zapis do tiff
imwrite(obraz_jpg,'zJPGdoPNG.png');  %zapis do png

%% Nieliniowe przekszta³cenie obrazu
%%Pierwiastkowanie
o_rgb = obraz_jpg;
o_drgb = double(o_rgb)/255;
o_gray = rgb2gray(obraz_jpg);
o_dgray = double(o_gray)/255;
figure(3)
subplot(2,2,1)
imshow(o_rgb);
title('Obraz barwny')
subplot(2,2,2)
imshow(sqrt(o_drgb));
title('Obr. barwny spierwiastkowany');
subplot(2,2,3)
imshow(o_gray);
title('Obraz w odcieniu szaroœci')
subplot(2,2,4)
imshow(sqrt(o_dgray));
title('Obr. szary spierwiastkowany');

%% Potêgowanie
figure(4)
subplot(2,2,1)
imshow(o_rgb);
title('Obraz barwny')
subplot(2,2,2)
imshow(o_drgb.^2);
title('Obr. barwny potêgowany');
subplot(2,2,3)
imshow(o_gray);
title('Obraz w odcieniu szarosci')
subplot(2,2,4)
imshow(o_dgray.^2);
title('Obr. szary potêgowany');

%% Podwojenie
figure(5)
subplot(2,2,1)
imshow(o_rgb);
title('Obraz barwny')
subplot(2,2,2)
imshow(o_drgb.*2);
title('Obr. barwny podwojony');
subplot(2,2,3)
imshow(o_gray);
title('Obraz w odcnieu szarosci')
subplot(2,2,4)
imshow(o_dgray.*2);
title('Obr. szary podwojony');

%% Konwersja obrazów
%%RGB na hsv
rgb_hsv=rgb2hsv(o_rgb);
hsv_rgb=hsv2rgb(rgb_hsv);
figure(6)
subplot(1,2,1)
imshow(rgb_hsv);
title('Obraz HSV')
subplot(1,2,2)
imshow(hsv_rgb);
title('Obraz RGB');

%%RGB na NTSC
rgb_ntsc=rgb2ntsc(o_rgb);
ntsc_rgb=ntsc2rgb(rgb_ntsc);
figure(7)
subplot(1,2,1)
imshow(rgb_ntsc);
title('Obraz NTSC')
subplot(1,2,2)
imshow(ntsc_rgb);
title('Obraz RGB');

%%RGB na YCbCr
rgb_ycbcr=rgb2ycbcr(o_rgb);
ycbcr_rgb=ycbcr2rgb(rgb_ycbcr);
figure(8)
subplot(1,2,1)
imshow(rgb_ycbcr);
title('Obraz YCbCr')
subplot(1,2,2)
imshow(ycbcr_rgb);
title('Obraz RGB');

%% Wyœwietlenie kolorów we wspó³rzêdnych 3D
%%Barwy
obrazR=obraz_jpg(:,:,1);    %Kana³ czerwony
obrazG=obraz_jpg(:,:,2);    %Kana³ zielony
obrazB=obraz_jpg(:,:,3);    %Kana³ niebieski
[w,k]=size(obrazR);
 
%Wyswietlanie kanalu czerwonego
r=0:1:255;
rr=r';
z=zeros(256,3);
z(:,1)=rr;
red=z/255;
 
%Wyswietlanie kanalu zielonego
r2=0:1:255;
rr2=r2';
z2=zeros(256,3);
z2(:,2)=rr2;
green=z2/255;   
 
%Wyswietlanie kanalu niebieskiego
r3=0:1:255;
rr3=r3';
z3=zeros(256,3);
z3(:,3)=rr3;
blue=z3/255;

figure(9)
mesh(1:k,w:-1:1,obrazR,'facecolor','texturemap','edgecolor','none','cdatamapping','direct');
colormap(red);
view(3);
colorbar('vert');
title('Wykres 3D-kanal czerwony');
xlabel('Kolumna');
ylabel('Wiersz');
zlabel('Natê¿enie'); 
figure(10)
mesh(1:k,w:-1:1,obrazG,'facecolor','texturemap','edgecolor','none','cdatamapping','direct');
colormap(green);
view(3);
colorbar('vert');
title('Wykres 3D-kanal zielony');
xlabel('Kolumna');
ylabel('Wiersz');
zlabel('Natê¿enie');
 
figure(11)
mesh(1:k,w:-1:1,obrazB,'facecolor','texturemap','edgecolor','none','cdatamapping','direct');
colormap(blue);
view(3);
colorbar('vert');
title('Wykres 3D-kanal niebieski');
xlabel('Kolumna');
ylabel('Wiersz');
zlabel('Natê¿enie');

 
%% Histogramy
figure(12)
dane_hist=imhist(obrazR);
colormap(red);
imhist(obrazR);
colorbar('location','southoutside');
grid on;
title('Histogram - kanal czerwony');
 
figure(13)
dane_hist=imhist(obrazG);
colormap(green);
imhist(obrazG);
colorbar('location','southoutside');
grid on;
title('Histogram - kanal zielony');
 
figure(14)
dane_hist=imhist(obrazB);
colormap(blue);
imhist(obrazB);
colorbar('location','southoutside');
grid on;
title('Histogram - kanal niebieski');

%% Wykreslenie profilu barwnego RGB wzd³u¿ wybranego odcinka
figure(15)
imshow(obraz_jpg);
c=improfile(obraz_jpg,[730 740],[275 325]);
line([730 740],[275 325],'Color','r','Linewidth',5);
figure(16)
plot(c(:,:,1),'r');
hold on;
plot(c(:,:,2),'g');
hold on;
plot(c(:,:,3),'b');
hold on;
title('Profile barwne RGB wzd³u¿ danej linii');
grid on;
xlabel('D³ugoœæ odcinka');
ylabel('Intensywnosc pikseli');

%% Modyfikacja wybranego kana³u barwnego
ObrazR=obraz_jpg(:,:,1);
ObrazG=obraz_jpg(:,:,2);
ObrazB=obraz_jpg(:,:,3)/5;
obraz_mod=cat(3,ObrazR,ObrazG,ObrazB);
figure(17)
imshow(obraz_mod);
title('Zmodyfikowany niebieski kana³ barwny');

%% Pseudokolorowanie
[ObrazRGB]=imread('miasto.jpg');
[x1,map1]=rgb2ind(ObrazRGB,16);%16 iloœæ poziomów danej palety
[x2,map2]=rgb2ind(ObrazRGB,128);%128 iloœæ poziomów danej palety
[x3,map3]=rgb2ind(ObrazRGB,255);%255 iloœæ poziomów danej palety
figure(18)
subplot(1,3,1);
imshow(x1,colormap(prism));
title('16 kolorow');
subplot(1,3,2);
imshow(x2,colormap(prism));
title('128 kolorow');
subplot(1,3,3);
imshow(x3,colormap(prism));
title('255 kolorow');

figure(19)
subplot(1,3,1);
imshow(x1,colormap(copper));
title('16 kolorow');
subplot(1,3,2);
imshow(x2,colormap(copper));
title('128 kolorow');
subplot(1,3,3);
imshow(x3,colormap(copper));
title('255 kolorow');

figure(20)
subplot(1,3,1);
imshow(x1,colormap(jet));
title('16 kolorow');
subplot(1,3,2);
imshow(x2,colormap(jet));
title('128 kolorow');
subplot(1,3,3);
imshow(x3,colormap(jet));
title('255 kolorow');

%% Obracanie obrazu
J11=imrotate(o_rgb,-45,'nearest','crop');
J12=imrotate(o_gray,-45,'nearest','loose');
J21=imrotate(o_rgb,-45,'bilinear','crop');
J22=imrotate(o_gray,-45,'bilinear','loose');
J31=imrotate(o_rgb,-45,'bicubic','crop');
J32=imrotate(o_gray,-45,'bicubic','loose');
figure(21)
subplot(1,2,1)
imshow(J11)
title('Obraz rgb obrócony o 45°')
subplot(1,2,2)
imshow(J12)
title('Obraz szary obrócony o 45°')
figure(22)
subplot(1,2,1)
imshow(J21)
title('Obraz rgb obrócony o 45°')
subplot(1,2,2)
imshow(J22)
title('Obraz szary obrócony o 45°')
figure(23)
subplot(1,2,1)
imshow(J31)
title('Obraz rgb obrócony o 45°')
subplot(1,2,2)
imshow(J32)
title('Obraz szary obrócony o 45°')
%% Maketfrom
T = maketform('affine',[.5 0 0; .5 2 0; 0 0 1]);
tformfwd([10 20],T);
T1 = imtransform(o_gray,T);
figure(24)
imshow(T1)
%% Imersize
IS12=imresize(o_rgb,0.5,'nearest');
IS22=imresize(o_rgb,0.5,'bilinear');
IS32=imresize(o_rgb,0.5,'bicubic');
IS11=imresize(o_gray,1.5,'nearest');
IS21=imresize(o_gray,1.5,'bilinear');
IS31=imresize(o_gray,1.5,'bicubic');
figure(25)
subplot(3,1,1)
imshow(IS11)
title('Obraz szary powiêkszony')
subplot(3,1,2)
imshow(IS21)
title('Obraz szary powiêkszony')
subplot(3,1,3)
imshow(IS31)
title('Obraz szary powiêkszony')
figure(26)
subplot(3,1,1)
imshow(IS12)
title('Obraz rgb pomniejszony')
subplot(3,1,2)
imshow(IS22)
title('Obraz rgb pomniejszony')
subplot(3,1,3)
imshow(IS32)
title('Obraz rgb pomniejszony')
%% Reshape
[i,j]=size(o_gray);
Br=reshape(o_gray,j,i);
figure(27)
imshow(Br)
%% Funkcje flipud, fliplr
Be=flipud(o_gray);
Be2=fliplr(o_gray);
figure(28)
subplot(1,2,1)
imshow(Be);
title('zamiana kolumn')
subplot(1,2,2)
imshow(Be2);
title('zamiana wierszy')
%% Funkcja imcrop
[w,k]=size(o_gray); % wymiary obrazu w odcieniach szaroœci
a=k/5; % obraz dzielony na 5 czêœci równe
figure(29)
title('Obraz sary podzielony na 5 czêœci')
for ii = 1:5
    subplot(1,5,ii);
    cropp=imcrop(o_gray,[(ii-1)*a 0 a w]);
    imshow(cropp); % druga czêœæ obrazu
    title(ii)
end

k1=imcrop(o_gray,[0 0 a w]);
k2=imcrop(o_gray,[a 0 a w]);
k3=imcrop(o_gray,[2*a 0 a w]);
k4=imcrop(o_gray,[3*a 0 a w]);
k5=imcrop(o_gray,[4*a 0 a w]); 
%%scalony
k=cat(2,k1,k2,k3,k4,k5);
figure(30)
imshow(k)
title('Scalony obraz')
%% Opracje arytmetyczne
%Suma
I1 = imread('czolg.bmp');
J1 = imread('miasto.jpg');
I2 = rgb2gray(I1);
J2= rgb2gray(J1);
K1 = imadd(I1,J1);
K2 = imadd(I2,J2);
figure(31)
subplot(1,2,1)
imshow(K1)
title('suma obrazów barwnych')
subplot(1,2,2)
imshow(K2)
title('suma obrazów szarych')
%% Ró¿nica
K3 = imabsdiff(I1,J1);
K4 = imabsdiff(I2,J2);
figure(32)
subplot(1,2,1)
imshow(K3)
title('ró¿nica obrazów barwnych')
subplot(1,2,2)
imshow(K4)
title('ró¿nica obrazów szarych')
%% œrednia arytmetyczna
K5 = imdivide(I1,J1);
K6 = imdivide(I2,J2);
figure(33)
subplot(1,2,1)
imshow(K5)
title('Œr. arytmetyczna obrazów barwnych')
subplot(1,2,2)
imshow(K6)
title('Œr. arytmetyczna obrazów szarych')
%% Iloczyn
K7 = immultiply(I1,J1);
K8 = immultiply(I2,J2);
figure(34)
subplot(1,2,1)
imshow(K7)
title('Iloczyn obrazów barwnych')
subplot(1,2,2)
imshow(K8)
title('Iloczyn szarych')
%% Zadanie2
%% Przeksztalcenie do postaci binarnej
obrazRGB=imread('lew.png');
[Y,map]=imread('lew.png');
obrazG=rgb2gray(obrazRGB);
I=13;
N=3;
%prog przyporzadkowania
H_I=imhist(obrazG);%historgram
[max_I,ind]=max(H_I);
level=(ind)/255;

%a)BW=im2bw(I,level) obraz intensywnosci do bin
BW_a=im2bw(obrazG,level);      %level z przedzia³u 0-1;
%b)BW=im2bw(K,map,level) obraz indexowany do bin
BW_b=im2bw(Y,map,level);
%c) Bw=im2bw(RGB,level) rgb do bin
BW_c=im2bw(obrazRGB,level);
 
figure(35)
subplot(2,2,1);
imshow(obrazG);
title('Obraz szary');
subplot(2,2,2);
imshow(BW_a);
title('BW=im2b2(I,level)');
subplot(2,2,3);
imshow(BW_b);
title('BW=im2bw(K,map,level)');
subplot(2,2,4);
imshow(BW_c);
title('BW=im2bw(RGB,level)');

%% Graythresh
obrazRGB=imread('lew.png');
[Y,map]=imread('lew.png');
obrazG=rgb2gray(obrazRGB);
level2=graythresh(obrazG);

BW_GRAY=im2bw(obrazG,level2);
figure(36)
subplot(1,2,1);
imshow(obrazG);
title('Obraz szary');
subplot(1,2,2);
imshow(BW_GRAY);
title('Obraz graythresh');

%% Binaryzacja
j=1;
for i=110:20:210
    BW_e(:,:,:,j)=im2bw(obrazG,i/255); %wiersze,kolumny,g³êbia obrazu,j-numer binaryzowanego obrazu
    j=j+1;
 
end
figure(37)
montage(BW_e,'Size',[1 6]);
title('Binaryzacja dolna obrazu z progiem od 110 do 210 z krokiem 20');

%% Dzielenie na 4 czesci
[w k q] =size(obrazG);
p1=obrazG(1:w/2,1:k/2);
p3=obrazG(w/2+1:w,1:k/2);
p2=obrazG(1:w/2,k/2+1:k);
p4=obrazG(w/2+1:w,k/2+1:k);

figure(38)
subplot(2,2,1)
imshow(p1);
subplot(2,2,2);
imshow(p2);
subplot(2,2,3);
imshow(p3);
subplot(2,2,4);
imshow(p4);
%% Dzielenie na 16 roznych czesci
I=13;
N=3;
for i=1:(I+N)
    p(:,:,i)= obrazG(:,(i-1)*k/(I+N)+1:(i)*k/(I+N));
    p_h(:,:,i)= histeq(p(:,:,i));
    %Obraz podzielony na I+N równych czêœci
    figure(39)
    subplot(1, (I+N),i)
    imshow(p(:,:,i));
    % podzielony na równe czêœci z kazdym oddzielnym wyrównaniem
    figure(40)
    subplot(1,(I+N), i)
    imshow(p_h(:,:,i));
   end
%% Zadanie 3
R=imread('zad3.png');
R2=rgb2gray(R);
level=graythresh(R2); 
R_bw=im2bw(R,level);
figure(41);
subplot(1,3,1);
imshow(R);
title('Obraz orginalny');
subplot(1,3,2);
imshow(R2)
title('Obraz w odcieniach szaroœci');
subplot(1,3,3);
imshow(R_bw);
title('Obraz binarny')  

%% Strel
SE1=strel('diamond',5);
SE2=strel('disk',10,6);
SE3=strel('line',30,25);
SE4=strel('octagon',9);
SE5=strel('rectangle',[15,20]);
SE6=strel('square',15);
SE7=strel('arbitrary',[0,1,1,1,0,1]);
SE8=strel('pair',[2,-1]);
SE9=strel('cube',5);
%
figure(42)
subplot(3,3,1);
imshow(getnhood(SE1),'InitialMagnification','fit');
title('Diamond');
subplot(3,3,2);
imshow(getnhood(SE2),'InitialMagnification','fit');
title('Disc');
subplot(3,3,3);
imshow(getnhood(SE3),'InitialMagnification','fit');
title('Line');
subplot(3,3,4);
imshow(getnhood(SE4),'InitialMagnification','fit');
title('Octagon');
subplot(3,3,5);
imshow(getnhood(SE5),'InitialMagnification','fit');
title('Rectangle');
subplot(3,3,6);
imshow(getnhood(SE6),'InitialMagnification','fit');
title('Square');
subplot(3,3,7);
imshow(getnhood(SE7),'InitialMagnification','fit');
title('Arbitrary');
subplot(3,3,8);
imshow(getnhood(SE8),'InitialMagnification','fit');
title('Pair');
subplot(3,3,9);
imshow(getnhood(SE9),'InitialMagnification','fit');
title('Cube'); 
%% b) operacja erozji
A_er2=imerode(R_bw,SE1);
figure(43)
imshow(A_er2);
title('Operacja erozji')

%operacja dylatacji - odwrotnoœæ erozji
A_odw2=imdilate(R_bw,SE1);
figure(44)
imshow(A_odw2);
title('Operacja dylatacji')
 
A_otw2=imopen(R_bw,SE1);
figure(45)
imshow(A_otw2);title('Operacja otwarcia')
A_zam2=imclose(R_bw,SE1);
figure(46)
imshow(A_zam2);title('Operacja zamkniêcia')

%% Zadanie 4
v=VideoReader ('Film2.avi') 
A=read(v,3);
info=aviinfo('Film2.avi')
 
filmG=rgb2gray(A);
level=graythresh(filmG); %Ustalenie progu met Otsu->optymalny próg binaryzacji
A_bw=~im2bw(filmG,level); %binaryzacja z progiem z metody Otsu
figure(47)
subplot(1,3,1);
imshow(A);
title({'3 klatka filmu';'-obraz RGB'})
subplot(1,3,2);
imshow(filmG);
title({'3 klatka filmu-';'w odcieniach szaroœci'})
subplot(1,3,3);
imshow(A_bw);
title({'3 klatka filmu'; 'po binaryzacji i negacji'});

figure(48)
imshow(A_bw);
s=regionprops(A_bw,'Centroid');
centroids=cat(1,s.Centroid);
hold on;title('Œrodek ciê¿koœci obektu');
plot(centroids(:,1),centroids(:,2),'b+','Markersize',15);

%%
[A_m,liczba]=bwlabel(A_bw);
figure(49)
imshow(A);
title(['Wykryto: ' num2str(liczba) ' obiekt(y)']);
Pole=bwarea(A_bw)%pole w pikselach/porównañ z regionprops

%%
for i=1:1:40
    A1=read(v,i);
    A1_gg=rgb2gray(A1);
    level1=graythresh(A1_gg);
    A1_bw=~im2bw(A1_gg,level1);
    c1=regionprops(A1_bw,'Centroid')
    c12=struct2cell(c1);
    c11=cell2mat(c12);
     figure(50);
    plot(-c11(1,1),c11(1,3),'k+','Markersize',15)
    hold on
    plot(-c11(1,2),c11(1,4),'r+','Markersize',15);
    kods1=(c11(1,3));
    kods2=kods1(2:1);
    kods3=c11(1,4);
    kods4=kods3(2:1);
    hold on;
    title('Tor ruchu obiektu');
    end

