#!/bin/bash

#SBATCH -p defq        # İşin çalıştırılması istenen kuyruk seçilir
#SBATCH -o %j.out      # Çalıştırılan kodun ekran çıktılarını içerir
#SBATCH -e %j.err      # Karşılaşılan hata mesajlarını içerir 
#SBATCH -n 1           # Talep edilen işlemci  çekirdek sayısı

source activate ic3

python /okyanus/users/deepdrone/DroneSwarm_MP_IC3/main_drone.py
