#!/bin/bash

#SBATCH -p longq        # İşin çalıştırılması istenen kuyruk seçilir
#SBATCH -o %j.out      # Çalıştırılan kodun ekran çıktılarını içerir
#SBATCH -e %j.err      # Karşılaşılan hata mesajlarını içerir
#SBATCH -n 1           # Talep edilen işlemci  çekirdek sayısı

source activate ic3

python /okyanus/users/deepdrone/DroneSwarm_MotionPlanning/main.py --env_name predator_prey --nagents 5 --nprocesses 1 --num_epochs 20000 --hid_size 128 --detach_gap 10 --lrate 0.001 --max_steps 5000 --ic3net --recurrent --scenario planning
