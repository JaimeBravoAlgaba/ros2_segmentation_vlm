# ros2_segmentation_vlm

Paquete ROS 2 para segmentacion semantica con modelos vision-lenguaje. El repositorio separa la inferencia pesada en un servidor TCP y deja los nodos ROS como clientes que envian imagenes, reciben mapas de clases y publican resultados en topics ROS.

## Funcionamiento general

El sistema tiene tres piezas principales:

1. Servidor de segmentacion (`ros2_segmentation_vlm/server/segmentation_server.py`)
   - Carga el backend de inferencia, actualmente `sam3`.
   - Escucha por TCP en `host:port`.
   - Recibe una configuracion inicial con los prompts/clases semanticas.
   - Recibe imagenes BGR y devuelve un `class_map` `uint8`, donde cada pixel contiene el ID de clase predicho.

2. Puente de segmentacion ROS (`ros2_segmentation_node`)
   - Se suscribe a un topic `sensor_msgs/Image`.
   - Envia cada imagen al servidor TCP.
   - Publica una imagen coloreada y un mapa de clases.
   - Usa los ficheros JSON de `config/` para convertir IDs de clase en colores y atributos.

3. Nodos de semantica y traversabilidad (`ros2_semantics_node` y `ros2_traversability_node`)
   - `ros2_semantics_node` consume datos de RTAB-Map (`/rtabmap/mapData` y `/rtabmap/cloud_map`).
   - Extrae imagenes RGB de los nodos de RTAB-Map, las segmenta y proyecta las clases sobre la nube global.
   - Publica una nube semantica con campos extra: `class_id`, `traversable`, `traversability` y `cost`.
   - `ros2_traversability_node` convierte esa nube semantica en un `nav_msgs/OccupancyGrid`.

El protocolo TCP interno esta documentado en `docs/semantic_segmentation_protocol.md`.

## Estructura relevante

```text
ros2_segmentation_vlm/
  ros2_segmentation_node.py       # puente imagen ROS -> servidor -> imagen segmentada
  ros2_semantics_node.py          # proyeccion semantica sobre cloud_map de RTAB-Map
  ros2_traversability_node.py     # nube semantica -> OccupancyGrid
  segmentation_protocol.py        # protocolo NPZ sobre TCP
  semantic_classes.py             # carga/validacion de clases semanticas
  server/
    segmentation_server.py        # servidor TCP de inferencia
    inference/sam3.py             # backend SAM3
config/
  demo_semantic_classes.json
  arena_semantic_classes.json
launch/
  segmentation_bridge.launch.py
  semantics.launch.py
rviz/
  ros2_segmentation_vlm.rviz
```

## Requisitos

- ROS 2 Humble o compatible.
- `colcon` y `ament_python`.
- Dependencias ROS declaradas en `package.xml`: `rclpy`, `sensor_msgs`, `sensor_msgs_py`, `nav_msgs`, `rtabmap_msgs`, `cv_bridge`, `tf2_ros`, entre otras.
- Dependencias Python para inferencia: `numpy<2`, `opencv-python`, `Pillow`, `torch` y el paquete/modulo `sam3` usado por `server/inference/sam3.py`.
- Un entorno con acceso al modelo SAM3 y sus pesos segun la instalacion de SAM3 que uses.

Nota sobre NumPy:

ROS 2 Humble suele compilar `cv_bridge` contra NumPy 1.x. Si el entorno de usuario carga NumPy 2.x desde `~/.local/lib/python3.10/site-packages`, los nodos ROS pueden fallar al importar con errores como `_ARRAY_API not found`.

Los launch files de este paquete establecen `PYTHONNOUSERSITE=1` para los nodos ROS, de forma que prioricen los paquetes Python del sistema/ROS frente a instalaciones del user-site.

## Compilacion

Desde la raiz del workspace:

```bash
cd ~/go2_ws
colcon build --packages-select ros2_segmentation_vlm
source install/setup.bash
```

Si cambias ficheros Python o launch files, recompila y vuelve a hacer `source install/setup.bash`.

## Configuracion de clases semanticas

Las clases se definen en JSON. Cada entrada incluye:

- `id`: entero entre 0 y 254. El valor 255 queda reservado para desconocido/fondo.
- `name`: prompt textual enviado al modelo.
- `color`: color RGB usado para visualizar la clase.
- `traversable`: booleano opcional para indicar si es transitable.
- `traversability`: valor opcional entre 0.0 y 1.0.
- `cost`: coste opcional, mayor o igual que 0.0.

Ejemplo:

```json
{
  "unknown_class_id": 255,
  "unknown_color": [0, 0, 0],
  "classes": [
    {
      "id": 0,
      "name": "grass",
      "color": [0, 255, 0],
      "traversable": true,
      "traversability": 0.65,
      "cost": 0.35
    }
  ]
}
```

## Lanzamiento: segmentacion de imagen ROS

Este modo sirve para tomar una imagen de ROS, segmentarla y publicar el resultado.

### 1. Arrancar el servidor de segmentacion

En una terminal con el entorno Python donde este disponible SAM3:

```bash
cd ~/go2_ws/src/ros2_segmentation_vlm
python3 ros2_segmentation_vlm/server/segmentation_server.py --host 127.0.0.1 --port 8765 --method sam3
```

El servidor debe quedarse escuchando antes de lanzar el nodo ROS.

### 2. Lanzar el puente ROS

En otra terminal:

```bash
cd ~/go2_ws
source install/setup.bash
ros2 launch ros2_segmentation_vlm segmentation_bridge.launch.py
```

Por defecto:

- Entrada: `/camera/color/image_raw`
- Imagen segmentada coloreada: `/segmentation/color/image`
- Servidor: `127.0.0.1:8765`
- RViz: activado

Ejemplo con topics personalizados:

```bash
ros2 launch ros2_segmentation_vlm segmentation_bridge.launch.py \
  input_topic:=/camera/color/image_raw \
  output_topic:=/segmentation/color/image \
  host:=127.0.0.1 \
  port:=8765 \
  rviz:=true
```

El nodo tambien publica el mapa de clases en `/segmentation/class` como imagen `mono8`.

## Lanzamiento: mapa semantico y traversabilidad

Este modo requiere que RTAB-Map este publicando `MapData` y la nube global.

### 1. Arrancar el servidor de segmentacion

```bash
cd ~/go2_ws/src/ros2_segmentation_vlm
python3 ros2_segmentation_vlm/server/segmentation_server.py --host 127.0.0.1 --port 8765 --method sam3
```

### 2. Arrancar RTAB-Map y sus sensores

Asegurate de que existen, como minimo:

```bash
ros2 topic echo /rtabmap/mapData --once
ros2 topic echo /rtabmap/cloud_map --once
```

### 3. Lanzar semantica + traversabilidad

```bash
cd ~/go2_ws
source install/setup.bash
ros2 launch ros2_segmentation_vlm semantics.launch.py \
  semantic_classes_path:=/home/jaime/go2_ws/src/ros2_segmentation_vlm/config/arena_semantic_classes.json
```

Topics por defecto:

- Entrada RTAB-Map: `/rtabmap/mapData`
- Nube RTAB-Map: `/rtabmap/cloud_map`
- Nube semantica: `/semantics/cloud`
- Mapa de traversabilidad: `/semantics/map/traversability`

Ejemplo completo con parametros:

```bash
ros2 launch ros2_segmentation_vlm semantics.launch.py \
  segmentation_server_host:=127.0.0.1 \
  segmentation_server_port:=8765 \
  semantic_classes_path:=/home/jaime/go2_ws/src/ros2_segmentation_vlm/config/arena_semantic_classes.json \
  map_data_topic:=/rtabmap/mapData \
  cloud_map_topic:=/rtabmap/cloud_map \
  semantics_cloud_topic:=/semantics/cloud \
  traversability_map_topic:=/semantics/map/traversability \
  traversability_resolution:=0.10
```

## Ejecucion directa de nodos

Tambien se pueden lanzar los ejecutables individualmente:

```bash
ros2 run ros2_segmentation_vlm ros2_segmentation_node --ros-args \
  -p host:=127.0.0.1 \
  -p port:=8765 \
  -p input_topic:=/camera/color/image_raw \
  -p output_topic:=/segmentation/color/image
```

```bash
ros2 run ros2_segmentation_vlm ros2_semantics_node --ros-args \
  -p segmentation_server_host:=127.0.0.1 \
  -p segmentation_server_port:=8765 \
  -p semantic_classes_path:=/home/jaime/go2_ws/src/ros2_segmentation_vlm/config/arena_semantic_classes.json
```

```bash
ros2 run ros2_segmentation_vlm ros2_traversability_node --ros-args \
  -p input_cloud_topic:=/semantics/cloud \
  -p output_map_topic:=/semantics/map/traversability \
  -p resolution:=0.10
```

## Comprobacion rapida

Servidor:

```bash
ss -ltnp | grep 8765
```

Topics del puente de imagen:

```bash
ros2 topic list | grep segmentation
ros2 topic echo /segmentation/class --once
```

Topics del mapa semantico:

```bash
ros2 topic list | grep semantics
ros2 topic echo /semantics/map/traversability --once
```

Visualizacion:

```bash
rviz2 -d ~/go2_ws/install/ros2_segmentation_vlm/share/ros2_segmentation_vlm/rviz/ros2_segmentation_vlm.rviz
```

## Problemas frecuentes

- El nodo ROS no conecta con el servidor: comprueba que `segmentation_server.py` esta en marcha y que `host`/`port` coinciden.
- El servidor tarda al arrancar: es normal si tiene que cargar SAM3 y mover el modelo a GPU.
- No aparece salida segmentada: verifica que el topic de entrada publica imagenes y que el servidor recibio primero el mensaje de configuracion.
- Error con `cv_bridge` y NumPy: usa los launch files del paquete o exporta `PYTHONNOUSERSITE=1` antes de ejecutar nodos ROS.
- No se genera mapa semantico: revisa que RTAB-Map publique `MapData` con imagen RGB, `CameraInfo`, `local_transform` y poses de grafo.
- No se publica traversabilidad: la nube `/semantics/cloud` debe contener el campo `traversability`.
