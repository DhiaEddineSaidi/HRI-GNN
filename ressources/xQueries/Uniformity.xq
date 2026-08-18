for $doc in collection()
for $panel at $panel_id in $doc//Panel
for $layout in $panel/Struct/Component/Layout
where $layout/@Uniformity = '3'

let $entities := $layout/Entity
let $different_sizes  := count(distinct-values($entities/@Size)) > 1
let $different_colors := count(distinct-values($entities/@Color)) > 1
let $different_angles := count(distinct-values($entities/@Angle)) > 1
let $different_types  := count(distinct-values($entities/@Type)) > 1

let $has_anomaly := $different_sizes or $different_colors or $different_angles or $different_types

(: On déclenche l'alerte si une anomalie existe ET que le numéro du PANEL est <= 8 :)
where $has_anomaly and $panel_id > 8

return (
  concat("Fichier : ", db:path($panel)),
  concat("ALERTE - Anomalie détectée dans le PANEL N° : ", $panel_id),
  $layout
)
