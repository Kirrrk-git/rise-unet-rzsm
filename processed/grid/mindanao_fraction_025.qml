<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" opacity="0.75" band="1" classificationMin="0" classificationMax="1">
      <rasterTransparency>
        <singleValuePixelList>
          <pixelListEntry min="0" max="0" percentTransparent="100"/>
        </singleValuePixelList>
      </rasterTransparency>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0" classificationMode="1">
          <colorramp type="gradient" name="custom">
            <prop k="color1" v="254,240,217,0"/>
            <prop k="color2" v="179,0,0,220"/>
            <prop k="stops" v="0.25;253,204,138,100:0.50;252,141,89,160:0.75;227,74,51,200"/>
          </colorramp>
          <item value="0.00" label="0.0 (Ocean Buffer - Transparent)" color="#fef0d9" alpha="0"/>
          <item value="0.25" label="0.25 (Coastal Islet)" color="#fdcc8a" alpha="100"/>
          <item value="0.50" label="0.50 (Evaluation Threshold)" color="#fc8d59" alpha="160"/>
          <item value="0.75" label="0.75 (Coastal Land)" color="#e34a33" alpha="200"/>
          <item value="1.00" label="1.00 (Interior Land)" color="#b30000" alpha="220"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeGreen="128" colorizeOn="0" colorizeRed="255" colorizeBlue="128" saturation="0" grayscaleMode="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
