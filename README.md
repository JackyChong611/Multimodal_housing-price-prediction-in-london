# Multimodal Deep Learning for London House-Price Prediction

## 🏙️ Overview 
London house prices are notoriously difficult to predict because many intangible and visual factors aren't captured by traditional datasets. Features like a home's curb appeal, interior condition, natural lighting, and neighborhood ambiance can strongly influence price but are often invisible in purely tabular data.

This project addresses that gap by training a multimodal deep learning model that integrates diverse data sources to predict property prices. We combine:

● **Rightmove listings** – Structured data about each property (e.g. size, type, location).

● **Police.UK crime statistics** – Neighborhood safety indicators.

● **Listing descriptions** – Unstructured text from the property listing.

● **Property photographs** – Images of the property's interior and exterior.

● **Satellite imagery** – Aerial view of the property and surrounding area.

● **Google Street View** – 360° street-level panoramas of the property’s vicinity.

**Model Architecture**: Each modality is processed with a dedicated sub-network. We use CNNs (ResNet-18) for images (property photos, satellite, Street View), a BERT-based transformer for textual descriptions, and a feed-forward MLP for tabular features. These outputs are concatenated in a late-fusion layer, enabling the model to learn a joint representation before the final price regression layer.
