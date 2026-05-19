import json
import jsonschema

from napytau.import_export.import_export_error import ImportExportError
from napytau.import_export.model.dataset import DataSet
from napytau.util.coalesce import coalesce

_SCHEMA = """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NapytauPersistedData",
  "description": "Research data conforming to the Napytau schema",
  "type": "object",
  "properties": {
    "relativeVelocity": {
      "type": "number",
      "description": "The relative velocity of the particles",
      "minimum": 0,
      "exclusiveMaximum": 1
    },
    "relativeVelocityError": {
      "type": "number",
      "description": "The error in the relative velocity of the particles",
      "minimum": 0,
      "exclusiveMaximum": 1
    },
    "datapoints": {
      "type": "array",
      "description": "The data points measured in the experiment",
      "items": {
        "type": "object",
        "properties": {
          "distance": {
            "type": "number",
            "description": "The distance of the particles",
            "minimum": 0
          },
          "distanceError": {
            "type": "number",
            "description": "The error in the distance between the particles",
            "minimum": 0
          },
          "calibration": {
            "type": "number",
            "description": "The calibration factor for the data point",
            "minimum": 0
          },
          "calibrationError": {
            "type": "number",
            "description": "The error in the calibration factor for the data point",
            "minimum": 0
          },
          "shiftedIntensity": {
            "type": "number",
            "description": "The shifted intensity of the data point",
            "minimum": 0
          },
          "shiftedIntensityError": {
            "type": "number",
            "description": "The error in the shifted intensity of the data point",
            "minimum": 0
          },
          "unshiftedIntensity": {
            "type": "number",
            "description": "The unshifted intensity of the data point",
            "minimum": 0
          },
          "unshiftedIntensityError": {
            "type": "number",
            "description": "The error in the unshifted intensity of the data point",
            "minimum": 0
          },
          "feedingShiftedIntensity": {
            "type": "number",
            "description": "The feeding shifted intensity of the data point",
            "minimum": 0
          },
          "feedingShiftedIntensityError": {
            "type": "number",
            "description": "Error in the feeding shifted intensity of the data point",
            "minimum": 0
          },
          "feedingUnshiftedIntensity": {
            "type": "number",
            "description": "The measured feeding unshifted intensity of the data point",
            "minimum": 0
          },
          "feedingUnshiftedIntensityError": {
            "type": "number",
            "description": "Error in the feeding unshifted intensity of the data point",
            "minimum": 0
          }
        },
        "required": [
          "distance",
          "distanceError",
          "calibration",
          "calibrationError",
          "shiftedIntensity",
          "shiftedIntensityError",
          "unshiftedIntensity",
          "unshiftedIntensityError"
        ]
      }
    },
    "setups": {
      "type": "array",
      "description": "A list of setups previously configured in Napytau",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the setup"
          },
          "tauFactor": {
            "type": "number",
            "description": "The tau factor of the setup",
            "minimum": 0
          },
          "polynomialCount": {
            "type": "number",
            "description": "The polynomial count of the setup",
            "minimum": 0
          },
          "datapointSetups": {
            "type": "array",
            "description": "Setup data for each datapoint of the experiment",
            "items": {
              "type": "object",
              "properties": {
                "distance": {
                  "type": "number",
                  "description": "Distance of the particles at time of the data point",
                  "minimum": 0
                },
                "active": {
                  "type": "boolean",
                  "description": "Whether the data point is active or not"
                }
              },
              "required": [
                "distance",
                "active"
              ]
            }
          },
          "samplingPoints": {
          "type": "array",
          "description": "The sampling points for the setup",
            "items": {
                "type": "number",
                "minimum": 0
            }
          }
        },
        "required": [
          "name",
          "tauFactor",
          "polynomialCount",
          "datapointSetups",
          "samplingPoints"
        ]
      }
    }
  },
  "required": [
    "relativeVelocity",
    "relativeVelocityError",
    "datapoints",
    "setups"
  ]
}
"""


class NapytauFormatJsonService:
    @staticmethod
    def parse_json_data(json_string: str) -> dict:
        """
        Parses the provided json data into a dictionary
        """

        try:
            json_data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ImportExportError(f"Provided json data could not be parsed: {e}")

        return dict(json_data)

    @staticmethod
    def validate_against_schema(json_data: dict) -> bool:
        """
        Validates the provided json data against the napytau json schema
        """

        schema = json.loads(_SCHEMA)

        try:
            jsonschema.validate(instance=json_data, schema=schema)
        except jsonschema.ValidationError as e:
            raise ImportExportError(
                f"Provided json data does not match the napytau json schema: {e}"
            )

        return True

    @staticmethod
    def create_calculation_data_json_string(dataset: DataSet) -> str:
        """
        Creates a json string from the provided dataset
        """

        try:
            json_data = json.dumps(
                obj={
                    "tauFactor": coalesce(dataset.get_tau_factor()),
                    "weightedMeanTau": coalesce(dataset.get_weighted_mean_tau()).value,
                    "weightedMeanTauError": coalesce(
                        dataset.get_weighted_mean_tau()
                    ).error,  # noqa E501
                    "datapoints": list(
                        map(
                            lambda datapoint: {
                                "distance": datapoint.distance.value,
                                "distanceError": datapoint.distance.error,
                                "tau": coalesce(datapoint.tau).value,
                                "tauError": coalesce(datapoint.tau).error,
                                "shiftedIntensity": coalesce(
                                    datapoint.shifted_intensity
                                ).value,  # noqa E501
                                "shiftedIntensityError": coalesce(
                                    datapoint.shifted_intensity
                                ).error,  # noqa E501
                                "unshiftedIntensity": coalesce(
                                    datapoint.unshifted_intensity
                                ).value,  # noqa E501
                                "unshiftedIntensityError": coalesce(
                                    datapoint.unshifted_intensity
                                ).error,  # noqa E501
                            }
                            if datapoint.feeding_shifted_intensity is None
                            else {
                                "distance": datapoint.distance.value,
                                "distanceError": datapoint.distance.error,
                                "tau": coalesce(datapoint.tau).value,
                                "tauError": coalesce(datapoint.tau).error,
                                "shiftedIntensity": coalesce(
                                    datapoint.shifted_intensity
                                ).value,  # noqa E501
                                "shiftedIntensityError": coalesce(
                                    datapoint.shifted_intensity
                                ).error,  # noqa E501
                                "unshiftedIntensity": coalesce(
                                    datapoint.unshifted_intensity
                                ).value,  # noqa E501
                                "unshiftedIntensityError": coalesce(
                                    datapoint.unshifted_intensity
                                ).error,  # noqa E501
                                "feedingShiftedIntensity": coalesce(
                                    datapoint.feeding_shifted_intensity
                                ).value,  # noqa E501
                                "feedingShiftedIntensityError": coalesce(
                                    datapoint.feeding_shifted_intensity
                                ).error,  # noqa E501
                                "feedingUnshiftedIntensity": coalesce(
                                    datapoint.feeding_unshifted_intensity
                                ).value,  # noqa E501
                                "feedingUnshiftedIntensityError": coalesce(
                                    datapoint.feeding_unshifted_intensity
                                ).error,  # noqa E501
                            },
                            dataset.get_datapoints(),
                        ),
                    ),
                    "samplingPoints": dataset.get_sampling_points(),
                    "polynomials": list(
                        map(
                            lambda polynomial: {
                                "coefficients": polynomial.coefficients,
                                "knots": polynomial.knots,
                                "degree": polynomial.degree,
                            },
                            coalesce(dataset.get_polynomials()),
                        ),
                    ),
                },
                indent=2,
            )
        except ValueError as e:
            raise ImportExportError(
                f"Provided dataset could not be converted to json: {e}"
            )

        return json_data
