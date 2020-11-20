#include "baseoperators.h"

namespace dd
{

  namespace basegraph

  {

    std::vector<std::vector<int>>
    utils::get_changed_dims(const std::vector<std::vector<int>> &new_dims,
                            const std::vector<std::vector<int>> &old_dims)
    {
      std::vector<std::vector<int>> all_changed;
      for (unsigned int i = 0; i < new_dims.size(); ++i)
        {
          std::vector<int> changed;
          for (unsigned int j = 0; j < new_dims[i].size(); ++j)
            if (old_dims[i][j] != new_dims[i][j])
              changed.push_back(j);
          all_changed.push_back(changed);
        }
      return all_changed;
    }

    bool utils::nblobs_changed(const std::vector<std::vector<int>> &new_dims,
                               const std::vector<std::vector<int>> &old_dims)
    {
      return new_dims.size() != old_dims.size();
    }

    bool
    utils::blob_ndims_changed(const std::vector<std::vector<int>> &new_dims,
                              const std::vector<std::vector<int>> &old_dims)
    {
      for (unsigned int i = 0; i < new_dims.size(); ++i)
        if (new_dims[i].size() != old_dims[i].size())
          return true;
      return false;
    }

    bool utils::n_in_blob_changed(
        BaseGraph::VertexProperty &v,
        const std::vector<std::vector<int>> &new_inputsdims)
    {
      return nblobs_changed(v.inputsdims, new_inputsdims);
    }

    bool utils::n_out_blob_changed(
        BaseGraph::VertexProperty &v,
        const std::vector<std::vector<int>> &new_outputsdims)
    {
      return nblobs_changed(v.outputsdims, new_outputsdims);
    }

    bool utils::in_blob_ndims_changed(
        BaseGraph::VertexProperty &v,
        const std::vector<std::vector<int>> &new_inputsdims)
    {
      return blob_ndims_changed(new_inputsdims, v.inputsdims);
    }

    bool utils::out_blob_ndims_changed(
        BaseGraph::VertexProperty &v,
        const std::vector<std::vector<int>> &new_outputsdims)
    {
      return blob_ndims_changed(new_outputsdims, v.outputsdims);
    }

    bool utils::general_shape_changed(
        BaseGraph::VertexProperty &v,
        const std::vector<std::vector<int>> &inputsdims,
        const std::vector<std::vector<int>> &outputsdims)
    {
      return n_in_blob_changed(v, inputsdims)
             || n_out_blob_changed(v, outputsdims)
             || in_blob_ndims_changed(v, inputsdims)
             || out_blob_ndims_changed(v, outputsdims);
    }

    namespace op
    {

      void rnn::compute_outputs_dims(
          const BaseGraph::VertexProperty &v,
          const std::vector<std::vector<int>> &inputsdims,
          std::vector<std::vector<int>> &outputsdims)
      {
        if (inputsdims.size() > 1)
          throw new BaseGraphException(
              "LSTM/RNN layer has more than one input!!");

        std::vector<int> outputdims;

        // main output
        outputdims.push_back(inputsdims[0][0]); // batchsize
        outputdims.push_back(inputsdims[0][1]); // timesteps
        outputdims.push_back(v.num_output);     // hidden_size
        outputsdims.push_back(outputdims);

        // hidden value
        outputdims.push_back(inputsdims[0][0]); // batchsize
        outputdims.push_back(v.num_output);     // hidden_size
        outputsdims.push_back(outputdims);

        // cell value
        outputdims.push_back(inputsdims[0][0]); // batchsize
        outputdims.push_back(v.num_output);     // hidden_size
        outputsdims.push_back(outputdims);
      }

      void rnn::update_alloc_status(
          BaseGraph::VertexProperty &v,
          const std::vector<std::vector<int>> &inputsdims,
          const std::vector<std::vector<int>> &outputsdims)
      {
        if (inputsdims.size() > 1)
          throw new BaseGraphException(
              "LSTM/RNN layer has more than one input!!");

        if (basegraph::utils::general_shape_changed(v, inputsdims,
                                                    outputsdims))
          {
            v.alloc_needed = true;
            return;
          }

        std::vector<std::vector<int>> changed_inputs_dims
            = basegraph::utils::get_changed_dims(v.inputsdims, inputsdims);

        if (changed_inputs_dims[0][0] > 1)
          // 0 is batch size, 1 is timestep => no need to realloc
          v.alloc_needed = true;

        std::vector<std::vector<int>> changed_outputs_dims
            = basegraph::utils::get_changed_dims(v.outputsdims, outputsdims);
        if (changed_inputs_dims[0][0] > 1)
          // 0 is batch size, 1 is timestep => no need to realloc
          v.alloc_needed = true;
      }

      void
      ip::compute_outputs_dims(const BaseGraph::VertexProperty &v,
                               const std::vector<std::vector<int>> &inputsdims,
                               std::vector<std::vector<int>> &outputsdims)
      {
        std::vector<int> outputdims;
        outputdims.push_back(inputsdims[0][0]); // batchsize
        unsigned int axis = (v.axis == -1) ? 1 : v.axis;
        for (unsigned int i = 1; i < axis; ++i)
          outputdims.push_back(inputsdims[0][i]);
        outputdims.push_back(v.num_output);

        outputsdims.push_back(outputdims);
      }

      void
      ip::update_alloc_status(BaseGraph::VertexProperty &v,
                              const std::vector<std::vector<int>> &inputsdims,
                              const std::vector<std::vector<int>> &outputsdims)
      {
        if (basegraph::utils::general_shape_changed(v, inputsdims,
                                                    outputsdims))
          {
            v.alloc_needed = true;
            return;
          }

        std::vector<std::vector<int>> changed_inputs_dims
            = basegraph::utils::get_changed_dims(v.inputsdims, inputsdims);
        if (changed_inputs_dims[0][0] > 0)
          // 0 is batch size => no need to realloc
          v.alloc_needed = true;

        std::vector<std::vector<int>> changed_outputs_dims
            = basegraph::utils::get_changed_dims(v.outputsdims, outputsdims);
        if (changed_outputs_dims[0][0] > 0)
          // 0 is batch size => no need to realloc
          v.alloc_needed = true;
      }

      void tile::compute_outputs_dims(
          const BaseGraph::VertexProperty &v,
          const std::vector<std::vector<int>> &inputsdims,
          std::vector<std::vector<int>> &outputsdims)
      {
        (void)v;
        std::vector<int> outputdims;
        for (unsigned int i = 0; i < inputsdims[0].size(); ++i)
          outputdims.push_back(inputsdims[0][i]);
        outputdims[v.axis] = outputdims[v.axis] * v.tiles;

        outputsdims.push_back(outputdims);
      }

      void tile::update_alloc_status(
          BaseGraph::VertexProperty &v,
          const std::vector<std::vector<int>> &inputsdims,
          const std::vector<std::vector<int>> &outputsdims)
      {
        (void)v;
        (void)inputsdims;
        (void)outputsdims;
      }

      void relu::compute_outputs_dims(
          const BaseGraph::VertexProperty &v,
          const std::vector<std::vector<int>> &inputsdims,
          std::vector<std::vector<int>> &outputsdims)
      {
        (void)v;
        outputsdims = inputsdims;
      }

      void relu::update_alloc_status(
          BaseGraph::VertexProperty &v,
          const std::vector<std::vector<int>> &inputsdims,
          const std::vector<std::vector<int>> &outputsdims)
      {
        (void)v;
        (void)inputsdims;
        (void)outputsdims;
      }

      void dispatcher::compute_outputs_dims(
          const BaseGraph::VertexProperty &v,
          const std::vector<std::vector<int>> &inputsdims,
          std::vector<std::vector<int>> &outputsdims)
      {

        std::string type = v.type;
        if (type == "LSTM" || type == "RNN")
          rnn::compute_outputs_dims(v, inputsdims, outputsdims);
        else if (type == "InnerProduct")
          ip::compute_outputs_dims(v, inputsdims, outputsdims);
        else if (type == "Tile")
          tile::compute_outputs_dims(v, inputsdims, outputsdims);
        else if (type == "ReLU")
          relu::compute_outputs_dims(v, inputsdims, outputsdims);
        else
          throw BaseGraphException(
              "compute output dims : unknown operator type " + type);
      }

      void dispatcher::update_alloc_status(
          BaseGraph::VertexProperty &v,
          const std::vector<std::vector<int>> &inputsdims,
          const std::vector<std::vector<int>> &outputsdims)
      {
        std::string type = v.type;
        if (type == "LSTM" || type == "RNN")
          rnn::update_alloc_status(v, inputsdims, outputsdims);
        else if (type == "InnerProduct")
          ip::update_alloc_status(v, inputsdims, outputsdims);
        else if (type == "Tile")
          tile::update_alloc_status(v, inputsdims, outputsdims);
        else if (type == "ReLU")
          relu::update_alloc_status(v, inputsdims, outputsdims);
        else
          throw BaseGraphException(
              "update alloc status  : unknown operator type " + type);
      }
    }
  }
}
