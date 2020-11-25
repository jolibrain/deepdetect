#ifndef BASE_OPERATORS_H
#define BASE_OPERATORS_H

#include "basegraph.h"

namespace dd
{

  namespace graph

  {
    namespace utils
    {
      /**
       * \brief get changed dimensions indexes
       */
      std::vector<std::vector<int>>
      get_changed_dims(const std::vector<std::vector<int>> &new_dims,
                       const std::vector<std::vector<int>> &old_dims);

      /**
       * \brief checks if number of blobs have changed
       */
      bool nblobs_changed(const std::vector<std::vector<int>> &new_dims,
                          const std::vector<std::vector<int>> &old_dims);

      /**
       * \brief checks if number of dimensions in all blobs have changed
       */
      bool blob_ndims_changed(const std::vector<std::vector<int>> &new_dims,
                              const std::vector<std::vector<int>> &old_dims);

      /**
       * \brief checks if number of input blob have changed
       */
      bool
      n_in_blob_changed(VertexProperty &v,
                        const std::vector<std::vector<int>> &new_inputsdims);
      /**
       * \brief checks if number of output blob have changed
       */
      bool
      n_out_blob_changed(VertexProperty &v,
                         const std::vector<std::vector<int>> &new_outputsdims);

      /**
       * \brief checks if number of dimensions in input blobs chave changed
       */
      bool in_blob_ndims_changed(
          VertexProperty &v,
          const std::vector<std::vector<int>> &new_inputsdims);

      /**
       * \brief checks if number of dimensions in output blobs chave changed
       */
      bool out_blob_ndims_changed(
          VertexProperty &v,
          const std::vector<std::vector<int>> &new_outputsdims);

      /**
       * \brief checks if shape have changed (combines all above)
       */
      bool
      general_shape_changed(VertexProperty &v,
                            const std::vector<std::vector<int>> &inputsdims,
                            const std::vector<std::vector<int>> &outputsdims);
    }

    /**
     * \brief contains all operator specifics
     */
    namespace op
    {
      /**
       * \brief contains all RNN/LSTM  specifics
       */
      namespace rnn
      {

        /**
         * \brief computes output dims for RNN / LTSM
         */
        void
        compute_outputs_dims(const VertexProperty &v,
                             const std::vector<std::vector<int>> &inputsdims,
                             std::vector<std::vector<int>> &outputsdims);

        /**
         * \brief checks if dims changed and if realloc is needed, for RNN /
         * LSTM
         */
        void
        update_alloc_status(VertexProperty &v,
                            const std::vector<std::vector<int>> &inputsdims,
                            const std::vector<std::vector<int>> &outputsdims);
      }

      /**
       * \brief contains all InnerProduct/Linear  specifics
       */
      namespace ip
      {
        /**
         * \brief computes output dims for InnerProduct/Linear
         */
        void
        compute_outputs_dims(const VertexProperty &v,
                             const std::vector<std::vector<int>> &inputsdims,
                             std::vector<std::vector<int>> &outputsdims);

        /**
         * \brief checks if dims changed and if realloc is needed, for
         * InnerProduct / Linear
         */
        void
        update_alloc_status(VertexProperty &v,
                            const std::vector<std::vector<int>> &inputsdims,
                            const std::vector<std::vector<int>> &outputsdims);
      }

      /**
       * \brief contains all Tile/Repeat  specifics
       */
      namespace tile
      {
        /**
         * \brief computes output dims for Tile/Repeat
         */
        void
        update_alloc_status(VertexProperty &v,
                            const std::vector<std::vector<int>> &inputsdims,
                            const std::vector<std::vector<int>> &outputsdims);
        /**
         * \brief checks if dims changed and if realloc is needed, for
         * Tile/Repeat
         */
        void
        compute_outputs_dims(const VertexProperty &v,
                             const std::vector<std::vector<int>> &inputsdims,
                             std::vector<std::vector<int>> &outputsdims);
      }

      /**
       * \brief contains all relu  specifics
       */
      namespace relu
      {
        /**
         * \brief computes output dims for Tile/Repeat
         */
        void
        update_alloc_status(VertexProperty &v,
                            const std::vector<std::vector<int>> &inputsdims,
                            const std::vector<std::vector<int>> &outputsdims);
        /**
         * \brief checks if dims changed and if realloc is needed, for
         * Tile/Repeat
         */
        void
        compute_outputs_dims(const VertexProperty &v,
                             const std::vector<std::vector<int>> &inputsdims,
                             std::vector<std::vector<int>> &outputsdims);
      }

      /**
       * \brief dispatches specific functions wrt type contained in graph
       */
      namespace dispatcher
      {
        /**
         * \brief dispatches output dimensions computation to operator specific
         * functions
         */
        void
        compute_outputs_dims(const VertexProperty &v,
                             const std::vector<std::vector<int>> &inputsdims,
                             std::vector<std::vector<int>> &outputsdims);

        /**
         * \brief dispatches check for realloc needto operator specific
         * functions
         */
        void
        update_alloc_status(VertexProperty &v,
                            const std::vector<std::vector<int>> &inputsdims,
                            const std::vector<std::vector<int>> &outputsdims);
      }
    }
  }
}
#endif
