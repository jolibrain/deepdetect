/**
 * DeepDetect
 * Copyright (c) 2019 Emmanuel Benazera
 * Author: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CHAIN_ACTIONS_H
#define CHAIN_ACTIONS_H

#include "apidata.h"
#include "chain.h"
#include <memory>
#include "dd_spdlog.h"

namespace dd
{

  class ActionBadParamException : public std::exception
  {
  public:
    ActionBadParamException(const std::string &s) : _s(s)
    {
    }
    ~ActionBadParamException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  class ActionInternalException : public std::exception
  {
  public:
    ActionInternalException(const std::string &s) : _s(s)
    {
    }
    ~ActionInternalException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  class ChainAction
  {
  public:
    ChainAction(oatpp::Object<DTO::ChainCall> call_dto,
                const std::shared_ptr<spdlog::logger> chain_logger)
        : _chain_logger(chain_logger)
    {
      _action_id = call_dto->id;
      _action_type = call_dto->action->type;
      _params = call_dto->action->parameters;
    }

    ~ChainAction()
    {
    }

    std::string genid(const std::string &uri, const std::string &local_id)
    {
      std::string str = uri + local_id;
      return std::to_string(std::hash<std::string>{}(str));
    }

    /** Apply an action to a model output.
     * \param model_out Output of the previous model.
     * \param cdata Chain data object containing all the output of previous
     * models and actions. This method should add the action result to cdata
     * using the `add_action_data()` method. If the action creates data that
     * should appear in the result, they are stored in the "output" field. If
     * the action creates data that should be used by the next model, they are
     * stored in other fields, as handled in `services.h:chain_service()`.
     * */
    void apply(APIData &model_out, ChainData &cdata);

    std::string _action_id;
    std::string _action_type;
    oatpp::Object<DTO::ChainActionParams> _params;
    bool _in_place = false;
    std::shared_ptr<spdlog::logger> _chain_logger;
  };

  class ImgsCropAction : public ChainAction
  {
  public:
    ImgsCropAction(oatpp::Object<DTO::ChainCall> call_dto,
                   const std::shared_ptr<spdlog::logger> chain_logger)
        : ChainAction(call_dto, chain_logger)
    {
    }

    ~ImgsCropAction()
    {
    }

    void apply(APIData &model_out, ChainData &cdata);
  };

  class ImgsRotateAction : public ChainAction
  {
  public:
    ImgsRotateAction(oatpp::Object<DTO::ChainCall> call_dto,
                     const std::shared_ptr<spdlog::logger> chain_logger)
        : ChainAction(call_dto, chain_logger)
    {
    }

    ~ImgsRotateAction()
    {
    }

    void apply(APIData &model_out, ChainData &cdata);
  };

  /** Recompose origin image with inference result, for example in crop + GAN
   * pipeline */
  class ImgsCropRecomposeAction : public ChainAction
  {
  public:
    ImgsCropRecomposeAction(oatpp::Object<DTO::ChainCall> call_dto,
                            const std::shared_ptr<spdlog::logger> chain_logger)
        : ChainAction(call_dto, chain_logger)
    {
    }

    ~ImgsCropRecomposeAction()
    {
    }

    void apply(APIData &model_out, ChainData &cdata);
  };

  class ImgsDrawBBoxAction : public ChainAction
  {
  public:
    ImgsDrawBBoxAction(oatpp::Object<DTO::ChainCall> call_dto,
                       const std::shared_ptr<spdlog::logger> chain_logger)
        : ChainAction(call_dto, chain_logger)
    {
    }

    ~ImgsDrawBBoxAction()
    {
    }

    void apply(APIData &model_out, ChainData &cdata);
  };

  class ClassFilter : public ChainAction
  {
  public:
    ClassFilter(oatpp::Object<DTO::ChainCall> call_dto,
                const std::shared_ptr<spdlog::logger> chain_logger)
        : ChainAction(call_dto, chain_logger)
    {
      _in_place = true;
    }
    ~ClassFilter()
    {
    }

    void apply(APIData &model_out, ChainData &cdata);
  };

  template <typename T>
  inline void apply_action(oatpp::Object<DTO::ChainCall> call_dto,
                           APIData &model_out, ChainData &cdata,
                           const std::shared_ptr<spdlog::logger> &chain_logger)
  {
    T act(call_dto, chain_logger);
    act.apply(model_out, cdata);
  }

  typedef std::function<void(oatpp::Object<DTO::ChainCall>, APIData &,
                             ChainData &,
                             const std::shared_ptr<spdlog::logger> &)>
      action_function;

  class ChainActionFactory
  {
  public:
    static void *add_chain_action(const std::string &action_name,
                                  const action_function &func);

  private:
    static std::unordered_map<std::string, action_function> &
    get_action_registry();

  public:
    ChainActionFactory(APIData adc)
        : _call_dto(adc.createSharedDTO<DTO::ChainCall>())
    {
    }

    ChainActionFactory(oatpp::Object<DTO::ChainCall> call_dto)
        : _call_dto(call_dto)
    {
    }
    ~ChainActionFactory()
    {
    }

    void apply_action(const std::string &action_type, APIData &model_out,
                      ChainData &cdata,
                      const std::shared_ptr<spdlog::logger> &chain_logger);

    oatpp::Object<DTO::ChainCall> _call_dto;
  };

  /** Add an action type to DeepDetect chains. The action transforms the output
   * of a model. This output can then be used by another model or embedded in
   * the API result. */
#define CHAIN_ACTION(ActionName, ActionType)                                  \
  namespace                                                                   \
  {                                                                           \
    void *_register_action_##ActionType                                       \
        = ChainActionFactory::add_chain_action(ActionName,                    \
                                               apply_action<ActionType>);     \
  }

} // end of namespace

#endif
