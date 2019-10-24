/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Julien Chicha
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

#ifndef CAFFE2LIBSTATE_H
#define CAFFE2LIBSTATE_H

#include "utils/apitools.h"

// Create getters, setters and check callbacks
#define _REGISTER_CONFIG(type, name, def)				\
									\
private:								\
									\
 type _##name##_default = def;						\
 type _##name##_current = def;						\
 type _##name##_last;							\
 bool name##_changed() const {						\
   return _##name##_last != _##name##_current;				\
 }									\
 void name##_backup() {							\
   _##name##_last = _##name##_current;					\
 }									\
 void name##_reset() {							\
   _##name##_current = _##name##_default;				\
 }									\
 void init_##name() {							\
   _changed.push_back(&Caffe2LibState::name##_changed);			\
   _backup.push_back(&Caffe2LibState::name##_backup);			\
   _reset.push_back(&Caffe2LibState::name##_reset);			\
 }									\
									\
public:						                        \
									\
 inline const type &name() const {					\
   return _##name##_current;						\
 }									\
 void set_##name(const type &value) {					\
   _##name##_current = value;						\
 }									\
 void set_default_##name(const type &value) {				\
   _##name##_default = value;						\
 }

// Special setters using an APIData
#define REGISTER_CONFIG(type, name, def)				\
 _REGISTER_CONFIG(type, name, def)					\
 void set_##name(const APIData &ad, bool force_get=false) {		\
   if (force_get || ad.has(#name)) {					\
     set_##name(ad.get(#name).get<type>());				\
   }									\
 }									\
 void set_default_##name(const APIData &ad, bool force_get=false) {	\
   if (force_get || ad.has(#name)) {					\
     set_default_##name(ad.get(#name).get<type>());			\
   }									\
 }

// Even more special setters using an APIData
#define REGISTER_CONFIG_FLOAT(name, def)				\
 _REGISTER_CONFIG(float, name, def)					\
 void set_##name(const APIData &ad, bool force_get=false) {		\
   if (force_get || ad.has(#name)) {					\
     apitools::get_float(ad, #name, _##name##_current);			\
   }									\
 }									\
 void set_default_##name(const APIData &ad, bool force_get=false) {	\
   if (force_get || ad.has(#name)) {					\
     apitools::get_float(ad, #name, _##name##_default);			\
   }									\
 }

namespace dd {

  /**
   * \brief Contains informations about the current configuration of the nets :
   *           - training or prediction
   *           - cpu or gpu
   *           - gpu ids
   *           - ...
   *        Each one has a getter and two setter (current value, and default value).
   *        (Note that setters can use an APIData to retrieve the value themselves)
   *        The goals of this class are to allow a simple flag management,
   *        and to provide a quick way to know if the nets need
   *        to be reconfigured between two api call.
   */
  class Caffe2LibState {

  private:

    bool _force_init = true;
    std::vector<bool(Caffe2LibState::*)()const> _changed;
    std::vector<void(Caffe2LibState::*)()> _backup;
    std::vector<void(Caffe2LibState::*)()> _reset;

    // Declare here every parameter needed to configure a net
    // (type, name, default value)

    REGISTER_CONFIG(bool, is_gpu, false);
    REGISTER_CONFIG(bool, is_training, false);
    REGISTER_CONFIG(bool, is_testing, false);
    REGISTER_CONFIG(bool, resume, false);
    REGISTER_CONFIG(std::vector<int>, gpu_ids, {0});

    REGISTER_CONFIG(std::string, extract_layer, "");
    REGISTER_CONFIG(bool, bbox, false);
    REGISTER_CONFIG(bool, mask, false);

    REGISTER_CONFIG(std::string, lr_policy, "fixed");
    REGISTER_CONFIG_FLOAT(base_lr, 0.01);
    REGISTER_CONFIG(int, stepsize, 1);
    REGISTER_CONFIG_FLOAT(gamma, 1);
    REGISTER_CONFIG_FLOAT(power, 1);
    REGISTER_CONFIG(int, max_iter, -1);

    REGISTER_CONFIG(std::string, solver_type, "sgd");
    REGISTER_CONFIG_FLOAT(momentum, -1.f);
    REGISTER_CONFIG_FLOAT(rms_decay, -1.f);

  public:

    // For every declared parameter call the corresponding init function
    Caffe2LibState() {

      init_is_gpu();
      init_is_training();
      init_is_testing();
      init_resume();
      init_gpu_ids();

      init_extract_layer();
      init_bbox();
      init_mask();

      init_lr_policy();
      init_base_lr();
      init_stepsize();
      init_gamma();
      init_power();
      init_max_iter();

      init_solver_type();
      init_momentum();
      init_rms_decay();
    }

    /**
     * \brief Is the current configuration different from the last backuped one ?
     * @return True they differ and False otherwise
     */
    bool changed() const {
      bool b = _force_init;
      for (auto f : _changed) {
	b |= (this->*f)();
      }
      return b;
    }

    /**
     * \brief Put the configuration in an 'Uninitialized' state
     *        until the backup() function is called.
     *        Should be used when the nets are in an unstable state
     *        (e.g. an unexpected error occured during the initialization)
     *        and they need to be reconfigured before being used.
     */
    void force_init() {
      _force_init = true;
    }

    /**
     * \brief Tag the current configuration as being the last configuration used.
     */
    void backup() {
      _force_init = false;
      for (auto f : _backup) {
	(this->*f)();
      }
    }

    /**
     * \brief Set current configuration to the default one.
     */
    void reset() {
      for (auto f : _reset) {
	(this->*f)();
      }
    }

  };
}

#undef _REGISTER_CONFIG
#undef REGISTER_CONFIG
#undef REGISTER_CONFIG_FLOAT

#endif
