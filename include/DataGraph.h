#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "DataEnums.h"
#include "tvm_api.h"

namespace taskvm
{
  /* todo or consider
     - should we merge DataSource, DataUser and DataNode in a single class
     - what do we allow to do after construction? For now, outputs and updates are
       declare at construction time and cannot be added after, while inputs can be
       added at any time. It makes sense for inputs because we can consider DataUser
       that are aggregating the outputs of many function. It could make sense for
       outputs and updates if we want a derived function to add outputs/updates to a
       base function. Adding inputs/updates/outputs outside of the constructor makes
       it possible to create loops, so that we need to check that at some point.
     - related to the above issue, do we want to 'lock' a node, i.e. forbid to change
       its i/u/o once it has been added to the graph, so that there is no discrepencies
       between the graph and the nodes? Alternatively we can trust the programmer or
       trigger a reprocessing of the graph.
     - it seems that we are defining output and update enums for every (class of)
       DataNode derived function. Should we add a templated layer above DataNode to 
       specialize for these enums? For example, if Function declares two enums Output 
       and Update, it could inherit from DataSourceT<Output, Update> which in turns
       inherits from DataSource. This way, we know the enums and can completely 
       remove UnifiedEnumValue from the exposed API, while keeping it at the heart 
       of the data mechanism and thus retaining a commonn base class. 
       An alternative leading to the same result is to have DataSourceT being 
       templated by Function to force Function to declare Output and Update.
     - for now, errors are not really meaningful when the graph input-update-output is
       ill-formed. Better error report could be done by associating string with the
       enumerations, or being able to iterate over the enumerations.*/


  /** A list of outputs specified as a set of enum values*/
  class TVM_API DataSource
  {
  public:
    template<typename OutputEnum>
    DataSource(const std::vector<OutputEnum>& outputList);
    template<typename OutputEnum>
    DataSource(std::initializer_list<OutputEnum> l);
    template<typename OutputEnum>
    DataSource(OutputEnum output);

    virtual ~DataSource() = default;

    template<typename OutputEnum>
    bool hasOutput(OutputEnum e) const;

  private:
    std::set<internal::UnifiedEnumValue> outputs_;
  };


  /** Essentially a list of inputs, specified as couple (DataSource, enum value), 
    * where enum value is one of the value in the output list of DataSource.
    */
  class TVM_API DataUser
  {
  public:
    virtual ~DataUser() = default;

    template <typename InputEnum>
    void addInput(std::shared_ptr<DataSource> source, InputEnum input);
    template <typename InputEnum>
    void addInput(std::shared_ptr<DataSource> source, std::initializer_list<InputEnum> inputs);

    bool hasSource(const DataSource& source) const;
    template <typename InputEnum>
    bool hasInput(const DataSource& source, InputEnum input);
    const std::set<std::shared_ptr<DataSource>>& getSources() const;
    const std::vector<internal::UnifiedEnumValue>& getInputs(const DataSource& source) const;
    
    /** Returns the shared pointer corresponding to ds, if ds is among the sources of this.
      * Throw otherwise.
      * internal: this method might be questionable. It is a consequence of the choice we
      * made of storing only once a shared_ptr on a source, and use pointer/reference to
      * this source everywhere else.*/
    std::shared_ptr<DataSource> getSharedPtr(DataSource const* ds) const;

  private:
    //it would seem better to store directly pairs of (source,input)
    std::set<std::shared_ptr<DataSource>> sources_;
    std::map<const DataSource*, std::vector<internal::UnifiedEnumValue>> inputs_;
  };

  /** A class processing inputs into outputs through updates.
    * The goal of the class is to store the dependency relations between these
    * inputs, updates and outputs.
    * Three type of dependency are considered:
    * - between outputs and updates (i.e. how outputs depend on updates)
    * - within updates
    * - between updates and inputs.
    */
  class TVM_API DataNode: public DataSource, public DataUser
  {
  public:
    typedef std::vector<internal::UnifiedEnumValue> UpdateList;
    typedef std::vector<std::pair<const DataSource*, internal::UnifiedEnumValue>> InputList;
    typedef std::map<internal::UnifiedEnumValue, UpdateList> OuputDependencies;
    typedef std::map<internal::UnifiedEnumValue, UpdateList> InternalDependencies;
    typedef std::map<internal::UnifiedEnumValue, InputList> InputDependencies;
    
  public:
    template <typename outputEnum, typename UpdateEnum >
    DataNode(const std::vector<outputEnum>& outputList, const std::vector<UpdateEnum>& updateList);

    template <typename outputEnum, typename UpdateEnum >
    DataNode(std::initializer_list<outputEnum> outputList, std::initializer_list<UpdateEnum> updateList);
    
    virtual ~DataNode() = default;

    template <typename UpdateEnum>
    bool hasUpdate(UpdateEnum e) const;

    template <typename UpdateEnum> 
    void update(UpdateEnum e);

    void updateAll();

    /* note: it would make sense for the following methods to be const.
      However, because of the lazy filling of the dependencies, this would require 
      all fillXXXDependencies and addXXXDependency to be const and all 
      XXXDependciesAreSet_ and XXXDependencies_ to be mutable, which makes a lot
      less sense.*/
    const OuputDependencies&    outputDependencies() /*const*/;
    const InternalDependencies& internalDependencies() /*const*/;
    const InputDependencies&    inputDependencies() /*const*/;
    const UpdateList&           outputDependencies(internal::UnifiedEnumValue output) /*const*/;
    const UpdateList&           internalDependencies(internal::UnifiedEnumValue update) /*const*/;
    const InputList&            inputDependencies(internal::UnifiedEnumValue update) /*const*/;

  protected:
    virtual void update_(const internal::UnifiedEnumValue&) = 0;
    
    /** Specify which output depends on which update. Done by repeated calls to
      * addOutputDependency
      */
    virtual void fillOutputDependencies() = 0;

    /** Specify which update depends on which other update (if any). Done by 
      * repeated calls to addInternalDependency
      */
    virtual void fillInternalDependencies() = 0;

    /** Specify which update depends on which input. Done by repeated calls to
      * addInputDependency
      */
    virtual void fillUpdateDependencies() = 0;

    /** Helper method to describe dependencies between an output and an update.*/
    template <typename OutputEnum, typename UpdateEnum>
    void addOutputDependency(OutputEnum output, UpdateEnum udpate);
    template <typename OutputEnum, typename UpdateEnum>
    void addOutputDependency(std::initializer_list<OutputEnum> outputs, UpdateEnum udpate);
    template <typename OutputEnum, typename UpdateEnum>
    void addOutputDependency(OutputEnum output, std::initializer_list<UpdateEnum> updates);

    /** Helper method to describe dependencies between updates.*/
    template <typename UpdateEnum>
    void addInternalDependency(UpdateEnum dependentUpdate, UpdateEnum udpate);

    /** Helper method to describe dependencies between an update and an output.*/
    template <typename UpdateEnum, typename InputEnum>
    void addInputDependency(UpdateEnum dependentUpdate, const DataSource& source, InputEnum input);
    template <typename UpdateEnum, typename InputEnum>
    void addInputDependency(std::initializer_list<UpdateEnum> dependentUpdates, const DataSource& source, InputEnum input);

  private:
    std::set<internal::UnifiedEnumValue>  updates_;
    bool                                  outputDependenciesAreSet_;
    bool                                  internalDependenciesAreSet_;
    bool                                  inputDependenciesAreSet_;
    /** outputDependencies[output] gives the vector of updates output depends on*/
    OuputDependencies                     outputDependencies_;
    /** outputDependencies[update] gives the vector of other updates update depends on*/
    InternalDependencies                  internalDependencies_;
    /** outputDependencies[update] gives the vector of (source,input) update depends on*/
    InputDependencies                     inputDependencies_;
  };

  
  /** Helper class for UpdateGraph and UdaptePlan*/
  struct Update
  {
    std::shared_ptr<DataNode>   nodePtr;
    internal::UnifiedEnumValue  updateId;
  };

  /** Equivalent to what the specialization std::less<Update> would be*/
  struct CompareUpdate
  {
    bool operator()(const Update& u1, const Update& u2) const
    {
      return (u1.nodePtr < u2.nodePtr)
        || (u1.nodePtr == u2.nodePtr && u1.updateId < u2.updateId);
    }
  };

  /** A Direct Acyclic Graph representing the dependencies between updates (where
    * and edge (from, to) means that from depends on to).
    * The update present in the graph depend on the inputs of the DataUser added 
    * to the graph.
    */
  class TVM_API UpdateGraph
  {
  public:
    void add(std::shared_ptr<DataUser> user);

    /** Force the entire graph to be up-to-date.
      * Suppose that for any update, calling it twice in a row does not change the outputs.
      * Warning: this function is for debugging purpose and is extremely inefficient as
      * each update is called n times, where n is the total number of updates.*/
    void forceUpdate() const;

  private:
    /** Add the updates the given input depends on and return the id of these updates. 
      * If the added updates depend on other updates, they will be added reccursively.*/
    std::vector<int> add(std::shared_ptr<DataSource> source, internal::UnifiedEnumValue input);
    /** Add an update and returns its id. 
      * If the added update depends on other updates, they will be added reccursively.*/
    int addUpdate(Update u);
    void addEdge(int from, int to);

    std::map<Update, int, CompareUpdate> updateId_; //update to id
    std::vector<Update*> update_;                   //id to update
    std::vector<std::vector<int>> dependencies_;    //dependencies id of each update (graph edges)
    std::vector<bool> root_;                        //wether or not an update is a root in the graph

    std::map<std::pair<const DataSource*, internal::UnifiedEnumValue>, std::vector<int>> visitedInputs_;
    std::set<std::shared_ptr<DataNode>> nodes_; //set of all DataNode appearing during the graph construction

    friend class UpdatePlan;
  };

  /** An ordered sequence of updates such that an update with index i in the 
    * sequence does no depend on any update with index j>i.
    */
  class TVM_API UpdatePlan
  {
  public:
    UpdatePlan(const UpdateGraph& graph);

    void execute() const;

  private:
    void buildPlan(const UpdateGraph& graph);
    void recursiveBuild(const UpdateGraph& graph, size_t v);

    std::vector<Update> plan_;

    //temp values for building process
    std::vector<size_t> order_;
    std::vector<bool> visited_;
    std::vector<bool> stack_;
  };



  template<typename OutputEnum>
  inline DataSource::DataSource(const std::vector<OutputEnum>& outputList)
  {
    outputs_.insert(outputList.begin(), outputList.end());
  }

  template<typename OutputEnum>
  inline DataSource::DataSource(std::initializer_list<OutputEnum> outputList)
  {
    outputs_.insert(outputList.begin(), outputList.end());
  }

  template<typename OutputEnum>
  inline DataSource::DataSource(OutputEnum output)
  {
    outputs_.insert(output);
  }

  template<typename OutputEnum>
  inline bool DataSource::hasOutput(OutputEnum e) const
  {
    return outputs_.find(e) != outputs_.end();
  }

  template<typename InputEnum>
  inline void DataUser::addInput(std::shared_ptr<DataSource> source, InputEnum e)
  {
    if (source->hasOutput(e))
    {
      sources_.insert(source);
      inputs_[source.get()].push_back(e);
    }
    else
      throw std::logic_error("The source does not provide this output.");
  }

  template<typename InputEnum>
  inline void DataUser::addInput(std::shared_ptr<DataSource> source, std::initializer_list<InputEnum> inputs)
  {
    for (auto i : inputs)
      addInput(source, i);
  }

  template<typename InputEnum>
  inline bool DataUser::hasInput(const DataSource& source, InputEnum input)
  {
    auto& v = inputs_[&source];
    return hasSource(source) && std::find(v.begin(), v.end(), input) != v.end();
  }


  template <typename outputEnum, typename UpdateEnum >
  inline DataNode::DataNode(const std::vector<outputEnum>& outputList, const std::vector<UpdateEnum>& updateList)
    : DataSource(outputList)
    , outputDependenciesAreSet_(false)
    , internalDependenciesAreSet_(false)
    , inputDependenciesAreSet_(false)
  {
    updates_.insert(updateList.begin(), updateList.end());
  }

  template<typename outputEnum, typename UpdateEnum>
  inline DataNode::DataNode(std::initializer_list<outputEnum> outputList, std::initializer_list<UpdateEnum> updateList)
    :DataSource(outputList)
    , outputDependenciesAreSet_(false)
    , internalDependenciesAreSet_(false)
    , inputDependenciesAreSet_(false)
  {
    updates_.insert(updateList.begin(), updateList.end());
  }

  template<typename UpdateEnum>
  inline bool DataNode::hasUpdate(UpdateEnum e) const
  {
    return updates_.find(e) != updates_.end();
  }

  template<typename UpdateEnum>
  inline void DataNode::update(UpdateEnum e)
  {
    if (hasUpdate(e))
      update_(internal::UnifiedEnumValue(e));
    else
      throw std::logic_error("No such update.");
  }

  template<typename OutputEnum, typename UpdateEnum>
  inline void DataNode::addOutputDependency(OutputEnum output, UpdateEnum u)
  {
    if (hasOutput(output))
    {
      if (hasUpdate(u))
        outputDependencies_[internal::UnifiedEnumValue(output)].push_back(u);
      else
        throw std::logic_error("No such update.");
    }
    else
      throw std::logic_error("No such output.");
  }

  template<typename OutputEnum, typename UpdateEnum>
  inline void DataNode::addOutputDependency(std::initializer_list<OutputEnum> outputs, UpdateEnum u)
  {
    for (auto o : outputs)
      addOutputDependency(o, u);
  }

  template<typename OutputEnum, typename UpdateEnum>
  inline void DataNode::addOutputDependency(OutputEnum output, std::initializer_list<UpdateEnum> updates)
  {
    for (auto u : updates)
      addOutputDependency(output, u);
  }

  template<typename UpdateEnum>
  inline void DataNode::addInternalDependency(UpdateEnum dependentUpdate, UpdateEnum u)
  {
    if (hasUpdate(dependentUpdate) && hasUpdate(u))
      internalDependencies_[internal::UnifiedEnumValue(dependentUpdate)].push_back(u);
    else
      throw std::logic_error("No such update.");
  }

  template<typename UpdateEnum, typename InputEnum>
  inline void DataNode::addInputDependency(UpdateEnum dependentUpdate, const DataSource& source, InputEnum input)
  {
    if (hasUpdate(dependentUpdate))
    {
      if (hasSource(source))
      {
        if (hasInput(source, input))
          inputDependencies_[internal::UnifiedEnumValue(dependentUpdate)].push_back({&source,input});
        else
          throw std::logic_error("No such input.");
      }
      else
        throw std::logic_error("No such source.");
    }
    else
      throw std::logic_error("No such dependent update.");
  }

  template<typename UpdateEnum, typename InputEnum>
  inline void DataNode::addInputDependency(std::initializer_list<UpdateEnum> dependentUpdates, const DataSource & source, InputEnum input)
  {
    for (auto du : dependentUpdates)
      addInputDependency(du, source, input);
  }
}
