#pragma once

namespace core { class World; }

namespace sim {

/**
 * @brief Interface for objects that observe simulation frames.
 * 
 * Inspired by dismech's observer pattern, this allows decoupling of
 * frame-based computations (like entanglement) from the main simulation loop.
 * Observers can be registered with environments and will be notified at
 * appropriate intervals.
 */
class IFrameObserver {
public:
    virtual ~IFrameObserver() = default;

    /**
     * @brief Called when a frame should be observed.
     * @param world The current simulation state
     * @param frameIdx The current frame index
     * @param time The current simulation time
     */
    virtual void onFrame(const core::World& world, int frameIdx, double time) = 0;

    /**
     * @brief Returns whether this observer should be called on the given frame.
     * @param frameIdx The current frame index
     * @return true if onFrame should be called for this frame
     */
    virtual bool shouldObserve(int frameIdx) const = 0;
};

} // namespace sim
