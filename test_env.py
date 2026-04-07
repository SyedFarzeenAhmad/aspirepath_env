import asyncio
from server.app_logic import AspirePathEnv
from server.models import Action

async def test():
    env = AspirePathEnv()
    obs = await env.reset()
    print(f"Initial Observation: {obs}")
    
    # Simulate a mock action
    action = Action(
        recommended_stream="PCM", 
        career_cluster="STEM", 
        justification="High analytical ability and coding interest strongly support a STEM path with PCM."
    )
    next_obs = await env.step(action)
    print(f"Reward: {next_obs.reward}, Done: {next_obs.done}")
    print(f"Reward reasoning: {next_obs.metadata.get('reward_reasoning')}")

if __name__ == "__main__":
    asyncio.run(test())
