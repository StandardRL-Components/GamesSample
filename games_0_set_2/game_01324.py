
# Generated: 2025-08-27T16:46:58.975241
# Source Brief: brief_01324.md
# Brief Index: 1324

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character. Survive for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive in a deadly field of spikes. The spikes become more numerous and activate faster over time. Dodge them for 60 seconds to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.GAME_DURATION_SECONDS = 60
        self.GAME_DURATION_FRAMES = self.GAME_DURATION_SECONDS * self.FPS

        # Player settings
        self.PLAYER_RADIUS = 12
        self.PLAYER_SPEED = 5
        self.PLAYER_START_POS = [self.WIDTH // 2, self.HEIGHT // 2]

        # Spike settings
        self.SPIKE_SIZE = 15
        self.INITIAL_SPIKE_COUNT = 5
        self.SPIKE_INCREASE_INTERVAL = 5 * self.FPS  # Add a spike every 5 seconds
        self.INITIAL_ACTIVATION_DELAY = 2 * self.FPS # 2 seconds
        self.FINAL_ACTIVATION_DELAY = 0.5 * self.FPS # 0.5 seconds
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_SPIKE_INACTIVE = (100, 100, 110)
        self.COLOR_SPIKE_WARNING = (255, 100, 0)
        self.COLOR_SPIKE_ACTIVE = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.spikes = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_timer = None
        self.current_spike_count = None
        self.np_random = None
        self.touched_inactive_spikes = None
        
        # Initialize state variables
        self.reset()

        # Self-check to ensure the implementation follows the spec
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.player_pos = list(self.PLAYER_START_POS)
        self.spikes = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = self.GAME_DURATION_FRAMES
        self.current_spike_count = self.INITIAL_SPIKE_COUNT
        self.touched_inactive_spikes = set()

        self._spawn_spikes(self.current_spike_count)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but advance the clock and return the final state
            self.clock.tick(self.FPS)
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward per frame

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self._handle_player_movement(movement)
        self._update_spikes()
        self._update_difficulty()
        
        collision_reward, terminated = self._check_collisions()
        reward += collision_reward
        
        self.game_timer -= 1
        self.steps += 1
        
        win_condition = self.game_timer <= 0
        if win_condition:
            reward = 100.0  # Big reward for winning
            terminated = True
            
        self.game_over = terminated
        self.score += reward
        self.clock.tick(self.FPS)

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_spikes(self):
        for spike in self.spikes:
            if spike['state'] == 'inactive':
                spike['activation_timer'] -= 1
                if spike['activation_timer'] <= 0:
                    spike['state'] = 'active'
                    # sfx: Spike activation sound

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.SPIKE_INCREASE_INTERVAL == 0:
            self.current_spike_count += 1
            self._spawn_spikes(1)
            # sfx: New spike warning sound

    def _check_collisions(self):
        reward = 0
        terminated = False
        
        for i, spike in enumerate(self.spikes):
            dist = math.hypot(self.player_pos[0] - spike['pos'][0], self.player_pos[1] - spike['pos'][1])
            
            if dist < self.PLAYER_RADIUS + self.SPIKE_SIZE / 2:
                if spike['state'] == 'active':
                    reward = -100.0 # Game over penalty
                    terminated = True
                    # sfx: Player death explosion
                    return reward, terminated # Exit immediately on death
                
                elif spike['state'] == 'inactive' and i not in self.touched_inactive_spikes:
                    reward -= 5.0 # Penalty for touching inactive spike
                    self.touched_inactive_spikes.add(i)
                    # sfx: Minor collision zap
        
        return reward, terminated

    def _spawn_spikes(self, num_to_spawn):
        for _ in range(num_to_spawn):
            progress = self.steps / self.GAME_DURATION_FRAMES
            current_delay = self.INITIAL_ACTIVATION_DELAY - (progress * (self.INITIAL_ACTIVATION_DELAY - self.FINAL_ACTIVATION_DELAY))
            
            while True:
                pos = [
                    self.np_random.integers(self.SPIKE_SIZE, self.WIDTH - self.SPIKE_SIZE),
                    self.np_random.integers(self.SPIKE_SIZE, self.HEIGHT - self.SPIKE_SIZE)
                ]
                if math.hypot(pos[0] - self.player_pos[0], pos[1] - self.player_pos[1]) > self.PLAYER_RADIUS * 5:
                    break
            
            self.spikes.append({
                'pos': pos,
                'activation_timer': int(current_delay),
                'state': 'inactive'
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for spike in self.spikes:
            color = self.COLOR_SPIKE_INACTIVE
            if spike['state'] == 'active':
                color = self.COLOR_SPIKE_ACTIVE
            elif spike['activation_timer'] < self.FPS:
                flicker_speed = max(2, int(spike['activation_timer'] / 5))
                if (self.steps // flicker_speed) % 2 == 0:
                    color = self.COLOR_SPIKE_WARNING
            
            self._draw_triangle(self.screen, color, spike['pos'], self.SPIKE_SIZE)

        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        glow_radius = int(self.PLAYER_RADIUS * 1.5)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (player_x - glow_radius, player_y - glow_radius))
        
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _draw_triangle(self, surface, color, center, size):
        cx, cy = center
        p1 = (cx, cy - size * 0.8)
        p2 = (cx - size * 0.7, cy + size * 0.4)
        p3 = (cx + size * 0.7, cy + size * 0.4)
        pygame.gfxdraw.aapolygon(surface, [p1, p2, p3], color)
        pygame.gfxdraw.filled_polygon(surface, [p1, p2, p3], color)

    def _render_ui(self):
        total_seconds = max(0, self.game_timer // self.FPS)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        timer_text = f"{minutes:02d}:{seconds:02d}"
        text_surf = self.font.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        spike_text = f"Spikes: {self.current_spike_count}"
        spike_surf = self.font.render(spike_text, True, self.COLOR_TEXT)
        spike_rect = spike_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(spike_surf, spike_rect)

        if self.game_over:
            message = "YOU WON!" if self.game_timer <= 0 else "GAME OVER"
            color = (100, 255, 100) if self.game_timer <= 0 else (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left_seconds": max(0, self.game_timer // self.FPS),
            "spike_count": self.current_spike_count,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    # Set to a dummy driver for headless execution, as required for agents.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    
    # --- Example of running with a human player (requires a display) ---
    # To use this, comment out the dummy driver line above and uncomment this block.
    # Make sure you have a display environment (e.g., a desktop).
    #
    # pygame.display.init()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Spike Field Survival")
    # clock = pygame.time.Clock()

    # obs, info = env.reset()
    # done = False
    # total_reward = 0
    
    # while not done:
    #     movement = 0 # No-op by default
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     if keys[pygame.K_DOWN]: movement = 2
    #     if keys[pygame.K_LEFT]: movement = 3
    #     if keys[pygame.K_RIGHT]: movement = 4
        
    #     action = [movement, 0, 0] # Space and Shift are not used
        
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     total_reward += reward

    #     # Render the observation to the display window
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True

    # print(f"Game Over! Final Score: {total_reward:.2f}")
    # env.close()
    # pygame.quit()

    # --- Example of running a simple random agent ---
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0.0
    step_count = 0
    while not done:
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        if step_count > env.GAME_DURATION_FRAMES + 10: # Safety break
            break

    print(f"Random agent finished in {step_count} steps.")
    print(f"Final Score: {total_reward:.2f}")
    print(f"Final Info: {info}")
    env.close()