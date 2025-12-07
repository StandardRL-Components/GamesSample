
# Generated: 2025-08-28T05:31:52.458893
# Source Brief: brief_02633.md
# Brief Index: 2633

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to aim, ↑↓ to adjust power. Press Space to fire."

    # Must be a short, user-facing description of the game:
    game_description = "A target practice game. Destroy all targets with a limited number of projectiles. Plan your shots carefully!"

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and color constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_TARGET = (255, 50, 50)
        self.COLOR_TARGET_OUTLINE = (255, 100, 100)
        self.COLOR_LAUNCHER = (200, 200, 220)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 200, 0)
        self.COLOR_AIM_LINE = (255, 255, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game parameters
        self.launcher_pos = (self.WIDTH // 2, self.HEIGHT - 20)
        self.gravity = 0.15
        self.max_shots = 15
        self.num_targets = 8
        self.target_radius = 15
        
        # Initialize state variables
        self.launcher_angle = 0
        self.launch_power = 0
        self.shots_remaining = 0
        self.targets = []
        self.particles = []
        self.last_trajectory = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.shots_remaining = self.max_shots
        self.launcher_angle = -90  # Pointing straight up
        self.launch_power = 50  # 0-100 scale

        self.targets = []
        for _ in range(self.num_targets):
            # Place targets in the upper 2/3 of the screen
            x = self.np_random.integers(self.target_radius * 2, self.WIDTH - self.target_radius * 2)
            y = self.np_random.integers(self.target_radius * 2, self.HEIGHT * 2 // 3)
            self.targets.append(pygame.Vector2(x, y))

        self.particles = []
        self.last_trajectory = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        fired_shot = False

        # 1. Update launcher state based on action
        if movement == 1:  # Up
            self.launch_power = min(100, self.launch_power + 2)
        elif movement == 2:  # Down
            self.launch_power = max(10, self.launch_power - 2)
        elif movement == 3:  # Left
            self.launcher_angle = (self.launcher_angle - 2) % 360
        elif movement == 4:  # Right
            self.launcher_angle = (self.launcher_angle + 2) % 360

        # 2. Handle firing action
        if space_held and self.shots_remaining > 0:
            fired_shot = True
            self.shots_remaining -= 1
            # sfx_fire()

            power_scalar = 8 + (self.launch_power / 100.0) * 12
            angle_rad = math.radians(self.launcher_angle)
            
            proj_pos = pygame.Vector2(self.launcher_pos)
            proj_vel = pygame.Vector2(power_scalar * math.cos(angle_rad), power_scalar * math.sin(angle_rad))

            self.last_trajectory = []
            hit_target = False

            for _ in range(200): # Simulate projectile flight
                proj_pos += proj_vel
                proj_vel.y += self.gravity
                self.last_trajectory.append(tuple(proj_pos))

                for i, target_pos in reversed(list(enumerate(self.targets))):
                    if proj_pos.distance_to(target_pos) < self.target_radius:
                        # sfx_explosion()
                        self._create_explosion(target_pos)
                        self.targets.pop(i)
                        reward += 1.0
                        self.score += 100
                        hit_target = True
                        break
                if hit_target: break
                if not (0 < proj_pos.x < self.WIDTH and proj_pos.y < self.HEIGHT): break
            
            if not hit_target: reward -= 0.1

        # 3. Update particles and steps
        self._update_particles()
        self.steps += 1

        # 4. Check for termination conditions
        if (fired_shot and self.shots_remaining == 0 and len(self.targets) > 0) or len(self.targets) == 0:
            terminated = True
            self.game_over = True
            if len(self.targets) == 0:
                self.game_won = True
                reward += 100.0
                self.score += 1000
            else:
                reward -= 100.0
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
    
    def _create_explosion(self, pos):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = random.randint(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'max_life': life})

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if len(self.last_trajectory) > 1:
            for i in range(len(self.last_trajectory) - 1):
                alpha = int(255 * (i / len(self.last_trajectory)))
                color = (*self.COLOR_PROJECTILE, alpha)
                start_pos = (int(self.last_trajectory[i][0]), int(self.last_trajectory[i][1]))
                end_pos = (int(self.last_trajectory[i+1][0]), int(self.last_trajectory[i+1][1]))
                pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
        
        for pos in self.targets:
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.target_radius, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.target_radius, self.COLOR_TARGET_OUTLINE)

        launcher_end_x = self.launcher_pos[0] + 30 * math.cos(math.radians(self.launcher_angle))
        launcher_end_y = self.launcher_pos[1] + 30 * math.sin(math.radians(self.launcher_angle))
        pygame.draw.line(self.screen, self.COLOR_LAUNCHER, self.launcher_pos, (launcher_end_x, launcher_end_y), 6)
        pygame.gfxdraw.filled_circle(self.screen, self.launcher_pos[0], self.launcher_pos[1], 8, self.COLOR_LAUNCHER)

        if not self.game_over:
            aim_len = 50 + self.launch_power * 1.5
            aim_end_x = self.launcher_pos[0] + aim_len * math.cos(math.radians(self.launcher_angle))
            aim_end_y = self.launcher_pos[1] + aim_len * math.sin(math.radians(self.launcher_angle))
            self._draw_dashed_line(self.screen, self.COLOR_AIM_LINE, self.launcher_pos, (aim_end_x, aim_end_y))
        
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            size = int(5 * (p['life'] / p['max_life']))
            if size > 0:
                color = (*self.COLOR_EXPLOSION, alpha)
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                self.screen.blit(surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _draw_dashed_line(self, surf, color, start_pos, end_pos, width=1, dash_length=10):
        origin = pygame.Vector2(start_pos)
        target = pygame.Vector2(end_pos)
        displacement = target - origin
        if displacement.length() == 0: return
        n_dashes = int(displacement.length() / dash_length)
        for i in range(n_dashes):
            if i % 2 == 0:
                start = origin + displacement * (i / n_dashes)
                end = origin + displacement * ((i + 1) / n_dashes)
                pygame.draw.line(surf, color, start, end, width)

    def _render_ui(self):
        shots_text = self.font_small.render(f"SHOTS: {self.shots_remaining}/{self.max_shots}", True, self.COLOR_UI_TEXT)
        self.screen.blit(shots_text, (10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        targets_text = self.font_small.render(f"TARGETS: {len(self.targets)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(targets_text, (self.WIDTH // 2 - targets_text.get_width() // 2, 10))

        power_bar_width = 100
        power_bar_height = 15
        power_bar_x = self.launcher_pos[0] - power_bar_width // 2
        power_bar_y = self.HEIGHT - 45
        
        pygame.draw.rect(self.screen, (50, 50, 70), (power_bar_x, power_bar_y, power_bar_width, power_bar_height))
        current_power_width = (self.launch_power / 100) * power_bar_width
        power_color = (255, 100 + 155 * (self.launch_power / 100), 0)
        pygame.draw.rect(self.screen, power_color, (power_bar_x, power_bar_y, current_power_width, power_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (power_bar_x, power_bar_y, power_bar_width, power_bar_height), 1)
        
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shots_remaining": self.shots_remaining,
            "targets_remaining": len(self.targets)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this to verify implementation.
        '''
        print("Running implementation validation...")
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
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    env.validate_implementation()
    
    print("\n--- Testing Random Agent ---")
    obs, info = env.reset()
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1:02d}: Action={action}, Reward={reward:6.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print(f"Episode finished in {i+1} steps. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            break
    env.close()