
# Generated: 2025-08-28T04:00:57.748959
# Source Brief: brief_05114.md
# Brief Index: 5114

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to steer. Dodge the blue blocks to reach the finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A high-speed arcade racer. Navigate a perilous, procedurally generated track and survive until the end."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 900  # 30 seconds at 30 FPS
        self.MAX_HITS = 5
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_main = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG = (15, 19, 26)
        self.COLOR_ROAD = (40, 45, 55)
        self.COLOR_LINES = (70, 75, 85)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 50)
        self.COLOR_OBSTACLE = (60, 160, 255)
        self.COLOR_OBSTACLE_GLOW = (100, 200, 255, 50)
        self.COLOR_PARTICLE = (255, 200, 80)
        self.COLOR_TEXT = (230, 230, 230)
        
        # Player state
        self.player_pos = None
        self.player_width = 40
        self.player_height = 20
        self.player_speed = 8

        # Game state
        self.road_x = self.WIDTH // 2
        self.road_width = 300
        self.obstacles = []
        self.particles = []
        self.obstacle_speed = 12
        self.initial_spawn_rate = 0.1
        self.spawn_rate_increase = 0.01 / self.FPS # Per step
        self.current_spawn_rate = self.initial_spawn_rate
        
        self.steps = 0
        self.score = 0
        self.obstacles_hit = 0
        self.game_over = False
        self.won_race = False

        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.obstacles_hit = 0
        self.game_over = False
        self.won_race = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 60]
        self.obstacles.clear()
        self.particles.clear()
        self.current_spawn_rate = self.initial_spawn_rate
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        self.steps += 1
        
        # --- Action Handling ---
        movement = action[0]
        if movement == 3:  # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos[0] += self.player_speed
        
        # Clamp player position to the road
        road_left_edge = self.road_x - self.road_width // 2
        road_right_edge = self.road_x + self.road_width // 2
        self.player_pos[0] = np.clip(
            self.player_pos[0],
            road_left_edge + self.player_width // 2,
            road_right_edge - self.player_width // 2
        )
        
        # --- Game Logic ---
        self._update_obstacles()
        self._update_particles()
        
        # --- Collision Detection ---
        collision_this_step = self._check_collisions()
        
        # --- Termination Check ---
        self.won_race = self.steps >= self.MAX_STEPS and self.obstacles_hit < self.MAX_HITS
        lost_race = self.obstacles_hit >= self.MAX_HITS
        
        terminated = self.won_race or lost_race or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True

        # --- Reward Calculation ---
        reward = 0
        if self.won_race:
            reward = 100.0
        elif collision_this_step:
            reward = -1.0
        elif not terminated:
            reward = 0.1 # Survival reward
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_obstacles(self):
        # Move existing obstacles and remove off-screen ones
        self.obstacles = [
            obs for obs in self.obstacles if obs['rect'].top < self.HEIGHT
        ]
        for obs in self.obstacles:
            obs['rect'].y += self.obstacle_speed

        # Spawn new obstacles
        self.current_spawn_rate += self.spawn_rate_increase
        if self.np_random.random() < self.current_spawn_rate:
            road_left_edge = self.road_x - self.road_width // 2
            obs_width = self.np_random.integers(30, 60)
            obs_height = 20
            obs_x = self.np_random.integers(
                road_left_edge, self.road_x + self.road_width // 2 - obs_width
            )
            obs_y = -obs_height
            self.obstacles.append({
                'rect': pygame.Rect(obs_x, obs_y, obs_width, obs_height)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.player_width // 2,
            self.player_pos[1] - self.player_height // 2,
            self.player_width, self.player_height
        )
        
        collided_this_step = False
        for obs in self.obstacles[:]:
            if player_rect.colliderect(obs['rect']):
                self.obstacles.remove(obs)
                self.obstacles_hit += 1
                collided_this_step = True
                self._create_explosion(player_rect.center)
                # Sound effect placeholder: // CRASH!
                break # Only one collision per step
        return collided_this_step

    def _create_explosion(self, pos):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.integers(2, 5)
            })
            
    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Road
        road_rect = (
            self.road_x - self.road_width // 2, 0, self.road_width, self.HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_ROAD, road_rect)

        # Road lines
        line_length = 30
        line_gap = 20
        total_segment_length = line_length + line_gap
        scroll_offset = (self.steps * self.obstacle_speed) % total_segment_length
        for y in range(-total_segment_length, self.HEIGHT, total_segment_length):
            start_y = y + scroll_offset
            end_y = start_y + line_length
            if start_y < self.HEIGHT and end_y > 0:
                pygame.draw.line(self.screen, self.COLOR_LINES, (self.road_x, start_y), (self.road_x, end_y), 5)
        
        # Obstacles
        for obs in self.obstacles:
            self._draw_glowing_rect(self.screen, obs['rect'], self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)
        
        # Finish Line visual
        time_to_finish = self.MAX_STEPS - self.steps
        if time_to_finish < self.HEIGHT / self.obstacle_speed:
            finish_y = time_to_finish * self.obstacle_speed
            self._draw_checkered_line(finish_y)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30))
            color = self.COLOR_PARTICLE + (alpha,)
            self._draw_glowing_circle(
                int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color
            )

        # Player
        player_rect = pygame.Rect(
            self.player_pos[0] - self.player_width // 2,
            self.player_pos[1] - self.player_height // 2,
            self.player_width, self.player_height
        )
        self._draw_glowing_rect(self.screen, player_rect, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _draw_glowing_rect(self, surface, rect, color, glow_color):
        # Draw the glow effect first
        glow_rect = rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=5)
        surface.blit(s, glow_rect.topleft)
        # Draw the main rect on top
        pygame.draw.rect(surface, color, rect, border_radius=3)

    def _draw_glowing_circle(self, x, y, radius, color):
        if color[3] == 0: return # Skip fully transparent
        # Draw the glow effect
        glow_radius = radius * 2
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, (color[0], color[1], color[2], color[3] // 4))
        self.screen.blit(s, (x - glow_radius, y - glow_radius))
        # Draw the main circle
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _draw_checkered_line(self, y):
        check_size = 20
        road_left = self.road_x - self.road_width // 2
        for i, x in enumerate(range(road_left, road_left + self.road_width, check_size)):
            color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
            pygame.draw.rect(self.screen, color, (x, y, check_size, 10))

    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Hits
        hits_text = self.font_ui.render(f"HITS: {self.obstacles_hit}/{self.MAX_HITS}", True, self.COLOR_TEXT)
        hits_rect = hits_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(hits_text, hits_rect)

        # Game Over / Win message
        if self.game_over:
            if self.won_race:
                msg = "FINISH!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_main.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hits": self.obstacles_hit,
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # This part is for demonstration and will not run in a headless environment.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Arcade Racer")
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Map keyboard keys to actions
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            if keys[pygame.K_DOWN]: movement = 2
            if keys[pygame.K_LEFT]: movement = 3
            if keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Handle closing the window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            env.clock.tick(env.FPS)
        
        print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")

    except Exception as e:
        print("\nCould not create Pygame display. This is expected in a headless environment.")
        print("The environment is designed for headless use with RL agents.")
        print(f"Error: {e}")

    finally:
        env.close()