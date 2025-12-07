
# Generated: 2025-08-27T20:26:49.727062
# Source Brief: brief_02464.md
# Brief Index: 2464

        
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
        "Controls: ↑ to move up, ↓ to move down. Dodge the red obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated obstacle course as a swiftly moving line, "
        "dodging hazards to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.LEVEL_LENGTH = 1000

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 100)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 100, 100, 120)
        self.COLOR_FINISH = (50, 255, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_THRUSTER = (255, 180, 50)

        self._create_background_surface()
        
        # Initialize state variables
        self.player_y = 0
        self.player_height = 0
        self.player_x = 0
        self.player_speed = 0
        self.obstacles = []
        self.obstacle_speed = 0
        self.last_speed_increase_step = 0
        self.next_obstacle_spawn_score = 0
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.reset()
        
        self.validate_implementation()
    
    def _create_background_surface(self):
        """Creates a pre-rendered gradient background for performance."""
        self.background_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.background_surf, color, (0, y), (self.SCREEN_WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_y = self.SCREEN_HEIGHT / 2
        self.player_height = 40
        self.player_x = 100
        self.player_speed = 8

        # Game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        
        # Obstacle state
        self.obstacles = []
        self.obstacle_speed = 8.0
        self.last_speed_increase_step = 0
        self.next_obstacle_spawn_score = self.np_random.uniform(100, 150)

        # Visuals
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game has ended, return the final state without updates
            # This allows the final screen (e.g., "GAME OVER") to persist
            return self._get_observation(), 0.0, True, False, self._get_info()

        # 1. Handle Action
        movement = action[0]
        if movement == 1:  # Up
            self.player_y -= self.player_speed
            self._create_thruster_particles(direction='down')
        elif movement == 2:  # Down
            self.player_y += self.player_speed
            self._create_thruster_particles(direction='up')
        
        # Clamp player position to screen bounds
        half_h = self.player_height / 2
        self.player_y = np.clip(self.player_y, half_h, self.SCREEN_HEIGHT - half_h)

        # 2. Update Game State
        self.steps += 1
        # Score is distance traveled, scaled by speed for a sense of progress
        self.score += self.obstacle_speed / 10.0
        
        self._update_obstacles()
        self._spawn_obstacles()
        self._update_particles()
        
        # Increase difficulty over time
        if self.steps - self.last_speed_increase_step >= 100:
            self.obstacle_speed = min(20.0, self.obstacle_speed + 0.2)
            self.last_speed_increase_step = self.steps

        # 3. Check for Termination & Calculate Reward
        reward = 0.1  # Survival reward per step
        terminated = False
        
        if self._check_collision():
            self.game_over = True
            terminated = True
            reward = -10.0
            self._create_player_explosion()
            # Sound: Explosion
        
        if not self.game_over and self.score >= self.LEVEL_LENGTH:
            self.game_over = True
            self.victory = True
            terminated = True
            reward = 100.0
            # Sound: Victory fanfare
        
        if self.steps >= 10000:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed
        self.obstacles = [obs for obs in self.obstacles if obs['x'] + obs['width'] > 0]

    def _spawn_obstacles(self):
        if self.score >= self.next_obstacle_spawn_score:
            min_h, max_h = 80, 250
            height = self.np_random.uniform(min_h, max_h)
            
            self.obstacles.append({
                'x': self.SCREEN_WIDTH,
                'y': self.np_random.uniform(0, self.SCREEN_HEIGHT - height),
                'width': self.np_random.uniform(20, 40),
                'height': height,
            })
            # Sound: Obstacle spawn woosh
            
            min_gap = 120
            max_gap = 250
            gap_reduction_factor = self.obstacle_speed * 8
            
            current_min_gap = max(80, min_gap - gap_reduction_factor)
            current_max_gap = max(120, max_gap - gap_reduction_factor)
            
            self.next_obstacle_spawn_score += self.np_random.uniform(current_min_gap, current_max_gap)

    def _check_collision(self):
        player_top = self.player_y - self.player_height / 2
        player_bottom = self.player_y + self.player_height / 2
        
        for obs in self.obstacles:
            obs_left, obs_right = obs['x'], obs['x'] + obs['width']
            if obs_left <= self.player_x <= obs_right:
                obs_top, obs_bottom = obs['y'], obs['y'] + obs['height']
                if player_bottom > obs_top and player_top < obs_bottom:
                    return True
        return False

    def _get_observation(self):
        self.screen.blit(self.background_surf, (0, 0))
        
        self._render_finish_line()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_player(self):
        if not self.game_over or self.victory:
            p_top = int(self.player_y - self.player_height / 2)
            p_bottom = int(self.player_y + self.player_height / 2)
            
            # Glow effect
            pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, (self.player_x, p_top), (self.player_x, p_bottom), 9)
            # Main line
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (self.player_x, p_top), (self.player_x, p_bottom), 3)
            
    def _render_obstacles(self):
        for obs in self.obstacles:
            rect = pygame.Rect(int(obs['x']), int(obs['y']), int(obs['width']), int(obs['height']))
            
            # Glow effect
            glow_rect = rect.inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_OBSTACLE_GLOW, s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)

            # Main shape
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)

    def _render_finish_line(self):
        if self.score > self.LEVEL_LENGTH - self.SCREEN_WIDTH:
            line_x = self.SCREEN_WIDTH + (self.LEVEL_LENGTH - self.score)
            if 0 < line_x < self.SCREEN_WIDTH:
                pygame.draw.line(self.screen, self.COLOR_FINISH, (int(line_x), 0), (int(line_x), self.SCREEN_HEIGHT), 5)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            if self.victory:
                end_text = self.font_large.render("VICTORY", True, self.COLOR_FINISH)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_OBSTACLE)
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Particle System ---
    def _create_thruster_particles(self, direction):
        p_top = self.player_y - self.player_height / 2
        p_bottom = self.player_y + self.player_height / 2
        for _ in range(2):
            self.particles.append({
                'x': self.player_x,
                'y': p_top if direction == 'up' else p_bottom,
                'vx': self.np_random.uniform(-1, 1) - 3,
                'vy': self.np_random.uniform(-2, 2) + (5 if direction == 'up' else -5),
                'life': self.np_random.uniform(10, 20),
                'color': self.COLOR_THRUSTER,
            })

    def _create_player_explosion(self):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'x': self.player_x,
                'y': self.player_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.uniform(20, 40),
                'color': self.np_random.choice([self.COLOR_PLAYER, self.COLOR_THRUSTER, self.COLOR_OBSTACLE]),
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            radius = int(max(0, p['life'] / 5))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, p['color'])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Swift Line")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults to NO-OP
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # Unused actions
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Game Over! Final Score: {int(info['score'])}, Total Reward: {total_reward:.2f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate for human play
        clock.tick(30)
        
    env.close()