
# Generated: 2025-08-28T06:56:13.005636
# Source Brief: brief_03083.md
# Brief Index: 3083

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Survive the falling obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced survival game. Dodge the falling red obstacles for as long as you can. "
        "The longer you survive, the faster they come!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.W, self.H = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * self.FPS  # 30 seconds

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_OUTLINE = (200, 255, 220)
        self.OBSTACLE_COLORS = [(255, 80, 80), (240, 70, 70), (220, 60, 60)]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_WIN_TEXT = (100, 255, 100)
        self.COLOR_LOSE_TEXT = (255, 100, 100)

        # Player settings
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 6

        # Obstacle settings
        self.OBSTACLE_MIN_SIZE = 15
        self.OBSTACLE_MAX_SIZE = 40

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_trail = None
        self.obstacles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.spawn_timer = 0
        self.np_random = None
        
        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = np.array([self.W / 2.0, self.H - 50.0], dtype=np.float32)
        self.player_trail = deque(maxlen=15)
        self.obstacles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.spawn_timer = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        
        # Update player movement
        self._handle_player_movement(movement)
        
        # Update game world (obstacles)
        self._update_obstacles()
        
        # Check for collisions
        if self._check_collisions():
            self.game_over = True
            # sound placeholder: explosion_sound.play()

        # Calculate reward and termination status
        reward = 0
        terminated = self.game_over
        
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.win = True
            terminated = True
            reward = 100.0  # Goal-oriented reward for winning
            self.score += 100.0
            # sound placeholder: win_sound.play()
        elif not self.game_over:
            reward = 0.1  # Continuous feedback for survival
            self.score += 0.1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1:  # Up
            move_vec[1] = -1
        elif movement == 2:  # Down
            move_vec[1] = 1
        elif movement == 3:  # Left
            move_vec[0] = -1
        elif movement == 4:  # Right
            move_vec[0] = 1
        
        self.player_pos += move_vec * self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.W - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.H - self.PLAYER_SIZE / 2)
        
        if self.steps % 2 == 0:
            self.player_trail.append(self.player_pos.copy())

    def _update_obstacles(self):
        # Difficulty scaling: spawn rate and speed increase over time
        spawn_progress = min(1.0, self.steps / (15 * self.FPS))
        speed_progress = min(1.0, self.steps / self.MAX_STEPS)

        start_interval, end_interval = 2 * self.FPS, 0.8 * self.FPS
        current_interval = start_interval - (start_interval - end_interval) * spawn_progress
        
        start_min_speed, end_min_speed = 1.0, 2.0
        start_max_speed, end_max_speed = 2.0, 4.0
        current_min_speed = start_min_speed + (end_min_speed - start_min_speed) * speed_progress
        current_max_speed = start_max_speed + (end_max_speed - start_max_speed) * speed_progress

        # Spawn new obstacles
        self.spawn_timer += 1
        if self.spawn_timer >= current_interval:
            self.spawn_timer = 0
            obs_type = self.np_random.choice(['rect', 'circle', 'triangle'])
            pos = np.array([self.np_random.uniform(0, self.W), -self.OBSTACLE_MAX_SIZE], dtype=np.float32)
            speed = self.np_random.uniform(current_min_speed, current_max_speed)
            drift = self.np_random.uniform(-0.5, 0.5)
            color = random.choice(self.OBSTACLE_COLORS)
            
            if obs_type == 'rect':
                size = (self.np_random.integers(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE), 
                        self.np_random.integers(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE))
            else:
                size = self.np_random.uniform(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
            
            self.obstacles.append({'pos': pos, 'speed': speed, 'drift': drift, 'type': obs_type, 'size': size, 'color': color})
            # sound placeholder: spawn_sound.play()

        # Update and remove old obstacles
        for obs in self.obstacles:
            obs['pos'][1] += obs['speed']
            obs['pos'][0] += obs['drift']
        
        self.obstacles = [obs for obs in self.obstacles if obs['pos'][1] < self.H + self.OBSTACLE_MAX_SIZE]

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE / 2, self.player_pos[1] - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obs in self.obstacles:
            if obs['type'] == 'rect':
                obs_rect = pygame.Rect(obs['pos'][0] - obs['size'][0] / 2, obs['pos'][1] - obs['size'][1] / 2, obs['size'][0], obs['size'][1])
                if player_rect.colliderect(obs_rect):
                    return True
            elif obs['type'] == 'circle':
                dist = np.linalg.norm(self.player_pos - obs['pos'])
                if dist < self.PLAYER_SIZE / 2 + obs['size'] / 2:
                    return True
            elif obs['type'] == 'triangle':
                # Use simple AABB collision for triangles for performance
                obs_rect = pygame.Rect(obs['pos'][0] - obs['size'] / 2, obs['pos'][1] - obs['size'] / 2, obs['size'], obs['size'])
                if player_rect.colliderect(obs_rect):
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render player trail
        if len(self.player_trail) > 1:
            for i, pos in enumerate(self.player_trail):
                alpha = int(150 * (i / len(self.player_trail)))
                trail_color = (*self.COLOR_PLAYER[:3], alpha)
                size = self.PLAYER_SIZE * (i / len(self.player_trail)) * 0.8
                if size > 1:
                    s = pygame.Surface((size, size), pygame.SRCALPHA)
                    pygame.draw.rect(s, trail_color, s.get_rect(), border_radius=2)
                    self.screen.blit(s, (int(pos[0] - size/2), int(pos[1] - size/2)))

        # Render obstacles
        for obs in self.obstacles:
            x, y = int(obs['pos'][0]), int(obs['pos'][1])
            color = obs['color']
            if obs['type'] == 'rect':
                w, h = obs['size']
                pygame.draw.rect(self.screen, color, (x - w/2, y - h/2, w, h))
            elif obs['type'] == 'circle':
                r = int(obs['size'] / 2)
                pygame.gfxdraw.aacircle(self.screen, x, y, r, color)
                pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)
            elif obs['type'] == 'triangle':
                s = obs['size']
                p1 = (x, y - s * 0.577)
                p2 = (x - s/2, y + s * 0.288)
                p3 = (x + s/2, y + s * 0.288)
                points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=3)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.W - timer_text.get_width() - 10, 10))

        if self.game_over:
            msg_text_str = "YOU WIN!" if self.win else "GAME OVER"
            msg_color = self.COLOR_WIN_TEXT if self.win else self.COLOR_LOSE_TEXT
            msg_surf = self.font_msg.render(msg_text_str, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(msg_surf, msg_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Beginning implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")