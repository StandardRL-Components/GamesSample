
# Generated: 2025-08-27T12:24:42.273481
# Source Brief: brief_00034.md
# Brief Index: 34

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use ↑ and ↓ to move your line vertically. Dodge the red obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, minimalist reaction game. Guide your line through a "
        "treacherous, auto-scrolling corridor of obstacles to reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.W, self.H = 640, 400
        
        # Colors
        self.COLOR_BG_TOP = (20, 10, 40)
        self.COLOR_BG_BOTTOM = (40, 20, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 255)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 50, 50)
        self.COLOR_PARTICLE = (255, 200, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_PROGRESS_BAR = (100, 255, 100)
        self.COLOR_PROGRESS_BAR_BG = (50, 50, 50)

        # Player settings
        self.PLAYER_X = self.W * 0.2
        self.PLAYER_HEIGHT = 40
        self.PLAYER_ACCEL = 1.5
        self.PLAYER_FRICTION = 0.90
        self.PLAYER_LINE_WIDTH = 3

        # Game settings
        self.MAX_STEPS = 4000
        self.INITIAL_SCROLL_SPEED = 3.0
        self.SCROLL_ACCEL_INTERVAL = 50
        self.SCROLL_ACCEL_AMOUNT = 0.075
        self.STAGE_COUNT = 3
        self.UNITS_PER_STAGE = 100
        self.PIXELS_PER_UNIT = 40 # Total length = 3 * 100 * 40 = 12000 pixels
        self.OBSTACLE_WIDTH = 30
        self.OBSTACLE_MIN_DIST = 250
        self.OBSTACLE_MAX_DIST = 400
        self.OBSTACLE_MIN_GAP = 110
        self.OBSTACLE_MAX_GAP = 140
        self.NEAR_MISS_THRESHOLD = 15

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # --- State Variables ---
        self.player_y = 0
        self.player_vy = 0
        self.world_scroll = 0
        self.scroll_speed = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 0
        self.next_obstacle_scroll = 0
        self.np_random = None

        # This will be called once to initialize state
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_y = self.H / 2
        self.player_vy = 0
        self.world_scroll = 0
        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        
        # Use Gymnasium's np_random for reproducibility
        self.next_obstacle_scroll = self.W * 0.8
        self._generate_initial_obstacles()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        terminated = False

        # --- Update Game Logic ---
        self._update_player(action)
        self._update_world_scroll()
        self._update_particles()
        
        collision_detected, near_miss_reward = self._update_obstacles()
        reward += near_miss_reward

        # --- Handle Rewards and Termination ---
        if collision_detected:
            # sfx: player_explode.wav
            reward = -100.0
            self.game_over = True
            terminated = True
        else:
            # Survival reward
            reward += 0.1
            
            # Stage completion reward
            stage_progress_pixels = self.UNITS_PER_STAGE * self.PIXELS_PER_UNIT
            if self.current_stage < self.STAGE_COUNT and self.world_scroll >= self.current_stage * stage_progress_pixels:
                # sfx: stage_complete.wav
                self.current_stage += 1
                reward += 10.0
            
            # Game completion reward
            total_game_pixels = self.STAGE_COUNT * stage_progress_pixels
            if self.world_scroll >= total_game_pixels:
                # sfx: victory.wav
                reward += 100.0
                self.game_over = True
                terminated = True

        # Check for max steps termination
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, action):
        movement = action[0]
        
        if movement == 1:  # Up
            self.player_vy -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.player_vy += self.PLAYER_ACCEL
        
        self.player_vy *= self.PLAYER_FRICTION
        self.player_y += self.player_vy
        
        # Clamp player position to screen bounds
        player_half_h = self.PLAYER_HEIGHT / 2
        self.player_y = np.clip(self.player_y, player_half_h, self.H - player_half_h)

    def _update_world_scroll(self):
        self.world_scroll += self.scroll_speed
        # Increase speed over time
        if self.steps > 0 and self.steps % self.SCROLL_ACCEL_INTERVAL == 0:
            self.scroll_speed += self.SCROLL_ACCEL_AMOUNT

    def _update_particles(self):
        # Move, shrink, and fade particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

    def _update_obstacles(self):
        collision_detected = False
        near_miss_reward = 0.0

        player_top = self.player_y - self.PLAYER_HEIGHT / 2
        player_bottom = self.player_y + self.PLAYER_HEIGHT / 2

        for obs in self.obstacles:
            obs_x_on_screen = obs['x'] - self.world_scroll
            
            # Collision check
            if self.PLAYER_X > obs_x_on_screen and self.PLAYER_X < obs_x_on_screen + obs['w']:
                if not (player_bottom < obs['gap_start'] or player_top > obs['gap_end']):
                    collision_detected = True
                    # Spawn collision particles
                    for _ in range(30):
                        self._spawn_particle(self.PLAYER_X, self.player_y, is_collision=True)
                    break
            
            # Near miss check
            if not obs['rewarded'] and obs_x_on_screen + obs['w'] < self.PLAYER_X:
                obs['rewarded'] = True # Mark as passed
                
                # Check proximity to edges
                dist_to_top_edge = abs(player_bottom - obs['gap_start'])
                dist_to_bottom_edge = abs(player_top - obs['gap_end'])
                
                if min(dist_to_top_edge, dist_to_bottom_edge) < self.NEAR_MISS_THRESHOLD:
                    # sfx: near_miss.wav
                    near_miss_reward += 1.0
                    
                    # Spawn sparks at the closest point
                    spark_y = obs['gap_start'] if dist_to_top_edge < dist_to_bottom_edge else obs['gap_end']
                    for _ in range(5):
                        self._spawn_particle(self.PLAYER_X, spark_y, is_collision=False)

        if collision_detected:
            return True, 0.0

        # Remove obstacles that are off-screen
        self.obstacles = [obs for obs in self.obstacles if obs['x'] - self.world_scroll + obs['w'] > 0]
        
        # Generate new obstacles
        if self.world_scroll > self.next_obstacle_scroll - self.W:
            self._generate_obstacle_pair()

        return False, near_miss_reward

    def _generate_obstacle_pair(self):
        gap_size = self.np_random.uniform(self.OBSTACLE_MIN_GAP, self.OBSTACLE_MAX_GAP)
        gap_center_y = self.np_random.uniform(gap_size, self.H - gap_size)
        
        gap_start = gap_center_y - gap_size / 2
        gap_end = gap_center_y + gap_size / 2

        obstacle_x = self.next_obstacle_scroll
        
        self.obstacles.append({
            'x': obstacle_x,
            'w': self.OBSTACLE_WIDTH,
            'gap_start': gap_start,
            'gap_end': gap_end,
            'rewarded': False
        })
        
        dist_to_next = self.np_random.uniform(self.OBSTACLE_MIN_DIST, self.OBSTACLE_MAX_DIST)
        self.next_obstacle_scroll += dist_to_next

    def _generate_initial_obstacles(self):
        while self.next_obstacle_scroll < self.world_scroll + self.W * 1.5:
            self._generate_obstacle_pair()

    def _spawn_particle(self, x, y, is_collision):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 5) if is_collision else self.np_random.uniform(0.5, 2)
        life = self.np_random.integers(20, 40) if is_collision else self.np_random.integers(10, 20)
        size = self.np_random.uniform(2, 5) if is_collision else self.np_random.uniform(1, 3)

        self.particles.append({
            'x': x, 'y': y,
            'vx': math.cos(angle) * speed - (self.scroll_speed if not is_collision else 0),
            'vy': math.sin(angle) * speed,
            'life': life,
            'size': size
        })

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.H):
            # Simple vertical gradient
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.H
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.H
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.H
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.W, y))

    def _render_game(self):
        self._render_obstacles()
        self._render_particles()
        if not self.game_over:
            self._render_player()

    def _render_player(self):
        y_pos = int(self.player_y)
        half_h = int(self.PLAYER_HEIGHT / 2)
        x_pos = int(self.PLAYER_X)
        
        # Glow effect
        pygame.gfxdraw.vline(self.screen, x_pos-1, y_pos - half_h, y_pos + half_h, (*self.COLOR_PLAYER_GLOW, 60))
        pygame.gfxdraw.vline(self.screen, x_pos+1, y_pos - half_h, y_pos + half_h, (*self.COLOR_PLAYER_GLOW, 60))
        for i in range(1, self.PLAYER_LINE_WIDTH + 2):
            alpha = 100 - i * 20
            if alpha > 0:
                pygame.gfxdraw.line(self.screen, x_pos - i, y_pos - half_h, x_pos - i, y_pos + half_h, (*self.COLOR_PLAYER_GLOW, alpha))
                pygame.gfxdraw.line(self.screen, x_pos + i, y_pos - half_h, x_pos + i, y_pos + half_h, (*self.COLOR_PLAYER_GLOW, alpha))

        # Main line
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (x_pos, y_pos - half_h), (x_pos, y_pos + half_h), self.PLAYER_LINE_WIDTH)

    def _render_obstacles(self):
        for obs in self.obstacles:
            x = int(obs['x'] - self.world_scroll)
            w = int(obs['w'])
            
            # Top part
            top_rect = pygame.Rect(x, 0, w, obs['gap_start'])
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, top_rect)
            pygame.gfxdraw.rectangle(self.screen, top_rect, (*self.COLOR_OBSTACLE_GLOW, 100))

            # Bottom part
            bottom_y = obs['gap_end']
            bottom_h = self.H - bottom_y
            bottom_rect = pygame.Rect(x, bottom_y, w, bottom_h)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, bottom_rect)
            pygame.gfxdraw.rectangle(self.screen, bottom_rect, (*self.COLOR_OBSTACLE_GLOW, 100))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            if alpha > 0 and p['size'] > 0:
                color = (*self.COLOR_PARTICLE, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), color)
    
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_large.render(f"STAGE: {self.current_stage}/{self.STAGE_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.W - stage_text.get_width() - 10, 10))
        
        # Progress Bar
        stage_len_pixels = self.UNITS_PER_STAGE * self.PIXELS_PER_UNIT
        stage_start_pixels = (self.current_stage - 1) * stage_len_pixels
        progress_in_stage = (self.world_scroll - stage_start_pixels) / stage_len_pixels
        progress_in_stage = np.clip(progress_in_stage, 0, 1)

        bar_w = self.W - 20
        bar_h = 5
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (10, 40, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (10, 40, bar_w * progress_in_stage, bar_h))

        if self.game_over:
            total_game_pixels = self.STAGE_COUNT * self.UNITS_PER_STAGE * self.PIXELS_PER_UNIT
            if self.world_scroll >= total_game_pixels:
                msg = "GOAL!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "world_scroll": self.world_scroll
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium's auto_advance=True means the game state updates on every step(),
    # so we need a loop that calls step() at a consistent rate.
    # A no-op action is [0, 0, 0].
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Speed Line")
    screen = pygame.display.set_mode((env.W, env.H))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    print("--- Human Controls ---")
    print(env.user_guide)
    print("----------------------")
    
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting, or wait for 'R' key
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0.0

        clock.tick(30) # Run at 30 FPS

    env.close()