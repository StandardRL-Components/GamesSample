
# Generated: 2025-08-27T22:06:03.481766
# Source Brief: brief_03010.md
# Brief Index: 3010

        
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
        "Controls: ↑/↓ to move vertically. Press Space in time with the green boosts to gain speed. Avoid red obstacles!"
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "A fast-paced, retro-futuristic racer. Steer your vehicle through a hazardous track, hitting boosts in rhythm with the beat to maximize your speed and finish before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    MAX_CRASHES = 3
    TRACK_LENGTH = 15000 # Distance to finish line in pixels

    # --- Colors (Neon/Retro Theme) ---
    COLOR_BG = (10, 0, 20)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (200, 0, 0)
    COLOR_BOOST = (50, 255, 50)
    COLOR_BOOST_GLOW = (0, 200, 0)
    COLOR_TRACK = (255, 200, 0)
    COLOR_TRACK_GLOW = (200, 150, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_FINISH_LINE = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player_y = 0
        self.player_vy = 0
        self.world_scroll_x = 0
        self.base_scroll_speed = 0
        self.current_scroll_speed = 0
        self.speed_boost_timer = 0
        self.crashes = 0
        self.invulnerability_timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.obstacles = []
        self.boosts = []
        self.particles = []
        self.parallax_bg = []

        self.reset()
        
        # self.validate_implementation() # Uncomment for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crashes = 0
        
        self.player_y = self.HEIGHT / 2
        self.player_vy = 0
        
        self.world_scroll_x = 0
        self.base_scroll_speed = 5.0
        self.current_scroll_speed = self.base_scroll_speed
        self.speed_boost_timer = 0
        self.invulnerability_timer = 0
        
        self.obstacles.clear()
        self.boosts.clear()
        self.particles.clear()
        self.parallax_bg.clear()
        
        self._generate_track_elements()
        self._generate_parallax_bg()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01  # Small reward for surviving

        # --- Action Handling ---
        movement = action[0]
        space_pressed_this_frame = action[1] == 1
        
        # --- Player Physics ---
        player_accel = 0
        if movement == 1:  # Up
            player_accel = -0.8
        elif movement == 2:  # Down
            player_accel = 0.8
        
        self.player_vy += player_accel
        self.player_vy *= 0.85 # Dampening
        self.player_y += self.player_vy
        
        # Clamp player position within track boundaries
        track_top, track_bottom = self.HEIGHT * 0.2, self.HEIGHT * 0.8
        self.player_y = np.clip(self.player_y, track_top + 15, track_bottom - 15)

        # --- World & Entity Updates ---
        self.current_scroll_speed = self.base_scroll_speed + self.speed_boost_timer * 0.2
        self.world_scroll_x += self.current_scroll_speed
        
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= 1
        if self.invulnerability_timer > 0:
            self.invulnerability_timer -= 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 300 == 0:
            self.base_scroll_speed = min(15, self.base_scroll_speed + 0.5)

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0] - self.current_scroll_speed
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Interactions & Rewards ---
        player_rect = pygame.Rect(self.WIDTH * 0.2 - 10, self.player_y - 10, 20, 20)
        
        # Boosts
        for boost in self.boosts[:]:
            if boost['collected']:
                continue
            
            boost_x_on_screen = boost['x'] - self.world_scroll_x
            if boost_x_on_screen < -50: # Missed boost
                reward -= 0.2
                boost['collected'] = True # Mark as handled
                continue

            boost_rect = pygame.Rect(boost_x_on_screen - 10, boost['y'] - 10, 20, 20)
            if player_rect.colliderect(boost_rect):
                if space_pressed_this_frame:
                    # Rhythmic Hit!
                    reward += 5
                    self.score += 50
                    self.speed_boost_timer = 90 # 3 seconds of boost
                    self._create_particles(player_rect.center, 50, self.COLOR_BOOST, 5, 40)
                    # Sound: BOOST!
                else:
                    # Just collected, no bonus
                    self.score += 5
                boost['collected'] = True
        
        # Obstacles and Crashes
        if self.invulnerability_timer <= 0:
            for obstacle in self.obstacles:
                obs_x_on_screen = obstacle['x'] - self.world_scroll_x
                obstacle_rect = pygame.Rect(obs_x_on_screen, obstacle['y'], obstacle['w'], obstacle['h'])
                if player_rect.colliderect(obstacle_rect):
                    reward -= 10
                    self.crashes += 1
                    self.invulnerability_timer = 60 # 2 seconds of invulnerability
                    self.speed_boost_timer = 0 # Lose boost on crash
                    self.current_scroll_speed = self.base_scroll_speed / 2 # Slow down
                    self._create_particles(player_rect.center, 80, self.COLOR_OBSTACLE, 8, 60)
                    # Sound: CRASH!
                    break # Only one crash per frame

        self.steps += 1
        
        # --- Termination Check ---
        won = self.world_scroll_x >= self.TRACK_LENGTH
        lost_crashes = self.crashes >= self.MAX_CRASHES
        lost_time = self.steps >= self.MAX_STEPS
        
        terminated = won or lost_crashes or lost_time
        if won:
            reward += 50
            self.score += 1000
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crashes": self.crashes,
            "progress": min(1.0, self.world_scroll_x / self.TRACK_LENGTH),
        }
        
    def _generate_track_elements(self):
        # Generate a consistent track using the seeded RNG
        current_x = 500
        while current_x < self.TRACK_LENGTH:
            y_pos = self.np_random.uniform(self.HEIGHT * 0.25, self.HEIGHT * 0.75)
            
            # Place a cluster of boosts or an obstacle
            if self.np_random.random() < 0.7: # 70% chance for a boost cluster
                for i in range(self.np_random.integers(2, 5)):
                    self.boosts.append({
                        'x': current_x + i * 150, 
                        'y': self.np_random.uniform(self.HEIGHT * 0.25, self.HEIGHT * 0.75),
                        'collected': False
                    })
                current_x += self.np_random.integers(500, 800)
            else: # 30% chance for an obstacle
                self.obstacles.append({
                    'x': current_x,
                    'y': y_pos,
                    'w': self.np_random.integers(30, 60),
                    'h': self.np_random.integers(30, 60)
                })
                current_x += self.np_random.integers(400, 600)

    def _generate_parallax_bg(self):
        # Far layer
        for _ in range(30):
            self.parallax_bg.append({
                'x': self.np_random.uniform(0, self.TRACK_LENGTH * 1.5),
                'y': self.np_random.uniform(0, self.HEIGHT * 0.8),
                'w': self.np_random.uniform(20, 50),
                'h': self.np_random.uniform(100, self.HEIGHT),
                'speed_mult': 0.1,
                'color': (20, 10, 40)
            })
        # Mid layer
        for _ in range(20):
            self.parallax_bg.append({
                'x': self.np_random.uniform(0, self.TRACK_LENGTH * 1.5),
                'y': self.np_random.uniform(0, self.HEIGHT * 0.8),
                'w': self.np_random.uniform(30, 80),
                'h': self.np_random.uniform(100, self.HEIGHT),
                'speed_mult': 0.3,
                'color': (40, 20, 70)
            })

    def _render_game(self):
        # Parallax Background
        for item in self.parallax_bg:
            x = (item['x'] - self.world_scroll_x * item['speed_mult']) % self.WIDTH
            pygame.draw.rect(self.screen, item['color'], (x, item['y'], item['w'], item['h']))

        # Track Boundaries
        track_top, track_bottom = self.HEIGHT * 0.2, self.HEIGHT * 0.8
        for i in range(5):
            alpha = 150 - i * 30
            pygame.draw.line(self.screen, (*self.COLOR_TRACK_GLOW, alpha), (0, track_top - i), (self.WIDTH, track_top - i), 1)
            pygame.draw.line(self.screen, (*self.COLOR_TRACK_GLOW, alpha), (0, track_bottom + i), (self.WIDTH, track_bottom + i), 1)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, track_top), (self.WIDTH, track_top), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, track_bottom), (self.WIDTH, track_bottom), 2)
        
        # Finish Line
        finish_x = self.TRACK_LENGTH - self.world_scroll_x
        if finish_x < self.WIDTH + 50:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_x, track_top), (finish_x, track_bottom), 5)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['life'] * 0.2))

        # Obstacles
        for obs in self.obstacles:
            x = obs['x'] - self.world_scroll_x
            if -obs['w'] < x < self.WIDTH:
                rect = pygame.Rect(x, obs['y'], obs['w'], obs['h'])
                self._draw_glow_rect(self.screen, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW, rect, 10)

        # Boosts
        for boost in self.boosts:
            if not boost['collected']:
                x = boost['x'] - self.world_scroll_x
                if -20 < x < self.WIDTH:
                    pulse = abs(math.sin(self.steps * 0.2))
                    radius = 10 + pulse * 5
                    self._draw_glow_circle(self.screen, self.COLOR_BOOST, self.COLOR_BOOST_GLOW, (int(x), int(boost['y'])), int(radius), 15)

        # Player
        player_x = self.WIDTH * 0.2
        player_pos = (int(player_x), int(self.player_y))
        if self.invulnerability_timer > 0 and self.steps % 10 < 5:
            pass # Flicker effect
        else:
            self._draw_glow_circle(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, player_pos, 12, 20)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Time
        remaining_time = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_large.render(f"TIME: {remaining_time:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 20))

        # Crashes
        crash_text = self.font_small.render("CRASHES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(crash_text, (20, 60))
        for i in range(self.MAX_CRASHES - 1):
            color = self.COLOR_OBSTACLE if i < self.crashes else (80, 80, 80)
            pygame.draw.circle(self.screen, color, (110 + i * 25, 72), 8)

        # Progress Bar
        progress = min(1.0, self.world_scroll_x / self.TRACK_LENGTH)
        bar_width = self.WIDTH - 40
        pygame.draw.rect(self.screen, (50, 50, 50), (20, self.HEIGHT - 30, bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (20, self.HEIGHT - 30, bar_width * progress, 10))

    def _create_particles(self, pos, count, color, speed_range, life_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_range)
            life = self.np_random.integers(20, life_range)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'color': color
            })
            
    def _draw_glow_circle(self, surface, color, glow_color, center, radius, glow_strength):
        for i in range(glow_strength // 2):
            alpha = 100 - i * (100 / (glow_strength / 2))
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius + i, (*glow_color, alpha))
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)

    def _draw_glow_rect(self, surface, color, glow_color, rect, glow_strength):
        for i in range(glow_strength):
            alpha = 80 - i * (80 / glow_strength)
            glow_rect = rect.inflate(i*2, i*2)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*glow_color, alpha), (0, 0, *glow_rect.size), border_radius=5)
            surface.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=3)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
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

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            env.reset()
            pygame.time.wait(2000) # Pause before restarting

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(env.FPS)

    pygame.quit()