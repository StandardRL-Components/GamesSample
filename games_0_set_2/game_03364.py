
# Generated: 2025-08-27T23:08:45.569330
# Source Brief: brief_03364.md
# Brief Index: 3364

        
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
        "Controls: Press space to jump over obstacles. Timing is everything!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic rhythm runner. Jump over neon obstacles to the beat, complete three stages, and achieve a high score."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GROUND_Y = 340
        self.FPS = 30

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_combo = pygame.font.Font(None, 40)
        
        # Colors
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_GRID = (30, 20, 80)
        self.COLOR_ROAD = (40, 30, 90)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_OBSTACLE = (255, 255, 0)
        self.COLOR_SUCCESS = (0, 255, 150)
        self.COLOR_FAIL = (255, 50, 50)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_BEAT = (255, 0, 150)
        
        # Player properties
        self.PLAYER_X = 100
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 50
        self.JUMP_STRENGTH = -14
        self.GRAVITY = 0.7

        # Game mechanics
        self.MAX_MISSES = 3
        self.STAGE_DURATION_FRAMES = 60 * self.FPS
        
        # Initialize state variables
        self.np_random = None
        self.player_y = 0
        self.player_vy = 0
        self.is_jumping = False
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.missed_obstacles = 0
        self.combo = 1
        self.stage = 1
        self.stage_timer = 0
        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_rate = 0
        self.scroll_speed = 0
        self.beat_pulse = 0
        self.beat_speeds = [0.1, 0.15, 0.2] # Beat speed per stage
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.missed_obstacles = 0
        self.combo = 1
        self.stage = 1
        
        self.player_y = self.GROUND_Y - self.PLAYER_HEIGHT
        self.player_vy = 0
        self.is_jumping = False
        
        self.obstacles.clear()
        self.particles.clear()
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Sets parameters for the current stage."""
        self.stage_timer = self.STAGE_DURATION_FRAMES
        self.scroll_speed = 4.0 + (self.stage - 1) * 1.5
        self.obstacle_spawn_rate = 90 - (self.stage - 1) * 15 # Frames between spawns
        self.obstacle_spawn_timer = self.obstacle_spawn_rate

    def step(self, action):
        reward = 1.0  # Survival reward
        terminated = False
        
        # Unpack factorized action
        space_pressed = action[1] == 1
        
        # --- GAME LOGIC ---
        self.steps += 1
        self.stage_timer -= 1
        self.beat_pulse = (math.sin(self.steps * self.beat_speeds[self.stage - 1]) + 1) / 2.0

        # Handle player input
        if space_pressed and not self.is_jumping:
            self.is_jumping = True
            self.player_vy = self.JUMP_STRENGTH
            # sfx: jump_sound

        # Update player physics
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy
        if self.player_y >= self.GROUND_Y - self.PLAYER_HEIGHT:
            self.player_y = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_vy = 0
            if self.is_jumping:
                self.is_jumping = False
                # sfx: land_sound

        # Update and spawn obstacles
        self._update_obstacles()
        reward += self._check_obstacle_collisions()

        # Update difficulty (more frequent obstacles over time)
        if self.steps > 0 and self.steps % (15 * self.FPS) == 0:
            self.obstacle_spawn_rate = max(30, self.obstacle_spawn_rate * 0.9)

        # Update particles
        self._update_particles()
        
        # Check for stage/game end conditions
        if self.missed_obstacles >= self.MAX_MISSES:
            terminated = True
            # sfx: game_over_sound
        
        if self.stage_timer <= 0 and not terminated:
            if self.stage < 3:
                self.stage += 1
                reward += 50
                self.score += 500
                self._setup_stage()
                self._create_particles(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2, 50, self.COLOR_SUCCESS)
                # sfx: stage_complete_sound
            else:
                reward += 100
                self.score += 1000
                terminated = True
                # sfx: win_sound
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_obstacles(self):
        """Moves obstacles left and spawns new ones."""
        # Move existing obstacles
        for obstacle in self.obstacles:
            obstacle['x'] -= self.scroll_speed
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -obs['width']]

        # Spawn new obstacles
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            height = self.np_random.integers(40, 81)
            width = self.np_random.integers(30, 51)
            self.obstacles.append({
                'x': self.SCREEN_WIDTH,
                'y': self.GROUND_Y - height,
                'width': width,
                'height': height,
                'handled': False
            })
            self.obstacle_spawn_timer = self.np_random.integers(
                int(self.obstacle_spawn_rate * 0.8), 
                int(self.obstacle_spawn_rate * 1.2)
            )
            # sfx: obstacle_spawn_sound

    def _check_obstacle_collisions(self):
        """Checks for player-obstacle interactions and returns reward delta."""
        reward_delta = 0
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        for obstacle in self.obstacles:
            if obstacle['handled']:
                continue

            obs_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle['width'], obstacle['height'])

            # Check for collision (hit)
            if player_rect.colliderect(obs_rect):
                reward_delta -= 2
                self.missed_obstacles += 1
                self.combo = 1
                obstacle['handled'] = True
                self._create_particles(player_rect.centerx, player_rect.centery, 30, self.COLOR_FAIL)
                # sfx: hit_sound

            # Check for successful jump (pass)
            elif obstacle['x'] + obstacle['width'] < self.PLAYER_X:
                reward_delta += 5
                self.score += 10 * self.combo
                self.combo += 1
                obstacle['handled'] = True
                self._create_particles(
                    obstacle['x'] + obstacle['width'] / 2, 
                    obstacle['y'] - 10, 
                    15, 
                    self.COLOR_SUCCESS
                )
                # sfx: success_sound
        
        return reward_delta
        
    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # a little gravity on particles
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Pulsating grid
        pulse_alpha = 50 + self.beat_pulse * 50
        grid_color = (*self.COLOR_GRID, pulse_alpha)
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, grid_color, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, grid_color, (0, i), (self.SCREEN_WIDTH, i))

        # Receding road
        for i in range(10):
            y = self.GROUND_Y + i * i * 0.6
            if y > self.SCREEN_HEIGHT: break
            pygame.draw.line(self.screen, self.COLOR_ROAD, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, 
                (int(obs['x']), int(obs['y']), int(obs['width']), int(obs['height'])))
        
        # Render player with glow
        glow_size = int(self.PLAYER_WIDTH * (1.2 + self.beat_pulse * 0.2))
        glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
        glow_rect.center = (int(self.PLAYER_X + self.PLAYER_WIDTH / 2), int(self.player_y + self.PLAYER_HEIGHT / 2))
        
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surf, glow_rect.topleft)
        
        player_rect = (int(self.PLAYER_X), int(self.player_y), self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (*p['color'], alpha)
            radius = int(max(1, p['lifespan'] / 5))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Stage and Time
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/3", True, self.COLOR_UI)
        self.screen.blit(stage_text, (10, 40))
        time_text = self.font_ui.render(f"TIME: {self.stage_timer // self.FPS}", True, self.COLOR_UI)
        self.screen.blit(time_text, (10, 70))

        # Combo
        if self.combo > 1:
            combo_text = self.font_combo.render(f"x{self.combo}", True, self.COLOR_SUCCESS)
            text_rect = combo_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
            self.screen.blit(combo_text, text_rect)

        # Misses (Hearts)
        for i in range(self.MAX_MISSES):
            is_broken = i < self.missed_obstacles
            self._draw_heart(self.screen, 30 + i * 40, self.SCREEN_HEIGHT - 30, 15, self.COLOR_FAIL, broken=is_broken)
            
        # Beat indicator
        beat_size = int(10 + self.beat_pulse * 15)
        beat_alpha = int(100 + self.beat_pulse * 155)
        pygame.gfxdraw.filled_circle(self.screen, self.PLAYER_X + self.PLAYER_WIDTH // 2, self.GROUND_Y + 20, beat_size, (*self.COLOR_BEAT, beat_alpha))
        pygame.gfxdraw.aacircle(self.screen, self.PLAYER_X + self.PLAYER_WIDTH // 2, self.GROUND_Y + 20, beat_size, (*self.COLOR_BEAT, beat_alpha))

    def _draw_heart(self, surface, x, y, size, color, broken=False):
        """Draws a heart shape, optionally broken."""
        h = size * 0.7
        w = size
        p1 = (x, y + h)
        p2 = (x - w, y - h)
        p3 = (x + w, y - h)
        p4 = (x, y - h * 0.5)

        if not broken:
            pygame.gfxdraw.filled_circle(surface, int(x - w/2), int(y - h/2), int(w/2), color)
            pygame.gfxdraw.filled_circle(surface, int(x + w/2), int(y - h/2), int(w/2), color)
            pygame.gfxdraw.filled_polygon(surface, [(x,y), (x-w, y-h/2), (x+w, y-h/2)], color)
        else:
            offset = 5
            # Left half
            pygame.gfxdraw.filled_circle(surface, int(x - w/2 - offset), int(y - h/2), int(w/2), color)
            pygame.gfxdraw.filled_polygon(surface, [(x - offset, y), (x - w - offset, y-h/2), (x - offset, y-h/2)], color)
            # Right half
            pygame.gfxdraw.filled_circle(surface, int(x + w/2 + offset), int(y - h/2), int(w/2), color)
            pygame.gfxdraw.filled_polygon(surface, [(x + offset, y), (x + offset, y-h/2), (x+w+offset, y-h/2)], color)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "combo": self.combo,
            "misses": self.missed_obstacles,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need to create a window
    pygame.display.set_caption("Rhythm Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        "up": False, "down": False, "left": False, "right": False,
        "space": False, "shift": False
    }

    running = True
    while running:
        # --- Pygame event handling for human play ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: keys_held["up"] = True
                if event.key == pygame.K_DOWN: keys_held["down"] = True
                if event.key == pygame.K_LEFT: keys_held["left"] = True
                if event.key == pygame.K_RIGHT: keys_held["right"] = True
                if event.key == pygame.K_SPACE: keys_held["space"] = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held["shift"] = True
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: keys_held["up"] = False
                if event.key == pygame.K_DOWN: keys_held["down"] = False
                if event.key == pygame.K_LEFT: keys_held["left"] = False
                if event.key == pygame.K_RIGHT: keys_held["right"] = False
                if event.key == pygame.K_SPACE: keys_held["space"] = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held["shift"] = False

        # --- Construct action from human input ---
        movement = 0
        if keys_held["up"]: movement = 1
        elif keys_held["down"]: movement = 2
        elif keys_held["left"]: movement = 3
        elif keys_held["right"]: movement = 4
        
        space = 1 if keys_held["space"] else 0
        shift = 1 if keys_held["shift"] else 0
        
        action = [movement, space, shift]

        # --- Step the environment ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display window ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()