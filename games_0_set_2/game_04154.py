
# Generated: 2025-08-28T01:34:45.539210
# Source Brief: brief_04154.md
# Brief Index: 4154

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Press SPACE to jump. Hold UP while jumping for a higher jump. "
        "Hold DOWN while in the air to fall faster."
    )

    # Short, user-facing description of the game
    game_description = (
        "Hop your neon spaceship through a perilous, procedurally generated "
        "obstacle field. Time your jumps to survive and reach the finish line."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500
        self.LEVEL_LENGTH = 4000

        # Colors
        self.COLOR_BG = (20, 10, 40)
        self.COLOR_STAR = (100, 80, 120)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 50, 50)
        self.COLOR_FINISH = (50, 150, 255)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (0, 0, 0)
        self.COLOR_JUMP_PARTICLE = (200, 255, 200)
        self.COLOR_LAND_PARTICLE = (150, 150, 150)
        self.COLOR_DEATH_PARTICLE = (255, 100, 100)

        # Player Physics
        self.GROUND_LEVEL = self.HEIGHT - 50
        self.GRAVITY = 0.8
        self.FAST_FALL_GRAVITY = 1.5
        self.JUMP_NORMAL = -13
        self.JUMP_HIGH = -16
        self.PLAYER_X_POS = 100
        self.PLAYER_SIZE = 12

        # Obstacle settings
        self.INITIAL_OBSTACLE_SPEED = 5.0
        self.MIN_OBSTACLE_SPACING = 300
        self.MAX_OBSTACLE_SPACING = 550
        self.OBSTACLE_WIDTH = 40
        self.OBSTACLE_HEIGHTS = [60, 90, 120]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup (Headless) ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.distance_traveled = 0.0
        self.camera_x = 0.0

        self.player_y = 0
        self.player_vy = 0
        self.on_ground = True
        self.prev_space_held = False

        self.obstacles = []
        self.obstacle_speed = 0.0
        self.obstacle_freq_multiplier = 1.0
        self.next_obstacle_spawn_dist = 0.0

        self.particles = []
        self.stars = []
        
        # Initialize state for the first time
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.distance_traveled = 0.0
        self.camera_x = 0.0

        self.player_y = self.GROUND_LEVEL
        self.player_vy = 0
        self.on_ground = True
        self.prev_space_held = False

        self.obstacles = []
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_freq_multiplier = 1.0
        self.next_obstacle_spawn_dist = self.WIDTH * 1.5

        self.particles = []
        
        # Generate a static starfield for consistent background
        if not self.stars:
            for _ in range(150):
                self.stars.append({
                    'x': self.np_random.uniform(0, self.WIDTH),
                    'y': self.np_random.uniform(0, self.HEIGHT),
                    'speed_factor': self.np_random.uniform(0.1, 0.5),
                    'size': self.np_random.choice([1, 2])
                })
        
        # Ensure start is safe
        for _ in range(3):
            self._spawn_obstacle()
            self.next_obstacle_spawn_dist += self.np_random.uniform(self.MIN_OBSTACLE_SPACING, self.MAX_OBSTACLE_SPACING)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        if not self.game_over:
            reward += 0.01  # Small reward for surviving a step
            self.score += 0.01

            # --- Update Game Logic ---
            self._update_player(movement, space_held)
            self._update_world()
            self._update_obstacles()
            self._update_particles()

            # --- Difficulty Scaling ---
            if self.steps > 0:
                if self.steps % 50 == 0:
                    self.obstacle_speed += 0.05
                if self.steps % 100 == 0:
                    self.obstacle_freq_multiplier *= 0.99 # Reduces spacing
            
            # --- Collision & Reward Logic ---
            player_rect = pygame.Rect(self.PLAYER_X_POS - self.PLAYER_SIZE, self.player_y - self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
            
            # Obstacle collision and scoring
            for obs in self.obstacles:
                obs_rect = pygame.Rect(obs['x'] - self.camera_x, obs['y'], self.OBSTACLE_WIDTH, obs['h'])
                if not obs['scored'] and obs_rect.right < player_rect.left:
                    obs['scored'] = True
                    reward += 1.0
                    self.score += 1.0
                
                if player_rect.colliderect(obs_rect):
                    self.game_over = True
                    reward = -50.0
                    self.score -= 50.0
                    self._create_explosion(self.PLAYER_X_POS, self.player_y)
                    # Sound effect placeholder: play('explosion')
                    break

        # --- Termination Conditions ---
        terminated = self.game_over
        if not terminated and self.distance_traveled >= self.LEVEL_LENGTH:
            terminated = True
            reward = 100.0
            self.score += 100.0
        
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.steps += 1
        self.prev_space_held = space_held
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_player(self, movement, space_held):
        # Jump on space press (rising edge)
        jump_requested = space_held and not self.prev_space_held
        if jump_requested and self.on_ground:
            self.on_ground = False
            if movement == 1: # Up
                self.player_vy = self.JUMP_HIGH
            else:
                self.player_vy = self.JUMP_NORMAL
            self._create_particles(self.PLAYER_X_POS, self.GROUND_LEVEL + 5, 20, self.COLOR_JUMP_PARTICLE, 'jump')
            # Sound effect placeholder: play('jump')

        # Apply gravity
        if not self.on_ground:
            if movement == 2: # Down (fast fall)
                self.player_vy += self.FAST_FALL_GRAVITY
            else:
                self.player_vy += self.GRAVITY
        
        # Update position and check for ground
        self.player_y += self.player_vy
        if self.player_y >= self.GROUND_LEVEL:
            if not self.on_ground: # Just landed
                self._create_particles(self.PLAYER_X_POS, self.GROUND_LEVEL, 10, self.COLOR_LAND_PARTICLE, 'land')
                # Sound effect placeholder: play('land')
            self.player_y = self.GROUND_LEVEL
            self.player_vy = 0
            self.on_ground = True

    def _update_world(self):
        self.distance_traveled += self.obstacle_speed
        self.camera_x += self.obstacle_speed

    def _update_obstacles(self):
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] - self.camera_x > -self.OBSTACLE_WIDTH]

        # Spawn new obstacles
        last_obstacle_x = self.obstacles[-1]['x'] if self.obstacles else 0
        if last_obstacle_x < self.camera_x + self.WIDTH + 100:
             self._spawn_obstacle()

    def _spawn_obstacle(self):
        last_obstacle_x = self.obstacles[-1]['x'] if self.obstacles else self.camera_x + self.WIDTH
        
        spacing = self.np_random.uniform(self.MIN_OBSTACLE_SPACING, self.MAX_OBSTACLE_SPACING) * self.obstacle_freq_multiplier
        
        new_x = last_obstacle_x + spacing
        height = self.np_random.choice(self.OBSTACLE_HEIGHTS)
        
        self.obstacles.append({
            'x': new_x,
            'y': self.GROUND_LEVEL - height,
            'h': height,
            'scored': False
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['type'] == 'jump' or p['type'] == 'death':
                p['vel'][1] += 0.2 # Gravity on some particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            screen_x = (star['x'] - self.camera_x * star['speed_factor']) % self.WIDTH
            pygame.draw.circle(self.screen, self.COLOR_STAR, (int(screen_x), int(star['y'])), star['size'])

    def _render_game_elements(self):
        # Finish Line
        finish_screen_x = self.LEVEL_LENGTH - self.camera_x
        if finish_screen_x < self.WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_FINISH, (finish_screen_x, 0, 20, self.HEIGHT))
            pygame.gfxdraw.vline(self.screen, int(finish_screen_x + 10), 0, self.HEIGHT, (*self.COLOR_FINISH, 100))

        # Obstacles
        for obs in self.obstacles:
            screen_x = obs['x'] - self.camera_x
            rect = pygame.Rect(screen_x, obs['y'], self.OBSTACLE_WIDTH, obs['h'])
            glow_rect = rect.inflate(8, 8)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=4)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Ground
        pygame.draw.line(self.screen, self.COLOR_STAR, (0, self.GROUND_LEVEL), (self.WIDTH, self.GROUND_LEVEL), 2)
        
        # Player
        if not self.game_over:
            player_pos = (int(self.PLAYER_X_POS), int(self.player_y))
            size = self.PLAYER_SIZE
            points = [
                (player_pos[0], player_pos[1] - size * 1.5),
                (player_pos[0] - size, player_pos[1] + size * 0.5),
                (player_pos[0] + size, player_pos[1] + size * 0.5)
            ]
            # Glow effect
            for i in range(10, 0, -2):
                alpha = 80 - i * 8
                pygame.gfxdraw.aapolygon(self.screen, points, (*self.COLOR_PLAYER_GLOW, alpha))

            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = f"SCORE: {int(self.score)}"
        dist_text = f"DISTANCE: {int(self.distance_traveled)} / {self.LEVEL_LENGTH}"

        self._draw_text(score_text, (10, 10))
        self._draw_text(dist_text, (self.WIDTH - self.font_ui.size(dist_text)[0] - 10, 10))
        
        if self.game_over:
            msg = "CRASHED"
            if self.distance_traveled >= self.LEVEL_LENGTH:
                msg = "FINISH!"
            
            text_surf = self.font_game_over.render(msg, True, self.COLOR_PLAYER)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos):
        shadow_surf = self.font_ui.render(text, True, self.COLOR_UI_SHADOW)
        text_surf = self.font_ui.render(text, True, self.COLOR_UI_TEXT)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_surf, pos)

    def _create_particles(self, x, y, count, color, p_type):
        for _ in range(count):
            if p_type == 'jump':
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(1, 4)]
            elif p_type == 'land':
                angle = self.np_random.uniform(math.pi, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else: # Generic
                vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)]
            
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [x, y], 'vel': vel, 'lifespan': lifespan, 
                'max_lifespan': lifespan, 'radius': self.np_random.uniform(1, 4),
                'color': color, 'type': p_type
            })

    def _create_explosion(self, x, y):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 8)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': [x, y], 'vel': vel, 'lifespan': lifespan, 
                'max_lifespan': lifespan, 'radius': self.np_random.uniform(2, 5),
                'color': self.COLOR_DEATH_PARTICLE, 'type': 'death'
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": self.distance_traveled,
            "game_over": self.game_over,
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


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    env.close()