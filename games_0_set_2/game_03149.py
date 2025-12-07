# Generated: 2025-08-28T07:07:38.853197
# Source Brief: brief_03149.md
# Brief Index: 3149

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space to jump over obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist side-scrolling platformer where precise timing is key to leaping over obstacles and reaching the end."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (100, 149, 237)  # Cornflower Blue
    COLOR_BG_BOTTOM = (75, 0, 130)   # Indigo
    COLOR_GROUND = (60, 60, 60)
    COLOR_PLAYER = (0, 191, 255)     # Deep Sky Blue
    COLOR_OBSTACLE = (255, 69, 0)     # Orangered
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE_LAND = (200, 200, 200)
    COLOR_PARTICLE_CRASH = (255, 100, 100)

    # Physics
    GRAVITY = 0.6
    JUMP_STRENGTH = -11
    SCROLL_SPEED = 4
    
    # Player
    PLAYER_X_POS = 100
    PLAYER_SIZE = 25

    # Level
    LEVEL_LENGTH = 5000
    GROUND_HEIGHT = 50
    MAX_STEPS = 1000
    MAX_HITS = 3

    # Obstacle Generation
    INITIAL_OBSTACLE_PROB = 0.015
    INITIAL_MIN_GAP = 300
    INITIAL_MAX_GAP = 500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        self.np_random = np.random.default_rng()
        
        # Initialize state variables
        self.player_pos_y = 0
        self.player_vel_y = 0
        self.on_ground = True
        self.obstacles = []
        self.particles = []
        self.world_scroll = 0
        self.hit_count = 0
        self.last_obstacle_x = 0
        self.obstacle_prob = self.INITIAL_OBSTACLE_PROB
        self.min_gap = self.INITIAL_MIN_GAP
        self.score = 0
        self.steps = 0
        self.game_over = False

        # Call reset to ensure a valid initial state
        self.reset()
        # Validate implementation after initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hit_count = 0
        self.world_scroll = 0

        self.player_pos_y = self.SCREEN_HEIGHT - self.GROUND_HEIGHT - self.PLAYER_SIZE
        self.player_vel_y = 0
        self.on_ground = True
        
        self.obstacles = []
        self.particles = []
        
        self.obstacle_prob = self.INITIAL_OBSTACLE_PROB
        self.min_gap = self.INITIAL_MIN_GAP
        self.last_obstacle_x = self.SCREEN_WIDTH # Start generating past the screen

        self._generate_initial_obstacles()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        space_pressed = action[1] == 1
        
        reward = 0.1 # Survival reward

        # 1. Handle Input
        if space_pressed and self.on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            # Sound: Jump sfx

        # 2. Update Player Physics
        self._update_player()

        # 3. Update World Scroll
        self.world_scroll += self.SCROLL_SPEED

        # 4. Update Obstacles and check collisions
        collision_penalty = self._update_obstacles()
        reward += collision_penalty

        # 5. Update Particles
        self._update_particles()
        
        # 6. Update Difficulty
        self._update_difficulty()

        # 7. Check Termination Conditions
        terminated = False
        if self.hit_count >= self.MAX_HITS:
            terminated = True
            # Sound: Game over sfx
        elif self.world_scroll >= self.LEVEL_LENGTH:
            terminated = True
            reward += 100.0  # Big win reward
            # Sound: Level complete sfx
        elif self.steps >= self.MAX_STEPS - 1:
            terminated = True

        if terminated:
            self.game_over = True
        
        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self):
        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_pos_y += self.player_vel_y

        ground_level = self.SCREEN_HEIGHT - self.GROUND_HEIGHT - self.PLAYER_SIZE
        
        # Ground collision
        if self.player_pos_y >= ground_level:
            if not self.on_ground: # Just landed
                self._create_particles(self.PLAYER_X_POS + self.PLAYER_SIZE / 2, ground_level + self.PLAYER_SIZE, self.COLOR_PARTICLE_LAND, 15, 3)
                # Sound: Landing sfx
            self.player_pos_y = ground_level
            self.player_vel_y = 0
            self.on_ground = True

    def _update_obstacles(self):
        collision_penalty = 0
        player_rect = pygame.Rect(self.PLAYER_X_POS, self.player_pos_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for obstacle in self.obstacles:
            obstacle_screen_x = obstacle['x'] - self.world_scroll
            obstacle_rect = pygame.Rect(obstacle_screen_x, obstacle['y'], obstacle['w'], obstacle['h'])

            if not obstacle['hit'] and player_rect.colliderect(obstacle_rect):
                self.hit_count += 1
                collision_penalty -= 5.0
                obstacle['hit'] = True
                self._create_particles(player_rect.centerx, player_rect.centery, self.COLOR_PARTICLE_CRASH, 30, 5)
                # Sound: Collision sfx
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] - self.world_scroll + obs['w'] > 0]
        
        # Generate new obstacles
        while self.last_obstacle_x < self.world_scroll + self.SCREEN_WIDTH + self.INITIAL_MAX_GAP:
            self._generate_obstacle()
            
        return collision_penalty

    def _generate_initial_obstacles(self):
        while self.last_obstacle_x < self.SCREEN_WIDTH * 2:
            self._generate_obstacle()

    def _generate_obstacle(self):
        gap = self.np_random.integers(self.min_gap, self.INITIAL_MAX_GAP)
        x_pos = self.last_obstacle_x + gap
        
        width = self.np_random.integers(20, 50)
        height = self.np_random.integers(30, 80)
        y_pos = self.SCREEN_HEIGHT - self.GROUND_HEIGHT - height
        
        self.obstacles.append({'x': x_pos, 'y': y_pos, 'w': width, 'h': height, 'hit': False})
        self.last_obstacle_x = x_pos + width

    def _update_difficulty(self):
        # Increase obstacle frequency every 200 steps
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_prob = min(self.INITIAL_OBSTACLE_PROB * 2, self.obstacle_prob * 1.05)

        # Decrease gap between obstacles every 300 steps
        if self.steps > 0 and self.steps % 300 == 0:
            self.min_gap = max(150, self.min_gap - 2)

    def _create_particles(self, x, y, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': [x, y], 'vel': vel, 'radius': radius, 'color': color, 'life': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Particle gravity
            p['radius'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_particles()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            # Interpolate color from top to bottom
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT, self.SCREEN_WIDTH, self.GROUND_HEIGHT))

        # Obstacles
        for obstacle in self.obstacles:
            obstacle_screen_x = obstacle['x'] - self.world_scroll
            obstacle_rect = (int(obstacle_screen_x), int(obstacle['y']), int(obstacle['w']), int(obstacle['h']))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle_rect)

        # Player
        player_rect = (int(self.PLAYER_X_POS), int(self.player_pos_y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Player glow effect
        glow_size = self.PLAYER_SIZE + 10
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 50), (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surf, (int(self.PLAYER_X_POS - 5), int(self.player_pos_y - 5)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _render_ui(self):
        # Hit counter
        hit_text = f"HITS: {self.hit_count} / {self.MAX_HITS}"
        text_surface = self.font_ui.render(hit_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Progress bar
        progress = min(1.0, self.world_scroll / self.LEVEL_LENGTH)
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 15
        
        # Bar background
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (10, self.SCREEN_HEIGHT - 25, bar_width, bar_height))
        # Bar fill
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.SCREEN_HEIGHT - 25, bar_width * progress, bar_height))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        if self.world_scroll >= self.LEVEL_LENGTH:
            msg = "LEVEL COMPLETE!"
        else:
            msg = "GAME OVER"
            
        text_surface = self.font_game_over.render(msg, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hits": self.hit_count,
            "progress": self.world_scroll / self.LEVEL_LENGTH,
        }

    def close(self):
        pygame.font.quit()
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
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

        # Test jump physics
        self.reset(seed=42)
        self.on_ground = True
        jump_action = [0, 1, 0] # Jump
        self.step(jump_action)
        # The velocity is set to JUMP_STRENGTH, then immediately updated by GRAVITY in the same step.
        # The check must account for this. Using isclose for float comparison.
        expected_vel = self.JUMP_STRENGTH + self.GRAVITY
        assert math.isclose(self.player_vel_y, expected_vel), f"Jump velocity is not correct. Expected ~{expected_vel}, got {self.player_vel_y}"
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Minimalist Platformer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    action = [0, 0, 0] # No-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # Reset action each frame
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS

        if terminated:
            # Wait a bit on the game over screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()