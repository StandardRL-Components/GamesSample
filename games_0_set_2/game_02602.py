
# Generated: 2025-08-27T20:51:56.797011
# Source Brief: brief_02602.md
# Brief Index: 2602

        
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
        "Controls: Use ←→ to run. Press ↑ or Space to jump. Avoid the ghosts and furniture."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling horror game. Escape the haunted house by jumping over obstacles to reach the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_FLOOR = (40, 35, 45)
    COLOR_WALL = (25, 22, 38)
    COLOR_PLAYER = (255, 60, 60)
    COLOR_PLAYER_TRAIL = (200, 50, 50)
    COLOR_GHOST = (220, 220, 255)
    COLOR_FURNITURE = (60, 45, 40)
    COLOR_TORCH_BASE = (255, 180, 20)
    COLOR_TORCH_LIGHT = (255, 150, 0)
    COLOR_EXIT = (100, 255, 100)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_SHADOW = (10, 12, 22)

    # Physics
    PLAYER_SPEED = 4.5
    GRAVITY = 0.6
    JUMP_STRENGTH = -12
    
    # Game settings
    MAX_STEPS = 1000
    INITIAL_LIVES = 5
    LEVEL_WIDTH_FACTOR = 15
    FLOOR_HEIGHT = 50

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # RNG
        self._np_random = None

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = None
        self.player_trail = None
        
        self.obstacles = None
        self.exit_rect = None
        self.torches = None
        
        self.camera_x = None
        self.lives = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.won = None
        
        self.particles = None
        self.obstacle_speed_modifier = None
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._np_random is None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Player state
        self.player_pos = [100, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - 30]
        self.player_vel = [0, 0]
        self.player_rect = pygame.Rect(0, 0, 20, 30)
        self.on_ground = False
        self.player_trail = []
        
        # Game state
        self.camera_x = 0
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.obstacle_speed_modifier = 1.0
        
        # World generation
        self._generate_level()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.obstacles = []
        self.torches = []
        level_width = self.SCREEN_WIDTH * self.LEVEL_WIDTH_FACTOR
        
        # Place obstacles
        current_x = self.SCREEN_WIDTH * 0.8
        while current_x < level_width - self.SCREEN_WIDTH:
            obstacle_type = self._np_random.choice(['furniture', 'ghost'])
            
            if obstacle_type == 'furniture':
                width = self._np_random.integers(40, 80)
                height = self._np_random.integers(30, 60)
                rect = pygame.Rect(current_x, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - height, width, height)
                self.obstacles.append({'type': 'furniture', 'rect': rect})
            
            elif obstacle_type == 'ghost':
                size = self._np_random.integers(25, 40)
                base_y = self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - self._np_random.integers(50, 150)
                rect = pygame.Rect(current_x, base_y, size, size)
                self.obstacles.append({
                    'type': 'ghost', 
                    'rect': rect, 
                    'base_y': base_y,
                    'amplitude': self._np_random.integers(20, 60),
                    'frequency': self._np_random.uniform(0.02, 0.05)
                })

            current_x += self._np_random.integers(200, 400)

        # Place torches
        for i in range(level_width // 300):
            self.torches.append((i * 300 + self._np_random.integers(-50, 50), self.SCREEN_HEIGHT // 2 + self._np_random.integers(-40, 40)))

        # Place exit
        self.exit_rect = pygame.Rect(level_width - 100, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - 100, 50, 100)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        
        player_moved_right = False
        prev_player_x = self.player_pos[0]

        # Horizontal movement
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
            player_moved_right = True

        # Jumping
        if (movement == 1 or space_held) and self.on_ground:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound
            self._create_particles(self.player_rect.midbottom, 15, (200, 200, 200), -math.pi/2, math.pi/4)

        # --- Physics and World Update ---
        self.steps += 1

        # Update player physics
        self.player_vel[1] += self.GRAVITY
        self.player_pos[1] += self.player_vel[1]
        
        # Floor collision
        if self.player_pos[1] > self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - self.player_rect.height:
            self.player_pos[1] = self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - self.player_rect.height
            if not self.on_ground:
                # sfx: land_sound
                self._create_particles(self.player_rect.midbottom, 8, (150, 150, 150), -math.pi/2, math.pi/6)
            self.on_ground = True
            self.player_vel[1] = 0

        # Update camera to follow player
        self.camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 3
        
        # Keep player within horizontal bounds of the level
        level_width = self.SCREEN_WIDTH * self.LEVEL_WIDTH_FACTOR
        self.player_pos[0] = max(self.player_rect.width / 2, self.player_pos[0])
        self.player_pos[0] = min(level_width - self.player_rect.width / 2, self.player_pos[0])
        
        # Update player rect
        self.player_rect.center = (self.player_pos[0], self.player_pos[1] + self.player_rect.height / 2)
        
        # Update player trail
        if self.steps % 2 == 0:
            self.player_trail.append(self.player_rect.copy())
            if len(self.player_trail) > 5:
                self.player_trail.pop(0)

        # Update obstacles
        for obs in self.obstacles:
            if obs['type'] == 'ghost':
                obs['rect'].y = obs['base_y'] + obs['amplitude'] * math.sin(obs['frequency'] * self.steps)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # Difficulty scaling
        if self.steps > 0 and self.steps % 50 == 0:
            self.obstacle_speed_modifier += 0.05
            # This is a placeholder; currently obstacles don't move horizontally.

        # --- Collisions and Rewards ---
        reward = 0
        
        # Reward for moving right
        if self.player_pos[0] > prev_player_x:
            reward += 0.1

        # Check obstacle collisions
        collided = False
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['rect']):
                collided = True
                break
        
        if collided:
            self.lives -= 1
            reward -= 5
            # sfx: hit_sound
            self._create_particles(self.player_rect.center, 30, self.COLOR_PLAYER, 0, math.pi*2)
            # Reset player position to avoid continuous collision
            self.player_pos = [self.player_pos[0] - 50, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - 30]
            self.player_vel = [0, 0]
        
        # Check for win condition
        if self.player_rect.colliderect(self.exit_rect):
            self.won = True
            self.game_over = True
            reward += 50
            # sfx: win_sound

        # --- Termination ---
        terminated = self._check_termination()
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.lives <= 0:
            self.game_over = True
            # sfx: game_over_sound
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_pos": self.player_pos.copy(),
            "won": self.won
        }

    def _render_game(self):
        # --- Background Elements ---
        # Wall
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT))
        
        # Flickering Torches
        for tx, ty in self.torches:
            screen_x = int(tx - self.camera_x)
            if -50 < screen_x < self.SCREEN_WIDTH + 50:
                light_radius = 40 + math.sin(self.steps * 0.1 + tx) * 5 + self._np_random.uniform(-2, 2)
                light_color = (*self.COLOR_TORCH_LIGHT, 60)
                
                surf = pygame.Surface((light_radius*2, light_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, light_color, (light_radius, light_radius), light_radius)
                self.screen.blit(surf, (screen_x - light_radius, ty - light_radius), special_flags=pygame.BLEND_RGBA_ADD)
                
                pygame.gfxdraw.filled_circle(self.screen, screen_x, ty, 5, self.COLOR_TORCH_BASE)

        # --- Foreground Elements ---
        # Floor
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (0, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT, self.SCREEN_WIDTH, self.FLOOR_HEIGHT))

        # --- Dynamic Elements ---
        # Exit door
        exit_screen_rect = self.exit_rect.move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_screen_rect)
        pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_EXIT), exit_screen_rect, 3)

        # Obstacles
        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].move(-self.camera_x, 0)
            if obs_screen_rect.right > 0 and obs_screen_rect.left < self.SCREEN_WIDTH:
                if obs['type'] == 'furniture':
                    pygame.draw.rect(self.screen, self.COLOR_FURNITURE, obs_screen_rect)
                    pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_FURNITURE), obs_screen_rect, 2)
                elif obs['type'] == 'ghost':
                    # Shadow
                    shadow_rect = pygame.Rect(obs_screen_rect.x, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - 5, obs_screen_rect.width, 10)
                    pygame.gfxdraw.filled_ellipse(self.screen, shadow_rect.centerx, shadow_rect.centery, shadow_rect.width // 2, shadow_rect.height // 2, (*self.COLOR_SHADOW, 150))
                    # Ghost body
                    pygame.gfxdraw.filled_circle(self.screen, obs_screen_rect.centerx, obs_screen_rect.centery, obs_screen_rect.width // 2, (*self.COLOR_GHOST, 120))
                    pygame.gfxdraw.aacircle(self.screen, obs_screen_rect.centerx, obs_screen_rect.centery, obs_screen_rect.width // 2, (*self.COLOR_GHOST, 180))

        # Player Trail
        for i, trail_rect in enumerate(self.player_trail):
            alpha = (i + 1) * 30
            trail_surf = pygame.Surface(trail_rect.size, pygame.SRCALPHA)
            trail_surf.fill((*self.COLOR_PLAYER_TRAIL, alpha))
            screen_trail_rect = trail_rect.move(-self.camera_x, 0)
            self.screen.blit(trail_surf, screen_trail_rect.topleft)

        # Player Shadow
        shadow_y = self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - 5
        shadow_width = int(self.player_rect.width * max(0.5, 1 - (shadow_y - self.player_rect.bottom) / 200))
        shadow_height = int(10 * max(0.5, 1 - (shadow_y - self.player_rect.bottom) / 200))
        player_screen_x = int(self.player_rect.centerx - self.camera_x)
        pygame.gfxdraw.filled_ellipse(self.screen, player_screen_x, shadow_y, shadow_width, shadow_height, (*self.COLOR_SHADOW, 150))

        # Player
        player_screen_rect = self.player_rect.move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_screen_rect)
        
        # Particles
        self._render_particles()

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(surf, (int(p['pos'][0] - self.camera_x - p['size']), int(p['pos'][1] - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Lives display
        lives_text = "LIVES: " + "♥ " * self.lives
        text_surface = self.font_ui.render(lives_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WON!" if self.won else "GAME OVER"
            color = self.COLOR_EXIT if self.won else self.COLOR_PLAYER
            
            text_surface = self.font_game_over.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, count, color, angle_start, angle_spread):
        for _ in range(count):
            angle = angle_start + self._np_random.uniform(-angle_spread, angle_spread)
            speed = self._np_random.uniform(1, 4)
            life = self._np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self._np_random.integers(2, 4)
            })

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Haunted House Escape")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    done = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Human Input Handling ---
        movement = 0 # no-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        # Down action has no effect, so it's not mapped
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose
        frame_to_display = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_display)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(30)
        
        if done:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Info: {info}")
            # Wait for reset
            pass

    env.close()