import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:13:21.776927
# Source Brief: brief_02189.md
# Brief Index: 2189
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate shifting quantum platforms by manipulating time crystals and size 
    to reach the end of each timeline.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a series of shifting quantum platforms to reach the exit portal. "
        "Change your size and manipulate time to overcome obstacles."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press Shift to toggle size and Space to use a time crystal."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 19, 26)
    COLOR_PLAYER = (255, 215, 0)
    COLOR_PLAYER_GLOW = (255, 215, 0, 50)
    COLOR_PLATFORM_NORMAL = (0, 191, 255)
    COLOR_PLATFORM_REVERSED = (124, 252, 0)
    COLOR_PLATFORM_ZERO_G = (255, 69, 0)
    COLOR_PLATFORM_FILL = (40, 40, 60, 150)
    COLOR_CRYSTAL = (255, 255, 255)
    COLOR_PORTAL = (148, 0, 211)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE_JUMP = (0, 191, 255)
    COLOR_PARTICLE_LAND = (200, 200, 200)
    COLOR_PARTICLE_CRYSTAL = (255, 255, 255)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Physics
    FPS = 60
    GRAVITY_NORMAL = 0.4
    PLAYER_JUMP_STRENGTH_LARGE = -10
    PLAYER_JUMP_STRENGTH_SMALL = -8
    PLAYER_HORZ_SPEED = 4
    MAX_FALL_SPEED = 12

    # Player
    PLAYER_SIZE_LARGE = (24, 40)
    PLAYER_SIZE_SMALL = (16, 28)

    # Game
    MAX_STEPS = 2000
    NUM_PLATFORMS = 10
    PLATFORM_BASE_SHIFT_TIME = 120 # in steps (2 seconds at 60fps)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_is_large = True
        self.player_on_ground = False
        self.player_current_platform_idx = -1
        
        self.platforms = []
        self.exit_portal_rect = pygame.Rect(0,0,0,0)
        
        self.time_crystals = 0
        self.platform_shift_frequency_factor = 1.0

        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False

        self.camera_x = 0
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_is_large = True
        self.player_on_ground = True
        self.player_current_platform_idx = 0
        
        self.time_crystals = 3
        self.platform_shift_frequency_factor = 1.0

        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False

        self._generate_level()
        
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform['rect'].centerx, start_platform['rect'].top - self.PLAYER_SIZE_LARGE[1])
        self.player_vel = pygame.Vector2(0, 0)
        self.camera_x = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal Movement
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_HORZ_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_HORZ_SPEED
        else:
            self.player_vel.x = 0

        # Jump
        if movement == 1 and self.player_on_ground: # Up
            jump_strength = self.PLAYER_JUMP_STRENGTH_LARGE if self.player_is_large else self.PLAYER_JUMP_STRENGTH_SMALL
            self.player_vel.y = jump_strength
            self.player_on_ground = False
            # Sfx: Jump sound
            self._create_jump_particles(self.player_pos + (0, self._get_player_rect().height))
            reward += 0.1

        # Toggle Size
        if shift_held and not self.last_shift_held:
            self.player_is_large = not self.player_is_large
            # Sfx: Size change sound
        
        # Use Time Crystal
        if space_held and not self.last_space_held and self.time_crystals > 0:
            reward += self._use_time_crystal()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Game Logic & Physics ---
        self._update_platforms()
        self._update_player_physics()
        self._update_particles()
        self._update_camera()

        # --- Reward Calculation ---
        if not self.player_on_ground and self.player_vel.y > 0:
            reward -= 0.01 # Small penalty for falling

        # --- Termination Check ---
        terminated = False
        if self.player_pos.y > self.SCREEN_HEIGHT + 50: # Fell off
            terminated = True
            reward -= 100
            # Sfx: Falling scream/fail sound
        
        if self._get_player_rect().colliderect(self.exit_portal_rect): # Reached exit
            terminated = True
            self.win = True
            reward += 100
            # Sfx: Win fanfare
        
        truncated = False
        if self.steps >= self.MAX_STEPS: # Timeout
            truncated = True
            reward -= 100

        self.game_over = terminated or truncated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_player_rect(self):
        size = self.PLAYER_SIZE_LARGE if self.player_is_large else self.PLAYER_SIZE_SMALL
        return pygame.Rect(self.player_pos.x, self.player_pos.y, size[0], size[1])

    def _update_player_physics(self):
        # Apply gravity
        gravity = self.GRAVITY_NORMAL
        if self.player_current_platform_idx != -1:
            current_platform = self.platforms[self.player_current_platform_idx]
            if self.player_on_ground:
                if current_platform['state'] == 1: # Reversed
                    gravity = -self.GRAVITY_NORMAL
                elif current_platform['state'] == 2: # Zero-G
                    gravity = 0

        self.player_vel.y += gravity
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Move player
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y
        
        player_rect = self._get_player_rect()

        # Boundary checks
        if player_rect.left < 0:
            player_rect.left = 0
            self.player_pos.x = player_rect.x
        # No right boundary, camera follows

        # Collision with platforms
        was_on_ground = self.player_on_ground
        self.player_on_ground = False
        for i, p in enumerate(self.platforms):
            if player_rect.colliderect(p['rect']) and self.player_vel.y >= 0:
                # Check if player was above the platform in the previous frame
                if player_rect.bottom - self.player_vel.y <= p['rect'].top + 1:
                    self.player_pos.y = p['rect'].top - player_rect.height
                    self.player_vel.y = 0
                    self.player_on_ground = True
                    if not was_on_ground: # Just landed
                        # Sfx: Land sound
                        self._create_land_particles(pygame.Vector2(player_rect.midbottom))
                    if i != self.player_current_platform_idx:
                        self.score += 5 # Reward for reaching a new platform
                    self.player_current_platform_idx = i
                    break
        if not self.player_on_ground:
            self.player_current_platform_idx = -1


    def _update_platforms(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.platform_shift_frequency_factor *= 0.95 # Increases frequency by making time shorter

        for p in self.platforms[1:]: # First platform is static
            p['timer'] -= 1
            if p['timer'] <= 0:
                p['state'] = (p['state'] + 1) % 3
                p['timer'] = int(self.PLATFORM_BASE_SHIFT_TIME * self.platform_shift_frequency_factor)
                # Sfx: Platform shift sound

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        # Smooth camera movement
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

    def _use_time_crystal(self):
        self.time_crystals -= 1
        reward = 1
        
        # Find the next platform
        next_platform_idx = self.player_current_platform_idx + 1
        if self.player_current_platform_idx != -1 and next_platform_idx < len(self.platforms):
            next_platform = self.platforms[next_platform_idx]
            # Force it to a favorable (normal) state
            next_platform['state'] = 0 # Normal Gravity
            next_platform['timer'] = int(self.PLATFORM_BASE_SHIFT_TIME * self.platform_shift_frequency_factor)
            # Sfx: Crystal use sound
            self._create_crystal_use_particles(pygame.Vector2(next_platform['rect'].center))
        
        return reward

    def _generate_level(self):
        self.platforms = []
        px, py = 80, self.SCREEN_HEIGHT - 60
        
        # First platform is static and normal
        start_platform = {
            'rect': pygame.Rect(px, py, 150, 20),
            'state': 0, # Normal
            'timer': 99999,
            'color_map': {
                0: self.COLOR_PLATFORM_NORMAL,
                1: self.COLOR_PLATFORM_REVERSED,
                2: self.COLOR_PLATFORM_ZERO_G
            }
        }
        self.platforms.append(start_platform)

        for i in range(1, self.NUM_PLATFORMS):
            px += self.np_random.integers(140, 220)
            py += self.np_random.integers(-80, 80)
            py = np.clip(py, 100, self.SCREEN_HEIGHT - 60)
            width = self.np_random.integers(80, 150)
            
            platform = {
                'rect': pygame.Rect(px, py, width, 20),
                'state': self.np_random.integers(0, 3),
                'timer': self.np_random.integers(60, self.PLATFORM_BASE_SHIFT_TIME + 1),
                'color_map': {
                    0: self.COLOR_PLATFORM_NORMAL,
                    1: self.COLOR_PLATFORM_REVERSED,
                    2: self.COLOR_PLATFORM_ZERO_G
                }
            }
            self.platforms.append(platform)
        
        # Exit Portal on the last platform
        last_platform = self.platforms[-1]
        self.exit_portal_rect = pygame.Rect(last_platform.get('rect').centerx - 20, last_platform.get('rect').top - 60, 40, 60)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_crystals": self.time_crystals,
            "player_is_large": self.player_is_large,
            "current_platform": self.player_current_platform_idx
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_platforms()
        self._render_portal()
        self._render_particles()
        self._render_player()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Faint timeline lines
        for i in range(1, 8):
            y = self.SCREEN_HEIGHT * i / 8
            pygame.draw.line(self.screen, (30, 35, 45), (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_player(self):
        player_rect_world = self._get_player_rect()
        player_rect_screen = player_rect_world.copy()
        player_rect_screen.x -= int(self.camera_x)

        # Glow effect
        glow_radius = int(player_rect_screen.width * 1.5)
        pygame.gfxdraw.filled_circle(
            self.screen,
            player_rect_screen.centerx,
            player_rect_screen.centery,
            glow_radius,
            self.COLOR_PLAYER_GLOW
        )
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_screen, border_radius=4)
    
    def _render_platforms(self):
        for p in self.platforms:
            rect_screen = p['rect'].copy()
            rect_screen.x -= int(self.camera_x)
            
            color = p['color_map'][p['state']]
            
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_FILL, rect_screen, border_radius=4)
            pygame.draw.rect(self.screen, color, rect_screen, 2, border_radius=4)

    def _render_portal(self):
        rect_screen = self.exit_portal_rect.copy()
        rect_screen.x -= int(self.camera_x)

        # Create a swirling effect with random particles
        for _ in range(3):
            p_pos = pygame.Vector2(
                rect_screen.x + self.np_random.uniform(0, rect_screen.width),
                rect_screen.y + self.np_random.uniform(0, rect_screen.height)
            )
            self.particles.append({
                'pos': p_pos,
                'vel': pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)),
                'life': 10,
                'color': self.COLOR_PORTAL,
                'radius': self.np_random.uniform(1, 4)
            })
        
        pygame.draw.rect(self.screen, self.COLOR_PORTAL, rect_screen, 2, border_radius=8)

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 35))

        # Time Crystals
        crystal_text = self.font_ui.render("CRYSTALS:", True, self.COLOR_TEXT)
        self.screen.blit(crystal_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.time_crystals):
            points = [
                (self.SCREEN_WIDTH - 50 + i * 20, 10),
                (self.SCREEN_WIDTH - 45 + i * 20, 17),
                (self.SCREEN_WIDTH - 50 + i * 20, 24),
                (self.SCREEN_WIDTH - 55 + i * 20, 17)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "TIMELINE COLLAPSED"
        if self.win:
            message = "TIMELINE STABILIZED"
            
        text = self.font_game_over.render(message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _create_jump_particles(self, pos):
        for _ in range(10):
            vel = pygame.Vector2(self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(0.5, 2.5))
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': 15, 'color': self.COLOR_PARTICLE_JUMP, 'radius': self.np_random.uniform(1, 3)})

    def _create_land_particles(self, pos):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 0))
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': 20, 'color': self.COLOR_PARTICLE_LAND, 'radius': self.np_random.uniform(1, 4)})

    def _create_crystal_use_particles(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': 25, 'color': self.COLOR_PARTICLE_CRYSTAL, 'radius': self.np_random.uniform(1, 3)})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            pos_screen = p['pos'].copy()
            pos_screen.x -= int(self.camera_x)
            pygame.draw.circle(self.screen, p['color'], pos_screen, max(0, int(p['radius'])))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # The original main block had a different display setup. 
    # To run this file directly, you might need to unset the dummy video driver.
    # For example:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for display
    pygame.display.set_caption("Quantum Jumper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    while running:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1 # up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2 # down (no-op in step)
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3 # left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4 # right
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to reset.")

        clock.tick(GameEnv.FPS)

    env.close()