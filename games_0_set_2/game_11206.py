import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:46:23.127254
# Source Brief: brief_01206.md
# Brief Index: 1206
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A 2D platformer where a shape-shifting blob collects gems.

    The player controls a blob that can be a square, circle, or triangle.
    Each shape has unique movement properties (speed and jump height).
    The goal is to collect 100 gems within a 60-second time limit.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up/jump, 2=down, 3=left, 4=right)
    - action[1]: Shape-shift (1=press, 0=release)
    - action[2]: Unused (shift key)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Rewards:
    - +0.1 for each gem collected.
    - +100 for collecting all 100 gems (victory).
    - 0 terminal reward for running out of time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a shape-shifting blob to collect all the gems before time runs out. "
        "Each shape has unique abilities for platforming challenges."
    )
    user_guide = (
        "Use ←→ arrow keys to move, ↑ to jump, and press space or shift to change your shape."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 19, 25)
    COLOR_PLATFORM = (60, 68, 81)
    COLOR_PLATFORM_OUTLINE = (85, 95, 112)
    COLOR_GEM = (255, 223, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    
    SHAPE_PROPS = {
        0: {"color": (230, 57, 70), "speed": 3, "jump": 10, "size": 30}, # Square (Red)
        1: {"color": (129, 199, 132), "speed": 4, "jump": 12, "size": 32}, # Circle (Green)
        2: {"color": (66, 165, 245), "speed": 5, "jump": 14, "size": 34}, # Triangle (Blue)
    }

    # Physics
    GRAVITY = 0.6
    FRICTION = 0.85
    MAX_FALL_SPEED = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.terminated = False
        
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_shape = 0
        self.is_grounded = False
        self.player_rect = pygame.Rect(0, 0, 0, 0)

        self.platforms = []
        self.gems = []
        self.particles = []
        
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.terminated = False
        
        self.player_shape = 0
        player_size = self.SHAPE_PROPS[self.player_shape]['size']
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT - 60.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_rect = pygame.Rect(0, 0, player_size, player_size)
        self.is_grounded = False
        
        self.particles = []
        self.prev_space_held = False
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        self._handle_input(action)
        self._update_physics_and_collisions()
        reward += self._update_gems()
        self._update_particles()
        
        self.time_remaining -= 1.0 / self.FPS
        
        # Check termination conditions
        win = self.score >= 100
        timeout = self.time_remaining <= 0
        self.terminated = win or timeout
        
        if win:
            reward += 100.0 # Victory bonus
            # sfx: game win
        elif timeout:
            # sfx: game over
            pass
            
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_level(self):
        self.platforms = [
            pygame.Rect(0, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH, 20), # Floor
            pygame.Rect(100, 300, 150, 20),
            pygame.Rect(400, 300, 150, 20),
            pygame.Rect(250, 220, 140, 20),
            pygame.Rect(0, 150, 120, 20),
            pygame.Rect(self.SCREEN_WIDTH - 120, 150, 120, 20),
            pygame.Rect(180, 80, 280, 20)
        ]
        
        self.gems = []
        gem_size = 12
        attempts = 0
        while len(self.gems) < 100 and attempts < 5000:
            attempts += 1
            gem_rect = pygame.Rect(
                self.np_random.integers(20, self.SCREEN_WIDTH - 20),
                self.np_random.integers(40, self.SCREEN_HEIGHT - 40),
                gem_size, gem_size
            )
            if gem_rect.collidelist(self.platforms) == -1 and gem_rect.collidelist(self.gems) == -1:
                self.gems.append(gem_rect)

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        shape_props = self.SHAPE_PROPS[self.player_shape]
        
        # Horizontal Movement
        if movement == 3: # Left
            self.player_vel[0] -= 1.0
        elif movement == 4: # Right
            self.player_vel[0] += 1.0
        
        # Jump
        if movement == 1 and self.is_grounded: # Up
            self.player_vel[1] = -shape_props['jump']
            self.is_grounded = False
            self._create_particles(self.player_rect.midbottom, 15, (200, 200, 200), 2)
            # sfx: jump
        
        # Shape-shifting
        if space_held and not self.prev_space_held:
            self.player_shape = (self.player_shape + 1) % len(self.SHAPE_PROPS)
            self._create_particles(self.player_pos, 30, self.SHAPE_PROPS[self.player_shape]['color'], 4)
            # sfx: shape shift
        self.prev_space_held = space_held

    def _update_physics_and_collisions(self):
        shape_props = self.SHAPE_PROPS[self.player_shape]
        player_size = shape_props['size']
        
        # Apply friction
        self.player_vel[0] *= self.FRICTION
        if abs(self.player_vel[0]) < 0.1: self.player_vel[0] = 0
        
        # Clamp horizontal speed
        self.player_vel[0] = np.clip(self.player_vel[0], -shape_props['speed'], shape_props['speed'])
        
        # Apply gravity
        if not self.is_grounded:
            self.player_vel[1] += self.GRAVITY
            self.player_vel[1] = min(self.player_vel[1], self.MAX_FALL_SPEED)

        # Update position and handle collisions
        self.player_rect = pygame.Rect(self.player_pos[0] - player_size/2, self.player_pos[1] - player_size/2, player_size, player_size)
        
        # Horizontal movement and collision
        self.player_pos[0] += self.player_vel[0]
        self.player_rect.centerx = int(self.player_pos[0])
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel[0] > 0: # Moving right
                    self.player_rect.right = plat.left
                elif self.player_vel[0] < 0: # Moving left
                    self.player_rect.left = plat.right
                self.player_pos[0] = self.player_rect.centerx
                self.player_vel[0] = 0

        # Vertical movement and collision
        self.player_pos[1] += self.player_vel[1]
        self.player_rect.centery = int(self.player_pos[1])
        self.is_grounded = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel[1] > 0: # Moving down
                    self.player_rect.bottom = plat.top
                    self.is_grounded = True
                    if self.player_vel[1] > 5: # Hard landing
                        self._create_particles(self.player_rect.midbottom, 5, (200, 200, 200), 1)
                        # sfx: land
                    self.player_vel[1] = 0
                elif self.player_vel[1] < 0: # Moving up
                    self.player_rect.top = plat.bottom
                    self.player_vel[1] = 0
                self.player_pos[1] = self.player_rect.centery

        # Screen boundaries
        if self.player_rect.left < 0:
            self.player_rect.left = 0
            self.player_pos[0] = self.player_rect.centerx
        if self.player_rect.right > self.SCREEN_WIDTH:
            self.player_rect.right = self.SCREEN_WIDTH
            self.player_pos[0] = self.player_rect.centerx
        if self.player_rect.top < 0:
            self.player_rect.top = 0
            self.player_pos[1] = self.player_rect.centery
            self.player_vel[1] = 0

    def _update_gems(self):
        collected_reward = 0.0
        gems_to_remove = []
        for gem in self.gems:
            if self.player_rect.colliderect(gem):
                gems_to_remove.append(gem)
                self.score += 1
                collected_reward += 0.1
                self._create_particles(gem.center, 10, self.COLOR_GEM, 3)
                # sfx: gem collect
        self.gems = [gem for gem in self.gems if gem not in gems_to_remove]
        return collected_reward

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Particle gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, plat, 2)
            
        # Render Gems
        pulse = math.sin(self.steps * 0.2) * 2
        for gem in self.gems:
            glow_size = gem.width + pulse
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_GEM, 60), glow_surf.get_rect())
            self.screen.blit(glow_surf, (gem.centerx - glow_size/2, gem.centery - glow_size/2))
            pygame.draw.rect(self.screen, self.COLOR_GEM, gem)

        # Render Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30.0))
            color = (*p['color'], alpha)
            size = int(5 * (p['lifetime'] / 30.0))
            if size > 1:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

        # Render Player
        shape_props = self.SHAPE_PROPS[self.player_shape]
        color = shape_props['color']
        size = shape_props['size']
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow effect
        glow_size = int(size * 1.5)
        glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 50), (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (pos[0] - glow_size, pos[1] - glow_size))

        if self.player_shape == 0: # Square
            rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
            pygame.draw.rect(self.screen, color, rect)
        elif self.player_shape == 1: # Circle
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(size/2), color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size/2), color)
        elif self.player_shape == 2: # Triangle
            points = [
                (pos[0], pos[1] - size/2 * 1.1),
                (pos[0] - size/2, pos[1] + size/2 * 0.8),
                (pos[0] + size/2, pos[1] + size/2 * 0.8),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        gem_text = self.font_ui.render(f"GEMS: {self.score}/100", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (10, 10))
        
        time_color = self.COLOR_UI_TEXT if self.time_remaining > 10 else (230, 57, 70)
        time_text = self.font_ui.render(f"TIME: {max(0, self.time_remaining):.1f}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.terminated:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            if self.score >= 100:
                end_text = self.font_game_over.render("VICTORY!", True, self.COLOR_GEM)
            else:
                end_text = self.font_game_over.render("TIME UP", True, (230, 57, 70))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for internal validation and can be removed or modified.
        # It's not part of the standard gym.Env interface.
        print("Running internal validation...")
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test observation space  
            # Reset first to ensure a valid state
            self.reset()
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
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            raise


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display. If you are in a headless environment,
    # you might need to remove os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # or set up a virtual display like Xvfb.
    
    # For local testing, we might want a real display.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # The main game window is created here, after re-enabling the video driver.
    pygame.display.set_caption("Shape Shifter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    running = True
    total_reward = 0
    
    print("--- Manual Control ---")
    print("A/D or Left/Right: Move")
    print("W or Up Arrow: Jump")
    print("Left Shift or Space: Transform Shape")
    print("Q: Quit")
    
    while running:
        movement_action = 0 # 0=none
        transform_action = 0 # 0=released
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement_action = 3 # left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement_action = 4 # right
            
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement_action = 1 # up/jump
        
        # Map both space and shift to the transform action
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_SPACE]:
            transform_action = 1

        action = [movement_action, transform_action, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        env.clock.tick(env.FPS)

    env.close()